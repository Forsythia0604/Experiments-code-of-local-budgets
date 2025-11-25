import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import gammaln
from torch.distributions import StudentT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. Generate 2D logistic data
# =========================

def generate_2d_logistic_data(n_samples=5000, seed=0):
    """
    x ~ N(0, I_2)
    y ~ Bernoulli( sigma(w_true^T x) )
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 2).astype(np.float32)

    # true parameter (for data generation only)
    w_true = np.array([2.0, -3.0], dtype=np.float32)
    logits = X @ w_true
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(np.float32)

    return X, y, w_true


# =========================
# 2. Tsallis function f_alpha (α=1 => KL)
# =========================

def f_alpha(x, alpha=1.0):
    """
    Tsallis function:
      alpha != 1: (x^alpha - alpha x + (alpha - 1)) / (alpha - 1)
      alpha = 1 : x log x - x + 1  (KL case)
    x: numpy array, x >= 0
    """
    eps = 1e-16
    x_safe = np.clip(x, eps, None)
    if alpha == 1.0:
        return x_safe * np.log(x_safe) - x_safe + 1.0
    else:
        return (np.power(x_safe, alpha) - alpha * x_safe + (alpha - 1.0)) / (alpha - 1.0)


# =========================
# 3. 2D Gaussian & Student-t pdfs (numpy)
# =========================

def gaussian_pdf_2d_np(theta, mean, cov):
    """
    2D Gaussian density:
      theta: (..., 2)
      mean:  (2,)
      cov:   (2,2)
    returns (...,) densities
    """
    theta = np.asarray(theta)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    dim = 2

    diff = theta - mean  # (..., 2)
    quad = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    norm_const = 1.0 / ((2.0 * np.pi) ** (dim / 2.0) * np.sqrt(det_cov))
    return norm_const * np.exp(-0.5 * quad)


def student_t_pdf_2d_np(theta, mean, cov, df):
    """
    2D Student-t density with df, location mean, scale matrix cov (Σ):

      p(x) = Γ((ν+d)/2) / [ Γ(ν/2) (νπ)^{d/2} |Σ|^{1/2} ]
             * ( 1 + (1/ν)(x-μ)^T Σ^{-1} (x-μ) )^{-(ν+d)/2}
    """
    theta = np.asarray(theta)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    d = 2
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)

    diff = theta - mean           # (..., 2)
    quad = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)

    df = float(df)
    log_norm = (
        gammaln((df + d) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * (d * np.log(df * np.pi) + np.log(det_cov))
    )
    log_kernel = -0.5 * (df + d) * np.log1p(quad / df)
    log_pdf = log_norm + log_kernel

    return np.exp(log_pdf)


# =========================
# 4. Torch logpdf helpers (Gaussian / Student-t)
# =========================

def gaussian_logpdf_indep_torch(theta, loc, log_sigma):
    """
    Independent 2D Gaussian log-density in torch:
      q(theta) = ∏ N(theta_i | loc_i, sigma_i^2)
    theta: (S,2), loc: (2,), log_sigma: (2,)
    returns: (S,) log-densities
    """
    theta = theta.float()
    device = theta.device

    loc = loc.to(device)
    log_sigma = log_sigma.to(device)

    var = torch.exp(2.0 * log_sigma)  # (2,)
    diff = theta - loc
    log2pi = torch.log(torch.tensor(2.0 * np.pi, device=device))

    # per-dim contribution: (diff^2 / var) + log(2π) + log(var)
    term = (diff ** 2) / var + log2pi + torch.log(var)
    return -0.5 * torch.sum(term, dim=1)


def student_t_logpdf_indep_torch(theta, loc, log_scale, df):
    """
    Independent 2D Student-t log-density in torch (same df for both dims):

      t_df(x | μ, s) = Γ((ν+1)/2) / [ Γ(ν/2) sqrt(νπ) s ]
                       * (1 + ((x-μ)^2)/(ν s^2))^{-(ν+1)/2}

    For 2D independent:
      q(theta) = ∏ t_df(theta_i).
    """
    theta = theta.float()
    device = theta.device

    loc = loc.to(device)
    log_scale = log_scale.to(device)
    df_t = torch.tensor(df, dtype=torch.float32, device=device)

    # standardise
    scale = torch.exp(log_scale)
    z = (theta - loc) / scale  # (S,2)
    quad = z ** 2

    log_pi = torch.log(torch.tensor(np.pi, device=device))
    # constant per-dimension
    c = (
        torch.lgamma((df_t + 1.0) / 2.0)
        - torch.lgamma(df_t / 2.0)
        - 0.5 * (torch.log(df_t) + log_pi)
    )
    # per-dim log-density
    log_pdf_dim = c - log_scale - 0.5 * (df_t + 1.0) * torch.log1p(quad / df_t)
    # sum over dimensions
    return torch.sum(log_pdf_dim, dim=1)


# =========================
# 5. Compute true posterior p_post on grid (for a given prior)
# =========================

def compute_true_posterior_grid(
    X, y,
    prior_type,
    prior_std=1.0,
    prior_df=3.0,
    theta1_min=-6.0, theta1_max=6.0,
    theta2_min=-6.0, theta2_max=6.0,
    num_points=201
):
    """
    Compute p_post(θ | X,y) on a 2D grid for a given prior type:

      p_post(θ) ∝ p_prior(θ) * p(y|X,θ)

    prior_type: 'gauss' or 'student'
    """
    theta1 = np.linspace(theta1_min, theta1_max, num_points)
    theta2 = np.linspace(theta2_min, theta2_max, num_points)
    T1, T2 = np.meshgrid(theta1, theta2)
    grid = np.stack([T1, T2], axis=-1)   # (ny,nx,2)
    grid_flat = grid.reshape(-1, 2)      # (M,2)

    # prior log-density
    if prior_type == "gauss":
        mean_p = np.array([0.0, 0.0], dtype=np.float64)
        cov_p = (prior_std ** 2) * np.eye(2, dtype=np.float64)
        log_prior = np.log(gaussian_pdf_2d_np(grid_flat, mean_p, cov_p) + 1e-30)
    elif prior_type == "student":
        mean_p = np.array([0.0, 0.0], dtype=np.float64)
        cov_p = (prior_std ** 2) * np.eye(2, dtype=np.float64)
        log_prior = np.log(student_t_pdf_2d_np(grid_flat, mean_p, cov_p, df=prior_df) + 1e-30)
    else:
        raise ValueError("Unknown prior_type: {}".format(prior_type))

    # logistic log-likelihood
    # logits: (M,N) = grid_flat @ X^T
    logits = grid_flat @ X.T  # (M,N)
    # log σ(z) = -log(1+exp(-z)), log(1-σ(z)) = -log(1+exp(z))
    log_sigmoid = -np.log1p(np.exp(-logits))
    log1m_sigmoid = -np.log1p(np.exp(logits))
    y_row = y.reshape(1, -1)  # (1,N)
    log_lik = (y_row * log_sigmoid + (1.0 - y_row) * log1m_sigmoid).sum(axis=1)  # (M,)

    log_post_unnorm = log_prior + log_lik  # (M,)

    # stabilise & normalise
    max_log = np.max(log_post_unnorm)
    post_unnorm = np.exp(log_post_unnorm - max_log)
    post_grid = post_unnorm.reshape(T1.shape)

    dtheta1 = theta1[1] - theta1[0]
    dtheta2 = theta2[1] - theta2[0]
    Z = np.sum(post_grid) * dtheta1 * dtheta2
    p_post_grid = post_grid / (Z + 1e-30)

    return theta1, theta2, p_post_grid


# =========================
# 6. VI models
#    Case A: prior Student-t, q Gaussian
#    Case B: prior Gaussian, q Student-t
# =========================

class VIGaussianQ_StudentPrior(nn.Module):
    """
    Case A:
      prior p_prior(θ): Student-t (df_prior, scale_prior)
      q(θ): diagonal Gaussian N(mu, diag(sigma^2))
    """

    def __init__(self, prior_df=3.0, prior_scale=1.0):
        super().__init__()
        # variational Gaussian q parameters
        self.mu = nn.Parameter(torch.zeros(2))
        self.log_sigma = nn.Parameter(torch.zeros(2))

        # Student-t prior parameters
        self.prior_df = float(prior_df)
        self.prior_loc = torch.zeros(2)
        self.prior_log_scale = torch.full((2,), np.log(prior_scale), dtype=torch.float32)

    def sample_theta(self, num_samples=1):
        eps = torch.randn(num_samples, 2)
        sigma = torch.exp(self.log_sigma)
        theta = self.mu + sigma * eps
        return theta  # (S,2)

    def forward(self, X_np, y_np, num_mc_samples=5):
        """
        -ELBO = E_q[NLL] + KL(q||prior)/N
        """
        X = torch.from_numpy(X_np).float()
        y = torch.from_numpy(y_np).float()
        N = X.shape[0]

        theta_samples = self.sample_theta(num_mc_samples)  # (S,2)

        # NLL under logistic model
        nlls = []
        for s in range(num_mc_samples):
            theta_s = theta_samples[s]   # (2,)
            logits = X @ theta_s         # (N,)
            nll = nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="mean"
            )
            nlls.append(nll)
        nll_mean = torch.stack(nlls).mean()

        # MC estimate of KL(q||prior) = E_q[log q - log p_prior]
        log_q = gaussian_logpdf_indep_torch(theta_samples, self.mu, self.log_sigma)
        log_p = student_t_logpdf_indep_torch(
            theta_samples, self.prior_loc, self.prior_log_scale, self.prior_df
        )
        kl_mc = torch.mean(log_q - log_p)

        loss = nll_mean + kl_mc / N
        return loss, nll_mean.item(), kl_mc.item()


class VIStudentQ_GaussianPrior(nn.Module):
    """
    Case B:
      prior p_prior(θ): Gaussian N(0, prior_std^2 I)
      q(θ): diagonal Student-t with df_q, location mu, scale diag(exp(log_scale))
    """

    def __init__(self, prior_std=1.0, df_q=3.0):
        super().__init__()
        # variational Student-t q parameters
        self.mu = nn.Parameter(torch.zeros(2))
        self.log_scale = nn.Parameter(torch.zeros(2))
        self.df_q = float(df_q)

        # Gaussian prior parameters
        self.prior_loc = torch.zeros(2)
        self.prior_log_sigma = torch.full((2,), np.log(prior_std), dtype=torch.float32)

    def sample_theta(self, num_samples=1):
        """
        Sample from q(θ): independent Student-t per dimension, then scale+shift.
        """
        df_t = torch.tensor(self.df_q)
        dist = StudentT(df_t)
        t_std = dist.sample((num_samples, 2))  # (S,2)
        scale = torch.exp(self.log_scale)
        theta = self.mu + scale * t_std
        return theta

    def forward(self, X_np, y_np, num_mc_samples=5):
        """
        -ELBO = E_q[NLL] + KL(q||prior)/N
        """
        X = torch.from_numpy(X_np).float()
        y = torch.from_numpy(y_np).float()
        N = X.shape[0]

        theta_samples = self.sample_theta(num_mc_samples)  # (S,2)

        # NLL under logistic model
        nlls = []
        for s in range(num_mc_samples):
            theta_s = theta_samples[s]
            logits = X @ theta_s
            nll = nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="mean"
            )
            nlls.append(nll)
        nll_mean = torch.stack(nlls).mean()

        # MC estimate KL(q||prior) = E_q[log q - log p_prior]
        log_q = student_t_logpdf_indep_torch(
            theta_samples, self.mu, self.log_scale, self.df_q
        )
        log_p = gaussian_logpdf_indep_torch(
            theta_samples, self.prior_loc, self.prior_log_sigma
        )
        kl_mc = torch.mean(log_q - log_p)

        loss = nll_mean + kl_mc / N
        return loss, nll_mean.item(), kl_mc.item()


# =========================
# 7. Training wrappers for Case A / Case B
# =========================

def train_vi_case_A(
    X, y,
    prior_df=3.0, prior_scale=1.0,
    num_epochs=2000, lr=1e-2,
    num_mc_samples=5, print_every=200
):
    model = VIGaussianQ_StudentPrior(prior_df=prior_df, prior_scale=prior_scale)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss, nll, kl = model(X, y, num_mc_samples=num_mc_samples)
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            print(f"[Case A] Epoch {epoch:4d} | loss = {loss.item():.4f} | "
                  f"NLL = {nll:.4f} | KL(q||prior) = {kl:.4f}")

    return model


def train_vi_case_B(
    X, y,
    prior_std=1.0, df_q=3.0,
    num_epochs=2000, lr=1e-2,
    num_mc_samples=5, print_every=200
):
    model = VIStudentQ_GaussianPrior(prior_std=prior_std, df_q=df_q)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss, nll, kl = model(X, y, num_mc_samples=num_mc_samples)
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            print(f"[Case B] Epoch {epoch:4d} | loss = {loss.item():.4f} | "
                  f"NLL = {nll:.4f} | KL(q||prior) = {kl:.4f}")

    return model


# =========================
# 8. Budgets from p_post grid and q grid
# =========================

def compute_budgets_from_grids(theta1, theta2, p_grid, q_grid, alpha=1.0):
    """
    Given p_post(θ) grid and q(θ) grid, compute

      R^p_q = p/q, R^q_p = q/p
      Γ_fwd(θ) = f_alpha(R^p_q)   (for p||q)
      Γ_rev(θ) = f_alpha(R^q_p)   (for q||p)
    """
    eps = 1e-16
    p_safe = np.clip(p_grid, eps, None)
    q_safe = np.clip(q_grid, eps, None)

    R_p_q = p_safe / q_safe
    R_q_p = q_safe / p_safe

    Gamma_fwd = f_alpha(R_p_q, alpha=alpha)
    Gamma_rev = f_alpha(R_q_p, alpha=alpha)
    return Gamma_fwd, Gamma_rev


def approximate_kl_and_budgets(theta1, theta2,
                               p_grid, q_grid,
                               Gamma_fwd, Gamma_rev):
    """
    Approximate:
      KL(q||p) = ∫ q log(q/p) dθ
      KL(p||q) = ∫ p log(p/q) dθ
      ∫ Γ_fwd d q   (≈ KL(p||q) for α=1)
      ∫ Γ_rev d p   (≈ KL(q||p) for α=1)
    by rectangle rule on the grid.
    """
    dtheta1 = theta1[1] - theta1[0]
    dtheta2 = theta2[1] - theta2[0]

    eps = 1e-16
    p_safe = np.clip(p_grid, eps, None)
    q_safe = np.clip(q_grid, eps, None)

    # KL(q||p)
    integrand_qp = q_safe * (np.log(q_safe) - np.log(p_safe))
    KL_q_p = np.sum(integrand_qp) * dtheta1 * dtheta2

    # KL(p||q)
    integrand_pq = p_safe * (np.log(p_safe) - np.log(q_safe))
    KL_p_q = np.sum(integrand_pq) * dtheta1 * dtheta2

    # ∫ Γ_fwd d q
    int_G_fwd_dq = np.sum(Gamma_fwd * q_safe) * dtheta1 * dtheta2

    # ∫ Γ_rev d p
    int_G_rev_dp = np.sum(Gamma_rev * p_safe) * dtheta1 * dtheta2

    return KL_q_p, KL_p_q, int_G_fwd_dq, int_G_rev_dp


# =========================
# 9. Plotting helper
# =========================

def get_green_to_red_cmap():
    """
    Custom colormap: small values = light green, large values = deep red.
    """
    return LinearSegmentedColormap.from_list(
        "green_to_red",
        [
            "#e0ffe0",  # very light green
            "#b8ff98",  # light green
            "#ffff80",  # yellow
            "#ffb366",  # orange
            "#ff6666",  # red
            "#800000",  # dark red
        ]
    )


def plot_budget_heatmap(theta1, theta2, Gamma,
                        vmax_percentile=99.5,
                        title=r"Budget $\Gamma(\theta)$"):
    T1, T2 = np.meshgrid(theta1, theta2)
    vmax = np.percentile(Gamma, vmax_percentile)
    Gamma_clip = np.clip(Gamma, 0.0, vmax)

    cmap = get_green_to_red_cmap()

    plt.figure(figsize=(6, 5))
    im = plt.pcolormesh(T1, T2, Gamma_clip, shading="auto", cmap=cmap)
    plt.colorbar(im, label=r"$\Gamma(\theta)$ (clipped)")
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================
# 10. Main: two cases
# =========================

def main():
    # 1. generate data
    X, y, w_true = generate_2d_logistic_data(n_samples=500, seed=0)
    print("True parameter w_true:", w_true)

    # grid settings for posterior / budgets
    theta1_min, theta1_max = -6.0, 6.0
    theta2_min, theta2_max = -6.0, 6.0
    num_points = 201
    alpha = 1.0  # KL case

    # ----------------------------------------------------
    # Case A: prior Student-t, variational q = Gaussian
    # ----------------------------------------------------
    prior_df_A = 3.0
    prior_scale_A = 1.0

    print("\n=== Training Case A: prior Student-t, variational Gaussian q ===")
    model_A = train_vi_case_A(
        X, y,
        prior_df=prior_df_A,
        prior_scale=prior_scale_A,
        num_epochs=2000,
        lr=1e-2,
        num_mc_samples=5,
        print_every=200
    )

    mu_A = model_A.mu.detach().numpy().astype(np.float64)
    sigma_A = torch.exp(model_A.log_sigma).detach().numpy().astype(np.float64)
    print("Case A: learned q Gaussian mu:", mu_A)
    print("Case A: learned q Gaussian sigma:", sigma_A)

    # true posterior p_post under Student-t prior
    theta1_A, theta2_A, p_post_A = compute_true_posterior_grid(
        X, y,
        prior_type="student",
        prior_std=prior_scale_A,
        prior_df=prior_df_A,
        theta1_min=theta1_min, theta1_max=theta1_max,
        theta2_min=theta2_min, theta2_max=theta2_max,
        num_points=num_points
    )

    # q Gaussian density on the same grid
    T1_A, T2_A = np.meshgrid(theta1_A, theta2_A)
    grid_A = np.stack([T1_A, T2_A], axis=-1)
    cov_q_A = np.diag(sigma_A**2)
    q_grid_A = gaussian_pdf_2d_np(grid_A, mu_A, cov_q_A)

    # budgets between p_post_A and q_A
    Gamma_fwd_A, Gamma_rev_A = compute_budgets_from_grids(
        theta1_A, theta2_A,
        p_post_A, q_grid_A,
        alpha=alpha
    )

    KL_q_p_A, KL_p_q_A, int_G_fwd_dq_A, int_G_rev_dp_A = approximate_kl_and_budgets(
        theta1_A, theta2_A,
        p_post_A, q_grid_A,
        Gamma_fwd_A, Gamma_rev_A
    )

    print("\n=== Case A: q vs p_post (Student-t prior) ===")
    print(f"KL(q || p_post)           ≈ {KL_q_p_A:.4f}")
    print(f"KL(p_post || q)           ≈ {KL_p_q_A:.4f}")
    print(f"∫ Gamma_fwd d q  (p||q)   ≈ {int_G_fwd_dq_A:.4f}")
    print(f"∫ Gamma_rev d p  (q||p)   ≈ {int_G_rev_dp_A:.4f}")
    print("=================================================\n")

    plot_budget_heatmap(
        theta1_A, theta2_A, Gamma_fwd_A,
        vmax_percentile=99.5,
        title=r"Case A: Forward budget $\Gamma_1(\theta)$ for $p_{\text{post}}\|\;q$"
    )
    plot_budget_heatmap(
        theta1_A, theta2_A, Gamma_rev_A,
        vmax_percentile=99.5,
        title=r"Case A: Reverse budget $\Gamma_1^*(\theta)$ for $q\|\;p_{\text{post}}$"
    )

    # ----------------------------------------------------
    # Case B: prior Gaussian, variational q = Student-t
    # ----------------------------------------------------
    prior_std_B = 1.0
    df_q_B = 3.0

    print("\n=== Training Case B: prior Gaussian, variational Student-t q ===")
    model_B = train_vi_case_B(
        X, y,
        prior_std=prior_std_B,
        df_q=df_q_B,
        num_epochs=2000,
        lr=1e-2,
        num_mc_samples=5,
        print_every=200
    )

    mu_B = model_B.mu.detach().numpy().astype(np.float64)
    scale_B = torch.exp(model_B.log_scale).detach().numpy().astype(np.float64)
    print("Case B: learned q Student-t mu:", mu_B)
    print("Case B: learned q Student-t scale:", scale_B)

    # true posterior p_post under Gaussian prior
    theta1_B, theta2_B, p_post_B = compute_true_posterior_grid(
        X, y,
        prior_type="gauss",
        prior_std=prior_std_B,
        prior_df=1.0,  # unused for gauss
        theta1_min=theta1_min, theta1_max=theta1_max,
        theta2_min=theta2_min, theta2_max=theta2_max,
        num_points=num_points
    )

    # q Student-t density on the same grid
    T1_B, T2_B = np.meshgrid(theta1_B, theta2_B)
    grid_B = np.stack([T1_B, T2_B], axis=-1)
    cov_q_B = np.diag(scale_B**2)
    q_grid_B = student_t_pdf_2d_np(grid_B, mu_B, cov_q_B, df=df_q_B)

    # budgets between p_post_B and q_B
    Gamma_fwd_B, Gamma_rev_B = compute_budgets_from_grids(
        theta1_B, theta2_B,
        p_post_B, q_grid_B,
        alpha=alpha
    )

    KL_q_p_B, KL_p_q_B, int_G_fwd_dq_B, int_G_rev_dp_B = approximate_kl_and_budgets(
        theta1_B, theta2_B,
        p_post_B, q_grid_B,
        Gamma_fwd_B, Gamma_rev_B
    )

    print("\n=== Case B: q vs p_post (Gaussian prior) ===")
    print(f"KL(q || p_post)           ≈ {KL_q_p_B:.4f}")
    print(f"KL(p_post || q)           ≈ {KL_p_q_B:.4f}")
    print(f"∫ Gamma_fwd d q  (p||q)   ≈ {int_G_fwd_dq_B:.4f}")
    print(f"∫ Gamma_rev d p  (q||p)   ≈ {int_G_rev_dp_B:.4f}")
    print("=================================================\n")

    plot_budget_heatmap(
        theta1_B, theta2_B, Gamma_fwd_B,
        vmax_percentile=99.5,
        title=r"Case B: Forward budget $\Gamma_1(\theta)$ for $p_{\text{post}}\|\;q$"
    )
    plot_budget_heatmap(
        theta1_B, theta2_B, Gamma_rev_B,
        vmax_percentile=99.5,
        title=r"Case B: Reverse budget $\Gamma_1^*(\theta)$ for $q\|\;p_{\text{post}}$"
    )


if __name__ == "__main__":
    main()
