import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
#  1. Generate 2D logistic data
# =========================

def generate_2d_logistic_data(n_samples=500, seed=0):
    """
    Simple 2D logistic regression data:
      x ~ N(0, I_2)
      y ~ Bernoulli( sigma(w_true^T x) )
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, 2).astype(np.float32)

    # True parameter (only used for data generation)
    w_true = np.array([2.0, -3.0], dtype=np.float32)
    logits = X @ w_true
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs).astype(np.float32)

    return X, y, w_true


# =========================
#  2. Tsallis f_alpha (α=1 -> KL)
# =========================

def f_alpha(x, alpha=1.0):
    """
    Tsallis function family:
      alpha != 1: (x^alpha - alpha x + (alpha - 1)) / (alpha - 1)
      alpha = 1 : x log x - x + 1
    x: np.ndarray or torch.Tensor
    """
    eps = 1e-12
    if isinstance(x, np.ndarray):
        x_safe = np.clip(x, eps, None)
        if alpha == 1.0:
            return x_safe * np.log(x_safe) - x_safe + 1.0
        else:
            return (np.power(x_safe, alpha) - alpha * x_safe + (alpha - 1.0)) / (alpha - 1.0)
    else:
        x_safe = torch.clamp(x, min=eps)
        if alpha == 1.0:
            return x_safe * torch.log(x_safe) - x_safe + 1.0
        else:
            return (x_safe**alpha - alpha * x_safe + (alpha - 1.0)) / (alpha - 1.0)


# =========================
#  3. 2D Gaussian density (numpy)
# =========================

def gaussian_pdf_2d_np(theta, mean, cov):
    """
    2D Gaussian density:
      theta: (..., 2)
      mean:  (2,)
      cov:   (2,2)
    Returns an array of shape (...,) with density values.
    """
    theta = np.asarray(theta)
    mean = np.asarray(mean)
    cov = np.asarray(cov)

    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    dim = 2

    diff = theta - mean            # (..., 2)
    quad = np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
    norm_const = 1.0 / ((2.0 * np.pi)**(dim / 2.0) * np.sqrt(det_cov))
    return norm_const * np.exp(-0.5 * quad)


# =========================
#  4. Variational logistic regression: q(θ) = N(μ, diag(σ^2))
# =========================

class VariationalLogisticRegression(nn.Module):
    def __init__(self, prior_std=1.0):
        super().__init__()
        # Variational parameters: mu, log_sigma ∈ R^2
        self.mu = nn.Parameter(torch.zeros(2))
        self.log_sigma = nn.Parameter(torch.zeros(2))

        # Prior p(θ) = N(0, prior_std^2 I)
        self.prior_std = prior_std
        self.prior_var = prior_std**2

    def sample_theta(self, num_samples=1):
        """
        Sample θ from q(θ):
          θ = μ + σ * ε
        """
        eps = torch.randn(num_samples, 2)
        sigma = torch.exp(self.log_sigma)
        theta = self.mu + sigma * eps
        return theta  # (S, 2)

    def kl_q_prior(self):
        """
        KL(q||p_prior) for diagonal Gaussians:
          q = N(μ, diag(σ^2))
          p = N(0, prior_std^2 I)
        """
        sigma2 = torch.exp(2.0 * self.log_sigma)  # σ^2
        mu2 = self.mu**2
        prior_var = self.prior_var

        # KL = 0.5 * sum( (σ^2 + μ^2)/prior_var - 1 + log(prior_var) - log(σ^2) )
        prior_var_t = self.mu.new_tensor(prior_var)
        kl = 0.5 * torch.sum(
            (sigma2 + mu2) / prior_var_t
            - 1.0
            + torch.log(prior_var_t)
            - torch.log(sigma2)
        )
        return kl

    def forward(self, X_np, y_np, num_mc_samples=1):
        """
        Compute -ELBO (used as the loss):
          -ELBO ≈ E_q[ NLL ] + KL(q||prior)/N
        Here the ELBO is the global quantity corresponding to KL(q||p_post).
        """
        X = torch.from_numpy(X_np)  # (N,2)
        y = torch.from_numpy(y_np)  # (N,)
        N = X.shape[0]

        theta_samples = self.sample_theta(num_mc_samples)  # (S,2)

        nlls = []
        for s in range(num_mc_samples):
            theta_s = theta_samples[s]           # (2,)
            logits = X @ theta_s                # (N,)
            nll = nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="mean"
            )
            nlls.append(nll)

        nll_mean = torch.stack(nlls).mean()
        kl = self.kl_q_prior()
        loss = nll_mean + kl / N  # negative ELBO

        return loss, nll_mean.item(), kl.item()


def train_variational_lr(X, y, prior_std=1.0,
                         num_epochs=2000, lr=1e-2,
                         num_mc_samples=1, print_every=200):
    """
    Train variational logistic regression using the ELBO,
    obtaining q(θ) = N(μ, diag(σ^2)).
    """
    model = VariationalLogisticRegression(prior_std=prior_std)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        loss, nll, kl = model(X, y, num_mc_samples=num_mc_samples)
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0:
            print(f"Epoch {epoch:4d} | -ELBO(loss) = {loss.item():.4f} | "
                  f"NLL = {nll:.4f} | KL(q||prior) = {kl:.4f}")

    return model


# =========================
#  5. Grid approximation of true posterior p(θ|X,y)
# =========================

def compute_true_posterior_grid(X, y, prior_std=1.0,
                                theta1_min=-4.0, theta1_max=4.0,
                                theta2_min=-4.0, theta2_max=4.0,
                                num_points=200):
    """
    On a 2D grid, compute the unnormalized posterior via Bayes' rule:
      p(θ|X,y) ∝ p(θ) p(y|X,θ)
    Then normalize numerically so that
      ∑ p(θ_ij) Δθ1 Δθ2 ≈ 1.
    """
    theta1 = np.linspace(theta1_min, theta1_max, num_points)
    theta2 = np.linspace(theta2_min, theta2_max, num_points)
    T1, T2 = np.meshgrid(theta1, theta2)      # (ny, nx)
    grid = np.stack([T1, T2], axis=-1)        # (ny, nx, 2)
    grid_flat = grid.reshape(-1, 2)           # (M, 2)
    M = grid_flat.shape[0]

    # Prior p(θ) = N(0, prior_std^2 I)
    mean_p = np.array([0.0, 0.0], dtype=np.float32)
    cov_p = np.diag([prior_std**2, prior_std**2])
    log_prior = np.log(gaussian_pdf_2d_np(grid_flat, mean_p, cov_p) + 1e-30)  # (M,)

    # Likelihood log p(y|X,θ)
    # logits: (M,N) = grid_flat @ X^T
    logits = grid_flat @ X.T  # (M,N)
    # log σ(z) = -log(1+exp(-z)), log(1-σ(z)) = -log(1+exp(z))
    log_sigmoid = -np.log1p(np.exp(-logits))
    log1m_sigmoid = -np.log1p(np.exp(logits))

    y_row = y.reshape(1, -1)  # (1,N)
    log_lik = (y_row * log_sigmoid + (1.0 - y_row) * log1m_sigmoid).sum(axis=1)  # (M,)

    # Unnormalized log posterior
    log_post_unnorm = log_prior + log_lik  # (M,)

    # Numerically stable normalization
    max_log = np.max(log_post_unnorm)
    post_unnorm = np.exp(log_post_unnorm - max_log)  # (M,)
    post_grid = post_unnorm.reshape(T1.shape)        # (ny,nx)

    # Normalize so that the integral ≈ 1
    dtheta1 = theta1[1] - theta1[0]
    dtheta2 = theta2[1] - theta2[0]
    Z = np.sum(post_grid) * dtheta1 * dtheta2
    p_post_grid = post_grid / (Z + 1e-30)

    return theta1, theta2, p_post_grid


# =========================
#  6. q(θ) and forward / reverse budgets
# =========================

def compute_q_and_budgets_grid(model,
                               p_post_grid,
                               theta1, theta2,
                               alpha=1.0):
    """
    Given:
      - model: provides μ, σ of the variational q
      - p_post_grid: grid of true posterior density p(θ|X,y)
      - theta1, theta2: grid coordinates

    Compute:
      - q_grid(θ)
      - forward ratio  R^p_q = p / q
      - reverse ratio  R^q_p = q / p
      - forward budget Γ_fwd(θ)  = f_alpha(R^p_q)   (for p||q)
      - reverse budget Γ_rev(θ)  = f_alpha(R^q_p)   (for q||p)
    """
    # Parameters of q
    mu = model.mu.detach().numpy()                         # (2,)
    sigma = torch.exp(model.log_sigma).detach().numpy()    # (2,)
    cov_q = np.diag(sigma**2)

    # Grid
    T1, T2 = np.meshgrid(theta1, theta2)
    grid = np.stack([T1, T2], axis=-1)  # (ny,nx,2)

    # Variational density q(θ)
    q_grid = gaussian_pdf_2d_np(grid, mu, cov_q)

    eps = 1e-12
    q_safe = np.clip(q_grid, eps, None)
    p_safe = np.clip(p_post_grid, eps, None)

    # Forward & reverse ratios
    R_p_q = p_safe / q_safe  # R^p_q
    R_q_p = q_safe / p_safe  # R^q_p

    # Budgets
    Gamma_fwd = f_alpha(R_p_q, alpha=alpha)  # forward budget, for p||q
    Gamma_rev = f_alpha(R_q_p, alpha=alpha)  # reverse budget, for q||p

    return q_grid, R_p_q, R_q_p, Gamma_fwd, Gamma_rev


# =========================
#  7. Approximate KL and budget integrals
# =========================

def approximate_kl_and_budgets(theta1, theta2,
                               p_grid, q_grid,
                               Gamma_fwd, Gamma_rev):
    """
    Approximate via grid:
      KL(q||p) = ∫ q log(q/p) dθ
      KL(p||q) = ∫ p log(p/q) dθ

    Also compute:
      ∫ Γ_fwd d q   (theoretically ≈ KL(p||q), α=1)
      ∫ Γ_rev d p   (theoretically ≈ KL(q||p), α=1)
    """
    dtheta1 = theta1[1] - theta1[0]
    dtheta2 = theta2[1] - theta2[0]

    eps = 1e-12
    p_safe = np.clip(p_grid, eps, None)
    q_safe = np.clip(q_grid, eps, None)

    # KL(q||p)
    integrand_qp = q_safe * (np.log(q_safe) - np.log(p_safe))
    KL_q_p = np.sum(integrand_qp) * dtheta1 * dtheta2

    # KL(p||q)
    integrand_pq = p_safe * (np.log(p_safe) - np.log(q_safe))
    KL_p_q = np.sum(integrand_pq) * dtheta1 * dtheta2

    # ∫ Γ_fwd d q
    integral_Gamma_fwd_dq = np.sum(Gamma_fwd * q_safe) * dtheta1 * dtheta2

    # ∫ Γ_rev d p
    integral_Gamma_rev_dp = np.sum(Gamma_rev * p_safe) * dtheta1 * dtheta2

    return KL_q_p, KL_p_q, integral_Gamma_fwd_dq, integral_Gamma_rev_dp


# =========================
#  8. Plot budget heatmap
# =========================

def plot_budget_heatmap(theta1, theta2, Gamma,
                        vmax_percentile=99.5,
                        title=r"Budget $\Gamma(\theta)$"):
    T1, T2 = np.meshgrid(theta1, theta2)

    # Clip extremely high values to avoid a few points blowing up the colormap
    vmax = np.percentile(Gamma, vmax_percentile)
    Gamma_clip = np.clip(Gamma, 0.0, vmax)

    # Custom colormap: small values = light green, large values = dark red
    custom_cmap = LinearSegmentedColormap.from_list(
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

    plt.figure(figsize=(6, 5))
    im = plt.pcolormesh(T1, T2, Gamma_clip, shading="auto", cmap=custom_cmap)
    plt.colorbar(im, label=r"$\Gamma(\theta)$ (clipped)")
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =========================
#  9. Main pipeline
# =========================

def main():
    # 1. Generate data
    X, y, w_true = generate_2d_logistic_data(n_samples=500, seed=0)
    print("True parameter w_true:", w_true)

    # 2. Train variational posterior q(θ) with ELBO
    prior_std = 1.0
    model = train_variational_lr(
        X, y,
        prior_std=prior_std,
        num_epochs=2000,      # can be increased to check more thorough convergence
        lr=1e-2,
        num_mc_samples=1,
        print_every=200
    )

    print("Learned mu:", model.mu.detach().numpy())
    print("Learned log_sigma:", model.log_sigma.detach().numpy())

    # 3. Grid approximation of true posterior p(θ|X,y)
    theta1_min, theta1_max = -6.0, 6.0
    theta2_min, theta2_max = -6.0, 6.0
    num_points = 200

    theta1, theta2, p_post_grid = compute_true_posterior_grid(
        X, y,
        prior_std=prior_std,
        theta1_min=theta1_min, theta1_max=theta1_max,
        theta2_min=theta2_min, theta2_max=theta2_max,
        num_points=num_points
    )

    # 4. Compute q(θ) and forward / reverse budgets
    alpha = 1.0  # start with KL case
    q_grid, R_p_q, R_q_p, Gamma_fwd, Gamma_rev = compute_q_and_budgets_grid(
        model,
        p_post_grid,
        theta1, theta2,
        alpha=alpha
    )

    # 5. Approximate global KL and budget integrals
    KL_q_p, KL_p_q, int_G_fwd_dq, int_G_rev_dp = approximate_kl_and_budgets(
        theta1, theta2,
        p_post_grid, q_grid,
        Gamma_fwd, Gamma_rev
    )

    print("\n=== Global divergences & budget integrals (grid approximation) ===")
    print(f"KL(q || p_post)           ≈ {KL_q_p:.4f}")
    print(f"KL(p_post || q)           ≈ {KL_p_q:.4f}")
    print(f"∫ Gamma_fwd d q  (p||q)   ≈ {int_G_fwd_dq:.4f}")
    print(f"∫ Gamma_rev d p  (q||p)   ≈ {int_G_rev_dp:.4f}")
    print("===================================================================")

    # 6. Plot forward / reverse budget heatmaps
    plot_budget_heatmap(
        theta1, theta2, Gamma_fwd,
        vmax_percentile=99.0,
        title=rf"Forward budget $\Gamma_1(\theta)$ for $p\|\;q$"
    )

    plot_budget_heatmap(
        theta1, theta2, Gamma_rev,
        vmax_percentile=99.0,
        title=rf"Reverse budget $\Gamma_1^*(\theta)$ for $q\|\;p$"
    )


if __name__ == "__main__":
    main()
