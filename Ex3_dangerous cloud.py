import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# =========================
#  0. Utility functions
# =========================

def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


def f_alpha(x, alpha=1.0):
    """
    Tsallis function family:
      alpha != 1: (x^alpha - alpha x + (alpha - 1)) / (alpha - 1)
      alpha = 1 : x log x - x + 1
    Supports both numpy and torch.
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


def log_gaussian_diag_pdf_torch(theta, mu, log_sigma):
    """
    log N(theta; mu, diag(exp(2*log_sigma)))  (torch version)
      theta: (S,P)
      mu:    (P,)
      log_sigma: (P,)
    """
    var_inv = torch.exp(-2.0 * log_sigma)      # 1 / σ^2
    diff = theta - mu                          # (S,P)
    quad = (diff**2 * var_inv).sum(dim=1)      # (S,)
    D = theta.shape[1]
    log2pi = np.log(2.0 * np.pi)
    log_det = (2.0 * log_sigma).sum()
    return -0.5 * (quad + log_det + D * log2pi)


# =========================
#  1. 1D regression data
# =========================

def generate_regression_data(n_samples=120, noise_std=0.1, seed=0):
    """
    1D toy regression:
      x ~ Uniform(-1,1)
      y = sin(3x) + ε, ε ~ N(0, noise_std^2)
    """
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 1)).astype(np.float32)
    y_clean = np.sin(3.0 * x)
    y = y_clean + noise_std * rng.randn(n_samples, 1).astype(np.float32)
    return x, y


# =========================
#  2. Larger BNN: 1 -> 50 -> 50 -> 1
# =========================

class LargeBNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim1=50, hidden_dim2=50, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        out = self.fc3(h2)
        return out


def flatten_net_params(net):
    return torch.cat([p.detach().flatten() for p in net.parameters()])


def get_param_shapes(net):
    return [p.shape for p in net.parameters()]


def unflatten_theta(theta, shapes):
    """
    theta: (..., P)
    shapes: list of torch.Size
    Returns a list[tensor] with the same order as net.parameters().
    """
    params = []
    idx = 0
    for shape in shapes:
        num = int(np.prod(shape))
        param = theta[..., idx:idx + num].view(*shape)
        params.append(param)
        idx += num
    return params


def bnn_forward_from_flat(theta, x, shapes):
    """
    Forward pass using flattened parameters theta:
      theta: (P,) or (S,P)
      x: (N,1) torch tensor
      shapes: [W1,b1,W2,b2,W3,b3]
    """
    if theta.dim() == 1:
        params = unflatten_theta(theta, shapes)
        W1, b1, W2, b2, W3, b3 = params
        h1 = torch.tanh(x @ W1.t() + b1)
        h2 = torch.tanh(h1 @ W2.t() + b2)
        out = h2 @ W3.t() + b3
        return out
    else:
        S = theta.shape[0]
        outs = []
        for s in range(S):
            params = unflatten_theta(theta[s], shapes)
            W1, b1, W2, b2, W3, b3 = params
            h1 = torch.tanh(x @ W1.t() + b1)
            h2 = torch.tanh(h1 @ W2.t() + b2)
            out = h2 @ W3.t() + b3
            outs.append(out)
        return torch.stack(outs, dim=0)  # (S,N,1)


# =========================
#  3. SGLD approximation of true posterior (strong L1)
# =========================

def run_sgld(net, X_train, Y_train,
             num_steps=20000,
             burn_in=5000,
             thinning=20,
             lr=5e-5,
             noise_std=0.1,
             lambda_l1_true=10.0):
    """
    SGLD:
      U(θ) = NLL(θ) + λ_true * ||θ||_1
    Returns sgld_samples: (M,P)
    """
    net.train()
    X = torch.from_numpy(X_train)
    Y = torch.from_numpy(Y_train)
    N = X.shape[0]

    params = list(net.parameters())
    shapes = get_param_shapes(net)

    samples = []
    step = 0

    for t in range(num_steps):
        step += 1
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

        pred = net(X)                          # (N,1)
        mse = ((pred - Y)**2).mean()
        nll = mse / (2.0 * noise_std**2) * N   # -loglik ~ N * NLL

        # Strong Laplace prior: λ_true is very large
        l1 = 0.0
        for p in params:
            l1 = l1 + torch.abs(p).sum()
        prior_term = lambda_l1_true * l1

        U = nll + prior_term
        U.backward()

        with torch.no_grad():
            for p in params:
                grad = p.grad
                noise = torch.randn_like(p)
                p.add_(-0.5 * lr * grad + torch.sqrt(torch.tensor(lr)) * noise)

        if step > burn_in and (step - burn_in) % thinning == 0:
            flat = flatten_net_params(net)
            samples.append(flat.cpu().numpy())

        if step % 2000 == 0:
            print(f"[SGLD] step {step}/{num_steps}, U = {U.item():.4f}")

    samples = np.stack(samples, axis=0)
    print(f"[SGLD] collected {samples.shape[0]} samples after burn-in.")
    return samples, shapes


# =========================
#  4. Mean-field Gaussian VI (weak L1)
# =========================

class MeanFieldVI(nn.Module):
    def __init__(self, P):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(P))
        self.log_sigma = nn.Parameter(torch.zeros(P))

    def sample_theta(self, num_samples=1):
        eps = torch.randn(num_samples, self.mu.shape[0])
        sigma = torch.exp(self.log_sigma)
        return self.mu + sigma * eps


def train_vi(X_train, Y_train, shapes,
             noise_std=0.1,
             lambda_l1_vi=0.5,   # Note: deliberately weaker than the true prior
             num_epochs=4000,
             lr=1e-3,
             num_mc_samples=5,
             print_every=200):
    """
    VI: loss ≈ E_q[NLL] + KL(q||prior_vi)/N
    prior_vi is a Laplace prior that is weaker than the true prior, to create mismatch.
    """
    X = torch.from_numpy(X_train)
    Y = torch.from_numpy(Y_train)
    N = X.shape[0]

    P = sum(int(np.prod(s)) for s in shapes)
    vi = MeanFieldVI(P)
    opt = optim.Adam(vi.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        opt.zero_grad()
        theta_samples = vi.sample_theta(num_mc_samples)   # (S,P)

        preds = bnn_forward_from_flat(theta_samples, X, shapes)  # (S,N,1)
        mse = ((preds - Y)**2).mean(dim=(1, 2))
        nll = mse / (2.0 * noise_std**2) * N
        nll_mean = nll.mean()

        log_q = log_gaussian_diag_pdf_torch(theta_samples,
                                            vi.mu, vi.log_sigma)

        # VI uses a significantly weaker L1 regularization (λ_vi << λ_true) to create mismatch
        l1 = torch.abs(theta_samples).sum(dim=1)
        log_p_prior_vi = -lambda_l1_vi * l1

        kl_est = (log_q - log_p_prior_vi).mean()      # up to const
        loss = nll_mean + kl_est / N

        loss.backward()
        opt.step()

        if epoch % print_every == 0:
            print(f"[VI] epoch {epoch}/{num_epochs} | loss = {loss.item():.4f} "
                  f"| NLL = {nll_mean.item():.4f} | MC-KL/N ≈ {kl_est.item()/N:.4f}")

    return vi


# =========================
#  5. Estimating marginal density at zero & budgets & local mass
# =========================

def gaussian_1d_pdf(x, mean, std):
    return 1.0 / (np.sqrt(2.0 * np.pi) * std) * np.exp(-0.5 * ((x - mean) / std)**2)


def kde_density_at_zero(samples_1d, bandwidth=None):
    """
    Simple Gaussian KDE estimate of the density at x=0:
      p_hat(0) = (1/(n h)) Σ φ( -x_i / h )
    """
    samples_1d = np.asarray(samples_1d)
    n = samples_1d.shape[0]
    if n == 0:
        return 0.0
    std = samples_1d.std(ddof=1) + 1e-8
    if bandwidth is None:
        h = 1.06 * std * n**(-1.0 / 5.0)
        h = max(h, 1e-3)
    else:
        h = bandwidth
    z = samples_1d / h
    density = (np.exp(-0.5 * z**2).sum()) / (n * h * np.sqrt(2.0 * np.pi))
    return float(density)


def compute_zero_budgets(sgld_samples, vi_mu, vi_log_sigma, alpha=1.0):
    """
    sgld_samples: (M,P) np.array
    vi_mu, vi_log_sigma: (P,) np.array
    Returns:
      Gamma_fwd(j) = f_1( p_j(0) / q_j(0) )
      Gamma_rev(j) = f_1( q_j(0) / p_j(0) )
    """
    P = vi_mu.shape[0]
    vi_sigma = np.exp(vi_log_sigma)
    Gamma_fwd = np.zeros(P)
    Gamma_rev = np.zeros(P)

    for j in range(P):
        samples_j = sgld_samples[:, j]
        p0 = kde_density_at_zero(samples_j)
        q0 = gaussian_1d_pdf(0.0, vi_mu[j], vi_sigma[j])

        if p0 <= 0.0 or q0 <= 0.0:
            Gamma_fwd[j] = 0.0
            Gamma_rev[j] = 0.0
            continue

        R_p_q = p0 / q0
        R_q_p = q0 / p0

        Gamma_fwd[j] = f_alpha(np.array([R_p_q]), alpha=alpha)[0]
        Gamma_rev[j] = f_alpha(np.array([R_q_p]), alpha=alpha)[0]

    return Gamma_fwd, Gamma_rev


def compute_local_mass_zero(sgld_samples, eps=0.05):
    """
    For each weight j, compute the empirical posterior mass in a neighborhood of 0:
      m_j = P(|theta_j| <= eps) ≈ proportion of samples with |theta_j| <= eps
    sgld_samples: (M, P) numpy array
    eps: neighborhood radius
    Returns:
      local_mass: (P,) numpy array
    """
    sgld_samples = np.asarray(sgld_samples)
    abs_vals = np.abs(sgld_samples)
    local_mass = (abs_vals <= eps).mean(axis=0)
    return local_mass


# =========================
#  6. Visualization
# =========================

def plot_sorted_budgets(Gamma, title):
    P = Gamma.shape[0]
    sorted_vals = np.sort(Gamma)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(P), sorted_vals, marker="o", linestyle="none", markersize=2)
    plt.xlabel("Weight index (sorted)")
    plt.ylabel(r"$\Gamma_1(0)$")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_hist_budgets(Gamma, title):
    plt.figure(figsize=(6, 4))
    plt.hist(Gamma, bins=40, log=True)
    plt.xlabel(r"$\Gamma_1(0)$")
    plt.ylabel("count (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_danger_cloud(Gamma_fwd, local_mass,
                      gamma_percentile=90.0,
                      mass_threshold=0.1,
                      title=r"Dangerous weights cloud at zero"):
    """
    "Dangerous weights cloud" plot:
      - x-axis: local posterior mass near zero m_j
      - y-axis: forward budget Γ_1,j(0)

    Highlight weights such that:
      Γ_1,j(0) is in the top gamma_percentile%,
      and m_j >= mass_threshold
      (i.e., non-negligible mass + large mismatch).
    """
    P = Gamma_fwd.shape[0]
    gamma_thr = np.percentile(Gamma_fwd, gamma_percentile)
    danger_mask = (Gamma_fwd >= gamma_thr) & (local_mass >= mass_threshold)

    print(f"[Danger cloud] gamma threshold (top {gamma_percentile}%) = {gamma_thr:.4e}")
    print(f"[Danger cloud] mass threshold = {mass_threshold:.4f}")
    print(f"[Danger cloud] #dangerous weights = {danger_mask.sum()} / {P}")

    plt.figure(figsize=(6, 4))
    # background points
    plt.scatter(local_mass, Gamma_fwd,
                alpha=0.3, s=10, label="all weights")
    # dangerous points
    if danger_mask.any():
        plt.scatter(local_mass[danger_mask], Gamma_fwd[danger_mask],
                    alpha=0.9, s=25, marker="o", label="dangerous weights")

    plt.xlabel(r"local posterior mass near zero: $m_j = P(|\theta_j|< \varepsilon)$")
    plt.ylabel(r"forward budget at zero: $\Gamma_{1,j}(0)$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return danger_mask


# =========================
#  7. Main pipeline
# =========================

def main():
    set_seed(0)

    # ---- Data ----
    X_train, Y_train = generate_regression_data(
        n_samples=120, noise_std=0.1, seed=0
    )
    print("X_train shape:", X_train.shape, "Y_train shape:", Y_train.shape)

    # ---- SGLD: true posterior with strong L1 prior ----
    net = LargeBNN(input_dim=1, hidden_dim1=50, hidden_dim2=50, output_dim=1)
    lambda_l1_true = 10.0   # Very strong sparsity prior in the true model
    sgld_samples, shapes = run_sgld(
        net,
        X_train, Y_train,
        num_steps=20000,
        burn_in=5000,
        thinning=20,
        lr=5e-5,
        noise_std=0.1,
        lambda_l1_true=lambda_l1_true
    )
    print("SGLD samples shape:", sgld_samples.shape)

    # ---- VI: mean-field Gaussian, weaker L1 prior ----
    lambda_l1_vi = 0.5      # VI uses a clearly weaker L1 prior (to create mismatch)
    vi = train_vi(
        X_train, Y_train,
        shapes,
        noise_std=0.1,
        lambda_l1_vi=lambda_l1_vi,
        num_epochs=4000,
        lr=1e-3,
        num_mc_samples=5,
        print_every=200
    )

    vi_mu = vi.mu.detach().numpy()
    vi_log_sigma = vi.log_sigma.detach().numpy()

    # ---- Compute per-weight budgets at zero ----
    Gamma_fwd, Gamma_rev = compute_zero_budgets(
        sgld_samples, vi_mu, vi_log_sigma, alpha=1.0
    )

    # ---- Compute local posterior mass near zero ----
    local_mass = compute_local_mass_zero(sgld_samples, eps=0.05)

    print("\n=== Budgets at zero (per weight) ===")
    print("Forward Γ1(0) stats: min={:.4e}, median={:.4e}, max={:.4e}".format(
        Gamma_fwd.min(), np.median(Gamma_fwd), Gamma_fwd.max()))
    print("Reverse Γ1*(0) stats: min={:.4e}, median={:.4e}, max={:.4e}".format(
        Gamma_rev.min(), np.median(Gamma_rev), Gamma_rev.max()))
    print("Local mass near zero stats: min={:.4f}, median={:.4f}, max={:.4f}".format(
        local_mass.min(), np.median(local_mass), local_mass.max()))

    # ---- Dangerous weights cloud plot ----
    danger_mask = plot_danger_cloud(
        Gamma_fwd, local_mass,
        gamma_percentile=90.0,   # top 10% Γ is considered "large"
        mass_threshold=0.1,      # m_j >= 0.1 is considered "non-negligible mass"
        title=r"Dangerous weights cloud (forward budget at zero)"
    )

    print(f"Number of dangerous weights = {danger_mask.sum()}")

    # ---- Other diagnostic plots ----
    plot_sorted_budgets(Gamma_fwd,
                        r"Forward budgets at zero: $\Gamma_1(0)$ (p\|\;q)")
    plot_hist_budgets(Gamma_fwd,
                      r"Histogram of forward budgets at zero")


if __name__ == "__main__":
    main()
