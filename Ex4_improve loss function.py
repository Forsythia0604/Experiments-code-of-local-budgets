import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ======================
# Global settings
# ======================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPARSITY_THRESH = 1e-3  # |mu| < 1e-3 is counted as "sparse"


def set_seed(seed=42):
    """Use a fixed seed to make the results more stable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================
# 1. Generate synthetic regression data: train / val / test
# ======================

def make_synthetic_regression(n_train=1600, n_val=400, n_test=500):
    """
    1D synthetic regression:
      x = 1D uniform grid,
      y = sin(x) + noise.
    Returns: (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    n_total = n_train + n_val + n_test
    x = torch.linspace(-3.0, 3.0, n_total).unsqueeze(1)
    y_clean = torch.sin(x)
    y = y_clean + 0.1 * torch.randn_like(y_clean)

    x_train = x[:n_train]
    y_train = y[:n_train]

    x_val = x[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]

    x_test = x[n_train + n_val:]
    y_test = y[n_train + n_val:]

    return x_train, y_train, x_val, y_val, x_test, y_test


# ======================
# 2. Variational linear layer: q = N(mu, sigma^2), p = spike & slab Gaussian mixture
# ======================

class VariationalLinear(nn.Module):
    """
    Mean-field variational linear layer:
    q(w_ij) = N(mu_ij, sigma_ij^2)
    prior p(w_ij) = pi N(0, sigma_spike^2) + (1-pi) N(0, sigma_slab^2)
    KL is estimated by Monte Carlo: E_q[log q - log p]
    """

    def __init__(
        self,
        in_features,
        out_features,
        prior_pi=0.9,
        prior_sigma_spike=0.05,
        prior_sigma_slab=1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Prior hyperparameters (not trained)
        self.prior_pi = float(prior_pi)
        self.prior_sigma_spike = float(prior_sigma_spike)
        self.prior_sigma_slab = float(prior_sigma_slab)

        # Variational parameters
        self.mu_weight = nn.Parameter(
            torch.Tensor(out_features, in_features).normal_(0.0, 0.1)
        )
        self.rho_weight = nn.Parameter(
            torch.Tensor(out_features, in_features).fill_(-3.0)
        )  # after softplus, sigma ~ 0.05
        self.mu_bias = nn.Parameter(
            torch.Tensor(out_features).normal_(0.0, 0.1)
        )
        self.rho_bias = nn.Parameter(
            torch.Tensor(out_features).fill_(-3.0)
        )

        # Store the KL estimate corresponding to the current forward pass
        self._last_kl = torch.tensor(0.0)

    def _sigma(self, rho):
        # Softplus, ensures sigma > 0
        return torch.log1p(torch.exp(rho))

    def _log_normal_pdf(self, x, mu, sigma):
        return (
            -0.5 * math.log(2.0 * math.pi)
            - torch.log(sigma)
            - 0.5 * ((x - mu) / sigma) ** 2
        )

    def _log_prior_mixture(self, w):
        """
        log p(w) for the mixture prior:
        p(w) = pi N(0, sigma_spike^2) + (1-pi) N(0, sigma_slab^2)
        """
        pi = torch.tensor(self.prior_pi, device=w.device)
        sigma_spike = torch.tensor(self.prior_sigma_spike, device=w.device)
        sigma_slab = torch.tensor(self.prior_sigma_slab, device=w.device)

        log_comp1 = self._log_normal_pdf(w, 0.0, sigma_spike) + torch.log(pi)
        log_comp2 = self._log_normal_pdf(w, 0.0, sigma_slab) + torch.log(1.0 - pi)
        # log-sum-exp
        log_p = torch.logaddexp(log_comp1, log_comp2)
        return log_p

    def sample_weight_and_bias_with_logprob(self):
        sigma_w = self._sigma(self.rho_weight)
        sigma_b = self._sigma(self.rho_bias)

        eps_w = torch.randn_like(self.mu_weight)
        eps_b = torch.randn_like(self.mu_bias)

        w = self.mu_weight + sigma_w * eps_w
        b = self.mu_bias + sigma_b * eps_b

        # log q
        log_q_w = self._log_normal_pdf(w, self.mu_weight, sigma_w)
        log_q_b = self._log_normal_pdf(b, self.mu_bias, sigma_b)

        # log p (mixture prior)
        log_p_w = self._log_prior_mixture(w)
        log_p_b = self._log_prior_mixture(b)

        kl = (log_q_w - log_p_w).sum() + (log_q_b - log_p_b).sum()
        self._last_kl = kl

        return w, b

    def forward(self, x):
        w, b = self.sample_weight_and_bias_with_logprob()
        return F.linear(x, w, b)

    def kl_divergence(self):
        # Return the Monte Carlo KL estimate for the latest forward pass
        return self._last_kl

    def budget_term_zero(self):
        """
        Approximate the budget at 0:
          R(0) = p(0) / q(0),
          f1(R) = R log R - R + 1,
        Then average over all parameters.
        """
        sigma_w = self._sigma(self.rho_weight)
        sigma_b = self._sigma(self.rho_bias)

        # log density of the prior at 0 (mixture)
        x0 = torch.tensor(0.0, device=self.mu_weight.device)
        pi = torch.tensor(self.prior_pi, device=self.mu_weight.device)
        sigma_spike = torch.tensor(self.prior_sigma_spike, device=self.mu_weight.device)
        sigma_slab = torch.tensor(self.prior_sigma_slab, device=self.mu_weight.device)

        log_comp1_0 = self._log_normal_pdf(x0, 0.0, sigma_spike) + torch.log(pi)
        log_comp2_0 = self._log_normal_pdf(x0, 0.0, sigma_slab) + torch.log(1.0 - pi)
        log_p0 = torch.logaddexp(log_comp1_0, log_comp2_0)  # scalar

        # log density of q at 0
        log_q0_w = self._log_normal_pdf(x0, self.mu_weight, sigma_w)
        log_q0_b = self._log_normal_pdf(x0, self.mu_bias, sigma_b)

        log_R0_w = log_p0 - log_q0_w
        log_R0_b = log_p0 - log_q0_b

        log_R0_w = torch.clamp(log_R0_w, -30.0, 30.0)
        log_R0_b = torch.clamp(log_R0_b, -30.0, 30.0)

        R0_w = torch.exp(log_R0_w)
        R0_b = torch.exp(log_R0_b)

        f1_R0_w = R0_w * log_R0_w - R0_w + 1.0
        f1_R0_b = R0_b * log_R0_b - R0_b + 1.0

        B0 = 0.5 * (f1_R0_w.mean() + f1_R0_b.mean())
        return B0

    def get_mu_vector(self):
        return torch.cat(
            [self.mu_weight.view(-1), self.mu_bias.view(-1)], dim=0
        )


# ======================
# 3. Small BNN model
# ======================

class VariationalMLP(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=32,
        prior_pi=0.9,
        prior_sigma_spike=0.05,
        prior_sigma_slab=1.0,
    ):
        super().__init__()
        self.fc1 = VariationalLinear(
            input_dim,
            hidden_dim,
            prior_pi=prior_pi,
            prior_sigma_spike=prior_sigma_spike,
            prior_sigma_slab=prior_sigma_slab,
        )
        self.fc2 = VariationalLinear(
            hidden_dim,
            hidden_dim,
            prior_pi=prior_pi,
            prior_sigma_spike=prior_sigma_spike,
            prior_sigma_slab=prior_sigma_slab,
        )
        self.fc3 = VariationalLinear(
            hidden_dim,
            1,
            prior_pi=prior_pi,
            prior_sigma_spike=prior_sigma_spike,
            prior_sigma_slab=prior_sigma_slab,
        )

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x

    def forward_mean(self, x):
        """
        Deterministic prediction using the posterior means (mu),
        used at validation / test time.
        """
        x = F.linear(x, self.fc1.mu_weight, self.fc1.mu_bias)
        x = torch.tanh(x)
        x = F.linear(x, self.fc2.mu_weight, self.fc2.mu_bias)
        x = torch.tanh(x)
        x = F.linear(x, self.fc3.mu_weight, self.fc3.mu_bias)
        return x

    def kl_divergence(self):
        # After a forward pass, collect the Monte Carlo KL from all layers
        return (
            self.fc1.kl_divergence()
            + self.fc2.kl_divergence()
            + self.fc3.kl_divergence()
        )

    def budget_zero(self):
        B0_1 = self.fc1.budget_term_zero()
        B0_2 = self.fc2.budget_term_zero()
        B0_3 = self.fc3.budget_term_zero()
        return (B0_1 + B0_2 + B0_3) / 3.0

    def all_mu_vector(self):
        return torch.cat(
            [
                self.fc1.get_mu_vector(),
                self.fc2.get_mu_vector(),
                self.fc3.get_mu_vector(),
            ],
            dim=0,
        )


# ======================
# 4. Training loop: track val MSE & sparsity
# ======================

def train_bnn(
    lambda_B0=0.0,
    n_epochs=400,
    batch_size=128,
    kl_weight=1.0,
    log_interval=20,
    seed=42,
):
    """
    lambda_B0 = 0  corresponds to pure ELBO;
    lambda_B0 > 0 adds a budget penalty at 0.
    Returns:
      - model
      - epoch_history: epochs where we logged (1, 20, 40, ...)
      - mse_history: validation MSE at those epochs
      - sparsity_history: sparsity at those epochs
      - best_val_mse: minimum val MSE over the whole training
      - best_sparsity: sparsity at the epoch achieving the best val MSE
    """
    set_seed(seed)

    x_train, y_train, x_val, y_val, x_test, y_test = make_synthetic_regression()
    train_ds = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = VariationalMLP(
        input_dim=1,
        hidden_dim=32,
        prior_pi=0.9,
        prior_sigma_spike=0.05,
        prior_sigma_slab=1.0,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    N_train = x_train.shape[0]
    obs_sigma = 0.1

    # Move val data to device in advance
    x_val_d = x_val.to(device)
    y_val_d = y_val.to(device)

    epoch_history = []
    mse_history = []
    sparsity_history = []

    best_val_mse = float("inf")
    best_sparsity = None
    best_epoch = None

    use_budget_penalty = lambda_B0 > 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        total_nll = 0.0
        total_kl = 0.0
        total_B0 = 0.0
        num_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            # Forward pass: internally samples one set of weights and computes its KL
            y_pred = model(xb)

            nll = 0.5 * (
                (yb - y_pred) ** 2 / (obs_sigma ** 2)
                + math.log(2.0 * math.pi * obs_sigma ** 2)
            )
            nll = nll.mean()

            kl = model.kl_divergence() / N_train

            if use_budget_penalty:
                B0 = model.budget_zero()
                loss = nll + kl_weight * kl + lambda_B0 * B0
            else:
                B0 = torch.tensor(0.0, device=device)
                loss = nll + kl_weight * kl

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_nll += nll.item()
            total_kl += kl.item()
            total_B0 += B0.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_nll = total_nll / num_batches
        avg_kl = total_kl / num_batches
        avg_B0 = total_B0 / num_batches

        # ===== Evaluate on validation data at every epoch
        #       (for early stopping & sparsity tracking) =====
        model.eval()
        with torch.no_grad():
            y_pred_val = model.forward_mean(x_val_d)
            mse_val = F.mse_loss(y_pred_val, y_val_d).item()

            mu_vec = model.all_mu_vector().detach().cpu()
            sparsity = (mu_vec.abs() < SPARSITY_THRESH).float().mean().item()

        # Update "best" (early-stopping style)
        if mse_val < best_val_mse:
            best_val_mse = mse_val
            best_sparsity = sparsity
            best_epoch = epoch

        # Only log and store curve points at some epochs
        if epoch == 1 or epoch % log_interval == 0:
            epoch_history.append(epoch)
            mse_history.append(mse_val)
            sparsity_history.append(sparsity)

            tag = "ELBO+Budget" if use_budget_penalty else "ELBO only"
            print(
                f"[{tag}, lambda_B0={lambda_B0}, seed={seed}] "
                f"Epoch {epoch:4d} | "
                f"loss={avg_loss:.4f}, nll={avg_nll:.4f}, kl={avg_kl:.4f}, "
                f"B0={avg_B0:.4f}, val MSE={mse_val:.4f}, sparsity={sparsity:.3f}"
            )

    print(
        f"==> [lambda_B0={lambda_B0}, seed={seed}] "
        f"Best val MSE={best_val_mse:.4f} at epoch {best_epoch}, "
        f"sparsity={best_sparsity:.3f}"
    )

    return model, epoch_history, mse_history, sparsity_history, best_val_mse, best_sparsity


# ======================
# 5. Main script: multi-seed average + scatter plot
# ======================

if __name__ == "__main__":
    n_epochs = 200
    log_interval = 20

    # --------- B. Sweep λ_B0 from 0 to 1 with step 0.1 (multi-seed average) ---------
    lambdas_sweep = [i / 10 for i in range(0, 11)]  # 0.00, 0.1, ..., 1.00
    seeds_sweep = [0, 1, 2]  # seeds for the sweep; can be increased

    scatter_results = []  # (lambda_B0, mean_best_mse, mean_best_spars)

    def run_multi_seed_for_scatter(lambda_B0, seeds):
        """
        For a fixed λ_B0 and multiple seeds:
        on each training trajectory, pick the point with the smallest val MSE,
        then average these best results across seeds.
        """
        best_mse_list = []
        best_spars_list = []

        for s in seeds:
            print(f"\n=== Scatter: lambda_B0 = {lambda_B0}, seed = {s} ===")
            _, _, _, _, best_mse, best_spars = train_bnn(
                lambda_B0=lambda_B0,
                n_epochs=n_epochs,
                kl_weight=1.0,
                log_interval=log_interval,
                seed=s,
            )
            best_mse_list.append(best_mse)
            best_spars_list.append(best_spars)

        mean_best_mse = float(np.mean(best_mse_list))
        mean_best_spars = float(np.mean(best_spars_list))
        return mean_best_mse, mean_best_spars

    for lam in lambdas_sweep:
        mean_mse, mean_spars = run_multi_seed_for_scatter(lam, seeds_sweep)
        scatter_results.append((lam, mean_mse, mean_spars))
        print(
            f"[Sweep] lambda_B0={lam:.2f}, "
            f"mean best val MSE={mean_mse:.4f}, mean best sparsity={mean_spars:.4f}"
        )

    # 3) MSE vs Sparsity scatter plot (multi-seed average)
    plt.figure()
    for lam, best_mse, best_spars in scatter_results:
        plt.scatter(best_spars, best_mse, s=15)
        # Only annotate some λ values (e.g., 0.0, 0.1, 0.2, ...)
        if abs(lam * 100 - round(lam * 100)) < 1e-6 and int(round(lam * 100)) % 10 == 0:
            plt.text(best_spars, best_mse, f"{lam:.1f}", fontsize=8,
                     ha="left", va="bottom")
    plt.xlabel(f"Sparsity (|mu| < {SPARSITY_THRESH})")
    plt.ylabel("Best validation MSE (posterior mean)")
    plt.title("MSE vs Sparsity (lambda_B0 in [0,1], averaged over seeds)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
