"""
Physics-informed neural network (PINN) for the fractional system, following
Raissi et al. (2019) with the fractional residual formulation of Pang et al.
(2019). The Caputo derivative is discretised on the collocation grid with the
L1 scheme (Lin & Xu, 2007; Eq. 17 of the manuscript):

    C D^alpha K(t_n) ~= (dt)^{-alpha} / Gamma(2 - alpha)
                        * sum_{j=0}^{n-1} b_j [K(t_{n-j}) - K(t_{n-j-1})],
    b_j = (j+1)^{1-alpha} - j^{1-alpha}.

Each state variable is approximated by an independent feedforward network with
5 hidden layers of 64 tanh units. Input is the normalised time t/tmax. Training
uses Adam followed by L-BFGS.

Requires PyTorch (see requirements.txt). The ABM/equilibrium code does not.
"""
import numpy as np
from math import gamma as _gamma

from params import BASELINE, IC
from model import rhs_components


def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "pinn.py requires PyTorch. Install it with `pip install torch` "
            "(a CUDA-enabled build is recommended; see requirements.txt)."
        ) from exc


def l1_weight_matrix(N, dt, alpha):
    """Dense (N, N) matrix A such that (A @ v)[n] approximates C D^alpha v(t_n)
    on the uniform grid, using the L1 scheme. Row 0 is zero."""
    g = dt ** (-alpha) / _gamma(2.0 - alpha)
    jj = np.arange(N, dtype=float)
    # b_j = (j+1)^{1-alpha} - j^{1-alpha}; the j=0 lower term is 0 for all
    # alpha in (0,1] (avoid numpy's 0**0 = 1 at alpha = 1, which gives b_0 = 1).
    low = np.where(jj > 0, jj ** (1.0 - alpha), 0.0)
    b = (jj + 1.0) ** (1.0 - alpha) - low          # b_j, j = 0..N-1
    A = np.zeros((N, N))
    for n in range(1, N):
        for j in range(n):
            A[n, n - j] += b[j]
            A[n, n - j - 1] -= b[j]
    return g * A


def make_mlp(torch, hidden_layers=5, width=64):
    import torch.nn as nn
    layers = [nn.Linear(1, width), nn.Tanh()]
    for _ in range(hidden_layers - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, 1)]
    return nn.Sequential(*layers)


def train_pinn(alpha=0.85, u=None, p=None, Nr=1000, tmax=50.0,
               adam_epochs=10000, lbfgs_iters=1000, lr=1e-3, lambda_r=1.0,
               seed=0, device="cpu", dtype="float64", verbose=False):
    """Train the fPINN and return (t, Y_pinn) on the collocation grid,
    where Y_pinn has shape (Nr, 3) = [K_hat, P_hat, T_hat]."""
    torch = _require_torch()
    td = torch.float64 if dtype == "float64" else torch.float32
    torch.manual_seed(seed)
    np.random.seed(seed)

    if p is None:
        p = BASELINE.copy()
    if u is None:
        u = p["u"]

    t = np.linspace(0.0, tmax, Nr)
    dt = t[1] - t[0]
    A = torch.tensor(l1_weight_matrix(Nr, dt, alpha), dtype=td, device=device)
    t_in = torch.tensor((t / tmax).reshape(-1, 1), dtype=td, device=device)

    K0, P0, T0 = IC["K0"], IC["P0"], IC["T0"]
    netK = make_mlp(torch).to(device).to(td)
    netP = make_mlp(torch).to(device).to(td)
    netT = make_mlp(torch).to(device).to(td)
    params = list(netK.parameters()) + list(netP.parameters()) + list(netT.parameters())

    def loss_fn():
        K = netK(t_in); P = netP(t_in); T = netT(t_in)          # (Nr, 1)
        DK = A @ K; DP = A @ P; DT = A @ T                       # Caputo derivs
        dK, dP, dT = rhs_components(K, P, T, u, p)               # RHS (Eqs. 3-5)
        res = (((DK - dK)[1:]) ** 2).mean() \
            + (((DP - dP)[1:]) ** 2).mean() \
            + (((DT - dT)[1:]) ** 2).mean()
        ic = (K[0] - K0) ** 2 + (P[0] - P0) ** 2 + (T[0] - T0) ** 2
        return ic.squeeze() + lambda_r * res

    # --- Adam ---
    opt = torch.optim.Adam(params, lr=lr)
    for epoch in range(adam_epochs):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        if verbose and epoch % max(1, adam_epochs // 10) == 0:
            print(f"  [Adam] epoch {epoch:6d}  loss={loss.item():.3e}")

    # --- L-BFGS fine-tuning ---
    if lbfgs_iters > 0:
        opt2 = torch.optim.LBFGS(params, max_iter=lbfgs_iters,
                                 line_search_fn="strong_wolfe")

        def closure():
            opt2.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss

        opt2.step(closure)
        if verbose:
            print(f"  [L-BFGS] final loss={loss_fn().item():.3e}")

    with torch.no_grad():
        K = netK(t_in).cpu().numpy().ravel()
        P = netP(t_in).cpu().numpy().ravel()
        T = netT(t_in).cpu().numpy().ravel()
    return t, np.stack([K, P, T], axis=1)


def pinn_vs_abm_mae(alpha=0.85, u=None, p=None, Nr=1000, tmax=50.0,
                    seed=0, device="cpu", **kw):
    """Train the PINN and return per-variable MAE against the ABM reference
    sampled on the same grid."""
    from abm import abm_solve
    if p is None:
        p = BASELINE.copy()
    if u is None:
        u = p["u"]
    t, Ypinn = train_pinn(alpha=alpha, u=u, p=p, Nr=Nr, tmax=tmax,
                          seed=seed, device=device, **kw)
    # ABM reference on a fine grid, then interpolate to the PINN grid
    t_ref, Yref = abm_solve(alpha, u, p, h=0.01, tmax=tmax)
    Yint = np.stack([np.interp(t, t_ref, Yref[:, i]) for i in range(3)], axis=1)
    mae = np.abs(Ypinn - Yint).mean(axis=0)   # [MAE_K, MAE_P, MAE_T]
    return mae


if __name__ == "__main__":
    # Quick smoke test (small config).
    mae = pinn_vs_abm_mae(alpha=0.85, Nr=200, adam_epochs=300, lbfgs_iters=50,
                          verbose=True)
    print("Smoke-test MAE [K, P, T]:", np.round(mae, 5))
