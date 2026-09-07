# Fractional Green Economic–Ecological Model with PINN Approximation

Reference implementation for the manuscript

> **PINN-Based Fractional Sustainability Dynamics: A Fractional Green
> Economic–Ecological Model with Pollution Memory and Physics-Informed Neural
> Network Approximation** — H. Zulfiqar & M. Sultana (CMDE-2605-3670-2-2).

The code implements the three-compartment Caputo fractional model of capital,
pollution, and green technology; the Adams–Bashforth–Moulton (ABM) reference
solver; the equilibrium/stability analysis (corrected Proposition 5.2 and
Appendix A); and a PyTorch physics-informed neural network (PINN) using the L1
fractional-residual scheme. It regenerates every figure and table in the paper.

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

PyTorch is required **only** for the PINN (`pinn.py`, Figure 3, Tables 4 and 6).
Everything else runs on NumPy/SciPy/Matplotlib alone. A CUDA-enabled GPU is
recommended for the full training schedule (10⁴ Adam epochs + 10³ L-BFGS
iterations, ≈4 min on a single NVIDIA A100); on CPU, use the `--quick` flag.

## Quick start

```bash
python equilibrium.py          # equilibrium, eigenvalues, stability, u^opt
python make_figures.py         # Figures 1–4  (Fig. 3 needs torch)
python make_tables.py --seeds 10   # ABM validation + Tables 4–6
python run_all.py              # everything, with a printed summary
```

Add `--quick` to `make_figures.py`, `make_tables.py`, or `run_all.py` for a fast
smoke test that shrinks the PINN training budget and collocation count.

## Files

| File | Contents |
|------|----------|
| `params.py` | Baseline parameters (Table 3), initial conditions, defaults |
| `model.py` | Model RHS (Eqs. 3–5); backend-agnostic for NumPy and torch |
| `abm.py` | Fractional ABM predictor–corrector solver; RK4 for α=1 validation |
| `equilibrium.py` | Interior equilibrium, Jacobian, stability check, capital-maximising u^opt |
| `pinn.py` | PyTorch fPINN with the L1 Caputo scheme (Eq. 17); training + MAE |
| `make_figures.py` | Regenerates Figures 1–4 |
| `make_tables.py` | Regenerates the ABM validation, Table 4 (seeds), Table 5, Table 6 |
| `run_all.py` | Runs the summary, all figures, and all tables |

## Reproducibility map

| Paper item | Produced by |
|------------|-------------|
| Figure 1 (memory effect) | `make_figures.py` → `figures/fig1_memory.pdf` |
| Figure 2 (steady state vs u, u^opt) | `make_figures.py` → `figures/fig2_steadystate.pdf` |
| Figure 3 (PINN vs ABM) | `make_figures.py` → `figures/fig3_pinn_abm.pdf` |
| Figure 4 (phase portrait + stability cone) | `make_figures.py` → `figures/fig4_phase_stability.pdf` |
| ABM validation (α=1 vs RK4) | `make_tables.py` |
| Table 4 (PINN MAE, mean ± s.d. over seeds) | `make_tables.py --seeds 10` |
| Table 5 (ABM step-size refinement) | `make_tables.py` |
| Table 6 (PINN collocation study) | `make_tables.py` |

## Notes on the PINN

Each state variable is approximated by an independent feedforward network with
5 hidden layers of 64 `tanh` units; the input is the normalised time `t/tmax`.
The Caputo derivative is discretised on the collocation grid with the L1 scheme

```
C D^alpha K(t_n) ≈ (Δt)^{-α}/Γ(2-α) · Σ_{j=0}^{n-1} b_j [K(t_{n-j}) − K(t_{n-j-1})],
b_j = (j+1)^{1-α} − j^{1-α},
```

assembled once as a sparse-in-structure weight matrix (`pinn.l1_weight_matrix`).
The loss is `L_IC + λ_r L_res` with `λ_r = 1` and `N_r = 1000` uniform
collocation points. Training uses Adam (lr `1e-3`) followed by L-BFGS with a
strong-Wolfe line search. A small positive floor is applied before the
fractional powers to keep training numerically stable (it never activates on the
strictly positive physical trajectory).

## License

MIT — see `LICENSE`.
