"""
Regenerate the numerical tables of the manuscript.

  - ABM validation:  alpha = 1 ABM vs RK4 (l-infinity error)
  - Table 4:  PINN vs ABM MAE for alpha in {0.70, 0.85, 1.00}, mean +/- s.d.
              over several random seeds  (needs torch)
  - Table 5:  ABM step-size refinement (l-infinity vs a fine reference)
  - Table 6:  PINN collocation study, MAE vs N_r  (needs torch)

LaTeX table bodies are printed to stdout so they can be pasted into the paper.

Usage:
    python make_tables.py --seeds 10                  # paper setting
    python make_tables.py --seeds 3 --quick           # fast check
"""
import argparse
import numpy as np

from params import BASELINE
from abm import abm_solve, rk4_solve


def linf(a, b):
    return float(np.max(np.abs(a - b)))


def abm_validation(p, tmax=50.0, h=0.01):
    t, Yabm = abm_solve(1.0, p["u"], p, h=h, tmax=tmax)
    _, Yrk4 = rk4_solve(p["u"], p, h=h, tmax=tmax)
    err = linf(Yabm, Yrk4)
    print(f"\n[ABM validation]  alpha=1, h={h}:  l-inf(ABM, RK4) = {err:.2e}")
    return err


def abm_step_study(p, tmax=50.0, hs=(0.04, 0.02, 0.01, 0.005), h_ref=0.0025, alpha=0.85):
    t_ref, Y_ref = abm_solve(alpha, p["u"], p, h=h_ref, tmax=tmax)
    print("\n[Table 5] ABM step-size refinement (l-inf vs h_ref = %.4f, alpha=%.2f)" % (h_ref, alpha))
    row = []
    for h in hs:
        t, Y = abm_solve(alpha, p["u"], p, h=h, tmax=tmax)
        Yi = np.stack([np.interp(t_ref, t, Y[:, i]) for i in range(3)], axis=1)
        e = linf(Y_ref, Yi)
        row.append(e)
        print(f"    h={h:<7} l-inf={e:.2e}")
    print("    LaTeX: " + " & ".join(f"${e:.1e}$".replace("e-0", r"\times10^{-").replace("e-", r"\times10^{-") + "}" for e in row))
    return dict(zip(hs, row))


def pinn_table4(p, alphas=(0.70, 0.85, 1.00), seeds=10, tmax=50.0, quick=False):
    try:
        from pinn import pinn_vs_abm_mae
    except ImportError as e:
        print(f"\n[Table 4] SKIPPED ({e}). Install PyTorch to regenerate.")
        return None
    kw = dict(Nr=200, adam_epochs=500, lbfgs_iters=100) if quick \
        else dict(Nr=1000, adam_epochs=10000, lbfgs_iters=1000)
    print(f"\n[Table 4] PINN vs ABM MAE (mean +/- s.d. over {seeds} seeds)"
          + ("  [QUICK]" if quick else ""))
    results = {}
    for a in alphas:
        maes = np.array([pinn_vs_abm_mae(alpha=a, p=p, seed=sd, tmax=tmax, **kw)
                         for sd in range(seeds)])
        m, sd = maes.mean(0), maes.std(0)
        results[a] = (m, sd)
        print(f"  alpha={a:.2f}:  "
              f"K={m[0]:.2e}+/-{sd[0]:.1e}  "
              f"P={m[1]:.2e}+/-{sd[1]:.1e}  "
              f"T={m[2]:.2e}+/-{sd[2]:.1e}")
    return results


def pinn_collocation(p, Nrs=(250, 500, 1000, 2000), alpha=0.85, seed=0, tmax=50.0, quick=False):
    try:
        from pinn import pinn_vs_abm_mae
    except ImportError as e:
        print(f"\n[Table 6] SKIPPED ({e}). Install PyTorch to regenerate.")
        return None
    ep = dict(adam_epochs=500, lbfgs_iters=100) if quick \
        else dict(adam_epochs=10000, lbfgs_iters=1000)
    print("\n[Table 6] PINN collocation study (mean MAE over K,P,T, alpha=%.2f)" % alpha)
    out = {}
    for Nr in Nrs:
        mae = pinn_vs_abm_mae(alpha=alpha, p=p, Nr=Nr, seed=seed, tmax=tmax, **ep)
        out[Nr] = float(mae.mean())
        print(f"    N_r={Nr:<6} MAE={out[Nr]:.2e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    p = BASELINE.copy()
    abm_validation(p)
    abm_step_study(p)
    pinn_table4(p, seeds=args.seeds, quick=args.quick)
    pinn_collocation(p, quick=args.quick)


if __name__ == "__main__":
    main()
