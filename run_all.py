"""
Reproduce everything: equilibrium/stability summary, all figures, all tables.

    python run_all.py            # paper settings (PINN training is slow on CPU)
    python run_all.py --quick    # fast settings for a smoke test
"""
import argparse
import subprocess
import sys

import numpy as np

from params import BASELINE
from equilibrium import (interior_equilibrium, jacobian, is_stable,
                         char_coeffs, u_opt_capital)


def summary():
    p = BASELINE
    K, P, T = interior_equilibrium(p["u"], p)
    J = jacobian(K, P, T, p["u"], p)
    stable, lam = is_stable(J, p["alpha"])
    uo, Ko = u_opt_capital(p)
    print("=" * 64)
    print("Equilibrium & stability (baseline parameters)")
    print("=" * 64)
    print(f"  E* = (K*, P*, T*) = ({K:.4f}, {P:.4f}, {T:.4f})")
    print(f"  eig(J*)          = {np.round(lam, 4)}")
    print(f"  |arg(lambda)|    = {np.round(np.abs(np.angle(lam)), 4)}  (> alpha*pi/2 = {p['alpha']*np.pi/2:.4f})")
    print(f"  locally stable   = {stable}")
    print(f"  (a1, a2, a3)     = {tuple(round(x, 5) for x in char_coeffs(J))}")
    print(f"  capital-max u^opt= {uo:.4f}   (K*max = {Ko:.4f})")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    summary()
    fig_cmd = [sys.executable, "make_figures.py"] + (["--quick"] if args.quick else [])
    tab_cmd = [sys.executable, "make_tables.py"] + (["--quick"] if args.quick else [])
    subprocess.run(fig_cmd, check=True)
    subprocess.run(tab_cmd, check=True)


if __name__ == "__main__":
    main()
