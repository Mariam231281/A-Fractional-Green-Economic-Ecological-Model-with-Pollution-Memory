"""
Regenerate Figures 1-4 of the manuscript.

    Figure 1  time evolution of K, P, T for four memory exponents  (ABM)
    Figure 2  steady-state K*, P*, T* vs u, with capital-max u^opt  (equilibrium)
    Figure 3  PINN vs ABM reference                                (PINN; needs torch)
    Figure 4  K-P phase portrait + fractional stability cone       (ABM + equilibrium)

Figures 1, 2, 4 need only NumPy/SciPy/Matplotlib. Figure 3 additionally needs
PyTorch; if torch is unavailable it is skipped with a message.

Usage:
    python make_figures.py            # all figures (Fig 3 uses paper settings)
    python make_figures.py --quick    # fast PINN settings for Fig 3
"""
import argparse
import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from params import BASELINE, IC, ALPHAS, TMAX
from abm import abm_solve
from equilibrium import interior_equilibrium, jacobian, is_stable, u_opt_capital

mpl.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "font.size": 10, "axes.linewidth": 0.8, "figure.dpi": 150,
    "savefig.bbox": "tight",
})
COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#7d3c98"]
STYLES = ["--", "-.", ":", "-"]
OUT = "figures"


def figure1(p):
    sols = {a: abm_solve(a, p["u"], p, h=0.02, tmax=TMAX) for a in ALPHAS}
    titles = [r"Capital $K(t)$", r"Pollution $P(t)$", r"Green technology $T(t)$"]
    ylabs = [r"$K(t)$", r"$P(t)$", r"$T(t)$"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for i in range(3):
        for a, cl, st in zip(ALPHAS, COLORS, STYLES):
            t, Y = sols[a]
            ax[i].plot(t, Y[:, i], st, color=cl, lw=1.4, label=rf"$\alpha={a:.2f}$")
        ax[i].set_title(titles[i]); ax[i].set_xlabel(r"Time $t$"); ax[i].set_ylabel(ylabs[i])
        ax[i].grid(alpha=0.25); ax[i].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_memory.pdf"); plt.close(fig)
    print("  Figure 1 -> figures/fig1_memory.pdf")
    return sols


def figure2(p):
    us = np.linspace(0.002, p["s"] - 0.002, 500)
    K = np.array([interior_equilibrium(u, p)[0] for u in us])
    P = np.array([interior_equilibrium(u, p)[1] for u in us])
    T = np.array([interior_equilibrium(u, p)[2] for u in us])
    uopt, _ = u_opt_capital(p)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for a, y, ttl, ylab, cl in zip(
            ax, [K, P, T], [r"Capital $K^*$", r"Pollution $P^*$", r"Technology $T^*$"],
            [r"$K^*$", r"$P^*$", r"$T^*$"], ["#1f77b4", "#d62728", "#2ca02c"]):
        a.plot(us, y, color=cl, lw=1.8)
        a.axvline(uopt, ls="--", color="0.4", lw=1.2, label=r"$u^{\mathrm{opt}}$")
        a.set_title(ttl); a.set_xlabel(r"Investment rate $u$"); a.set_ylabel(ylab)
        a.grid(alpha=0.25); a.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_steadystate.pdf"); plt.close(fig)
    print(f"  Figure 2 -> figures/fig2_steadystate.pdf   (u^opt = {uopt:.4f})")


def figure4(p, sols):
    Kst, Pst, Tst = interior_equilibrium(p["u"], p)
    J = jacobian(Kst, Pst, Tst, p["u"], p)
    _, eig = is_stable(J, p["alpha"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, cl, st in zip(ALPHAS, COLORS, STYLES):
        t, Y = sols[a]
        ax[0].plot(Y[:, 0], Y[:, 1], st, color=cl, lw=1.3, label=rf"$\alpha={a:.2f}$")
    ax[0].plot(IC["K0"], IC["P0"], "o", color="k", ms=6)
    ax[0].plot(Kst, Pst, "*", color="k", ms=13)
    ax[0].set_xlabel(r"Capital $K(t)$"); ax[0].set_ylabel(r"Pollution $P(t)$")
    ax[0].set_title(r"Phase portrait: $K$--$P$ plane")
    ax[0].grid(alpha=0.25); ax[0].legend(fontsize=8, frameon=False)

    th = np.linspace(0, 2 * np.pi, 300)
    ax[1].plot(np.cos(th), np.sin(th), ":", color="0.6", lw=0.8)
    for a, cl in zip(ALPHAS, COLORS):
        ang = a * np.pi / 2
        for sgn in (1, -1):
            ax[1].plot([0, 1.4 * math.cos(sgn * ang)], [0, 1.4 * math.sin(sgn * ang)],
                       color=cl, lw=1.3)
    ax[1].axhline(0, color="k", lw=0.8); ax[1].axvline(0, color="k", lw=0.8)
    ev = eig[np.argmax(np.abs(np.angle(eig)))]; sc = 1.0 / max(np.abs(eig))
    ax[1].plot(ev.real * sc, ev.imag * sc, "*", color="k", ms=14, label=r"Eigenvalue $\lambda^*$")
    ax[1].set_xlim(-1.5, 1.5); ax[1].set_ylim(-1.5, 1.5); ax[1].set_aspect("equal")
    ax[1].set_xlabel(r"Re$(\lambda)$"); ax[1].set_ylabel(r"Im$(\lambda)$")
    ax[1].set_title(r"Stability cone $|\arg(\lambda)|>\alpha\pi/2$")
    ax[1].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_phase_stability.pdf"); plt.close(fig)
    print("  Figure 4 -> figures/fig4_phase_stability.pdf")


def figure3(p, quick=False):
    try:
        from pinn import train_pinn
    except ImportError as e:
        print(f"  Figure 3 SKIPPED ({e}). Install PyTorch to regenerate it.")
        return
    kw = dict(Nr=200, adam_epochs=500, lbfgs_iters=100) if quick \
        else dict(Nr=1000, adam_epochs=10000, lbfgs_iters=1000)
    t, Yp = train_pinn(alpha=0.85, u=p["u"], p=p, tmax=TMAX, seed=0, **kw)
    t_ref, Yref = abm_solve(0.85, p["u"], p, h=0.01, tmax=TMAX)
    Yint = np.stack([np.interp(t, t_ref, Yref[:, i]) for i in range(3)], axis=1)
    titles = [r"Capital $K(t)$", r"Pollution $P(t)$", r"Green technology $T(t)$"]
    ylabs = [r"$K(t)$", r"$P(t)$", r"$T(t)$"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.1))
    for i in range(3):
        ax[i].plot(t, Yint[:, i], "-", color="#1f77b4", lw=1.8, label="ABM reference")
        ax[i].plot(t[::12], Yp[::12, i], "--", color="#d62728", lw=1.3, label="PINN")
        ax[i].set_title(titles[i]); ax[i].set_xlabel(r"Time $t$"); ax[i].set_ylabel(ylabs[i])
        ax[i].grid(alpha=0.25); ax[i].legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_pinn_abm.pdf"); plt.close(fig)
    mae = np.abs(Yp - Yint).mean(axis=0)
    print(f"  Figure 3 -> figures/fig3_pinn_abm.pdf   (MAE K,P,T = {np.round(mae,5)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast PINN settings for Fig 3")
    args = ap.parse_args()
    p = BASELINE.copy()
    print("Generating figures...")
    sols = figure1(p)
    figure2(p)
    figure4(p, sols)
    figure3(p, quick=args.quick)
    print("Done.")


if __name__ == "__main__":
    main()
