"""
Fractional Adams-Bashforth-Moulton (ABM) predictor-corrector solver for the
Caputo system (Diethelm, Ford & Freed, 2002), plus a classical RK4 reference
used to validate the alpha = 1 case.
"""
import numpy as np
from scipy.special import gamma as Gamma

from model import rhs
from params import IC


def abm_solve(alpha, u, p, h=0.01, tmax=50.0, y0=None):
    """Solve C D^alpha y = f(t, y) on [0, tmax] with step h.

    Returns (t, Y) where Y has shape (N+1, 3) holding [K, P, T].
    """
    if y0 is None:
        y0 = np.array([IC["K0"], IC["P0"], IC["T0"]], dtype=float)
    N = int(round(tmax / h))
    t = np.linspace(0.0, tmax, N + 1)
    Y = np.zeros((N + 1, 3))
    F = np.zeros((N + 1, 3))
    Y[0] = y0
    F[0] = rhs(Y[0], u, p)
    ha = h ** alpha

    for n in range(N):
        j = np.arange(n + 1)
        # --- predictor (fractional Adams-Bashforth) ---
        b = (n + 1 - j) ** alpha - (n - j) ** alpha
        yp = Y[0] + (ha / (alpha * Gamma(alpha))) * (b[:, None] * F[:n + 1]).sum(0)
        Fp = rhs(yp, u, p)
        # --- corrector (fractional Adams-Moulton) ---
        a = np.empty(n + 2)
        a[0] = n ** (alpha + 1) - (n - alpha) * (n + 1) ** alpha
        if n >= 1:
            k = np.arange(1, n + 1)
            a[1:n + 1] = ((n - k + 2) ** (alpha + 1)
                          + (n - k) ** (alpha + 1)
                          - 2 * (n - k + 1) ** (alpha + 1))
        a[n + 1] = 1.0
        conv = (a[:n + 1, None] * F[:n + 1]).sum(0) + a[n + 1] * Fp
        Y[n + 1] = Y[0] + (ha / Gamma(alpha + 2)) * conv
        F[n + 1] = rhs(Y[n + 1], u, p)
    return t, Y


def rk4_solve(u, p, h=0.01, tmax=50.0, y0=None):
    """Classical RK4 for the integer-order (alpha = 1) system, used only to
    validate the ABM implementation at alpha = 1."""
    if y0 is None:
        y0 = np.array([IC["K0"], IC["P0"], IC["T0"]], dtype=float)
    N = int(round(tmax / h))
    t = np.linspace(0.0, tmax, N + 1)
    Y = np.zeros((N + 1, 3))
    Y[0] = y0
    for n in range(N):
        y = Y[n]
        k1 = rhs(y, u, p)
        k2 = rhs(y + 0.5 * h * k1, u, p)
        k3 = rhs(y + 0.5 * h * k2, u, p)
        k4 = rhs(y + h * k3, u, p)
        Y[n + 1] = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, Y
