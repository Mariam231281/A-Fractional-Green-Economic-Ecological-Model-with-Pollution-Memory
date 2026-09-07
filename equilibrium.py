"""
Interior equilibrium, Jacobian, fractional stability check, and the
capital-maximising pollution-control rate u^opt.

Implements the corrected Proposition 5.2 (phi decreasing, psi increasing =>
unique interior equilibrium) and the corrected Appendix A coefficients.
"""
import numpy as np
from scipy.optimize import brentq, minimize_scalar


def interior_equilibrium(u, p):
    """Return (K*, P*, T*) for the interior equilibrium at investment rate u.

    Solves phi(K) = psi(K) with
        phi(K) = s A (omega u / xi)^rho K^{beta+rho-1}   (strictly decreasing)
        psi(K) = delta + u + gamma eta K / (mu + c K)     (strictly increasing)
    which cross exactly once for beta + rho < 1 (Proposition 5.2).
    """
    s, A, beta, rho = p["s"], p["A"], p["beta"], p["rho"]
    delta, gamma, u_ = p["delta"], p["gamma"], u
    eta, mu, theta, omega, xi = p["eta"], p["mu"], p["theta"], p["omega"], p["xi"]

    c = theta * omega * u_ / xi
    B = s * A * (omega * u_ / xi) ** rho
    expo = beta + rho - 1.0

    def g(K):
        return B * K ** expo - (delta + u_ + gamma * eta * K / (mu + c * K))

    K = brentq(g, 1e-9, 1e9, xtol=1e-12, rtol=1e-12)
    T = (omega * u_ / xi) * K
    P = eta * K / (mu + c * K)
    return K, P, T


def jacobian(K, P, T, u, p):
    """Jacobian (Eq. 13) evaluated at (K, P, T)."""
    s, A, beta, rho = p["s"], p["A"], p["beta"], p["rho"]
    delta, gamma = p["delta"], p["gamma"]
    eta, mu, theta, omega, xi = p["eta"], p["mu"], p["theta"], p["omega"], p["xi"]
    J = np.array([
        [s * A * beta * K ** (beta - 1) * T ** rho - delta - gamma * P - u,
         -gamma * K,
         s * A * rho * K ** beta * T ** (rho - 1)],
        [eta, -mu - theta * T, -theta * P],
        [omega * u, 0.0, -xi],
    ])
    return J


def char_coeffs(J):
    """Corrected Appendix A coefficients of det(lambda I - J) = lam^3
    + a1 lam^2 + a2 lam + a3, with a = j11, ..., h = j33 (j32 = 0)."""
    a, b, c = J[0, 0], J[0, 1], J[0, 2]
    d, e, f = J[1, 0], J[1, 1], J[1, 2]
    g, h = J[2, 0], J[2, 2]
    a1 = -(a + e + h)
    a2 = a * e + a * h + e * h - b * d - c * g
    a3 = -(a * e * h - b * d * h + b * f * g - c * e * g)
    return a1, a2, a3


def is_stable(J, alpha):
    """Matignon condition: |arg(lambda_i)| > alpha pi / 2 for all eigenvalues."""
    lam = np.linalg.eigvals(J)
    return bool(np.all(np.abs(np.angle(lam)) > alpha * np.pi / 2)), lam


def u_opt_capital(p, u_lo=1e-3, u_hi=None):
    """Capital-maximising pollution-control rate u^opt = argmax_u K*(u)
    (Eq. 13-14 of the revised manuscript), found numerically."""
    if u_hi is None:
        u_hi = p["s"] - 1e-3
    neg_K = lambda u: -interior_equilibrium(u, p)[0]
    res = minimize_scalar(neg_K, bounds=(u_lo, u_hi), method="bounded",
                          options={"xatol": 1e-6})
    return res.x, -res.fun


if __name__ == "__main__":
    from params import BASELINE
    p = BASELINE
    K, P, T = interior_equilibrium(p["u"], p)
    J = jacobian(K, P, T, p["u"], p)
    stable, lam = is_stable(J, p["alpha"])
    uo, Ko = u_opt_capital(p)
    print(f"Baseline equilibrium:  K*={K:.4f}  P*={P:.4f}  T*={T:.4f}")
    print(f"Eigenvalues of J*:     {np.round(lam, 4)}")
    print(f"|arg(lambda)|:         {np.round(np.abs(np.angle(lam)), 4)}  (> {p['alpha']*np.pi/2:.4f})")
    print(f"Locally stable:        {stable}")
    print(f"Char. coeffs (a1,a2,a3): {tuple(round(x,5) for x in char_coeffs(J))}")
    print(f"Capital-maximising u^opt = {uo:.4f}  (K*max = {Ko:.4f})")
