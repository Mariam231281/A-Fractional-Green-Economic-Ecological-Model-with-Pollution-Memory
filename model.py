"""
Right-hand side of the fractional green economic--ecological--technology model.

    C D^alpha K = s A K^beta T^rho - delta K - gamma P K - u K      (Eq. 3)
    C D^alpha P = eta K - mu P - theta T P                          (Eq. 4)
    C D^alpha T = omega u K - xi T                                  (Eq. 5)

`rhs_components` is written so it works transparently on NumPy arrays/scalars
and on PyTorch tensors, so the same equations drive the ABM solver, the
equilibrium analysis, and the PINN residual.
"""


def _pos(x, eps=1e-8):
    """Clamp to a small positive floor before taking fractional powers.

    The physical trajectory is strictly positive, so this floor is never
    active for the ABM/equilibrium code. During PINN training, however, the
    (unconstrained) networks can momentarily produce negative values, and
    a negative base raised to a fractional power yields NaN; the floor keeps
    training numerically stable without altering the modelled equations.
    """
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return torch.clamp(x, min=eps)
    except Exception:
        pass
    import numpy as np
    return np.maximum(x, eps)


def rhs_components(K, P, T, u, p):
    """Return (dK, dP, dT) = C D^alpha of each state (the RHS of Eqs. 3-5)."""
    s, A, beta, rho = p["s"], p["A"], p["beta"], p["rho"]
    delta, gamma, eta = p["delta"], p["gamma"], p["eta"]
    mu, theta, omega, xi = p["mu"], p["theta"], p["omega"], p["xi"]

    dK = s * A * _pos(K) ** beta * _pos(T) ** rho - delta * K - gamma * P * K - u * K
    dP = eta * K - mu * P - theta * T * P
    dT = omega * u * K - xi * T
    return dK, dP, dT


def rhs(y, u, p):
    """NumPy convenience wrapper: y = [K, P, T] -> np.array([dK, dP, dT])."""
    import numpy as np
    K, P, T = y
    dK, dP, dT = rhs_components(K, P, T, u, p)
    return np.array([dK, dP, dT])
