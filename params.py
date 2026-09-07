"""
Baseline parameters and initial conditions for the fractional
green economic--ecological--technology model.

Values correspond to Table 3 of the manuscript
(CMDE-2605-3670-2-2).
"""

# Baseline parameter set (Table 3)
BASELINE = {
    "alpha": 0.85,   # memory (Caputo order)
    "s":     0.30,   # savings / investment rate
    "A":     1.00,   # total factor productivity
    "beta":  0.40,   # capital output elasticity
    "rho":   0.20,   # technology productivity elasticity
    "delta": 0.05,   # capital depreciation rate
    "gamma": 0.10,   # pollution damage to capital productivity
    "u":     0.05,   # pollution-control investment rate
    "eta":   0.20,   # pollution emission coefficient
    "mu":    0.10,   # natural pollution decay rate
    "theta": 0.30,   # technology-pollution abatement rate
    "omega": 0.50,   # pollution-control R&D efficiency
    "xi":    0.08,   # technology obsolescence / depreciation
}

# Initial conditions (K0, P0, T0)
IC = {"K0": 1.0, "P0": 0.2, "T0": 0.1}

# Memory exponents used in the figures
ALPHAS = [0.60, 0.75, 0.90, 1.00]

# Default numerical settings
TMAX = 50.0
H_ABM = 0.01          # ABM step size used in the paper
NR_PINN = 1000        # PINN collocation points used in the paper
