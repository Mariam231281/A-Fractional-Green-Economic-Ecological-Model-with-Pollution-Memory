"""Background runner: real PINN ensemble for Table 4 (seeds) and Table 6
(collocation), full paper schedule, writing each result to results.jsonl
as soon as it completes."""
import json, time, sys
import numpy as np
from params import BASELINE
from pinn import pinn_vs_abm_mae

p = BASELINE.copy()
OUT = "results.jsonl"


def log(rec):
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()


def full(**kw):
    return pinn_vs_abm_mae(p=p, adam_epochs=10000, lbfgs_iters=1000,
                           device="cpu", **kw)


# ---- Table 4: MAE per variable, alpha in {0.70,0.85,1.00}, 2 seeds ----
for alpha in (0.70, 0.85, 1.00):
    for seed in (0, 1):
        t0 = time.time()
        m = full(alpha=alpha, Nr=1000, seed=seed)
        log({"kind": "table4", "alpha": alpha, "seed": seed,
             "mae": [float(x) for x in m], "sec": round(time.time() - t0)})

# ---- Table 6: collocation study, alpha=0.85, seed 0, mean MAE ----
for Nr in (250, 500, 2000):   # 1000 already covered by table4 alpha=0.85 seed0
    t0 = time.time()
    m = full(alpha=0.85, Nr=Nr, seed=0)
    log({"kind": "collocation", "Nr": Nr, "mae_mean": float(np.mean(m)),
         "sec": round(time.time() - t0)})

log({"kind": "DONE"})
