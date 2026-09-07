import sys, json, time, torch
torch.set_num_threads(4)
from params import BASELINE
from pinn import pinn_vs_abm_mae

alpha = float(sys.argv[1]); seed = int(sys.argv[2])
Nr = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
t0 = time.time()
m = pinn_vs_abm_mae(alpha=alpha, p=BASELINE.copy(), Nr=Nr,
                    adam_epochs=10000, lbfgs_iters=1000, seed=seed, device="cpu")
rec = {"alpha": alpha, "seed": seed, "Nr": Nr,
       "mae": [float(x) for x in m], "sec": round(time.time() - t0)}
with open("results.jsonl", "a") as f:
    f.write(json.dumps(rec) + "\n")
print(rec, flush=True)
