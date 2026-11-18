# Sensitivity of the moving-block bootstrap interval (ridge skill vs CLIPER) to the block length.
import os, sys, math
sys.path.insert(0, os.path.dirname(__file__)); from nbcode import load_pipeline
g = load_pipeline(9, data=os.environ.get("MPF_DATA", "data/Solcast_4cities_2020_2024_csmodel.csv")); globals().update(g)
import numpy as np, pandas as pd
RP = np.load("outputs_scripts/ridge_preds.npz"); rows = []
for c in CITIES:
    for H in CFG["HORIZONS"]:
        _, y, cs, m = test_targets(c, H); refs = ref_forecasts(c, H); pr = RP[f"{c}_{H}"]
        se_ = (m * (pr - y) ** 2).sum(1); sc_ = (m * (refs["CLIPER"] - y) ** 2).sum(1); n = len(se_)
        lag = H if H < 24 else math.ceil(H / 24); b0 = max(lag, 7 if H >= 24 else 24)
        for f in [0.5, 1, 2, 4]:
            b = max(int(round(b0 * f)), lag); nb = int(np.ceil(n / b)); rng = np.random.default_rng(0); sk = []
            for _ in range(2000):
                st = rng.integers(0, n - b + 1, nb); idx = (st[:, None] + np.arange(b)).ravel()[:n]
                sk.append(100 * (1 - np.sqrt(se_[idx].sum() / sc_[idx].sum())))
            lo, hi = np.percentile(sk, [2.5, 97.5])
            rows.append(dict(Region=c, H=H, factor=f, block=b, n_blocks=nb, CI_lo=lo, CI_hi=hi, width=hi - lo))
B = pd.DataFrame(rows); os.makedirs("tables", exist_ok=True); B.to_csv("tables/bootstrap_block_check.csv", index=False)
w = B.pivot_table(index=["Region", "H"], columns="factor", values="width")
r = w.div(w[1], axis=0)
print(r.groupby(level="H").median().round(2)); print(B[B.factor == 1].groupby("H").block.first())
