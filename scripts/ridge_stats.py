import os, math
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); from nbcode import load_pipeline
g = load_pipeline(9, data=os.environ.get("MPF_DATA", "data/Solcast_4cities_2020_2024.csv"))
globals().update(g)
import numpy as np, pandas as pd
RP = np.load("results/extra/ridge_preds.npz")
rng = np.random.default_rng(0)
def wl(p, y, m): return np.sum(m * (p - y) ** 2, 1) / np.maximum(m.sum(1), 1)
rows = []; LOSS = {}
for c in CITIES:
    for H in CFG["HORIZONS"]:
        _, y, cs, m = test_targets(c, H); refs = ref_forecasts(c, H); pr = RP[f"{c}_{H}"]
        lr, lc = wl(pr, y, m), wl(refs["CLIPER"], y, m)
        LOSS[(c, H)] = dict(ridge=lr, CLIPER=lc, SP=wl(refs["Smart persistence"], y, m), clim=wl(refs["Climatology"], y, m))
        lag = H if H < 24 else math.ceil(H / 24)
        dm, p = dm_test(lr, lc, lag)
        # moving-block bootstrap of skill (blocks of forecasts; block length = max(lag, 7 days or 24 h))
        n = len(lr); b = max(lag, 7 if H >= 24 else 24); nb = int(np.ceil(n / b))
        se_ = (m * (pr - y) ** 2).sum(1); sc_ = (m * (refs["CLIPER"] - y) ** 2).sum(1); nh = m.sum(1)
        sk = []
        for _ in range(2000):
            st = rng.integers(0, n - b + 1, nb); idx = (st[:, None] + np.arange(b)).ravel()[:n]
            sk.append(100 * (1 - np.sqrt(se_[idx].sum() / sc_[idx].sum())))
        r = metrics(y, pr, m); rc = metrics(y, refs["CLIPER"], m)
        skill = 100 * (1 - r["RMSE"] / rc["RMSE"])
        # DM-derived approximate CI (same method usable for models without stored forecasts)
        d = lr.mean() - lc.mean(); se = abs(d / dm) if abs(dm) > 1e-9 else np.nan
        Lc = lc.mean(); tq = 1.96
        lo_dm, hi_dm = [100 * (1 - np.sqrt(max(1 + (d + s * tq * se) / Lc, 0))) for s in (+1, -1)]
        rows.append(dict(Region=c, H=H, model="CLIPER+ridge", RMSE=r["RMSE"], Skill_pct=skill, DM=dm, p_value=p,
                         CI_lo_boot=np.percentile(sk, 2.5), CI_hi_boot=np.percentile(sk, 97.5), CI_lo_dm=lo_dm, CI_hi_dm=hi_dm, n_fc=n))
S = pd.DataFrame(rows); S.to_csv("results/extra/ridge_stats.csv", index=False)
print(S.round(3).to_string())
np.save("results/extra/losses_refs_ridge.npy", {f"{c}|{H}": v for (c, H), v in LOSS.items()}, allow_pickle=True)
