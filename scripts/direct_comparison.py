"""Paired comparison of MultiPatchFormer with PatchTST, DLinear, LSTM and CLIPER+ridge (1 h, 3 h, 6 h, 1 d).

Step 1 (train): regenerates the test-year forecasts of the four learning models with the notebook code
(same data, split, selected input lengths, seeds 0-2, early stopping on 2023). If the Kaggle output
folders are available, set MPF_PRED_DIRS to their 'predictions' folders and the saved forecasts are used
instead of retraining.
Step 2 (test): Diebold-Mariano tests (Newey-West HAC, Harvey correction) on the paired loss series
(seed-mean forecast, daytime MSE per forecast), per site and pooled over the four sites, with
Benjamini-Hochberg correction over all tests.

Usage:  python scripts/direct_comparison.py train  [model] [H]   (runs the missing trainings)
        python scripts/direct_comparison.py test
"""
import os, sys, json, math, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MPF_OUT", "outputs_direct")
from nbcode import load_pipeline, HERE

HS = [1, 3, 6, 24]   # horizons whose forecasts are missing are skipped (the paper uses 1, 6 and 24)
MODELS = ["MultiPatchFormer", "PatchTST", "DLinear", "LSTM"]
# input lengths selected on the 2023 validation year by the notebooks (tables 01_T09 / 02_T09)
BEST_L = {"MultiPatchFormer": {1: 24, 3: 24, 6: 24, 24: 96}, "PatchTST": {1: 24, 3: 24, 6: 24, 24: 96},
          "DLinear": {1: 24, 3: 72, 6: 72, 24: 96}, "LSTM": {1: 24, 3: 168, 6: 72, 24: 336}}
SEEDS = [int(x) for x in os.environ.get("MPF_SEEDS", "").split(",") if x]   # e.g. MPF_SEEDS=0 for a single-seed run
PRED_DIR = os.path.join(os.environ["MPF_OUT"], "predictions_direct")


def pipeline():
    g = load_pipeline(12, data=os.environ.get("MPF_DATA", "data/Solcast_4cities_2020_2024_csmodel.csv"))
    nb = json.load(open(os.path.join(HERE, "..", "notebooks", "02_MultiPatchFormer.ipynb")))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    exec(compile(cells[11], "nb02_cell11", "exec"), g)          # MultiPatchFormer definition
    return g


def train(only_m=None, only_h=None):
    g = pipeline()
    os.makedirs(PRED_DIR, exist_ok=True)
    for H in HS:
        for m in MODELS:
            if (only_m and m != only_m) or (only_h and H != only_h):
                continue
            for s in SEEDS or g["CFG"]["SEEDS"]:
                f = os.path.join(PRED_DIR, f"{m}__H{H}__s{s}.npz")
                if os.path.exists(f):
                    continue
                rec, ks = g["train_one"](m, BEST_L[m][H], H, s)
                np.savez_compressed(f, **{c: ks[c].astype(np.float32) for c in g["CITIES"]})
                print(f"{m} H={H} seed={s}: val RMSE {rec['val_RMSE']:.2f} W/m2, epochs {rec['epochs_run']}, "
                      f"{rec['train_time_s'] / 60:.1f} min", flush=True)


def load_k(m, H, seeds, cities):
    """Seed-stacked clear-sky-index forecasts {city: (n_seeds, N, H)}."""
    kag = [d for d in os.environ.get("MPF_PRED_DIRS", "").split(os.pathsep) if d]
    for d in kag:
        f = os.path.join(d, f"{m}__{cities[0]}__H{H}.npz")
        if os.path.exists(f):
            return {c: np.load(os.path.join(d, f"{m}__{c}__H{H}.npz"))["k"].astype(np.float32) for c in cities}
    z = [np.load(os.path.join(PRED_DIR, f"{m}__H{H}__s{s}.npz")) for s in seeds]
    return {c: np.stack([zz[c] for zz in z]) for c in cities}


def dm_hac(d, h):
    """DM statistic with Newey-West HAC variance (lag h-1, Bartlett) and Harvey small-sample correction."""
    from scipy import stats
    d = np.asarray(d, float); n = len(d); dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, h):
        v += 2 * (1 - k / h) * (dc[k:] @ dc[:-k]) / n
    dm = d.mean() / math.sqrt(max(v, 1e-30) / n)
    dm *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return dm, 2 * stats.t.sf(abs(dm), n - 1)


def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def test():
    g = load_pipeline(9, data=os.environ.get("MPF_DATA", "data/Solcast_4cities_2020_2024_csmodel.csv"))
    CITIES, seeds = g["CITIES"], SEEDS or g["CFG"]["SEEDS"]
    rid_f = os.environ.get("MPF_RIDGE_PRED", "outputs_scripts/ridge_preds.npz")   # written by ridge_correction.py
    rid = np.load(rid_f) if os.path.exists(rid_f) else None
    rows, skill = [], []
    for H in HS:
        try:
            for m in MODELS:
                load_k(m, H, seeds, CITIES)
        except FileNotFoundError:
            print(f"H={H}: forecasts missing, skipped"); continue
        lag = H if H < 24 else math.ceil(H / 24)
        loss, se = {}, {}
        for m in MODELS + ["CLIPER+ridge", "CLIPER"]:
            loss[m], se[m] = {}, {}
            for c in CITIES:
                iss, y, cs, msk = g["test_targets"](c, H)
                if m == "CLIPER":
                    pred = g["ref_forecasts"](c, H)["CLIPER"]
                elif m == "CLIPER+ridge":
                    if rid is None:
                        continue
                    pred = rid[f"{c}_{H}"]                                     # ridge forecasts are stored as GHI
                else:
                    pred = load_k(m, H, seeds, CITIES)[c].mean(0) * cs       # seed-mean forecast
                e2 = msk * (pred - y) ** 2
                loss[m][c] = e2.sum(1) / np.maximum(msk.sum(1), 1)
                se[m][c] = (e2.sum(), msk.sum())
        for m in MODELS + ["CLIPER+ridge", "CLIPER"]:
            if len(se[m]) == len(CITIES):
                sse = sum(v[0] for v in se[m].values()); n = sum(v[1] for v in se[m].values())
                skill.append(dict(model=m, H=H, pooled_RMSE=math.sqrt(sse / n)))
        for o in MODELS[1:] + ["CLIPER+ridge"]:
            if len(loss[o]) < len(CITIES):
                continue
            for c in CITIES + ["Pooled"]:
                cs_ = CITIES if c == "Pooled" else [c]
                # loss series (per-forecast daytime MSE); pooled: the four sites' series stacked (HAC lag unchanged)
                d = np.concatenate([loss["MultiPatchFormer"][cc] - loss[o][cc] for cc in cs_])
                dm, p = dm_hac(d, lag)
                # RMSE over all scored hours (same aggregation as the main tables)
                ra = math.sqrt(sum(se["MultiPatchFormer"][cc][0] for cc in cs_) / sum(se["MultiPatchFormer"][cc][1] for cc in cs_))
                rb = math.sqrt(sum(se[o][cc][0] for cc in cs_) / sum(se[o][cc][1] for cc in cs_))
                rows.append(dict(H=H, Region=c, other=o, n_forecasts=len(d), RMSE_MPF=ra, RMSE_other=rb,
                                 diff_pct=100 * (ra / rb - 1), DM=dm, p_value=p))
    T = pd.DataFrame(rows)
    T["q_BH"] = bh(T.p_value.values)
    T["result"] = np.where(T.q_BH < 0.05, np.where(T.DM < 0, "MPF better", "MPF worse"), "no significant difference")
    os.makedirs("tables", exist_ok=True)
    T.round(4).to_csv("tables/direct_mpf_vs_competitors.csv", index=False)
    S = pd.DataFrame(skill)
    ref = S[S.model == "CLIPER"].set_index("H").pooled_RMSE
    S["skill_vs_CLIPER_pct"] = 100 * (1 - S.pooled_RMSE / S.H.map(ref))
    S.round(4).to_csv("tables/direct_pooled_skill_check.csv", index=False)
    print(T.round(3).to_string()); print(S.round(3).to_string())


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2] if len(sys.argv) > 2 else None, int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        test()
