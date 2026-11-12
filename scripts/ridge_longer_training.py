# Simple residual-learning baseline: CLIPER + per-lead-time ridge correction (closed form, CPU).
import sys, os, json
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); from nbcode import load_pipeline
g = load_pipeline(9, data=os.environ.get("MPF_DATA_EXT", "data/Solcast_4cities_2017_2024.csv"), cfg_patch={"DATA_YEARS": (2017, 2024), "TRAIN_YEARS": (2017, 2022)})
globals().update(g)
import numpy as np, pandas as pd

FI = {f: FEATURES.index(f) for f in FEATURES}
print("features", FEATURES)

def feats(city, idx, H):
    """Small set of predictors known at issue time t (per lead h): returns (N, H, F) float32."""
    a = ARR[city]; N = len(idx); lead = np.arange(1, H + 1); out = idx[:, None] + lead
    kp, kc, _ = ref_components(city, idx, H)
    al = ALPHA[(city, kind(H))][:H][None, :]
    kref = np.clip(al * kp + (1 - al) * kc, 0, CFG["K_MAX"]).astype(np.float32)
    win = idx[:, None] + np.arange(-23, 1)
    kd, dd = a.k[win], a.day[win]
    kmean24 = (kd * dd).sum(1) / np.maximum(dd.sum(1), 1)
    sun = [(a.cs[idx - j] >= CFG["CS_MIN"]).astype(np.float32) for j in range(3)]
    kl = [a.k[idx - j] * sun[j] for j in range(3)]
    X = a.X
    rh, T, ws = X[idx, FI["relative_humidity"]], X[idx, FI["air_temp"]], X[idx, FI["wind_speed_10m"]]
    rh24 = X[win, FI["relative_humidity"]].mean(1)
    rep = lambda v: np.broadcast_to(v[:, None], (N, H))
    cols = [np.ones((N, H), np.float32), kref, kp, kc, rep(kmean24), rep(sun[0]), rep(kl[0]), rep(kl[1]), rep(kl[2]),
            rep(rh), rep(rh24), rep(T), rep(ws), X[out, FI["hour_sin"]], X[out, FI["hour_cos"]], X[out, FI["doy_sin"]], X[out, FI["doy_cos"]]]
    return np.stack([np.asarray(c, np.float32) for c in cols], -1), kref

def dataset(split, H, L):
    Fs, R, W, KR, meta = [], [], [], [], []
    for c in CITIES:
        a = ARR[c]
        i = a.windows(split, L, H)
        F, kref = feats(c, i, H)
        out = i[:, None] + np.arange(1, H + 1)
        Fs.append(F); R.append(a.k[out] - kref); W.append((a.cs[out] ** 2) * a.day[out]); KR.append(kref); meta.append((c, i))
    return Fs, R, W, KR, meta

LAMS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
def fit(Fs, R, W, lam):
    # pass 1: weighted mean / sd per lead and feature (no big concatenation)
    sw = sum(w.sum(0) for w in W)                                   # (H,)
    mu = sum(np.einsum("nhf,nh->hf", F, w) for F, w in zip(Fs, W)) / sw[:, None]; mu[:, 0] = 0
    var = sum(np.einsum("nhf,nh->hf", (F - mu) ** 2, w) for F, w in zip(Fs, W)) / sw[:, None]
    sd = np.sqrt(var) + 1e-6; sd[:, 0] = 1
    nf = Fs[0].shape[-1]; Hh = Fs[0].shape[1]
    A = np.zeros((Hh, nf, nf)); b = np.zeros((Hh, nf))
    for F, r, w in zip(Fs, R, W):
        Z = ((F - mu) / sd).astype(np.float32)
        A += np.einsum("nhf,nh,nhg->hfg", Z, w, Z, optimize=True); b += np.einsum("nhf,nh,nh->hf", Z, w, r, optimize=True)
    P = np.diag([0.0] + [1.0] * (nf - 1))
    B = np.stack([np.linalg.solve(A[h] + lam * sw[h] / 1e6 * P + 1e-9 * np.eye(nf), b[h]) for h in range(Hh)])
    return B, mu, sd

def apply(B, mu, sd, F, kref):
    Z = (F - mu) / sd
    return np.clip(kref + np.einsum("nhf,hf->nh", Z, B), 0, CFG["K_MAX"])

rows, PREDS = [], {}
for H in [h for h in CFG["HORIZONS"] if h <= 336]:
    L = input_lengths(H)[0]
    tr = dataset("train", H, L); va = dataset("val", H, L); te = dataset("test", H, L)
    best = None
    for lam in LAMS:
        B, mu, sd = fit(tr[0], tr[1], tr[2], lam)
        y = np.concatenate([ARR[c].ghi[i[:, None] + np.arange(1, H + 1)] for c, i in va[4]])
        cs = np.concatenate([ARR[c].cs[i[:, None] + np.arange(1, H + 1)] for c, i in va[4]])
        m = np.concatenate([ARR[c].day[i[:, None] + np.arange(1, H + 1)] for c, i in va[4]])
        p = np.concatenate([apply(B, mu, sd, F, kr) for F, kr in zip(va[0], va[3])]) * cs
        v = metrics(y, p, m)["RMSE"]
        if best is None or v < best[0]: best = (v, lam, B, mu, sd)
    v, lam, B, mu, sd = best
    for (c, i), F, kr in zip(te[4], te[0], te[3]):
        it, y, cs, m = test_targets(c, H)
        assert np.array_equal(it, i)
        k = apply(B, mu, sd, F, kr)
        PREDS[(c, H)] = (k * cs).astype(np.float32)
        refs = ref_forecasts(c, H)
        r = metrics(y, k * cs, m); rc = metrics(y, refs["CLIPER"], m)
        rows.append(dict(Region=c, H=H, model="CLIPER+ridge", lam=lam, val_RMSE=v, RMSE=r["RMSE"], MAE=r["MAE"], MBE=r["MBE"],
                         CLIPER_RMSE=rc["RMSE"], Skill_pct=100 * (1 - r["RMSE"] / rc["RMSE"])))
    # pooled
    Y, P_, Pc, M_ = [], [], [], []
    for c in CITIES:
        it, y, cs, m = test_targets(c, H); Y.append(y); P_.append(PREDS[(c, H)]); Pc.append(ref_forecasts(c, H)["CLIPER"]); M_.append(m)
    Y, P_, Pc, M_ = map(np.concatenate, (Y, P_, Pc, M_))
    r, rc = metrics(Y, P_, M_), metrics(Y, Pc, M_)
    rows.append(dict(Region="Pooled", H=H, model="CLIPER+ridge", lam=lam, RMSE=r["RMSE"], MAE=r["MAE"], MBE=r["MBE"], CLIPER_RMSE=rc["RMSE"], Skill_pct=100 * (1 - r["RMSE"] / rc["RMSE"])))
    print(H, lam, round(v, 2), [(x["Region"], round(x["Skill_pct"], 2)) for x in rows if x["H"] == H])
pd.DataFrame(rows).to_csv("results/extra/ridge_ext_results.csv", index=False)
np.savez_compressed("results/extra/ridge_ext_preds.npz", **{f"{c}_{H}": v for (c, H), v in PREDS.items()})
# references for sanity + later use
REF = []
for c in CITIES:
    for H in CFG["HORIZONS"]:
        it, y, cs, m = test_targets(c, H)
        for n, p in ref_forecasts(c, H).items():
            REF.append(dict(Region=c, H=H, model=n, **metrics(y, p, m)))
pd.DataFrame(REF).to_csv("results/extra/ref_ext.csv", index=False)
