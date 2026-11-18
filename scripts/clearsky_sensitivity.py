"""Re-conversion test of the EARLIER version of the study (supplementary Table S9); not needed for the final results.

In the earlier version every forecast of the clear-sky index k is converted to GHI with the Solcast clear-sky GHI
of the target hour. Solcast computes this value with REST2 and time-varying aerosol and water-vapour inputs, so the
value for a future hour is not exactly known at issue time. (The final version therefore uses scripts/make_clearsky.py and
refits/retrains all models.) This script converts the earlier k forecasts, fitted with the original Solcast file, with clear-sky
GHI that is fully known at issue time and rescores them on the same hours:
  (a) Ineichen-Perez model with the monthly Linke-turbidity climatology of pvlib ("Ineichen-clim");
  (b) the same, multiplied by the month x hour ratio of Solcast to Ineichen clear-sky GHI over the training years
      2020-2022 ("Ineichen-cal"; uses only past data);
  (c) (b) multiplied by the ratio of Solcast to (b) over the daytime hours of the 24 h before issue ("Ineichen-persist":
      the atmospheric state known at issue carried forward).
Forecasts: references and ridge correction (all horizons; deterministic) and the regenerated seed-0 runs of the
four neural models (1 h, 6 h, 1 d; scripts/direct_comparison.py).
Output: tables/clearsky_sensitivity.csv (pooled RMSE and skill against CLIPER under each conversion)."""
import os, sys, glob, math
sys.path.insert(0, os.path.dirname(__file__)); from nbcode import load_pipeline
import numpy as np, pandas as pd, pvlib

g = load_pipeline(9, data=os.environ.get("MPF_DATA", "data/Solcast_4cities_2020_2024.csv"))
CITIES, CFG = g["CITIES"], g["CFG"]
PRED_DIR = os.environ.get("MPF_PRED_DIR", "outputs_direct/predictions_direct")
RIDGE = np.load(os.environ.get("MPF_RIDGE_PRED", "outputs_scripts/ridge_preds.npz"))
NEURAL = ["MultiPatchFormer", "PatchTST", "DLinear", "LSTM"]

# ---- clear-sky GHI known at issue time (hourly mean of four 15-min values inside each Solcast period)
CS_OP = {}
PERSIST_RATIO = {}
for c in CITIES:
    d = g["CITY_DF"][c]; si = CFG["SITE_INFO"][c]
    loc = pvlib.location.Location(si["lat"], si["lon"], tz="UTC", altitude=si["elev_m"])
    pe = pd.DatetimeIndex(pd.to_datetime(d.period_end, utc=True))
    acc = np.zeros(len(pe))
    for m in [52.5, 37.5, 22.5, 7.5]:
        t = pe - pd.Timedelta(minutes=m)
        acc += loc.get_clearsky(t, model="ineichen")["ghi"].to_numpy()       # monthly Linke-turbidity climatology
    clim = np.nan_to_num(acc / 4)
    a = g["ARR"][c]
    tr = a.train_rows & (clim > 20) & (a.cs > 20)
    key = a.month * 24 + a.hour_start
    ratio = pd.Series(a.cs[tr]).groupby(key[tr]).sum() / pd.Series(clim[tr]).groupby(key[tr]).sum()
    cal = clim * pd.Series(key).map(ratio).fillna(1.0).to_numpy()
    CS_OP[c] = {"Solcast": a.cs.astype(float), "Ineichen-clim": clim, "Ineichen-cal": cal}
    # (c) "Ineichen-persist": Ineichen-cal times the ratio Solcast/Ineichen-cal of the last daytime hours known at issue
    #     (mean over the daytime hours of the 24 h before issue), i.e. the current aerosol state carried forward.
    r = np.where((cal > 50) & (a.cs > 50), a.cs / np.maximum(cal, 1), np.nan)
    num = pd.Series(np.nan_to_num(r) * (~np.isnan(r))).rolling(24, min_periods=1).sum()
    den = pd.Series((~np.isnan(r)).astype(float)).rolling(24, min_periods=1).sum()
    PERSIST_RATIO[c] = (num / den.replace(0, np.nan)).fillna(1.0).to_numpy()

rows = []
for H in CFG["HORIZONS"]:
    have_nn = all(os.path.exists(f"{PRED_DIR}/{m}__H{H}__s0.npz") for m in NEURAL)
    sse = {}
    for c in CITIES:
        iss, y, cs, msk = g["test_targets"](c, H)
        out = iss[:, None] + np.arange(1, H + 1)
        k = {}
        refs = g["ref_forecasts"](c, H)
        for r in ["Smart persistence", "Climatology", "CLIPER"]:
            k[r] = np.where(cs > 1, refs[r] / np.maximum(cs, 1), 0)
        k["CLIPER+ridge"] = np.where(cs > 1, RIDGE[f"{c}_{H}"] / np.maximum(cs, 1), 0)
        if have_nn:
            for m in NEURAL:
                k[m] = np.load(f"{PRED_DIR}/{m}__H{H}__s0.npz")[c]
        variants = dict(CS_OP[c]); variants["Ineichen-persist"] = None
        for v, csv in variants.items():
            cso = csv[out] if csv is not None else CS_OP[c]["Ineichen-cal"][out] * PERSIST_RATIO[c][iss][:, None]
            fc = {m: np.clip(kk, 0, CFG["K_MAX"]) * cso for m, kk in k.items()}
            fc["Seasonal persistence"] = refs["Seasonal persistence"]          # past GHI only: unaffected
            for m, f in fc.items():
                e = msk * (f - y) ** 2
                s = sse.setdefault((v, m), [0.0, 0])
                s[0] += e.sum(); s[1] += msk.sum()
    for (v, m), (s, n) in sse.items():
        rows.append(dict(H=H, clear_sky=v, model=m, pooled_RMSE=math.sqrt(s / n)))
T = pd.DataFrame(rows)
cl = T[T.model == "CLIPER"].set_index(["H", "clear_sky"]).pooled_RMSE
T["skill_vs_CLIPER_pct"] = 100 * (1 - T.pooled_RMSE / T.set_index(["H", "clear_sky"]).index.map(cl).to_numpy())
os.makedirs("tables", exist_ok=True); T.round(4).to_csv("tables/clearsky_sensitivity.csv", index=False)
pd.set_option("display.width", 220)
print(T.pivot_table(index=["H", "model"], columns="clear_sky", values="skill_vs_CLIPER_pct").round(2).to_string())
print(T[T.model == "CLIPER"].pivot_table(index="H", columns="clear_sky", values="pooled_RMSE").round(2).to_string())
