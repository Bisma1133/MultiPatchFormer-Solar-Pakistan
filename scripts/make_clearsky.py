"""Replaces the Solcast clear-sky GHI by a clear-sky GHI that is known at any issue time.

Solcast computes its historical clear-sky GHI with REST2 and time-varying aerosol, water-vapour and ozone inputs, so
the value for a future hour contains information that is not available when a forecast is issued. This script
computes, for every hourly period, the Ineichen-Perez clear-sky GHI with the monthly Linke-turbidity climatology of
pvlib (mean of four 15-min values inside the period), and adjusts it per site, month and hour of day by the ratio of
Solcast to Ineichen clear-sky GHI over the training years 2020-2022 (past data only). The result is written to
`clearsky_ghi`, which the notebooks use for the clear-sky index, the daytime mask, CLIPER and the conversion to GHI;
the original Solcast value is kept in `clearsky_ghi_solcast` (used only by the data-integrity checks).

Usage: python scripts/make_clearsky.py IN.csv OUT.csv
"""
import sys, json, os
import numpy as np, pandas as pd, pvlib

IN, OUT = sys.argv[1], sys.argv[2]
CAL_YEARS = (2020, 2022)
cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "config", "config.json")))
SITES = cfg["SITE_INFO"]
df = pd.read_csv(IN)
if "clearsky_ghi_solcast" in df.columns:
    raise SystemExit("input already converted")
df["clearsky_ghi_solcast"] = df["clearsky_ghi"]
out = []
for c, d in df.groupby("Region", sort=False):
    s = SITES[c]
    loc = pvlib.location.Location(s["lat"], s["lon"], tz="UTC", altitude=s["elev_m"])
    pe = pd.DatetimeIndex(pd.to_datetime(d.period_end, utc=True))
    acc = np.zeros(len(pe))
    for m in [52.5, 37.5, 22.5, 7.5]:
        acc += loc.get_clearsky(pe - pd.Timedelta(minutes=m), model="ineichen")["ghi"].to_numpy()
    ine = np.nan_to_num(acc / 4)
    loc_t = (pe - pd.Timedelta(minutes=30)).tz_convert("Asia/Karachi")
    key = loc_t.month * 24 + loc_t.hour
    cal = (loc_t.year >= CAL_YEARS[0]) & (loc_t.year <= CAL_YEARS[1]) & (ine > 20) & (d.clearsky_ghi_solcast.to_numpy() > 20)
    ratio = (pd.Series(d.clearsky_ghi_solcast.to_numpy()[cal]).groupby(key[cal]).sum()
             / pd.Series(ine[cal]).groupby(key[cal]).sum())
    r = pd.Series(key).map(ratio).fillna(1.0).to_numpy()
    d = d.copy(); d["clearsky_ghi"] = np.round(ine * r, 1)
    out.append(d)
    print(f"{c}: calibration ratio {ratio.min():.3f}-{ratio.max():.3f} (median {ratio.median():.3f}); "
          f"mean daytime clear-sky GHI Solcast {d.clearsky_ghi_solcast[d.clearsky_ghi_solcast > 20].mean():.1f}, "
          f"model {d.clearsky_ghi[d.clearsky_ghi > 20].mean():.1f} W/m2")
pd.concat(out).to_csv(OUT, index=False)
print("written", OUT)
