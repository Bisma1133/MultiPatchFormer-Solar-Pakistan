"""Data-integrity checks for the four Solcast series (supplementary Table S1).
For every site: request coordinates (from the original file name), record count and period, missing hours,
the longitude and latitude implied by the clear-sky GHI in the file (solar-noon time and day length),
and, for every pair of sites, the share of identical daytime GHI values and the daytime correlation.
Usage: python scripts/data_integrity.py [data/Solcast_4cities_2020_2024.csv]"""
import os, sys
import numpy as np, pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data/Solcast_4cities_2020_2024.csv"
FILES = {  # original Solcast file names (request coordinates, array type, tilt, azimuth, resolution)
    "Islamabad": "Islamabad_csv_33.699509_73.03629_fixed_23_180_PT60M.xlsx",
    "Karachi": "Karachi_csv_24.860734_67.001136_fixed_23_180_PT60M.csv",
    "Lahore": "Lahore_csv_31.52037_74.358747_fixed_23_180_PT60M.csv",
    "Multan": "Multan_csv_30.186356_71.488581_fixed_23_180_PT60M.csv"}
# label under which each series appeared in the earlier (faulty) compilation used before 13 Nov 2026
PREVIOUS = {"Islamabad": "Islamabad", "Karachi": "(absent)", "Lahore": "Karachi", "Multan": "Lahore and Multan"}


def cos_zenith(t, lat, lon):
    """NOAA approximation; t = UTC DatetimeIndex (centre of the hour); lat/lon arrays broadcast."""
    doy = t.dayofyear.values; hr = t.hour.values + t.minute.values / 60
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eot = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = 0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g)
    ha = np.radians((hr * 60 + eot + 4 * np.asarray(lon)[..., None]) / 4 - 180)
    la = np.radians(np.asarray(lat))[..., None]
    return np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)


def corr_at(t, cs, lat, lon):
    x = np.log(np.clip(cos_zenith(t, np.array([lat]), np.array([lon]))[0], 0.02, None)); y = np.log(cs)
    return float(np.corrcoef(x, y)[0, 1])


def implied_location(t, cs):
    best = (-1, None, None); y = np.log(cs)
    for step, box in [(0.5, (20, 40, 60, 80)), (0.05, None)]:
        la0, la1, lo0, lo1 = box if box else (best[1] - 0.6, best[1] + 0.6, best[2] - 0.6, best[2] + 0.6)
        LA, LO = np.meshgrid(np.arange(la0, la1 + 1e-9, step), np.arange(lo0, lo1 + 1e-9, step))
        cz = cos_zenith(t, LA.ravel(), LO.ravel())
        x = np.log(np.clip(cz, 0.02, None)); xm = x - x.mean(1, keepdims=True); ym = y - y.mean()
        r = (xm @ ym) / np.sqrt((xm ** 2).sum(1) * (ym ** 2).sum())
        i = int(np.argmax(r))
        best = (r[i], LA.ravel()[i], LO.ravel()[i])
    return round(float(best[1]), 2), round(float(best[2]), 2)

df = pd.read_csv(DATA)
if "clearsky_ghi_solcast" in df.columns:          # integrity checks use the clear-sky GHI delivered by Solcast
    df["clearsky_ghi"] = df["clearsky_ghi_solcast"]
df["t"] = pd.to_datetime(df.period_end, utc=True) - pd.Timedelta(minutes=30)       # centre of the hourly period
sites = list(FILES)
rows, piv = [], {}
for c in sites:
    d = df[df.Region == c].set_index("t").sort_index()
    piv[c] = d.ghi
    lat, lon = map(float, FILES[c].split("_")[2:4])
    full = pd.date_range("2020-01-01", "2025-01-01", freq="h", inclusive="left", tz="Asia/Karachi").tz_convert("UTC") - pd.Timedelta(hours=1) + pd.Timedelta(minutes=30)
    # location implied by the clear-sky GHI in the file: grid search for the (lat, lon) whose solar geometry
    # best explains log(clear-sky GHI) = a + b log(cos zenith) (one day in seven, sun well above the horizon)
    cs = d.clearsky_ghi
    sub = cs[(cs.index.dayofyear % 7 == 0) & (cs > 50)]
    lon_impl = implied_location(sub.index, sub.values)[1]          # timing fixes the longitude well; latitude is not identified
    best_site = min(FILES, key=lambda o: abs(float(FILES[o].split("_")[3]) - lon_impl))   # nearest request longitude
    rows.append(dict(Region=c, original_file=FILES[c], request_lat=lat, request_lon=lon, label_previous=PREVIOUS[c], label_corrected=c,
                     first=str(d.index.min()), last=str(d.index.max()), records=len(d), missing_hours=43848 - len(d),
                     implied_lon=lon_impl, best_matching_request=best_site))
S = pd.DataFrame(rows)
P = pd.DataFrame(piv)
cs_all = pd.DataFrame({c: df[df.Region == c].set_index("t").clearsky_ghi for c in sites})
pairs = []
for i, a in enumerate(sites):
    for b in sites[i + 1:]:
        m = (cs_all[a] >= 20) & (cs_all[b] >= 20) & P[a].notna() & P[b].notna()
        pairs.append(dict(site_a=a, site_b=b, daytime_hours=int(m.sum()), identical_daytime_pct=round(100 * (P[a][m] == P[b][m]).mean(), 2),
                          daytime_corr=round(P[a][m].corr(P[b][m]), 3)))
Q = pd.DataFrame(pairs)
os.makedirs("tables", exist_ok=True)
S.to_csv("tables/data_integrity_sites.csv", index=False); Q.to_csv("tables/data_integrity_pairs.csv", index=False)
pd.set_option("display.width", 250); print(S.to_string()); print(Q.to_string())
assert Q.identical_daytime_pct.max() < 5, "two sites share the same series"
assert (S.best_matching_request == S.Region).all(), "a file does not match its own request location"
assert (abs(S.implied_lon - S.request_lon) < 0.5).all(), "timing of clear-sky GHI does not match the request longitude"
print("All data-integrity checks passed.")
