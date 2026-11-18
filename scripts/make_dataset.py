"""Build the combined four-site file from the original Solcast downloads and check that the sites are independent.

Usage: python scripts/make_dataset.py RAW_DIR OUT_CSV [START_YEAR END_YEAR]
RAW_DIR holds one Solcast file per city, named as downloaded, e.g.
  Islamabad_csv_33.699509_73.03629_fixed_23_180_PT60M.(csv|xlsx)
The request coordinates and the array (fixed, tilt 23, azimuth 180) are read from the file name and printed.
Note: the Solcast downloads used in the paper end on 2024-11-29 for Karachi, Lahore and Multan. For Lahore and Multan,
29 Nov - 31 Dec 2024 were taken from an earlier download of the same Solcast series (identical in all overlapping hours);
Karachi has no data after 2024-11-29 00:00 UTC. The file used in the paper is archived on Zenodo (see README).
"""
import sys, glob, os, re, itertools, pandas as pd
raw, out = sys.argv[1], sys.argv[2]
y0, y1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (2020, 2024)
COLS = ["air_temp", "clearsky_dhi", "clearsky_dni", "clearsky_ghi", "clearsky_gti", "dhi", "dni", "ghi", "gti",
        "relative_humidity", "wind_direction_10m", "wind_speed_10m", "period_end", "period"]
s, e = f"{y0 - 1}-12-31T20:00:00+00:00", f"{y1}-12-31T19:00:00+00:00"     # local (UTC+5) years
parts = []
for f in sorted(glob.glob(os.path.join(raw, "*_PT60M.*"))):
    m = re.match(r"(?:.*-)?([A-Za-z]+)_csv_([\d.]+)_([\d.]+)_fixed_(\d+)_(\d+)_PT60M", os.path.basename(f))
    city, lat, lon, tilt, az = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    d = pd.read_excel(f) if f.endswith("xlsx") else pd.read_csv(f)
    d = d[COLS]; d = d[(d.period_end >= s) & (d.period_end <= e)].drop_duplicates("period_end").sort_values("period_end")
    d["Region"] = city; parts.append(d)
    print(f"{city}: lat {lat}, lon {lon}, tilt {tilt}, azimuth {az}; {len(d)} hours, {d.period_end.min()} to {d.period_end.max()}")
D = pd.concat(parts, ignore_index=True)
P = D.pivot(index="period_end", columns="Region", values="ghi"); C = D.pivot(index="period_end", columns="Region", values="clearsky_ghi")
for a, b in itertools.combinations(P.columns, 2):
    m = (C[a] >= 20) & (C[b] >= 20) & P[a].notna() & P[b].notna()
    same = (P[a][m] == P[b][m]).mean()
    print(f"{a}-{b}: identical daytime GHI in {100 * same:.2f}% of hours, correlation {P[a][m].corr(P[b][m]):.3f}")
    assert same < 0.5, f"{a} and {b} look like the same series"
D.to_csv(out, index=False); print("written", out, D.shape)
