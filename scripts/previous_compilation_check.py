"""Traces the series of the earlier (faulty) merged table to the corrected series (supplementary Table S2, bottom).
Usage: python scripts/previous_compilation_check.py <earlier_merged_table.csv> [data/Solcast_4cities_2020_2024.csv]
The earlier merged table is not part of the data release; it is kept by the authors and available on request."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
OLD = sys.argv[1]; NEWF = sys.argv[2] if len(sys.argv) > 2 else "data/Solcast_4cities_2020_2024.csv"
sys.argv = [sys.argv[0], NEWF]
src = open(os.path.join(os.path.dirname(__file__), "data_integrity.py")).read()
exec(src[src.index("def cos_zenith"):src.index("\ndf = pd.read_csv(DATA)")])          # cos_zenith, implied_location
df = pd.read_csv(OLD); df["t"] = pd.to_datetime(df.period_end, utc=True) - pd.Timedelta(minutes=30)
df = df[(df.t >= "2019-12-31 19:00") & (df.t < "2024-12-31 19:00")]
new = pd.read_csv(NEWF); new["t"] = pd.to_datetime(new.period_end, utc=True) - pd.Timedelta(minutes=30)
rows = []
for lab in ["Islamabad", "Karachi", "Lahore", "Multan"]:
    d = df[df.Region == lab].set_index("t").sort_index(); cs = d.clearsky_ghi
    sub = cs[(cs.index.dayofyear % 7 == 0) & (cs > 50)]
    m = {}
    for c in ["Islamabad", "Karachi", "Lahore", "Multan"]:
        n = new[new.Region == c].set_index("t").ghi
        j = d.ghi.to_frame("o").join(n.rename("n"), how="inner"); j = j[cs.reindex(j.index) >= 20]
        m[f"identical_with_{c}"] = round(100 * (j.o == j.n).mean(), 1)
    rows.append(dict(previous_label=lab, implied_lon=implied_location(sub.index, sub.values)[1], **m))
T = pd.DataFrame(rows); os.makedirs("tables", exist_ok=True)
T.to_csv("tables/previous_compilation_check.csv", index=False); print(T.to_string())
