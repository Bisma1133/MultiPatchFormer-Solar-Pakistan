"""Test forecasts per site and horizon (supplementary Table S7): issued forecasts, forecasts with at least one
scored hour, forecasts whose whole target window is real data, first and last issue time, scored hours.
Scored hours: daytime (clear-sky GHI >= 20 W/m2) hours with real data; gap-filled hours (used as inputs only)
are never scored. Every model and reference is scored on exactly these hours."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__)); from nbcode import load_pipeline
import numpy as np, pandas as pd
g = load_pipeline(9, data=os.environ.get("MPF_DATA", "data/Solcast_4cities_2020_2024_csmodel.csv"))
rows = []
for c in g["CITIES"]:
    a = g["ARR"][c]; real = a.valid & ~g["CITY_DF"][c]["gap_filled"].to_numpy(bool)
    for H in g["CFG"]["HORIZONS"]:
        iss, y, cs, m = g["test_targets"](c, H)
        out = iss[:, None] + np.arange(1, H + 1)
        scored = m.sum(1) > 0; complete = real[out].all(1)
        t = a.time_local
        rows.append(dict(Region=c, H=H, issued=len(iss), with_scored_hours=int(scored.sum()), complete_window=int(complete.sum()),
                         partial_window=int((scored & ~complete).sum()), no_scored_hour=int((~scored).sum()),
                         first_issue=str(t.iloc[iss[0]])[:16], last_issue=str(t.iloc[iss[scored][-1]])[:16], scored_hours=int(m.sum())))
T = pd.DataFrame(rows); os.makedirs("tables", exist_ok=True); T.to_csv("tables/forecast_counts.csv", index=False)
pd.set_option("display.width", 220); print(T.to_string())
