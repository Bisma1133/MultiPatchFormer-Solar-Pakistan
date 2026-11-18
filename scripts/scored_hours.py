"""Number of test forecasts and scored (daytime) hours per site and horizon; used to pool site RMSE values
(pooled RMSE = sqrt(sum_s n_s RMSE_s^2 / sum_s n_s)), e.g. for the pooled ablation table."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__)); from nbcode import load_pipeline
import pandas as pd
g = load_pipeline(9, data=os.environ.get("MPF_DATA", "data/Solcast_4cities_2020_2024_csmodel.csv"))
rows = []
for c in g["CITIES"]:
    for H in g["CFG"]["HORIZONS"]:
        iss, y, cs, m = g["test_targets"](c, H)
        rows.append(dict(Region=c, H=H, n_forecasts=len(iss), n_hours=int(m.sum())))
os.makedirs("tables", exist_ok=True); pd.DataFrame(rows).to_csv("tables/nscored.csv", index=False); print(pd.DataFrame(rows))
