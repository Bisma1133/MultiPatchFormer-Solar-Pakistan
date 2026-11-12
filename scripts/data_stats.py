"""Descriptive statistics per site (Table I of the paper)."""
import pandas as pd, numpy as np
D = pd.read_csv("data/Solcast_4cities_2020_2024.csv")
D["t"] = pd.to_datetime(D.period_end) + pd.Timedelta(hours=4); D["date"] = D.t.dt.date      # local period start
rows = []
for c, d in D.groupby("Region"):
    day = d.clearsky_ghi >= 20; k = (d.ghi / d.clearsky_ghi).where(day).clip(0, 1.4)
    fd = d.groupby("date").size(); fd = fd[fd == 24].index; dd = d[d.date.isin(fd)]
    rows.append(dict(Region=c, hours=len(d), H_ghi=dd.ghi.sum() / len(fd) / 1000, H_gti=dd.gti.sum() / len(fd) / 1000, ghi_day=d.ghi[day].mean(),
                     k_mean=k.mean(), k_std=k.std(), k_gt1=(k > 1).mean() * 100, T=d.air_temp.mean(), RH=d.relative_humidity.mean(), WS=d.wind_speed_10m.mean()))
pd.DataFrame(rows).set_index("Region").to_csv("results/data_stats.csv"); print(pd.DataFrame(rows).round(3))
