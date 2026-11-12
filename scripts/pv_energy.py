# Hourly PV energy model (PVWatts-style losses), POA = Solcast GTI (tilt 23 deg, south-facing) and, for comparison, horizontal GHI.
import pandas as pd, numpy as np, json
D = pd.read_csv("data/Solcast_4cities_2020_2024.csv")
D["t"] = pd.to_datetime(D.period_end) + pd.Timedelta(hours=5) - pd.Timedelta(hours=1)   # local period start
D["year"] = D.t.dt.year
P_DC = 10.0          # kWp
GAMMA = -0.0035      # 1/degC, temperature coefficient of power (modern crystalline module)
NOCT = 45.0          # degC
LOSS = 0.14          # PVWatts default system losses (soiling, shading, mismatch, wiring, LID, nameplate, availability)
ETA_INV = 0.96       # inverter efficiency
def energy(g, ta):
    tc = ta + (NOCT - 20) / 800 * g
    return P_DC * g / 1000 * (1 + GAMMA * (tc - 25)) * (1 - LOSS) * ETA_INV        # kWh per hour
rows = []
for c, d in D.groupby("Region"):
    d = d.copy()
    d["E_poa"] = energy(d.gti.clip(lower=0), d.air_temp); d["E_hor"] = energy(d.ghi.clip(lower=0), d.air_temp)
    d["date"] = d.t.dt.date
    full_days = d.groupby("date").size(); full_days = full_days[full_days == 24].index
    dd = d[d.date.isin(full_days)]
    yr = dd.groupby("year").agg(days=("date", "nunique"), Epoa=("E_poa", "sum"), Ehor=("E_hor", "sum"), Hpoa=("gti", "sum"), Hghi=("ghi", "sum"))
    yr["Epoa_y"] = yr.Epoa / yr.days * 365; yr["Ehor_y"] = yr.Ehor / yr.days * 365
    yr["Hpoa_d"] = yr.Hpoa / yr.days / 1000; yr["Hghi_d"] = yr.Hghi / yr.days / 1000
    tot = dict(Region=c, days=int(yr.days.sum()), Hghi_d=dd.ghi.sum() / yr.days.sum() / 1000, Hpoa_d=dd.gti.sum() / yr.days.sum() / 1000,
               Epoa=dd.E_poa.sum() / yr.days.sum() * 365, Ehor=dd.E_hor.sum() / yr.days.sum() * 365,
               Epoa_lo=yr.Epoa_y.min(), Epoa_hi=yr.Epoa_y.max(), Ehor_lo=yr.Ehor_y.min(), Ehor_hi=yr.Ehor_y.max())
    tot["PR_poa"] = tot["Epoa"] / (P_DC * tot["Hpoa_d"] * 365); tot["PR_hor"] = tot["Ehor"] / (P_DC * tot["Hghi_d"] * 365)
    tot["poa_gain_pct"] = 100 * (tot["Hpoa_d"] / tot["Hghi_d"] - 1)
    rows.append(tot)
PV = pd.DataFrame(rows); PV.to_csv("results/pv/pv_energy.csv", index=False); print(PV.round(3).to_string())
