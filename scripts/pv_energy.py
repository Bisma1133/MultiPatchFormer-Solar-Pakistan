import os; os.makedirs("tables", exist_ok=True)
# Hourly PV energy model (PVWatts-style losses), POA = Solcast GTI (tilt 23 deg, south-facing) and, for comparison, horizontal GHI.
import pandas as pd, numpy as np, json
D = pd.read_csv("data/Solcast_4cities_2020_2024_csmodel.csv")
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
    # Season-balanced annual values: mean daily value of each calendar month over all complete days of that month
    # (2020-2024), times the number of days in the month. A year with missing months (Karachi 2024 has no December)
    # therefore does not over-weight the remaining seasons. Year-to-year range: complete years only (all 12 months).
    dd = dd.assign(month=pd.to_datetime(dd.date).dt.month)
    dly = dd.groupby(["year", "month", "date"]).agg(Epoa=("E_poa", "sum"), Ehor=("E_hor", "sum"), Hpoa=("gti", "sum"), Hghi=("ghi", "sum")).reset_index()
    ndays = pd.Series({m: pd.Period(f"2021-{m:02d}").days_in_month for m in range(1, 13)})
    mon = dly.groupby("month")[["Epoa", "Ehor", "Hpoa", "Hghi"]].mean()
    ann = mon.mul(ndays, axis=0).sum()                         # one season-balanced year (365 days)
    ym = dly.groupby(["year", "month"])[["Epoa", "Ehor"]].mean()
    years = [y for y in sorted(dly.year.unique()) if ym.loc[y].shape[0] == 12]
    yr = pd.DataFrame({y: ym.loc[y].mul(ndays, axis=0).sum() for y in years}).T
    tot = dict(Region=c, days=int(dly.date.nunique()), complete_years=len(years), Hghi_d=ann.Hghi / 365 / 1000, Hpoa_d=ann.Hpoa / 365 / 1000,
               Epoa=ann.Epoa, Ehor=ann.Ehor, Epoa_lo=yr.Epoa.min(), Epoa_hi=yr.Epoa.max(), Ehor_lo=yr.Ehor.min(), Ehor_hi=yr.Ehor.max(),
               Epoa_daycount=dd.E_poa.sum() / dly.date.nunique() * 365)
    tot["PR_poa"] = tot["Epoa"] / (P_DC * tot["Hpoa_d"] * 365); tot["PR_hor"] = tot["Ehor"] / (P_DC * tot["Hghi_d"] * 365)
    tot["poa_gain_pct"] = 100 * (tot["Hpoa_d"] / tot["Hghi_d"] - 1)
    rows.append(tot)
PV = pd.DataFrame(rows); PV.to_csv("tables/pv_energy.csv", index=False); print(PV.round(3).to_string())
