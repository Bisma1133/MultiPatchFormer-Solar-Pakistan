"""PV economics under the NEPRA (Prosumer) Regulations 2026 (net billing).
Self-consumed energy avoids the import tariff; exported energy is credited at the national average
energy purchase price (NAEPP). The self-consumption share is an assumption (no load data)."""
import os, sys
import pandas as pd, numpy as np
PVF = sys.argv[1] if len(sys.argv) > 1 else "tables/pv_energy.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "tables/economics.csv"
PV = pd.read_csv(PVF).set_index("Region")
FX = 279.0
# capex PKR (10 kWp); import tariff avoided by self-consumption (PKR/kWh); export credit (PKR/kWh);
# self-consumption share; system losses; degradation /yr; O&M share of capex /yr
SC = {
    "Conservative": dict(capex=1_450_000, imp=33.38, exp=9.41, sc=0.40, loss=0.20, deg=0.008, om=0.015),
    "Base":         dict(capex=1_200_000, imp=33.38, exp=10.00, sc=0.60, loss=0.14, deg=0.005, om=0.010),
    "Optimistic":   dict(capex=900_000,   imp=41.78, exp=11.00, sc=0.80, loss=0.10, deg=0.004, om=0.005),
    # upper bound used in the previous version: every generated unit valued at the average tariff
    "All units at tariff": dict(capex=1_200_000, imp=34.00, exp=34.00, sc=1.00, loss=0.14, deg=0.005, om=0.010),
}


def payback_net(E1, value, capex, deg, om, Y=40):
    cum = 0.0
    for y in range(1, Y + 1):
        cf = E1 * (1 - deg) ** (y - 1) * value - om * capex
        if cf <= 0:
            return np.nan
        if cum + cf >= capex:
            return y - 1 + (capex - cum) / cf
        cum += cf
    return np.nan


rows = []
for s, p in SC.items():
    v = p["sc"] * p["imp"] + (1 - p["sc"]) * p["exp"]          # value of one generated kWh
    for c in ["Islamabad", "Karachi", "Lahore", "Multan"]:
        r = PV.loc[c]
        scale = (1 - p["loss"]) / (1 - 0.14)
        for plane, E, lo, hi in [("POA", r.Epoa, r.Epoa_lo, r.Epoa_hi), ("Horizontal", r.Ehor, r.Ehor_lo, r.Ehor_hi)]:
            E1 = E * scale
            gross_sc, gross_exp = E1 * p["sc"] * p["imp"], E1 * (1 - p["sc"]) * p["exp"]
            gross = gross_sc + gross_exp
            net = gross - p["om"] * p["capex"]
            rows.append(dict(scenario=s, Region=c, plane=plane, capex_pkr=p["capex"], capex_usd=p["capex"] / FX,
                             imp_pkr=p["imp"], exp_pkr=p["exp"], self_cons=p["sc"], value_pkr=v, value_usd=v / FX,
                             loss=p["loss"], deg=p["deg"], om=p["om"], E=E1, E_lo=lo * scale, E_hi=hi * scale,
                             specific_yield=E1 / 10, gross_sc_usd=gross_sc / FX, gross_exp_usd=gross_exp / FX,
                             gross_usd=gross / FX, om_usd=p["om"] * p["capex"] / FX, net_usd=net / FX,
                             payback_simple=p["capex"] / gross, payback_simple_lo=p["capex"] / (hi * scale * v),
                             payback_simple_hi=p["capex"] / (lo * scale * v),
                             payback_net=payback_net(E1, v, p["capex"], p["deg"], p["om"])))
EC = pd.DataFrame(rows)
EC.to_csv(OUT, index=False)
pd.set_option("display.width", 250)
print(EC[EC.plane == "POA"][["scenario", "Region", "value_pkr", "E", "gross_usd", "net_usd", "payback_simple", "payback_net"]].round(2).to_string())
