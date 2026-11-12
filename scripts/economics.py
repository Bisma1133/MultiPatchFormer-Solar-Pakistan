import pandas as pd, numpy as np
PV = pd.read_csv("results/pv/pv_energy.csv").set_index("Region")
FX = 279.0
SC = {  # capex PKR (10 kWp), avoided tariff PKR/kWh, system losses, degradation /yr, O&M share of capex /yr
    "Conservative": dict(capex=1_450_000, tariff=22.00, loss=0.20, deg=0.008, om=0.015),
    "Base":         dict(capex=1_200_000, tariff=34.00, loss=0.14, deg=0.005, om=0.010),
    "Optimistic":   dict(capex=900_000,   tariff=41.68, loss=0.10, deg=0.004, om=0.005)}
def payback_net(E1, tariff, capex, deg, om, Y=30):
    cum = 0.0
    for y in range(1, Y + 1):
        cf = E1 * (1 - deg) ** (y - 1) * tariff - om * capex
        if cum + cf >= capex:
            return y - 1 + (capex - cum) / cf
        cum += cf
    return np.nan
rows = []
for s, p in SC.items():
    for c in ["Islamabad", "Karachi", "Lahore", "Multan"]:
        r = PV.loc[c]
        scale = (1 - p["loss"]) / (1 - 0.14)
        for plane, E, lo, hi in [("POA", r.Epoa, r.Epoa_lo, r.Epoa_hi), ("Horizontal", r.Ehor, r.Ehor_lo, r.Ehor_hi)]:
            E1 = E * scale
            S = E1 * p["tariff"]
            rows.append(dict(scenario=s, Region=c, plane=plane, capex_pkr=p["capex"], capex_usd=p["capex"] / FX, capex_usd_per_kw=p["capex"] / FX / 10,
                             tariff_pkr=p["tariff"], tariff_usd=p["tariff"] / FX, loss=p["loss"], deg=p["deg"], om=p["om"],
                             E=E1, E_lo=lo * scale, E_hi=hi * scale, specific_yield=E1 / 10, S_pkr=S, S_usd=S / FX,
                             payback_simple=p["capex"] / S, payback_simple_lo=p["capex"] / (hi * scale * p["tariff"]), payback_simple_hi=p["capex"] / (lo * scale * p["tariff"]),
                             payback_net=payback_net(E1, p["tariff"], p["capex"], p["deg"], p["om"])))
EC = pd.DataFrame(rows); EC.to_csv("results/pv/economics2.csv", index=False)
print(EC[EC.plane == "POA"][["scenario", "Region", "E", "specific_yield", "S_usd", "payback_simple", "payback_net"]].round(2).to_string())
print(EC[(EC.plane == "Horizontal") & (EC.scenario == "Base")][["Region", "E", "specific_yield", "payback_simple", "payback_net"]].round(2))
