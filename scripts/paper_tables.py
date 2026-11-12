import pandas as pd, numpy as np, json
N = "results/"
R = pd.read_csv(N + "res_main.csv"); P = pd.read_csv(N + "res_pooled.csv"); S = pd.read_csv(N + "res_stats_vs_cliper.csv")
RR = pd.read_csv(N + "extra/ridge_results.csv"); ST = pd.read_csv(N + "data_stats.csv").set_index("Region")
AB = pd.read_csv(N + "res_ablation.csv"); ABA = pd.read_csv(N + "res_ablation_avg.csv"); BL = pd.read_csv(N + "res_blind.csv")
FB = pd.read_csv(N + "res_fallback.csv"); EC = pd.read_csv(N + "pv/economics2.csv"); PV = pd.read_csv(N + "pv/pv_energy.csv").set_index("Region")
EX = pd.read_csv(N + "extra/extended_years.csv")
SITES = ["Islamabad", "Karachi", "Lahore", "Multan"]; HS = [1, 3, 6, 24, 96, 168, 336, 672]
DEEP = ["MultiPatchFormer", "PatchTST", "DLinear", "LSTM"]; LM = DEEP + ["CLIPER+ridge"]
REFS = ["Seasonal persistence", "Smart persistence", "Climatology", "CLIPER"]
neg = lambda s: s.replace("-", "$-$")
T = {}
# sites
info = {"Islamabad": (33.70, 73.04, 540, "Cwa"), "Karachi": (24.86, 67.00, 10, "BWh"), "Lahore": (31.52, 74.36, 217, "BSh"), "Multan": (30.19, 71.49, 122, "BWh")}
L = []
for c in SITES:
    la, lo, el, k = info[c]; s = ST.loc[c]
    L.append(f"{c} & {la:.2f}$^\\circ$N & {lo:.2f}$^\\circ$E & {el} & {k} & {s.H_ghi:.2f} & {s.H_gti:.2f} & {s.ghi_day:.0f} & {s.k_mean:.2f} & {s.k_std:.2f} & {s["T"]:.1f} & {s.RH:.0f} & {s.WS:.1f} \\\\")
T["TAB_SITES"] = "\n".join(L)
# reference RMSE
L = []
for c in SITES + ["Pooled"]:
    L.append(r"\multicolumn{9}{l}{\textit{%s}}\\" % ("Pooled (four sites)" if c == "Pooled" else c))
    for r in REFS:
        d = P[P.model == r].set_index("H") if c == "Pooled" else R[(R.Region == c) & (R.model == r)].set_index("H")
        L.append(f"\\quad {r} & " + " & ".join(f"{d.loc[h, 'RMSE']:.1f}" for h in HS) + r" \\")
T["TAB_REF"] = "\n".join(L)
# skill table mean (sd) ; ridge deterministic
L = []
for c in SITES + ["Pooled"]:
    L.append(r"\multicolumn{9}{l}{\textit{%s}}\\" % ("Pooled (four sites)" if c == "Pooled" else c))
    cl = (P[P.model == "CLIPER"].set_index("H").RMSE if c == "Pooled" else R[(R.Region == c) & (R.model == "CLIPER")].set_index("H").RMSE)
    L.append(r"\quad \textit{CLIPER RMSE} & " + " & ".join(f"\\textit{{{cl.loc[h]:.1f}}}" for h in HS) + r" \\")
    vals = {}
    for m in LM:
        if m == "CLIPER+ridge":
            d = RR[RR.Region == c].set_index("H"); vals[m] = [(d.loc[h, "Skill_pct"], None) for h in HS]
        elif c == "Pooled":
            d = P[P.model == m].set_index("H"); vals[m] = [(d.loc[h, "Skill_pct"], d.loc[h, "Skill_pct_sd"]) for h in HS]
        else:
            d = R[(R.Region == c) & (R.model == m)].set_index("H"); vals[m] = [(d.loc[h, "Skill_pct"], d.loc[h, "Skill_pct_sd"]) for h in HS]
    for m in LM:
        cells = []
        for i in range(len(HS)):
            v, sd = vals[m][i]; best = max(vals[x][i][0] for x in LM)
            s = f"{v:.1f}" if sd is None else f"{v:.1f}\\,({sd:.1f})"
            s = neg(s)
            if np.isclose(v, best): s = r"\textbf{" + s + "}"
            if c != "Pooled" and m != "CLIPER+ridge":
                q = S[(S.model == m) & (S.Region == c) & (S.H == HS[i])].q_BH.iloc[0]
                if q < 0.05: s += "$^{*}$"
            if c != "Pooled" and m == "CLIPER+ridge":
                q = S[(S.model == m) & (S.Region == c) & (S.H == HS[i])].q_BH.iloc[0]
                if q < 0.05: s += "$^{*}$"
            cells.append(s)
        name = "Ridge correction" if m == "CLIPER+ridge" else m
        L.append(f"\\quad {name} & " + " & ".join(cells) + r" \\")
T["TAB_SKILL"] = "\n".join(L)
# significance summary
L = []
for m in LM:
    s = S[S.model == m]
    name = "Ridge correction" if m == "CLIPER+ridge" else m
    L.append(f"{name} & {(s.sig_raw == 'better').sum()} & {(s.sig_raw == 'worse').sum()} & {(s.sig_BH == 'better').sum()} & {(s.sig_BH == 'worse').sum()} \\\\")
T["TAB_SIG"] = "\n".join(L)
# ablation (four-site average + range of site changes)
names = {"full": "Full model", "no_multiscale": "w/o multi-scale", "no_channel_attention": "w/o channel attention", "no_multistep_decoder": "w/o multi-step decoder"}
L = []
for H, lab in [(6, "6 h"), (24, "1 d")]:
    a = ABA[ABA.H == H].set_index("variant"); ab = AB[AB.H == H]
    sdr = ab.RMSE_std
    L.append(r"\multicolumn{5}{l}{\textit{Horizon %s} (seed std of site RMSE: %.1f--%.1f W\,m$^{-2}$)}\\" % (lab, sdr.min(), sdr.max()))
    for v in ["full", "no_multiscale", "no_channel_attention", "no_multistep_decoder"]:
        r = a.loc[v]; site = ab[ab.variant == v].dRMSE_vs_full_pct
        d = "--" if v == "full" else neg(f"{r.dRMSE_vs_full_pct:+.1f}") + f" ({neg(f'{site.min():+.1f}')} to {neg(f'{site.max():+.1f}')})"
        L.append(f"{names[v]} & {int(r.params):,} & {r.RMSE:.2f} & {neg(f'{r.Skill_pct:.2f}')} & {d} \\\\")
T["TAB_ABL"] = "\n".join(L)
# blind test
L = []
for _, r in BL.iterrows():
    L.append(f"{r.Region} & {'6 h' if r.H == 6 else '1 d'} & {r.local_RMSE:.1f} & {r.blind_RMSE:.1f} & {r.CLIPER_RMSE:.1f} & {neg(f'{r.local_Skill_pct:.1f}')} & {neg(f'{r.blind_Skill_pct:.1f}')} \\\\")
T["TAB_BLIND"] = "\n".join(L)
# fallback
L = []
for _, r in FB.iterrows():
    L.append(f"{r.Region} & {'6 h' if r.H == 6 else '1 d'} & {r.CLIPER:.1f} & {r.Climatology:.1f} & {r.MPF_full:.1f} & {r.MPF_noK:.1f} & {neg(f'{r.noK_vs_clim_pct:+.1f}')} \\\\")
T["TAB_FALLBACK"] = "\n".join(L)
# extended training years
L = []
for _, r in EX.iterrows():
    hl = {1: "1 h", 3: "3 h", 6: "6 h", 24: "1 d", 96: "4 d", 168: "7 d", 336: "14 d"}[int(r.H)]
    L.append(f"{hl} & {r.CLIPER_3y:.2f} & {r.CLIPER_6y:.2f} & {r.ridge_3y:.2f} & {r.ridge_6y:.2f} \\\\")
T["TAB_EXT"] = "\n".join(L)
# economics base: POA and horizontal
L = []
for c in SITES:
    b = EC[(EC.scenario == "Base") & (EC.plane == "POA") & (EC.Region == c)].iloc[0]; h = EC[(EC.scenario == "Base") & (EC.plane == "Horizontal") & (EC.Region == c)].iloc[0]
    L.append(f"{c} & {PV.loc[c, 'Hpoa_d']:.2f} & {b.E:,.0f} ({b.E_lo:,.0f}--{b.E_hi:,.0f}) & {b.specific_yield:,.0f} & {h.specific_yield:,.0f} & {b.S_usd:,.0f} & {b.payback_simple:.1f} & {b.payback_net:.1f} \\\\")
T["TAB_ECON"] = "\n".join(L)
L = []
for s in ["Conservative", "Base", "Optimistic"]:
    e = EC[(EC.scenario == s) & (EC.plane == "POA")]; b = e.iloc[0]
    L.append(f"{s} & {b.capex_pkr:,.0f} ({b.capex_usd:,.0f}) & {b.tariff_pkr:.2f} ({b.tariff_usd:.3f}) & {100 * b.loss:.0f} & {100 * b.deg:.1f} & {100 * b.om:.1f} & " + " & ".join(f"{e[e.Region == c].payback_net.iloc[0]:.1f}" for c in SITES) + r" \\")
T["TAB_SCEN"] = "\n".join(L)
# energy error at 1 d: horizontal-plane conversion with site PR_hor
L = []
FCS = ["Smart persistence", "CLIPER", "MultiPatchFormer", "PatchTST", "DLinear", "LSTM", "CLIPER+ridge"]
rows = []
for c in SITES:
    g = R[(R.Region == c) & (R.H == 24)].set_index("model")
    meanobs = g.loc["CLIPER", "RMSE"] / (g.loc["CLIPER", "nRMSE_pct"] / 100)
    k = 10 * PV.loc[c, "PR_hor"] / 1000                        # kWh per (W/m2) per hour for 10 kWp
    for m in FCS:
        if m == "CLIPER+ridge":
            rr = RR[(RR.Region == c) & (RR.H == 24)].iloc[0]; rmse, mbe = rr.RMSE, rr.MBE
        else:
            rmse, mbe = g.loc[m, "RMSE"], g.loc[m, "MBE"]
        rows.append(dict(Region=c, model=m, rmse_kwh=k * rmse, bias_pct=100 * mbe / meanobs, meanobs=meanobs))
EE = pd.DataFrame(rows); EE.to_csv("results/energy_error.csv", index=False)
for m in FCS:
    e = EE[EE.model == m].set_index("Region")
    name = "Ridge correction" if m == "CLIPER+ridge" else m
    L.append(f"{name} & " + " & ".join(f"{e.loc[c, 'rmse_kwh']:.2f}" for c in SITES) + " & " + " & ".join(neg(f"{e.loc[c, 'bias_pct']:+.1f}") for c in SITES) + r" \\")
T["TAB_ENERGY"] = "\n".join(L)
import re
for k_ in T: T[k_] = re.sub(r'\$-\$0\.0(?![0-9])', '0.0', T[k_])
T["TAB_ABL"] = re.sub(r'\(([^()/]+) to ([^()/]+)\)', r'[\1, \2]', T["TAB_ABL"].replace('w/o channel attention', 'w/o channel attn.').replace('w/o multi-step decoder', 'w/o multi-step dec.'))
json.dump(T, open("results/paper_tables.json", "w"), indent=1)
print(T["TAB_SKILL"][:1500]); print(T["TAB_ENERGY"]); print(T["TAB_ABL"]); print(EE.groupby("model").bias_pct.agg(["min", "max"]).round(2)); print(EE.rmse_kwh.agg(["min", "max"]).round(2), EE.meanobs.round(0).unique())
