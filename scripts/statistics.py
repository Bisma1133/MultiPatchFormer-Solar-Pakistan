import pandas as pd, numpy as np
R = pd.read_csv("results/res_main.csv"); D = pd.read_csv("results/res_dm.csv"); RS = pd.read_csv("results/extra/ridge_stats.csv")
L = np.load("results/extra/losses_refs_ridge.npy", allow_pickle=True).item()
DEEP = ["MultiPatchFormer", "PatchTST", "DLinear", "LSTM"]; SITES = ["Islamabad", "Karachi", "Lahore", "Multan"]; HS = [1, 3, 6, 24, 96, 168, 336, 672]
D["model"] = D.comparison.str.split(" vs ").str[0]; D["other"] = D.comparison.str.split(" vs ").str[1]
rows = []
for m in DEEP:
    for c in SITES:
        for H in HS:
            g = R[(R.Region == c) & (R.H == H)].set_index("model")
            dm = D[(D.model == m) & (D.Region == c) & (D.H == H) & (D.other == "CLIPER")].iloc[0]
            rm, sdm, rc = g.loc[m, "RMSE"], g.loc[m, "RMSE_sd"], g.loc["CLIPER", "RMSE"]
            Lc = L[f"{c}|{H}"]["CLIPER"].mean()
            d = Lc * ((rm ** 2 + sdm ** 2) / rc ** 2 - 1)
            se = abs(d / dm.DM) if abs(dm.DM) > 1e-6 else np.nan
            lo, hi = [100 * (1 - np.sqrt(max(1 + (d + s * 1.96 * se) / Lc, 0))) for s in (+1, -1)]
            rows.append(dict(model=m, Region=c, H=H, RMSE=rm, RMSE_sd=sdm, Skill=g.loc[m, "Skill_pct"], Skill_sd=g.loc[m, "Skill_pct_sd"],
                             DM=dm.DM, p=dm.p_value, CI_lo=lo, CI_hi=hi))
for _, r in RS.iterrows():
    rows.append(dict(model="CLIPER+ridge", Region=r.Region, H=r.H, RMSE=r.RMSE, RMSE_sd=0, Skill=r.Skill_pct, Skill_sd=0, DM=r.DM, p=r.p_value, CI_lo=r.CI_lo_dm, CI_hi=r.CI_hi_dm, CI_lo_boot=r.CI_lo_boot, CI_hi_boot=r.CI_hi_boot))
S = pd.DataFrame(rows)
# Benjamini-Hochberg over the family of all learning-model-vs-CLIPER tests (5 models x 4 sites x 8 horizons = 160)
p = S.p.values; n = len(p); o = np.argsort(p); q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
S["q_BH"] = np.minimum(q, 1)
S["sig_raw"] = np.where(S.p < 0.05, np.where(S.DM < 0, "better", "worse"), "ns")
S["sig_BH"] = np.where(S.q_BH < 0.05, np.where(S.DM < 0, "better", "worse"), "ns")
S.to_csv("results/res_stats_vs_cliper.csv", index=False)
print("n tests", n)
print(pd.crosstab(S.model, [S.sig_raw]), "\n", pd.crosstab(S.model, [S.sig_BH]))
print(S[S.sig_BH != "ns"][["model", "Region", "H", "Skill", "CI_lo", "CI_hi", "p", "q_BH", "sig_BH"]].round(3).to_string())
# best learning model per site-horizon (seed-mean RMSE)
B = []
for c in SITES:
    for H in HS:
        g = S[(S.Region == c) & (S.H == H)].set_index("model")
        deep = g.loc[DEEP].RMSE.idxmin(); allm = g.RMSE.idxmin()
        cl = R[(R.Region == c) & (R.H == H) & (R.model == "CLIPER")].RMSE.iloc[0]
        B.append(dict(Region=c, H=H, best_deep=deep, best_all=allm, best_vs_cliper=g.RMSE.min() < cl, second_deep=g.loc[DEEP].RMSE.nsmallest(2).index[1],
                      gap_best_second=g.loc[DEEP].RMSE.nsmallest(2).diff().iloc[1]))
B = pd.DataFrame(B); B.to_csv("results/res_best.csv", index=False)
print(B.best_deep.value_counts(), B.best_all.value_counts(), "beats CLIPER:", B.best_vs_cliper.sum())
# --- direct comparison: MultiPatchFormer vs the strongest competing model (seed-based) ---
rows = []
for (c, H), g in S.groupby(["Region", "H"]):
    g = g.set_index("model"); b = g.drop("MultiPatchFormer").Skill.idxmax()
    d = g.loc[b, "Skill"] - g.loc["MultiPatchFormer", "Skill"]; sd = np.sqrt(g.loc[b, "Skill_sd"] ** 2 + g.loc["MultiPatchFormer", "Skill_sd"] ** 2)
    rows.append(dict(Region=c, H=H, best=b, delta=d, comb_sd=sd, ratio=d / sd if sd > 0 else np.inf))
pd.DataFrame(rows).to_csv("results/res_direct.csv", index=False)
# --- fallback when the clear-sky index is missing: MPF without k (climatology anchor) vs climatology ---
MI = pd.read_csv("results/res_missing.csv").set_index("missing_input"); rows = []
for H in [6, 24]:
    for c in SITES:
        g = R[(R.Region == c) & (R.H == H)].set_index("model").RMSE
        nok = g["MultiPatchFormer"] * (1 + MI.loc["k", f"{H}_{c}"] / 100)
        rows.append(dict(H=H, Region=c, CLIPER=g["CLIPER"], Climatology=g["Climatology"], SmartP=g["Smart persistence"], MPF_full=g["MultiPatchFormer"],
                         MPF_noK=round(nok, 2), noK_vs_clim_pct=round(100 * (1 - nok / g["Climatology"]), 2), clim_vs_cliper_pct=round(100 * (1 - g["Climatology"] / g["CLIPER"]), 2)))
pd.DataFrame(rows).to_csv("results/res_fallback.csv", index=False)
