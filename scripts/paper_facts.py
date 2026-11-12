import pandas as pd, numpy as np, json
N = "results/"
R = pd.read_csv(N + "res_main.csv"); P = pd.read_csv(N + "res_pooled.csv"); S = pd.read_csv(N + "res_stats_vs_cliper.csv")
RR = pd.read_csv(N + "extra/ridge_results.csv"); B = pd.read_csv(N + "res_best.csv"); E = pd.read_csv(N + "res_ensemble.csv")
AB = pd.read_csv(N + "res_ablation.csv"); ABA = pd.read_csv(N + "res_ablation_avg.csv"); BL = pd.read_csv(N + "res_blind.csv")
MI = pd.read_csv(N + "res_missing.csv").set_index("missing_input"); FB = pd.read_csv(N + "res_fallback.csv")
EX = pd.read_csv(N + "extra/extended_years.csv"); PV = pd.read_csv(N + "pv/pv_energy.csv").set_index("Region"); EC = pd.read_csv(N + "pv/economics2.csv")
CM = pd.read_csv(N + "res_compute_mpf.csv"); CB = pd.read_csv(N + "res_compute_base.csv"); ST = pd.read_csv(N + "data_stats.csv").set_index("Region")
DM = pd.read_csv(N + "res_dm.csv")
SITES = ["Islamabad", "Karachi", "Lahore", "Multan"]; HS = [1, 3, 6, 24, 96, 168, 336, 672]
DEEP = ["MultiPatchFormer", "PatchTST", "DLinear", "LSTM"]; LM = DEEP + ["CLIPER+ridge"]
F = {}
f1 = lambda v: f"{v:.1f}"; f2 = lambda v: f"{v:.2f}"
def rng(vals, fmt=f1, sep="--"):
    lo, hi = min(vals), max(vals); return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}{sep}{fmt(hi)}"
def neg(s): return s.replace("-", "$-$") if not s.startswith("$") else s
# pooled skill table incl ridge
PS = P.pivot(index="H", columns="model", values="Skill_pct")
PS["CLIPER+ridge"] = RR[RR.Region == "Pooled"].set_index("H").Skill_pct
PSD = P.pivot(index="H", columns="model", values="Skill_pct_sd")
PSP = P.pivot(index="H", columns="model", values="Skill_SP_pct")
# CLIPER vs smart persistence (pooled)
for H, k in [(1, "1h"), (6, "6h"), (24, "1d"), (672, "28d")]:
    F[f"CLIP_SP_{k}"] = f1(PSP.loc[H, "CLIPER"])
F["SEAS_INTRA_RANGE"] = rng([R[(R.model == "Seasonal persistence") & (R.H == h)].RMSE.min() for h in [1]] + [R[(R.model == "Seasonal persistence") & (R.H == 1)].RMSE.max()], f1)
# per-model pooled intraday range
for m, k in zip(LM, ["MPF", "PTST", "DLIN", "LSTM", "RIDGE"]):
    F[f"{k}_INTRA"] = rng([PS.loc[h, m] for h in [1, 3, 6]], f1)
    F[f"{k}_1D"] = f1(PS.loc[24, m]); F[f"{k}_4D28D"] = rng([PS.loc[h, m] for h in [96, 168, 336, 672]], f1)
F["ALL_INTRA_MAX"] = f1(max(PS.loc[h, m] for h in [1, 3, 6] for m in LM))
F["DEEP_4D28D"] = rng([PS.loc[h, m] for h in [96, 168, 336, 672] for m in DEEP], f1)
F["BEST_INTRA_SITE"] = f1(max(S[(S.H <= 6)].Skill))
# best counts
bd = B.best_deep.value_counts(); ba = B.best_all.value_counts()
for m, k in zip(LM, ["MPF", "PTST", "DLIN", "LSTM", "RIDGE"]):
    F[f"BESTDEEP_{k}"] = str(int(bd.get(m, 0))); F[f"BESTALL_{k}"] = str(int(ba.get(m, 0)))
F["N_CASES"] = str(len(B)); F["BEST_BEATS_CLIPER"] = str(int(B.best_vs_cliper.sum()))
# significance counts
for m, k in zip(LM, ["MPF", "PTST", "DLIN", "LSTM", "RIDGE"]):
    s = S[S.model == m]
    for col, t in [("sig_raw", "RAW"), ("sig_BH", "BH")]:
        F[f"SIGB_{t}_{k}"] = str(int((s[col] == "better").sum())); F[f"SIGW_{t}_{k}"] = str(int((s[col] == "worse").sum()))
F["N_TESTS"] = str(len(S))
# seed variability
sd = S[S.model.isin(DEEP)]
F["SEEDSD_MEAN"] = f2(sd.Skill_sd.mean()); F["SEEDSD_MAX"] = f2(sd.Skill_sd.max())
F["SEEDSD_MEDIAN"] = f2(sd.Skill_sd.median())
# ensemble vs single
ens_gain = []
for m in DEEP:
    for c in SITES:
        for H in HS:
            ens_gain.append(E[(E.Region == c) & (E.H == H)][m].iloc[0] - S[(S.model == m) & (S.Region == c) & (S.H == H)].Skill.iloc[0])
F["ENS_GAIN_RANGE"] = rng(ens_gain, f2); F["ENS_GAIN_POS"] = str(int(sum(g >= -0.005 for g in ens_gain))); F["ENS_N"] = str(len(ens_gain))
# baseline pairwise DM counts (BH within family of pairwise baseline tests)
PW = DM[DM.comparison.str.contains("vs PatchTST|vs DLinear|vs LSTM") & ~DM.comparison.str.startswith("MultiPatchFormer")].copy()
PW["pair"] = PW.comparison.apply(lambda s: " vs ".join(sorted(s.split(" vs "))))
PW = PW.drop_duplicates(["Region", "H", "pair"])
p = PW.p_value.values; n = len(p); o = np.argsort(p); q = np.empty(n); q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
PW["q"] = np.minimum(q, 1); F["PW_N"] = str(n); F["PW_SIG_BH"] = str(int((PW.q < 0.05).sum())); F["PW_SIG_RAW"] = str(int((PW.p_value < 0.05).sum()))
PW.to_csv("results/pairwise_baselines.csv", index=False)
# ablation
a6 = ABA[ABA.H == 6].set_index("variant"); a24 = ABA[ABA.H == 24].set_index("variant")
F["ABL_MAXABS_SITE"] = f1(AB.dRMSE_vs_full_pct.abs().max())
F["ABL_SEEDSD"] = rng(AB.RMSE_std, f1)
F["ABL_CA_PARAMS"] = f"{int(a6.loc['full', 'params'] - a6.loc['no_channel_attention', 'params']):,}"
F["ABL_CA_PCT6"] = f"{100 * (1 - a6.loc['no_channel_attention', 'params'] / a6.loc['full', 'params']):.0f}"
F["ABL_CA_PCT24"] = f"{100 * (1 - a24.loc['no_channel_attention', 'params'] / a24.loc['full', 'params']):.0f}"
F["ABL_AVG_RANGE"] = rng(list(ABA[ABA.variant != "full"].dRMSE_vs_full_pct), f1)
# blind test
BL["d"] = BL.blind_minus_local_pct
F["BLIND_INC_RANGE"] = rng(BL.d, f1); F["BLIND_SKILL_RANGE"] = rng(BL.blind_Skill_pct, f1)
F["BLIND_KAR_6"] = f1(BL[(BL.Region == "Karachi") & (BL.H == 6)].d.iloc[0]); F["BLIND_KAR_24"] = f1(BL[(BL.Region == "Karachi") & (BL.H == 24)].d.iloc[0])
F["BLIND_LAH_6"] = f1(BL[(BL.Region == "Lahore") & (BL.H == 6)].d.iloc[0]); F["BLIND_LAH_24"] = f1(BL[(BL.Region == "Lahore") & (BL.H == 24)].d.iloc[0])
# missing input
mk6 = [MI.loc["k", f"6_{c}"] for c in SITES]; mk24 = [MI.loc["k", f"24_{c}"] for c in SITES]
F["MISSK_6"] = rng(mk6, f1, "--"); F["MISSK_24"] = rng(mk24, f1)
oth = MI.drop(["k", "none"])
F["MISS_OTHER_6"] = f1(max(oth[[f"6_{c}" for c in SITES]].max())); F["MISS_OTHER_24"] = f1(max(oth[[f"24_{c}" for c in SITES]].max()))
F["MISS_OTHER_24_VAR"] = oth[[f"24_{c}" for c in SITES]].max(axis=1).idxmax()
F["FB_NOK_VS_CLIM_6"] = rng(FB[FB.H == 6].noK_vs_clim_pct, f1); F["FB_NOK_VS_CLIM_24"] = rng(FB[FB.H == 24].noK_vs_clim_pct, f1)
F["FB_CLIM_VS_CLIPER_6"] = rng(FB[FB.H == 6].clim_vs_cliper_pct, f1); F["FB_CLIM_VS_CLIPER_24"] = rng(FB[FB.H == 24].clim_vs_cliper_pct, f1)
# extended years
F["EXT_CLIPER_CHG"] = rng(EX.cliper_change_pct, f2); F["EXT_RIDGE_CHG"] = rng(EX.ridge_change_pct, f2)
# compute
F["MPF_GPU_MS"] = rng(CM[CM.device == "cuda"].ms_per_forecast_median, f1); F["MPF_CPU_MS"] = rng(CM[CM.device == "cpu"].ms_per_forecast_median, f1)
F["BASE_MS"] = rng(CB.ms_per_forecast_median, f1); F["MPF_TRAIN_MIN"] = f1(CM.mean_train_time_s.iloc[0] / 60); F["MPF_TRAIN_MAX"] = f1(CM.max_train_time_s.iloc[0] / 60)
F["BASE_TRAIN_MIN"] = rng(CB.drop_duplicates("model").mean_train_time_s / 60, f1); F["MPF_BEST_EPOCH"] = f1(CM.mean_best_epoch.iloc[0])
F["BASE_BEST_EPOCH"] = rng(CB.drop_duplicates("model").mean_best_epoch, f1)
# data stats
F["H_GHI_RANGE"] = rng(ST.H_ghi, f2); F["H_GTI_RANGE"] = rng(ST.H_gti, f2)
F["KSD_ISL"] = f"{ST.loc['Islamabad', 'k_std']:.2f}"; F["KSD_KAR"] = f"{ST.loc['Karachi', 'k_std']:.2f}"; F["KSD_MUL"] = f"{ST.loc['Multan', 'k_std']:.2f}"; F["KSD_LAH"] = f"{ST.loc['Lahore', 'k_std']:.2f}"
# PV / economics (POA, base)
b = EC[(EC.scenario == "Base") & (EC.plane == "POA")].set_index("Region"); bh = EC[(EC.scenario == "Base") & (EC.plane == "Horizontal")].set_index("Region")
F["SY_RANGE"] = rng(b.specific_yield, lambda v: f"{v:,.0f}"); F["SY_HOR_RANGE"] = rng(bh.specific_yield, lambda v: f"{v:,.0f}")
F["E_RANGE"] = rng(b.E, lambda v: f"{v / 1000:.1f}"); F["S_USD_RANGE"] = rng(b.S_usd, lambda v: f"{v:,.0f}")
F["PB_SIMPLE"] = rng(b.payback_simple, f1); F["PB_NET"] = rng(b.payback_net, f1); F["PB_SIMPLE_HOR"] = rng(bh.payback_simple, f1)
F["POA_GAIN"] = rng(PV.poa_gain_pct, f1); F["PR_EFF"] = rng(PV.PR_poa, f2)
cons = EC[(EC.scenario == "Conservative") & (EC.plane == "POA")]; opt = EC[(EC.scenario == "Optimistic") & (EC.plane == "POA")]
F["PB_CONS"] = rng(cons.payback_net, f1); F["PB_OPT"] = rng(opt.payback_net, f1)
F["IAV_E"] = rng([100 * (b.loc[c, "E_lo"] / b.loc[c, "E"] - 1) for c in SITES] + [100 * (b.loc[c, "E_hi"] / b.loc[c, "E"] - 1) for c in SITES], f1)
F["CAPEX_USD"] = f"{b.capex_usd.iloc[0]:,.0f}"; F["CAPEX_USD_KW"] = f"{b.capex_usd_per_kw.iloc[0]:,.0f}"; F["TARIFF_USD"] = f"{b.tariff_usd.iloc[0]:.3f}"
json.dump(F, open("results/facts.json", "w"), indent=1)
for k, v in F.items(): print(k, "=", v)
