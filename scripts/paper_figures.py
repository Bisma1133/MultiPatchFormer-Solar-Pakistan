import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
N = "results/"; O = "paper_figures/"; import os; os.makedirs(O, exist_ok=True)
R = pd.read_csv(N + "res_main.csv"); P = pd.read_csv(N + "res_pooled.csv"); RR = pd.read_csv(N + "extra/ridge_results.csv")
MI = pd.read_csv(N + "res_missing.csv").set_index("missing_input")
HS = [1, 3, 6, 24, 96, 168, 336, 672]; HL = ["1 h", "3 h", "6 h", "1 d", "4 d", "7 d", "14 d", "28 d"]; x = np.arange(len(HS))
SITES = ["Islamabad", "Karachi", "Lahore", "Multan"]
MODELS = ["MultiPatchFormer", "PatchTST", "DLinear", "LSTM", "Ridge correction"]
COL = {"MultiPatchFormer": "#1f4e9c", "PatchTST": "#d1495b", "DLinear": "#2a9d8f", "LSTM": "#e9a03b", "Ridge correction": "#555555"}
MK = {"MultiPatchFormer": "o", "PatchTST": "s", "DLinear": "^", "LSTM": "D", "Ridge correction": "x"}
def series(site, m):
    if m == "Ridge correction":
        d = RR[RR.Region == ("Pooled" if site is None else site)].set_index("H"); return d.loc[HS, "Skill_pct"].values, None
    d = (P[P.model == m] if site is None else R[(R.Region == site) & (R.model == m)]).set_index("H")
    return d.loc[HS, "Skill_pct"].values, d.loc[HS, "Skill_pct_sd"].values
fig, axs = plt.subplots(1, 5, figsize=(7.2, 2.7), sharey=True)
for ax, site, t in zip(axs, SITES + [None], SITES + ["Pooled (4 sites)"]):
    for m in MODELS:
        v, sd = series(site, m)
        ax.plot(x, v, marker=MK[m], ms=3, lw=1.1, color=COL[m], label=m, ls="--" if m == "Ridge correction" else "-")
        if sd is not None and m == "MultiPatchFormer": ax.fill_between(x, v - sd, v + sd, color=COL[m], alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.8); ax.axvspan(-0.4, 2.5, color="0.93", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(HL, rotation=65, fontsize=6.8); ax.set_title(t, fontsize=8.5); ax.grid(alpha=0.3, lw=0.5)
axs[0].set_ylabel("Skill vs CLIPER (%)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, ncol=5, fontsize=6.8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(w_pad=0.4, rect=(0, 0.07, 1, 1)); fig.savefig(O + "fig_skill_vs_horizon.pdf"); fig.savefig(O + "fig_skill_vs_horizon.png", dpi=250); plt.close()
# pooled RMSE: references + MPF + best learning model
fig, ax = plt.subplots(figsize=(3.4, 2.7))
for r, c, mk in [("Seasonal persistence", "0.55", "v"), ("Smart persistence", "#8c8c00", "<"), ("Climatology", "#17becf", ">"), ("CLIPER", "k", "o")]:
    ax.plot(x, P[P.model == r].set_index("H").loc[HS, "RMSE"], marker=mk, ms=3.5, lw=1.2, color=c, label=r)
ax.plot(x, P[P.model == "MultiPatchFormer"].set_index("H").loc[HS, "RMSE"], marker="o", ms=3, lw=1.1, ls="--", color=COL["MultiPatchFormer"], label="MultiPatchFormer")
ax.plot(x, P[P.model == "LSTM"].set_index("H").loc[HS, "RMSE"], marker="D", ms=3, lw=1.1, ls=":", color=COL["LSTM"], label="LSTM")
ax.set_xticks(x); ax.set_xticklabels(HL, rotation=60, fontsize=7.5); ax.set_ylabel("Pooled RMSE (W m$^{-2}$)"); ax.grid(alpha=0.3, lw=0.5)
ax.legend(fontsize=6.3, frameon=False); fig.tight_layout(); fig.savefig(O + "fig_rmse_vs_horizon.pdf"); fig.savefig(O + "fig_rmse_vs_horizon.png", dpi=250); plt.close()
# missing input
names = {"k": "clear-sky index $k$", "dni": "DNI", "dhi": "DHI", "gti": "GTI", "clearsky_ghi": "clear-sky GHI", "air_temp": "air temperature",
         "relative_humidity": "relative humidity", "wind_speed_10m": "wind speed", "wind_dir_sin": "wind dir. (sin)", "wind_dir_cos": "wind dir. (cos)",
         "hour_sin": "hour (sin)", "hour_cos": "hour (cos)", "doy_sin": "day of year (sin)", "doy_cos": "day of year (cos)"}
order = list(names)[::-1]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
for ax, H, t in zip(axs, [6, 24], ["Horizon 6 h", "Horizon 1 d"]):
    y = np.arange(len(order)); w = 0.2
    for i, (s, c) in enumerate(zip(SITES, ["#1f4e9c", "#e9a03b", "#2a9d8f", "#d1495b"])):
        ax.barh(y + (i - 1.5) * w, MI.loc[order, f"{H}_{s}"], height=w, color=c, label=s)
    ax.axvline(0, color="k", lw=0.7); ax.set_title(t, fontsize=9); ax.set_xlabel("Increase in test RMSE (%)"); ax.grid(axis="x", alpha=0.3, lw=0.5)
axs[0].set_yticks(np.arange(len(order))); axs[0].set_yticklabels([names[o] for o in order], fontsize=7.5)
axs[1].legend(fontsize=7, frameon=False, loc="lower right"); fig.tight_layout(); fig.savefig(O + "fig_missing_input.pdf"); fig.savefig(O + "fig_missing_input.png", dpi=250); plt.close()
# examples: rows Islamabad, Lahore, Multan of the notebook figure (Karachi row omitted, its cloudiest-week panel is truncated by the data gap)
im = Image.open("results/figures/02_F19b_example_forecasts_H6.png").convert("RGB"); W, Hh = im.size; r = Hh / 4
rows = [im.crop((0, int(i * r), W, int((i + 1) * r))) for i in (0, 2, 3)]
out = Image.new("RGB", (W, int(3 * r)), "white")
for i, rw in enumerate(rows): out.paste(rw, (0, int(i * r)))
out.save(O + "fig_examples_H6.png", dpi=(250, 250))
print("ok")
