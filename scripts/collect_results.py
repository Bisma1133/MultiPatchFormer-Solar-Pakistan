import pandas as pd, numpy as np, glob
T = "results/notebook_tables/"
def pm(df):
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    for c in list(df.columns):
        if not pd.api.types.is_numeric_dtype(df[c]) and df[c].astype(str).str.contains("±").any():
            s = df[c].astype(str).str.split("±", expand=True)
            df[c] = s[0].astype(float); df[c + "_sd"] = s[1].astype(float)
    return df
HS = [1, 3, 6, 24, 96, 168, 336, 672]
fl = [(f"nb01_{10+i}_c28.csv", h) for i, h in enumerate(HS)] + [(f"nb02_{12+i}_c28.csv", h) for i, h in enumerate(HS)]
R = pd.concat([pm(pd.read_csv(T + f)).assign(H=h) for f, h in fl])
# refs appear in both notebooks: check identical, keep one copy
refs = R[R.model.isin(["Seasonal persistence", "Smart persistence", "Climatology", "CLIPER"])]
chk = refs.groupby(["Region", "H", "model"]).RMSE.agg(["min", "max"]); assert (chk["max"] - chk["min"]).max() < 1e-9, chk
R = R.drop_duplicates(["Region", "H", "model"]).reset_index(drop=True)
R.to_csv("results/res_main.csv", index=False)
P = pd.concat([pm(pd.read_csv(T + "nb01_18_c28.csv")), pm(pd.read_csv(T + "nb02_20_c28.csv"))]).drop_duplicates(["H", "model"])
P.to_csv("results/res_pooled.csv", index=False)
CA = pd.concat([pm(pd.read_csv(T + "nb01_19_c28.csv")), pm(pd.read_csv(T + "nb02_21_c28.csv"))]).drop_duplicates(["H", "model"])
CA.to_csv("results/res_cityavg.csv", index=False)
E = pd.read_csv(T + "nb01_32_c36.csv").merge(pd.read_csv(T + "nb02_34_c36.csv"), on=["Region", "H"]).drop(columns=[c for c in ["model_x", "model_y", "model"] if c in []])
E = E[[c for c in E.columns if c not in ("model_x", "model_y")]]
E.to_csv("results/res_ensemble.csv", index=False)
D = pd.concat([pm(pd.read_csv(T + f)) for f in ["nb01_33_c38.csv", "nb01_34_c38.csv", "nb01_35_c38.csv", "nb02_35_c37.csv"]])
D.to_csv("results/res_dm.csv", index=False)
for n, f in [("ablation", "nb02_36_c39.csv"), ("ablation_avg", "nb02_37_c39.csv"), ("blind", "nb02_38_c41.csv"), ("missing", "nb02_39_c43.csv"),
             ("missing_skill", "nb02_40_c43.csv"), ("compute_mpf", "nb02_41_c45.csv"), ("compute_base", "nb01_36_c40.csv"), ("split", "nb01_06_c11.csv"),
             ("qc", "nb01_05_c9.csv"), ("dup", "nb01_04_c7.csv"), ("selL_base", "nb01_09_c24.csv"), ("selL_mpf", "nb02_11_c24.csv"),
             ("hyper_base", "nb01_07_c22.csv"), ("hyper_mpf", "nb02_09_c22.csv"), ("params_mpf", "nb02_07_c20.csv"),
             ("season24_base", "nb01_24_c31.csv"), ("season24_mpf", "nb02_26_c31.csv"), ("sky24_base", "nb01_26_c31.csv"), ("sky24_mpf", "nb02_28_c31.csv"),
             ("season1_base", "nb01_29_c34.csv"), ("season1_mpf", "nb02_31_c34.csv"), ("sky1_base", "nb01_31_c34.csv"), ("sky1_mpf", "nb02_33_c34.csv"),
             ("leadday_base", "nb01_22_c30.csv"), ("leadday_mpf", "nb02_24_c30.csv"), ("leadhour_base", "nb01_27_c34.csv"), ("leadhour_mpf", "nb02_29_c34.csv")]:
    pm(pd.read_csv(T + f)).to_csv("results/res_{n}.csv", index=False)
print(R.groupby("model").size()); print(P[P.H == 1][["model", "RMSE", "Skill_pct"]])
