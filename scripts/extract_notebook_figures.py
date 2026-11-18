"""Saves every figure shown in the executed notebooks to figures/<name>.png, using the figure name in its caption
(e.g. *02_F19b_example_forecasts_H6 — ...*). Run after the notebooks: python scripts/extract_notebook_figures.py"""
import json, base64, re, os, glob
os.makedirs("figures", exist_ok=True)
n = 0
for f in sorted(glob.glob("notebooks/0*.ipynb")):
    nb = json.load(open(f)); last, used, extra = None, set(), {}
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            d = o.get("data", {})
            md = d.get("text/markdown")
            if md:
                md = "".join(md) if isinstance(md, list) else md
                m = re.match(r"\s*\*`?(\d\d_F[0-9a-z]+_[^ *`]+)", md)
                if m: last = m.group(1)
            if "image/png" in d:
                img = d["image/png"]; img = base64.b64decode(img if isinstance(img, str) else "".join(img))
                if last and last not in used:
                    name = last; used.add(last)
                else:                      # further figures under the same caption (e.g. training curves per model)
                    extra[last] = extra.get(last, 0) + 1; name = f"{last}_part{extra[last]}"
                open(f"figures/{name}.png", "wb").write(img); n += 1
print(n, "figures written")
# figures shown without their own caption (training curves and Diebold-Mariano heatmaps, one per model)
REN = {"01_F01_validation_rmse_vs_input_length_part{}": "01_F02_training_curves_{}", "01_F19b_example_forecasts_H6_part{}": "01_T14_{}_diebold_mariano"}
for a, b in REN.items():
    for i, m in enumerate(["PatchTST", "DLinear", "LSTM"], 1):
        if os.path.exists(f"figures/{a.format(i)}.png"): os.replace(f"figures/{a.format(i)}.png", f"figures/{b.format(m)}.png")
for a, b in {"02_F01_validation_rmse_vs_input_length_part1": "02_F02_training_curves_MultiPatchFormer", "02_F19b_example_forecasts_H6_part1": "02_T14_diebold_mariano"}.items():
    if os.path.exists(f"figures/{a}.png"): os.replace(f"figures/{a}.png", f"figures/{b}.png")
