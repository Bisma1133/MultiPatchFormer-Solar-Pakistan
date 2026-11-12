"""Extract every result table shown in the executed notebooks 01 and 02 into results/notebook_tables/*.csv."""
import json, io, os, pandas as pd
os.makedirs("results/notebook_tables", exist_ok=True)
for nb, tag in [("notebooks/01_Baselines.ipynb", "nb01"), ("notebooks/02_MultiPatchFormer.ipynb", "nb02")]:
    j = json.load(open(nb)); n = 0
    for i, c in enumerate(j["cells"]):
        if c["cell_type"] != "code": continue
        for o in c.get("outputs", []):
            d = o.get("data", {})
            if "text/html" in d and "<table" in "".join(d["text/html"]):
                for t in pd.read_html(io.StringIO("".join(d["text/html"]))):
                    n += 1; t.to_csv(f"results/notebook_tables/{tag}_{n:02d}_c{i}.csv", index=False)
    print(tag, n, "tables")
