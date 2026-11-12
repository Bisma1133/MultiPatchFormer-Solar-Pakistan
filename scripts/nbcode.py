"""Run the data-preparation cells of notebook 01 so that scripts use exactly the same data, split, windows and references."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def load_pipeline(n_cells=9, data=None, out="outputs_scripts", cfg_patch=None):
    if data: os.environ["MPF_DATA"] = data
    os.environ.setdefault("MPF_OUT", out)
    nb = json.load(open(os.path.join(HERE, "..", "notebooks", "01_Baselines.ipynb")))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    g = {"__name__": "__main__"}
    for j, s in enumerate(cells[:n_cells]):
        exec(compile("\n".join(l for l in s.split("\n") if not l.strip().startswith(("!", "%"))), f"nb01_cell{j}", "exec"), g)
        if j == 2 and cfg_patch: g["CFG"].update(cfg_patch)
    return g
