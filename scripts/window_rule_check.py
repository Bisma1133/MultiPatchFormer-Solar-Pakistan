"""Complete vs partial target windows (supplementary Table S8, bottom block).

The final evaluation scores only forecasts whose whole target window is real data (no gap-filled hour) with at least
one scored hour. The earlier version also scored partly real windows. This script refits CLIPER and the ridge correction
under the earlier rule (MPF_PARTIAL_WINDOWS=1) and compares pooled CLIPER RMSE and ridge skill with tables/ridge_results.csv.
Run after scripts/ridge_correction.py:  python scripts/window_rule_check.py
"""
import os, subprocess, sys, pandas as pd
env = dict(os.environ, MPF_PARTIAL_WINDOWS="1", MPF_TABLES="outputs_scripts/partial_windows", MPF_SCRIPT_OUT="outputs_scripts/partial_windows",
           MPF_OUT="outputs_scripts/partial_windows")
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "ridge_correction.py")], env=env, check=True)
a = pd.read_csv("tables/ridge_results.csv"); b = pd.read_csv("outputs_scripts/partial_windows/ridge_results.csv")
a = a[a.Region == "Pooled"].set_index("H"); b = b[b.Region == "Pooled"].set_index("H")
T = pd.DataFrame({"CLIPER_complete": a.CLIPER_RMSE, "CLIPER_partial": b.CLIPER_RMSE,
                  "ridge_skill_complete": a.Skill_pct, "ridge_skill_partial": b.Skill_pct})
T["CLIPER_diff_pct"] = 100 * (T.CLIPER_partial / T.CLIPER_complete - 1)
T.round(4).to_csv("tables/complete_vs_partial_windows.csv"); print(T.round(3).to_string())
