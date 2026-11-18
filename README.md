# Reference-anchored multi-horizon solar forecasting for Pakistan

Code, configuration, executed notebooks and results (release **v1.1**, the version used for the revised manuscript) for the paper

> Bisma Ali and Hashir Moheed Kiani, "Multi-Horizon Solar Forecasting and Economic Analysis for Pakistan's Photovoltaic Integration: A Reference-Anchored Benchmark of a Multiscale Transformer", *Journal of Renewable and Sustainable Energy* (under review, JRSE26-AR-01851).

The study forecasts hourly global horizontal irradiance (GHI) for Islamabad, Karachi, Lahore and Multan from 1 hour to 28 days ahead. Every model forecasts the clear-sky index as a bounded correction of the CLIPER reference (climatology + persistence). The clear-sky GHI is computed with a model that is known at issue time (Ineichen–Perez, Linke-turbidity climatology, calibrated to Solcast on the training years only), and only forecasts whose whole target window is real data are scored. Training uses 2020–2022, every choice is made on the validation year 2023, and the models are scored on 2024 (earlier versions of the study had also reported 2024 results; see the paper, Sec. III D). Models: MultiPatchFormer, PatchTST, DLinear, LSTM (three seeds each) and a ridge-regression correction of CLIPER.

## 1. Data

* Product: Solcast **estimated actuals** (satellite-derived irradiance; weather variables from NWP/reanalysis), hourly (`PT60M`), `array_type=fixed`, `tilt=23`, `azimuth=180` (south-facing).
* Sites (request coordinates): Islamabad 33.699509, 73.03629; Karachi 24.860734, 67.001136; Lahore 31.52037, 74.358747; Multan 30.186356, 71.488581.
* Columns: `air_temp, clearsky_dhi, clearsky_dni, clearsky_ghi, clearsky_gti, dhi, dni, ghi, gti, relative_humidity, wind_direction_10m, wind_speed_10m, period_end, period, Region` (`period_end` in UTC).
* Original files, archived on Zenodo (record: https://zenodo.org/records/24181080): B. Ali, "Pakistan 4 cities solar forecasting dataset," Zenodo, Nov. 13, 2026, doi: [10.5281/zenodo.24181080](https://doi.org/10.5281/zenodo.24181080). This is the only dataset DOI of this project; any older DOI in earlier versions of this repository is obsolete.
  Files:
  * `Solcast_4cities_2020_2024.csv` – main experiments (2020–2024).
  * `Solcast_4cities_2017_2024.csv` – longer-training experiment (2017–2024).
* **Clear-sky GHI known at issue time.** The Solcast `clearsky_ghi` is computed with REST2 and the aerosol and water-vapour data of the target hour, so it is not known when a forecast is issued. Before running the notebooks, create the files actually used by the code:
  ```
  python scripts/make_clearsky.py data/Solcast_4cities_2020_2024.csv data/Solcast_4cities_2020_2024_csmodel.csv
  python scripts/make_clearsky.py data/Solcast_4cities_2017_2024.csv data/Solcast_4cities_2017_2024_csmodel.csv
  ```
  The script replaces `clearsky_ghi` by the Ineichen–Perez clear-sky GHI (pvlib, monthly Linke-turbidity climatology, mean of four quarter-hour values per hourly period), multiplied by the Solcast/Ineichen ratio per site, month and local hour over 2020–2022. The original value is kept as `clearsky_ghi_solcast` and is used only by the data-integrity checks. SHA-256 checksums of the original and derived files are in `data/SHA256SUMS.txt`.
* The Karachi file ends on 2024-11-29 00:00 UTC. The code fills gaps longer than 3 h from neighbouring days for the model inputs only; filled hours are never scored.
* **Data integrity.** An earlier merged table (used before v1.1) had two wrong labels: its "Karachi" series was the Lahore file and its "Lahore" series was a copy of Multan. The files above were rebuilt directly from the four original Solcast downloads. `scripts/data_integrity.py` checks every site (request coordinates from the original file name, longitude implied by the timing of the clear-sky GHI, missing hours, pairwise identical values and correlation) and stops if a check fails; `scripts/previous_compilation_check.py` maps every series of the old table to the corrected series (supplementary Tables S1–S3 of the paper).
* Licence: Solcast data are distributed under the provider's terms. If you cannot use the archived file, request the same series from the Solcast historical API with the parameters above and build the combined file with `scripts/make_dataset.py` (it also checks that no two sites share the same series).

Put the CSV files in `data/`. For a local run of the notebooks, set `MPF_DATA=data/Solcast_4cities_2020_2024_csmodel.csv` (the scripts use this file by default). On Kaggle, upload `Solcast_4cities_2020_2024_csmodel.csv` as a dataset; the notebooks find the CSV automatically. **Attach only this one combined CSV**: the notebooks pick the largest CSV that has the columns `ghi`, `clearsky_ghi` and `Region`.

## 2. Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.11–3.13, PyTorch 2.x; the exact versions used are listed in `requirements.txt`. The neural models were trained on Kaggle with 2 × NVIDIA T4 GPUs (about 2 h per notebook). The scripts in `scripts/` run on a CPU.

## 3. Execution order

| Step | Command | Output |
|---|---|---|
| 0 | `python scripts/make_dataset.py RAW_DIR data/Solcast_4cities_2020_2024.csv 2020 2024` (only if you start from raw Solcast files) | combined CSV + independence check |
| 0b | `python scripts/make_clearsky.py IN.csv OUT_csmodel.csv` for both files (Section 1) | `*_csmodel.csv` with the issue-time clear-sky GHI |
| 1 | run `notebooks/01_Baselines.ipynb` | references, PatchTST, DLinear, LSTM (`outputs/`; every forecast, observation and mask is saved in `outputs/predictions/`) |
| 2 | run `notebooks/02_MultiPatchFormer.ipynb` | MultiPatchFormer, ablation, blind test, missing-input test |
| 3 | `python scripts/extract_notebook_tables.py`, `python scripts/extract_notebook_figures.py` | `tables/nb01_*.csv`, `nb02_*.csv`, `figures/01_*`, `02_*` (tables and figures shown in the executed notebooks; `outputs/` is not needed) |
| 4 | `python scripts/collect_results.py` | `tables/res_*.csv` |
| 5 | `python scripts/ridge_correction.py` | ridge correction of CLIPER, `tables/ridge_results.csv` |
| 5b | `python scripts/window_rule_check.py` | `tables/complete_vs_partial_windows.csv` (complete vs earlier partly real target windows, Table S8) |
| 6 | `python scripts/ridge_stats.py` | `tables/ridge_stats.csv` (DM tests and intervals for the ridge model) |
| 7 | `python scripts/ridge_longer_training.py` (uses the 2017–2024 file) | `tables/ridge_ext_results.csv` (2017–2022 training) |
| 8 | `python scripts/stats_tests.py` | BH-corrected tests vs CLIPER, intervals, best models, seed-based comparison, fallback, longer-training table |
| 9 | `python scripts/bootstrap_block_check.py` | `tables/bootstrap_block_check.csv` (block-length sensitivity, Table S4); then `python scripts/forecast_counts.py` (forecasts, issue times and scored hours, Table S7) |
| 10 | `python scripts/direct_comparison.py train` then `python scripts/direct_comparison.py test` | direct paired DM tests of MultiPatchFormer vs PatchTST, DLinear, LSTM and ridge at 1 h, 6 h and 1 d (Table VII, S5; this release contains the results at 1 h and 6 h, `tables/direct_mpf_vs_competitors.csv`; 3 h and 1 d are added by running `train` for that horizon) and the reproduction check (Table S6). `train` regenerates the forecasts (set `MPF_SEEDS=0` for the single-seed run used in the paper; on a CPU this takes several hours, on a GPU about 1 h). If the `outputs/predictions` folders of the two Kaggle runs are available, set `MPF_PRED_DIRS=path1:path2` and run only `test` (three-seed version). |
| 10b | `python scripts/clearsky_sensitivity.py` (optional) | re-conversion test of the earlier version, which used the Solcast clear-sky GHI (Table S9); needs the original file and the earlier forecasts |
| 11 | `python scripts/data_integrity.py`, `scripts/data_stats.py`, `scripts/scored_hours.py` | integrity checks (Table S1–S2), Table I, scored hours per site |
| 12 | `python scripts/pv_energy.py`, `scripts/economics.py` | PV energy and payback under the 2026 net-billing rules (Tables XIII–XIV) |
| 13 | `python scripts/paper_tables.py`, `scripts/paper_figures.py`, `scripts/paper_facts.py` | all tables of the paper (`outputs_scripts/paper_tables.json`, incl. the pooled ablation), `figures/paper_*`, and the numbers of the paper |

All scripts are run from the repository root. Steps 3–9 and 11–13 reproduce every number in the paper from the executed notebooks and the data; we re-ran them from scratch and obtained identical tables. Step 10 retrains models, so its numbers can differ slightly between machines (GPU vs CPU arithmetic); supplementary Table S6 compares the single-seed CPU run used in the paper with the seed-0 notebook runs.

## 4. Configuration and seeds

* All settings are in one `CFG` block, identical in both notebooks (copy in `config/config.json`).
* Split: train 2020–2022, validation 2023, test 2024 (local time, UTC+5). Input length chosen on 2023 with seed 0 (validation tables `nb01_*`/`nb02_*` T08/T09 list the 2023 RMSE of every candidate and the selected value); final runs with seeds 0, 1, 2. Test scores of the candidate input lengths are saved for transparency only and are not read by any selection step.
* Training: global model on all four sites, AdamW, batch 256, early stopping (patience 5, max 40 epochs), learning rate 3e-4 (Transformers) or 1e-3 (DLinear, LSTM).
* `ALLOW_DUPLICATE_REGIONS=False`: the notebooks stop if two sites have the same series.
* `CFG["ECON"]` holds only auxiliary values that notebooks 01–02 do not use. The paper's economic values come from `scripts/economics.py`: NEPRA (Prosumer) Regulations 2026 (net billing); base case 60% self-consumption at PKR 33.38/kWh and exports at PKR 10/kWh, capital cost PKR 1,200,000 for 10 kWp, PKR 279/USD; conservative and optimistic scenarios and the 'all units at PKR 34.00/kWh' upper limit are defined at the top of the script.

## 5. Folder structure

```
notebooks/        executed notebooks 01 and 02 (Kaggle run of 18 Nov 2026, with all outputs)
scripts/          data assembly, ridge correction, statistics, PV/economics, paper tables and figures
config/           configuration used by the notebooks
data/             place the Solcast CSV files here (not included, see Section 1); SHA256SUMS.txt
tables/   all result tables (CSV): notebook tables (nb01_*, nb02_*) and paper results
figures/  notebook figures (01_*, 02_*) and the figures of the paper (paper_*)
```
Scripts write intermediate files (forecasts, JSON summaries) to `outputs_scripts/`, which is not part of the repository.

## Versions

* **v1.1** (this release): final version for the revised manuscript. Issue-time clear-sky GHI (`scripts/make_clearsky.py`), complete target windows only, all models refitted or retrained with three seeds, all forecasts saved, season-balanced annual PV energy.
* Earlier versions used the Solcast clear-sky GHI and also scored partly real target windows; they are superseded. Supplementary Table S8 of the paper compares the results.

## 6. Citation

See `CITATION.cff`. Licence: MIT (code). Data: see Section 1.
