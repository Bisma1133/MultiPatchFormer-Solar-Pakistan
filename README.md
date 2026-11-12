# Reference-anchored multi-horizon solar forecasting for Pakistan

Code, configuration, executed notebooks and results for the paper

> Bisma Ali and Hashir Moheed Kiani, "Multi-Horizon Solar Forecasting and Economic Analysis for Pakistan's Photovoltaic Integration: A Reference-Anchored Benchmark of a Multiscale Transformer", *Journal of Renewable and Sustainable Energy* (under review, JRSE26-AR-01851).

The study forecasts hourly global horizontal irradiance (GHI) for Islamabad, Karachi, Lahore and Multan from 1 hour to 28 days ahead. Every model forecasts the clear-sky index as a bounded correction of the CLIPER reference (climatology + persistence) and is tested once on the hidden year 2024. Models: MultiPatchFormer, PatchTST, DLinear, LSTM (three seeds each) and a ridge-regression correction of CLIPER.

## 1. Data

* Product: Solcast **estimated actuals** (satellite-derived irradiance; weather variables from NWP/reanalysis), hourly (`PT60M`), `array_type=fixed`, `tilt=23`, `azimuth=180` (south-facing).
* Sites (request coordinates): Islamabad 33.699509, 73.03629; Karachi 24.860734, 67.001136; Lahore 31.52037, 74.358747; Multan 30.186356, 71.488581.
* Columns: `air_temp, clearsky_dhi, clearsky_dni, clearsky_ghi, clearsky_gti, dhi, dni, ghi, gti, relative_humidity, wind_direction_10m, wind_speed_10m, period_end, period, Region` (`period_end` in UTC).
* Files used in the paper (archived on Zenodo, DOI 10.5281/zenodo.20691120):
  * `Solcast_4cities_2020_2024.csv` – main experiments (2020–2024).
  * `Solcast_4cities_2017_2024.csv` – longer-training experiment (2017–2024).
* The Karachi file ends on 2024-11-29 00:00 UTC.
* Licence: Solcast data are distributed under the provider's terms. If you cannot use the archived file, request the same series from the Solcast historical API with the parameters above and build the combined file with `scripts/make_dataset.py` (it also checks that no two sites share the same series).

Put the CSV files in `data/` (or set `MPF_DATA`). On Kaggle, attach the dataset; the notebooks find the CSV automatically. **Attach only one combined CSV**: the notebooks pick the largest CSV that has the columns `ghi`, `clearsky_ghi` and `Region`.

## 2. Installation

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Tested with Python 3.11–3.13, PyTorch 2.x. The neural models were trained on Kaggle with 2 × NVIDIA T4 GPUs (about 2 h per notebook). The scripts in `scripts/` run on a CPU.

## 3. Execution order

| Step | Command | Output |
|---|---|---|
| 0 | `python scripts/make_dataset.py RAW_DIR data/Solcast_4cities_2020_2024.csv 2020 2024` (only if you start from raw Solcast files) | combined CSV + independence check |
| 1 | run `notebooks/01_Baselines.ipynb` | references, PatchTST, DLinear, LSTM (`outputs/`) |
| 2 | run `notebooks/02_MultiPatchFormer.ipynb` | MultiPatchFormer, ablation, blind test, missing-input test |
| 3 | `python scripts/extract_notebook_tables.py` | `results/notebook_tables/` (tables shown in the executed notebooks) |
| 4 | `python scripts/collect_results.py` | `results/res_*.csv` |
| 5 | `python scripts/ridge_correction.py` | ridge correction of CLIPER, `results/extra/ridge_*` |
| 6 | `python scripts/ridge_stats.py` | DM tests, bootstrap and DM-based intervals for the ridge model |
| 7 | `python scripts/ridge_longer_training.py` (uses the 2017–2024 file) | CLIPER and ridge with 2017–2022 training |
| 8 | `python scripts/statistics.py` | BH-corrected tests, confidence intervals, best models, direct comparison, fallback |
| 9 | `python scripts/data_stats.py`, `scripts/pv_energy.py`, `scripts/economics.py` | Table I, PV energy and payback |
| 10 | `python scripts/paper_tables.py`, `scripts/paper_figures.py`, `scripts/paper_facts.py` | tables, figures and numbers of the paper |

All scripts are run from the repository root. Steps 3–10 reproduce every number in the paper from the executed notebooks; we checked that they give identical values.

## 4. Configuration and seeds

* All settings are in one `CFG` block, identical in both notebooks (copy in `config/config.json`).
* Split: train 2020–2022, validation 2023, test 2024 (local time, UTC+5). Input length chosen on 2023 with seed 0; final runs with seeds 0, 1, 2. The test year is scored once.
* Training: global model on all four sites, AdamW, batch 256, early stopping (patience 5, max 40 epochs), learning rate 3e-4 (Transformers) or 1e-3 (DLinear, LSTM).
* `ALLOW_DUPLICATE_REGIONS=False`: the notebooks stop if two sites have the same series.
* After the Kaggle runs, three economic constants in `CFG["ECON"]` (not used by notebooks 01–02) and the duplicate-region switch were set to their final values; this does not change any output. The paper's economic values come from `scripts/economics.py` (tariff PKR 34.00/kWh, capital cost PKR 1,200,000 for 10 kWp, PKR 279/USD).

## 5. Folder structure

```
notebooks/   executed notebooks 01 and 02 (with all outputs)
scripts/     data assembly, ridge correction, statistics, PV/economics, paper tables and figures
config/      configuration used by the notebooks
results/     notebook_tables/ (80 tables), figures/ (60 notebook figures), paper_figures/ (figures of the paper),
             res_*.csv and extra/, pv/ (all paper results), facts.json and paper_tables.json
data/        place the Solcast CSV files here (not included, see Section 1)
```

## 6. Citation

See `CITATION.cff`. Licence: MIT (code). Data: see Section 1.
