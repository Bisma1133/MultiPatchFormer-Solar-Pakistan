# Data

Place the combined Solcast files here (they are not included in the repository; Solcast data are distributed under the provider's terms):

* `Solcast_4cities_2020_2024.csv` (main experiments)
* `Solcast_4cities_2017_2024.csv` (longer-training experiment)

They are archived on Zenodo: B. Ali, "Pakistan 4 cities solar forecasting dataset," Zenodo, Nov. 13, 2026, doi: 10.5281/zenodo.24181080.

Then create the files with the clear-sky GHI known at issue time, which the notebooks and scripts use:

```
python scripts/make_clearsky.py data/Solcast_4cities_2020_2024.csv data/Solcast_4cities_2020_2024_csmodel.csv
python scripts/make_clearsky.py data/Solcast_4cities_2017_2024.csv data/Solcast_4cities_2017_2024_csmodel.csv
```

`SHA256SUMS.txt` lists the checksums of the original downloads and of the derived `*_csmodel.csv` files. See the main README for the Solcast request parameters, the site coordinates and how to rebuild the files with `scripts/make_dataset.py`.
