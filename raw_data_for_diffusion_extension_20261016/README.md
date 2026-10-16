# Raw data for plotting

This package contains unchanged source CSVs in `original/` and convenient merged/long-format files in `plot_ready/`.

## Recommended files

- `plot_ready/rate_panels_long.csv`: easiest source for alpha/gamma panel plots. Filter `metric` so different metrics remain in separate figures.
- `plot_ready/rate_panels_summary_long.csv`: means, standard errors, and pointwise 95% Student-t intervals.
- `plot_ready/slope_results.csv`: empirical log-log slopes and 2,000-resample bootstrap intervals.
- `plot_ready/reverse_discretization_raw.csv`: reverse-step experiment at n=4000, alpha=1, with three fixed private densities and ten reverse runs per step multiplier.
- `plot_ready/stability_plot_long.csv`: private/non-private score-stability data. `stability_dataset_units.csv` is the correct independent-unit table for inference.

## Row counts

- Alpha panel: 750 rows = 5 n values x 5 alpha values x 30 repetitions.
- Gamma panel: 750 rows = 5 n values x 5 gamma values x 30 repetitions.
- Reverse discretization: 210 rows = 3 fixed densities x 7 step multipliers x 10 reverse runs.
- Stability replacements: 450 rows.
- Stability dataset units: 180 rows after averaging replacements within each base dataset and replacement kind.

## Pairing and independence

- Alpha panel: the same training dataset is retained across alpha values within each `(n, rep)` pair. LMC randomness is independent across alpha.
- Gamma panel: gamma changes the target distribution, so datasets are independent across gamma.
- Stability: use the ten base datasets per n as independent units. Individual replacements within one base dataset are not independent repetitions.

## Metric cautions

- Population density TV, generated binned TV, and Wasserstein-1 are different quantities and should be plotted separately.
- Generated binned TV contains finite-particle and binning effects.
- The theoretical lines in `slope_results.csv` are reference exponents, not fitted estimates with confidence intervals.
- The private/non-private experiment concerns the score estimator. It does not measure final synthetic-sample stability.

The workbook `raw_data_for_plotting.xlsx` contains the same plot-ready tables as separate worksheets. The CSV files remain the preferred machine-readable source.
