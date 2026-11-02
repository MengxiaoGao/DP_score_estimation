# Figure generation

The scripts in this directory generate the paper figures from saved CSV files. Run all commands from the repository root.

## Setup

```bash
python3 -m pip install -r plotting/requirements.txt
```

All scripts save both PDF and PNG files. The common appearance is defined in `paper_style.py`: serif fonts, a blue line for the private estimator, a dark dashed line for ERM, and pointwise 95% Student-t confidence intervals. Synthetic sample-size panels use logarithmic axes.

## One-dimensional score estimation

```bash
python3 plotting/plot_score_1d.py \
  --privacy-csv outputs/score_1d/privacy/tables/score_raw.csv \
  --smoothness-csv outputs/score_1d/smoothness/tables/score_raw.csv \
  --stability-csv outputs/score_1d/stability/coefficient_stability.csv \
  --output-dir figures
```

This creates `score_alpha_panel`, `score_gamma_panel`, and `score_perturbation`. Each name is produced in PDF and PNG format. The default perturbation metric is coefficient MSE; use `--perturbation-metric private_score_squared` to plot score change instead.

## One-dimensional diffusion extension

```bash
python3 plotting/plot_diffusion_1d.py \
  --alpha-csv experiments/diffusion_1d/paper/outputs/alpha/raw.csv \
  --gamma-csv experiments/diffusion_1d/paper/outputs/gamma/raw.csv \
  --erm-alpha-csv experiments/diffusion_1d/paper/outputs/erm_baselines/alpha/raw.csv \
  --erm-gamma-csv experiments/diffusion_1d/paper/outputs/erm_baselines/gamma/raw.csv \
  --perturbation-csv experiments/diffusion_1d/paper/outputs/perturbation/raw.csv \
  --output-dir figures
```

This creates `diffusion_alpha_tv`, `diffusion_gamma_tv`, and `diffusion_perturbation_tv`. The default utility metric is generated binned TV. It can be changed with `--utility-metric` to `density_tv` or `ou_tv`. The perturbation figure accepts `density_tv_change`, `ou_tv_change`, or `generated_binned_tv_change` through `--perturbation-metric`.

## Two-dimensional synthetic experiments

```bash
python3 plotting/plot_synthetic_2d.py \
  --alpha-csv outputs/synthetic_2d/alpha/raw.csv \
  --gamma-csv outputs/synthetic_2d/gamma/raw.csv \
  --perturbation-csv outputs/synthetic_2d_perturbation/raw.csv \
  --output-dir figures
```

The alpha and gamma arguments can each receive one CSV or several batch CSVs. The script creates score-risk and diffusion-TV panels for both parameter grids, plus the perturbation-TV figure when a perturbation CSV is supplied. The diffusion metric is selected with `--diffusion-metric`; its default is `generated_binned_tv`.

## Two-dimensional NHANES experiment

```bash
python3 plotting/plot_nhanes_2d.py \
  --private-csv outputs/real_data_2d_summary/combined_private_metrics.csv \
  --nonprivate-csv outputs/real_data_2d_summary/combined_nonprivate_metrics.csv \
  --output-dir figures
```

This creates `nhanes_score`, `nhanes_main`, and `nhanes_perturbation`. To also draw density contours, provide the saved fit directory and NHANES data:

```bash
python3 plotting/plot_nhanes_2d.py \
  --private-csv outputs/real_data_2d_summary/combined_private_metrics.csv \
  --nonprivate-csv outputs/real_data_2d_summary/combined_nonprivate_metrics.csv \
  --results-root outputs/real_data_2d \
  --data-dir data/nhanes \
  --contour-split 0 \
  --output-dir figures
```

## LMC convergence diagnostics

```bash
python3 plotting/plot_rhat_grids.py \
  --privacy-csv outputs/score_1d/convergence/privacy/convergence.csv \
  --smoothness-csv outputs/score_1d/convergence/smoothness/convergence.csv \
  --output-dir figures/rhat
```

This creates privacy and smoothness R-hat heatmaps in PDF, PNG, and SVG format. If results are stored elsewhere, replace the example paths with the corresponding CSV files.