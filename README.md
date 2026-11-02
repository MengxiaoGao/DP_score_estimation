# DP score estimation and diffusion experiments

This repository contains numerical experiments for private score estimation and diffusion-based sampling, using synthetic data and public NHANES measurements.

## Experiment settings

The projected LMC is initialized at zero and run for 500,000 updates. The step size is $0.1/L$ for the 1D synthetic experiments, $0.01/L$ for the 2D synthetic experiments and $0.01/L$ for the NHANES experiment, where $L$ is the calibrated Lipschitz constant of the objective gradient.

The synthetic experiments use 100 repetitions for each configuration, sample sizes $n\in\{1000,2000,4000,8000,16000\}$. For diffusion, the total privacy budget is split equally between score estimation and the histogram.

**Privacy figure ($\alpha$ panel).** This figure shows how loss changes with sample size at different privacy budgets. It fixes $\gamma=0.5$ and compares score budgets $\alpha_{\mathrm{score}}\in\{0.5,1,2,4,8\}$, corresponding to total diffusion budgets $\alpha_{\mathrm{total}}\in\{1,2,4,8,16\}$.

**Smoothness figure ($\gamma$ panel).** This figure shows how loss changes with sample size for different target smoothness levels. It compares $\gamma\in\{0.2,0.4,0.6,0.8,1\}$, fixing the score budget at $\alpha_{\mathrm{score}}=2$ and the total diffusion budget at $\alpha_{\mathrm{total}}=4$.

## Installation

Use Python 3.12 (recommended) or 3.13. Run all commands below from the repository root: the directory containing this README.

```bash
python -m pip install -r requirements.txt
```

To generate figures, also install Matplotlib:

```bash
python -m pip install -r plotting/requirements.txt
```

The examples use `--workers 4` to run up to four jobs in parallel. Adjust this to suit your machine; use `--workers 1` for serial execution.

## 1. One-dimensional score estimation

Run the privacy ($\alpha$) and smoothness ($\gamma$) panels, respectively:

```bash
python experiments/score_1d/samples_and_privacy.py --workers 4 --output-dir outputs/score_1d/privacy
python experiments/score_1d/samples_and_smoothness.py --workers 4 --output-dir outputs/score_1d/smoothness
```

These compare score-estimation loss across sample sizes using the settings above. Non-private ERM baselines are included in these runs.

After completing the privacy ($\alpha$) panel, run the perturbation experiment, which replaces the first observation by $3$ and measures the change in the score-estimator coefficients:

```bash
python experiments/score_1d/coefficient_stability.py --workers 4 --source-dir outputs/score_1d/privacy --output-dir outputs/score_1d/stability
```

Run the convergence diagnostics separately:

```bash
python experiments/score_1d/generate_rhat_grids.py --workers 4 --output-dir outputs/score_1d/convergence
```

The convergence diagnostics use four chains per setting and the LMC settings given above. Results are saved under `outputs/score_1d/`.

## 2. One-dimensional diffusion

Run the privacy ($\alpha$) and smoothness ($\gamma$) panels, respectively:

```bash
python experiments/diffusion_1d/paper/runner.py --panel alpha --workers 4
python experiments/diffusion_1d/paper/runner.py --panel gamma --workers 4
```

These compare diffusion loss across sample sizes using the settings above. Run the corresponding non-private ERM baselines separately after completing both panels:

```bash
python experiments/diffusion_1d/paper/nonprivate_erm_baselines_1d.py --workers 4
```

After completing the privacy ($\alpha$) panel, run the perturbation experiment, which replaces the first observation by $3$ and measures early-stopped OU TV between neighboring fits:

```bash
python experiments/diffusion_1d/paper/perturbation_runner_1d.py --workers 4
```

Add `--with-reverse` to compute generated-output perturbation metrics as well.

Run the optional appendix checks separately:

```bash
python experiments/diffusion_1d/paper/appendix_checks.py density --workers 4
python experiments/diffusion_1d/paper/appendix_checks.py reverse --workers 4
```

Results are saved under `experiments/diffusion_1d/paper/outputs/`.

## 3. Two-dimensional synthetic data

Use `--profile paper` for the experiment settings described above.

Run both panels and summarize their results:

```bash
python experiments/synthetic_2d/experiment.py --profile paper --task alpha --workers 4 --output outputs/synthetic_2d
python experiments/synthetic_2d/experiment.py --profile paper --task gamma --workers 4 --output outputs/synthetic_2d
python experiments/synthetic_2d/experiment.py --task summarize --output outputs/synthetic_2d
```

These panels compute score and diffusion metrics for private fits and constrained non-private ERM.

After completing the alpha panel, run and summarize its perturbation experiment:

```bash
python experiments/synthetic_2d/perturbation.py batch --profile paper --alpha-output outputs/synthetic_2d --output outputs/synthetic_2d_perturbation --workers 4
python experiments/synthetic_2d/perturbation.py summarize --alpha-output outputs/synthetic_2d --output outputs/synthetic_2d_perturbation
```

The perturbation experiment replaces the first observation by $(3,3)$.

## 4. Two-dimensional NHANES data

This experiment uses public NHANES BMI and waist measurements across 100 random data splits. Download the required body-measurement and demographic files:

```bash
python scripts/fetch_nhanes.py --data-dir data/nhanes
```

Run all 500 jobs across 100 data splits, then combine the results:

```bash
python experiments/real_data_2d/experiment.py batch --job-start 0 --job-stop 500 --workers 4 --data-dir data/nhanes --output outputs/real_data_2d
python experiments/real_data_2d/experiment.py summarize --output outputs/real_data_2d --summary-dir outputs/real_data_2d_summary
```

## Saved results and restarting runs

Keep the complete output directory when restarting a run or reusing results for perturbation experiments.

If a run stops, repeat the same command with unchanged settings to reuse completed work.

Use `--help` with any script to see its options.

## Figures

The scripts in `plotting/` generate PDF and PNG figures from the saved results. See [plotting/README.md](plotting/README.md) for the 1D score, 2D score, 1D diffusion and NHANES plotting instructions. 

To plot the convergence grids generated above:

```bash
python plotting/plot_rhat_grids.py --csv-dir outputs/score_1d/convergence --output-dir figures/rhat_grids
```
