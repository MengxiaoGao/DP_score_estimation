# Score-estimation simulations

Code for score estimation simulations. The estimator uses quadratic B-splines and coordinate 
Gibbs sampling targeting the constrained exponential-mechanism distribution, 
rather than the projected Langevin Monte Carlo procedure.

## Setup

Requires Python 3.10 or later. From the directory containing the scripts, run: 

```bash
python -m pip install -r requirements.txt
```

Keep the Python files together:

| File | Purpose |
|---|---|
| `private_score_experiments_v3.py` | Score-estimation model, conditional samplers and constrained ERM |
| `experiment_common.py` | Shared target, calibration, Gibbs updates and risk evaluation |
| `check_convergence.py` | Convergence diagnostics and trace plot |
| `samples_and_privacy.py` | Sample size versus privacy experiment |
| `samples_and_smoothness.py` | Sample size versus smoothness experiment |

## Target and risk

The target score is

$$
s_{\gamma,K}(x)=-x+0.5(2^\gamma-1)
\sum_{k=1}^{K}2^{-k\gamma}\sin(2^kx),
\qquad
K=\left\lceil\frac{\log_2(0.5/10^{-3})}{\gamma}\right\rceil.
$$

The choice of $K$ bounds the omitted infinite-series score by $10^{-3}$. 

The risk is estimated as the squared score error averaged over 10,000 independent test observations.
Each main experiment uses 50 independently generated training datasets and one 
final Gibbs coefficient draw per private setting. 


## Convergence checks
Run 
```bash
python check_convergence.py --workers 2
```

The nine settings are:

| Sample size | Privacy level | Smoothness |
|---:|---|---|
| 16,000 | 0.5, 8 | 0.2, 0.6, 0.8 |
| 1,000 | 0.5, 8 | 0.8 |
| 8,000 | 2 | 0.4 |

Each setting initially uses one dataset and four chains, started at zero,
the constrained empirical minimiser, and constant coefficient vectors $\pm0.8B$,
where $B$ is the coefficient bound. We demonstrate results for 20,000, 50,000, 100,000 and 150,000 sweeps, discarding
the first 10% as burnin. 

Outputs in `results/convergence/`:

- `convergence.csv`: maximum coefficient R-hat, minimum coefficient bulk ESS,
  risk R-hat/ESS, mean risk and risk MCSE for all settings and checkpoints.
- `convergence.png`: a four-panel comparison of representative settings.
- `trace.png`: selected coefficient and risk traces for the setting with the
  largest maximum coefficient R-hat and lowest ESS at 100,000 sweeps

## Sample size versus privacy
Run 
```bash
python samples_and_privacy.py --workers 2 --sweeps 100000 --reps 50
```

We vary $n\in\{1000,2000,4000,8000,16000\}$, and $\alpha\in\{0.5,1,2,4,8\}$ and fix $\gamma=0.5$.

Outputs in `results/privacy/`:

- `privacy_risk.npy`: shape **(6, 5, 50)**, with axes
  `[privacy, sample size, repetition]`. Privacy index 5 contains ERM risks.
- `n_privacy_risk.png`: risk versus sample size, by privacy level.

## Sample size versus smoothness
Run
```bash
python samples_and_smoothness.py --workers 2 --sweeps 100000 --reps 50
```

We vary  $n\in\{1000,2000,4000,8000,16000\}$, $\gamma\in\{0.2,0.4,0.6,0.8\}$ and fix $\alpha=2$.

Outputs in `results/smoothness/`:

- `smoothness_risk.npy`: shape **(5, 4, 50)**, with axes
  `[sample size, smoothness, repetition]`.
- `smoothness_erm_risk.npy`: ERM risks with the same shape.
- `n_smoothness_risk.png`: private and ERM risk panels.
