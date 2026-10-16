# Projected-LMC diagnostics from the production zero start

## Design

The diagnostic re-creates four corner datasets from the recorded seeds. For each setting, four independent projected-LMC chains start from the exact production initialization `theta=0`. The production check uses 100,000 iterations and step factor 0.01, discards the first 50,000 iterations, and saves every 100th state. Screening references are rank-normalized split-Rhat <= 1.01 and bulk ESS >= 400.

Because all chains share the zero initial state, passing Rhat would not prove convergence; failure remains informative. The analysis therefore also reports time-block drift, output-relevant density TV, target potential, and step-factor sensitivity.

## Production setting: density-TV functional

| scenario         | functional   |     mean |   rank_split_rhat |   bulk_ess |   first_last_shift_sd |
|:-----------------|:-------------|---------:|------------------:|-----------:|----------------------:|
| alpha_n1000_a0p5 | density_tv   | 0.13293  |            1.0033 |    1217.9  |             0.093115  |
| alpha_n16000_a8  | density_tv   | 0.019431 |            1.0011 |     970.54 |             0.064355  |
| gamma_n1000_g0p2 | density_tv   | 0.087227 |            0.9989 |    1594.2  |             0.0061602 |
| gamma_n16000_g1  | density_tv   | 0.018863 |            1.002  |    1175.3  |             0.042214  |

The density-TV functional passes the reference rules at all four corners. This explains why the reported alpha/gamma density curves can look stable even when some parameter directions mix slowly.

## Production setting: score-function magnitude

| scenario         | functional         |   mean |   rank_split_rhat |   bulk_ess |   first_last_shift_sd |
|:-----------------|:-------------------|-------:|------------------:|-----------:|----------------------:|
| alpha_n1000_a0p5 | score_function_rms | 1.8089 |            1.0021 |     980.79 |              0.10269  |
| alpha_n16000_a8  | score_function_rms | 1.019  |            1.0206 |     211.77 |              0.077463 |
| gamma_n1000_g0p2 | score_function_rms | 2.0289 |            1.0013 |     739.16 |              0.029925 |
| gamma_n16000_g1  | score_function_rms | 1.078  |            1.0102 |     296.25 |              0.18185  |

The large-n score-function magnitude is borderline or fails the screening rule: alpha=8 has Rhat about 1.021 and ESS about 212; gamma=1 has Rhat about 1.010 and ESS about 296.

## Production setting: coefficient diagnostics

| scenario         |   all_max_rhat |   all_min_ess |   active_max_rhat |   active_min_ess |   worst_basis_center |   truncation_R |
|:-----------------|---------------:|--------------:|------------------:|-----------------:|---------------------:|---------------:|
| alpha_n1000_a0p5 |         1.0394 |       89.641  |            1.0349 |         169.86   |              -6.5438 |         3.2856 |
| alpha_n16000_a8  |         3.1445 |        4.5768 |            1.4751 |           8.6401 |               7.6076 |         3.8892 |
| gamma_n1000_g0p2 |         1.0793 |       56.959  |            1.0282 |          99.581  |               4.8248 |         3.2856 |
| gamma_n16000_g1  |         1.9445 |        5.8851 |            1.3947 |          10.543  |               6.0598 |         3.8892 |

The full coefficient vector does not mix adequately at the large-n corners. The slowest directions are near the outer cutoff region and are weakly informed by the empirical score-matching loss. However, the density-active subset also fails at large n, so the problem cannot be dismissed solely as unused tail coefficients.

## Step-factor sensitivity

| base_scenario    |   step_factor | functional         |     mean |   rank_split_rhat |   bulk_ess |
|:-----------------|--------------:|:-------------------|---------:|------------------:|-----------:|
| alpha_n1000_a0p5 |          0.01 | score_function_rms | 1.8089   |           1.0021  |     980.79 |
| alpha_n1000_a0p5 |          0.01 | density_tv         | 0.13293  |           1.0033  |    1217.9  |
| alpha_n16000_a8  |          0.01 | score_function_rms | 1.019    |           1.0206  |     211.77 |
| alpha_n16000_a8  |          0.01 | density_tv         | 0.019431 |           1.0011  |     970.54 |
| gamma_n1000_g0p2 |          0.01 | score_function_rms | 2.0289   |           1.0013  |     739.16 |
| gamma_n1000_g0p2 |          0.01 | density_tv         | 0.087227 |           0.9989  |    1594.2  |
| gamma_n16000_g1  |          0.01 | score_function_rms | 1.078    |           1.0102  |     296.25 |
| gamma_n16000_g1  |          0.01 | density_tv         | 0.018863 |           1.002   |    1175.3  |
| alpha_n1000_a0p5 |          0.02 | score_function_rms | 1.8414   |           1.0004  |    1404.4  |
| alpha_n1000_a0p5 |          0.02 | density_tv         | 0.13323  |           1.0014  |    1875.1  |
| alpha_n16000_a8  |          0.02 | score_function_rms | 1.0235   |           1.0121  |     330.28 |
| alpha_n16000_a8  |          0.02 | density_tv         | 0.019433 |           0.99943 |    1649.2  |
| gamma_n1000_g0p2 |          0.02 | score_function_rms | 2.0479   |           1.0001  |    1191.5  |
| gamma_n1000_g0p2 |          0.02 | density_tv         | 0.087415 |           0.99922 |    2000    |
| gamma_n16000_g1  |          0.02 | score_function_rms | 1.0705   |           1.0057  |     550.09 |
| gamma_n16000_g1  |          0.02 | density_tv         | 0.018904 |           1.0004  |    1844.1  |
| alpha_n1000_a0p5 |          0.1  | score_function_rms | 1.9637   |           1.0001  |    1923.2  |
| alpha_n1000_a0p5 |          0.1  | density_tv         | 0.13439  |           1.0012  |    2000    |
| alpha_n16000_a8  |          0.1  | score_function_rms | 1.03     |           1.0031  |    1147.2  |
| alpha_n16000_a8  |          0.1  | density_tv         | 0.019475 |           0.99971 |    2000    |
| gamma_n1000_g0p2 |          0.1  | score_function_rms | 2.155    |           0.99898 |    2000    |
| gamma_n1000_g0p2 |          0.1  | density_tv         | 0.088256 |           1.0009  |    2000    |
| gamma_n16000_g1  |          0.1  | score_function_rms | 1.0691   |           1.0009  |    1273.9  |
| gamma_n16000_g1  |          0.1  | density_tv         | 0.019151 |           1.001   |    2000    |

Increasing the step factor improves ESS and Rhat. It is not a drop-in global fix. At the small-n alpha=0.5 corner, step factor 0.1 reaches the 500-cycle projection cap and produces a maximum feasibility violation around 1.18e-5. It also changes score-function magnitude by about 6--9% at the two small-n corners, consistent with non-negligible unadjusted-Langevin discretization bias. Density TV moves much less (at most about 1.5%).

## Longer large-n check

| scenario                        | functional         |     mean |   rank_split_rhat |   bulk_ess |
|:--------------------------------|:-------------------|---------:|------------------:|-----------:|
| alpha_n16000_a8_sf0p1_steps500k | theta_rms          | 3.0246   |           1.0609  |     73.075 |
| alpha_n16000_a8_sf0p1_steps500k | score_function_rms | 1.0289   |           1.0018  |   1434.4   |
| alpha_n16000_a8_sf0p1_steps500k | density_tv         | 0.019488 |           1.0009  |   2000     |
| gamma_n16000_g1_sf0p1_steps500k | theta_rms          | 3.203    |           1.0063  |    267.85  |
| gamma_n16000_g1_sf0p1_steps500k | score_function_rms | 1.066    |           0.99932 |   1909.8   |
| gamma_n16000_g1_sf0p1_steps500k | density_tv         | 0.019141 |           1.0014  |   2000     |

| scenario                        |   all_max_rhat |   all_min_ess |   active_max_rhat |   active_min_ess |
|:--------------------------------|---------------:|--------------:|------------------:|-----------------:|
| alpha_n16000_a8_sf0p1_steps500k |         1.0798 |        44.711 |            1.0632 |           63.612 |
| gamma_n16000_g1_sf0p1_steps500k |         1.0577 |        84.97  |            1.0111 |          223.97  |

Even 500,000 steps at factor 0.1 do not validate the complete parameter draw at both large-n corners. Output-relevant density TV and score-function summaries are stable, but the alpha=8 active coefficient block still has maximum Rhat about 1.063 and minimum ESS about 64.

## Conclusion for the paper experiments

1. The current 100,000-step, factor-0.01 run is not demonstrated to be a well-mixed draw from the full private coefficient target at large n.
2. Density TV is much less sensitive to the slow directions and passes all four zero-start functional checks. The existing rate-panel shapes therefore remain useful preliminary numerical evidence, but they should not be presented as having a completed full-parameter mixing validation.
3. The private/non-private stability result should remain a diagnostic or appendix result until the score-parameter sampler is improved, because it directly compares LMC score outputs.
4. A production rerun should first address conditioning, for example with an explicitly justified ridge/preconditioner or a reduced active basis, and then re-run diagnostics from both zero and overdispersed starts. Simply changing the global step factor to 0.1 is not supported by these checks.

No formal alpha/gamma result files were modified by this diagnostic.
