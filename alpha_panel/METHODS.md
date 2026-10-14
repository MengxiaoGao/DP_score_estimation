# Revised diffusion-extension experiments

## Design and scope
The existing 750 private densities are retained (5 sample sizes × 5 privacy budgets × 30 independent datasets). Training data are paired across privacy budgets. Projected LMC is retained; no Gibbs or score-risk calculation is used. The reverse update is the exponential integrator in Eq. (1), with the time grid of Lemma 1.7 and kappa=delta². Scores are evaluated by validated 3,201-point interpolation of truncated Gaussian moments, with direct evaluation outside the grid. This remaining numerical approximation is disclosed; epsilon_0 is not claimed to be exactly zero. Finite-step LMC mixing/privacy certification limitations of the source experiment remain.

## Theory references
For d=1 and gamma=0.5, delta=max(n^(-3/8),(n alpha)^(-3/5)). All current alpha values exceed the nominal crossover alpha*=n^(-3/8). Thus -3/8 is the relevant log-log n reference in the displayed polynomial bound; -3/5 is not the appropriate alpha reference for this panel. Its nominal alpha reference is zero, but that is not a prediction of exactly constant finite-sample risk. Hidden constants, logarithmic factors, and lower-order privacy terms remain. The theorem is a class-wise upper bound, not an equality for this fixed finite-series target.

## Slope estimation
Fit OLS to log(group mean error) versus log(n), and separately versus log(alpha), using all five points. Percentile 95% intervals use 2,000 bootstrap resamples, stratified by n; each resampled dataset retains all its alpha values. Intervals describe sampling uncertainty of this descriptive slope, not uncertainty in a theoretical exponent. No TV exponent is assigned to Wasserstein-1. Binned TV is a finite-resolution empirical diagnostic and must not be identified with population TV.

## Histogram ablation
The histogram-only density uses exactly the same released cell masses as the reconstructed density. Its budget is alpha/2, not a separately optimized full-budget histogram baseline. Positive reconstruction-minus-histogram TV differences favor histogram; negative differences favor reconstruction. Report all settings, including unfavorable ones. Pointwise paired intervals are not simultaneous multiplicity-adjusted intervals.

## Discretization design
Fix n=4000, alpha=1 and preselect dataset repetitions 0,1,2. For each density use multipliers 128,32,8,2,1,1/2,1/4 and 10 reverse runs of 20,000 particles. Identical initial seeds across multipliers aid comparison, but changing step counts means this is not a nested Brownian coupling. IID samples have the same size and OU target. Plot the same binned-TV metric for the two sampling methods. Do not subtract the IID baseline as if that recovered population discretization TV. The nominal -1/2 discretization exponent is an upper-bound reference at fixed T,tau, not an expected slope after the empirical floor dominates.

## Reproduction
Run `python3 driver.py` to resume simulation and final figures. Run `python3 analyze.py` to rebuild summaries. Per-run JSON checkpoints and generated samples are saved. The grouped private-density snapshots are read-only inputs to these experiment entry points. See config.json, numerical_config.json, and the repository-level CHECKSUMS.sha256.
