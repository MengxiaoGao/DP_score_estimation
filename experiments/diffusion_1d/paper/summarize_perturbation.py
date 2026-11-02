#!/usr/bin/env python3
"""Summarise the paper-aligned 1D perturbation experiment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


import numpy as np
import pandas as pd
from scipy.stats import t as student_t


METRICS = {
    "score_displacement": "Squared score displacement",
    "ou_tv_change": "Early-stopped DM TV",
    "density_tv_change": "Reconstructed-density TV",
    "generated_binned_tv_change": "Generated binned TV",
}


def bootstrap_slope(frame: pd.DataFrame, column: str, seed: int, B: int = 2000):
    pivot = frame.pivot(index="rep", columns="n", values=column).sort_index(axis=1)
    x = np.log(pivot.columns.to_numpy(float))
    values = pivot.to_numpy(float)
    estimate = float(np.polyfit(x, np.log(values.mean(axis=0)), 1)[0])
    random = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        ids = random.integers(0, len(values), len(values))
        boot[b] = np.polyfit(x, np.log(values[ids].mean(axis=0)), 1)[0]
    return estimate, float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir.resolve()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(out / "raw.csv")
    available_metrics = [metric for metric in METRICS
                         if any(np.isfinite(raw[f"{method}_{metric}"]).any()
                                for method in ("private", "nonprivate"))]
    rows = []
    for method in ("private", "nonprivate"):
        for metric in available_metrics:
            column = f"{method}_{metric}"
            for (alpha, n), group in raw.groupby(["alpha_total", "n"]):
                mean = group[column].mean()
                se = group[column].std(ddof=1) / np.sqrt(len(group)) if len(group) > 1 else np.nan
                q = student_t.ppf(.975, len(group)-1) if len(group) > 1 else np.nan
                rows.append({
                    "method": method, "metric": metric, "alpha_total": alpha, "n": n,
                    "repetitions": len(group), "mean": mean, "se": se,
                    "ci_low": mean-q*se if len(group) > 1 else np.nan,
                    "ci_high": mean+q*se if len(group) > 1 else np.nan,
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "tables" / "summary.csv", index=False)

    slope_rows = []
    if raw.rep.nunique() > 1 and raw.n.nunique() > 1:
        for method in ("private", "nonprivate"):
            for metric in available_metrics:
                for alpha, group in raw.groupby("alpha_total"):
                    est, lo, hi = bootstrap_slope(
                        group, f"{method}_{metric}",
                        20261027 + int(alpha*100) + (0 if method == "private" else 1))
                    slope_rows.append({"method": method, "metric": metric,
                                       "alpha_total": alpha, "slope": est,
                                       "ci_low": lo, "ci_high": hi,
                                       "bootstrap_repetitions": 2000})
    pd.DataFrame(slope_rows).to_csv(out / "tables" / "slopes.csv", index=False)

    # Compare the two methods within each paired repetition.  Resampling whole
    # pairs preserves the common training dataset used by the experiment.
    comparison_rows = []
    for metric in available_metrics:
        pcol = f"private_{metric}"
        qcol = f"nonprivate_{metric}"
        for (alpha, n), group in raw.groupby(["alpha_total", "n"]):
            pvals = group.sort_values("rep")[pcol].to_numpy(float)
            qvals = group.sort_values("rep")[qcol].to_numpy(float)
            estimate = float(pvals.mean() / qvals.mean())
            difference = float(np.mean(pvals - qvals))
            random = np.random.default_rng(20261028 + int(alpha*100) + int(n))
            ids = random.integers(0, len(group), size=(2000, len(group)))
            boot_ratio = pvals[ids].mean(axis=1) / qvals[ids].mean(axis=1)
            boot_diff = (pvals[ids] - qvals[ids]).mean(axis=1)
            comparison_rows.append({
                "metric": metric, "alpha_total": alpha, "n": n,
                "repetitions": len(group),
                "private_mean": pvals.mean(), "nonprivate_mean": qvals.mean(),
                "private_to_nonprivate_ratio": estimate,
                "ratio_ci_low": np.quantile(boot_ratio, .025),
                "ratio_ci_high": np.quantile(boot_ratio, .975),
                "paired_mean_difference": difference,
                "difference_ci_low": np.quantile(boot_diff, .025),
                "difference_ci_high": np.quantile(boot_diff, .975),
                "fraction_private_lower": np.mean(pvals < qvals),
                "bootstrap_repetitions": 2000,
            })
    pd.DataFrame(comparison_rows).to_csv(
        out / "tables" / "private_vs_nonprivate.csv", index=False)



if __name__ == "__main__":
    main()
