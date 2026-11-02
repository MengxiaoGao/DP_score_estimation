#!/usr/bin/env python3
"""Recreate the paper's 1D score-risk and replace-one score plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_style import (
    ERM, PRIVATE, SECONDARY, THEORY_PRIVATE, THEORY_SAMPLING,
    draw_curve, log_panel, mean_ci, read_csv, require_columns,
    save_figure, scaled_reference, set_paper_style,
)


def _risk_panel(raw_path: Path, output_dir: Path, panel: str) -> None:
    raw = read_csv(raw_path, ["n", "gamma", "alpha", "rep", "score_risk", "is_erm"])
    if panel == "alpha":
        values = sorted(raw.loc[~raw["is_erm"].astype(bool), "alpha"].dropna().unique())
        fixed = "alpha"
        stem = "score_alpha_panel"
    else:
        values = sorted(raw.loc[~raw["is_erm"].astype(bool), "gamma"].dropna().unique())
        fixed = "gamma"
        stem = "score_gamma_panel"
    if len(values) != 5:
        raise ValueError(f"Expected five {fixed} values in {raw_path}, found {values}")

    fig, axes = plt.subplots(1, 5, figsize=(10.8, 2.25), sharex=True)
    legend_handles = None
    for ax, value in zip(axes, values):
        private = raw[(~raw["is_erm"].astype(bool)) & np.isclose(raw[fixed], value)]
        if panel == "alpha":
            erm = raw[raw["is_erm"].astype(bool)]
            gamma = float(private["gamma"].iloc[0])
            alpha = float(value)
        else:
            erm = raw[raw["is_erm"].astype(bool) & np.isclose(raw["gamma"], value)]
            gamma = float(value)
            alpha = float(private["alpha"].iloc[0])
        p = mean_ci(private, ["n"], "score_risk")
        e = mean_ci(erm, ["n"], "score_risk")
        draw_curve(ax, p, color=PRIVATE, label="Private")
        draw_curve(ax, e, color=ERM, label="Non-private ERM", linestyle="--", marker="s")

        n = p.sort_values("n")["n"].to_numpy(float)
        anchor = float(p.sort_values("n")["mean"].iloc[0])
        sampling = scaled_reference(n, n ** (-2 * gamma / (2 * gamma + 3)), anchor)
        privacy = scaled_reference(n, (n * alpha) ** (-2 * gamma / (gamma + 2)), anchor)
        ax.plot(n, sampling, color=THEORY_SAMPLING, linestyle=":",
                label=rf"$n^{{-2\gamma/(3+2\gamma)}}$")
        ax.plot(n, privacy, color=THEORY_PRIVATE, linestyle="-.",
                label=rf"$(n\alpha)^{{-2\gamma/(2+\gamma)}}$")
        ax.set_title(rf"$\{fixed}={value:g}$")
        log_panel(ax)
        if ax is axes[0]:
            ax.set_ylabel("Score-matching risk")
            legend_handles = ax.get_legend_handles_labels()
    if legend_handles:
        fig.legend(*legend_handles, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.23, top=0.76, wspace=0.30)
    save_figure(fig, output_dir, stem)


def _perturbation(stability_path: Path, output_dir: Path, metric: str) -> None:
    raw = read_csv(stability_path, ["n", "alpha", "rep", metric])
    private_col = metric
    erm_col = metric.replace("private_", "erm_", 1)
    require_columns(raw, [erm_col], stability_path)
    alphas = sorted(raw["alpha"].unique())
    colors = ["#2f6b93", "#218c74", "#b88a2b", "#dd6b20", "#b23a48"]
    fig, ax = plt.subplots(figsize=(4.55, 3.1))
    for alpha, color in zip(alphas, colors):
        summary = mean_ci(raw[np.isclose(raw["alpha"], alpha)], ["n"], private_col)
        draw_curve(ax, summary, color=color, label=rf"$\alpha={alpha:g}$")
    # The ERM result is repeated once per privacy budget in the stability
    # export. Use one paired alpha slice so the CI is based on repetitions,
    # rather than treating duplicated rows as independent observations.
    baseline = raw[np.isclose(raw["alpha"], alphas[0])]
    erm = mean_ci(baseline, ["n"], erm_col)
    draw_curve(ax, erm, color=ERM, label="Non-private ERM", linestyle="--", marker="s")
    log_panel(ax)
    ylabel = ("Mean squared coefficient change" if metric == "private_coefficient_mse"
              else "Mean squared score-function change")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.18, top=0.96)
    save_figure(fig, output_dir, "score_perturbation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privacy-csv", type=Path, required=True,
                        help="privacy/tables/score_raw.csv")
    parser.add_argument("--smoothness-csv", type=Path, required=True,
                        help="smoothness/tables/score_raw.csv")
    parser.add_argument("--stability-csv", type=Path, required=True,
                        help="coefficient_stability.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perturbation-metric",
                        choices=("private_coefficient_mse", "private_score_squared"),
                        default="private_coefficient_mse")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_paper_style()
    _risk_panel(args.privacy_csv, args.output_dir, "alpha")
    _risk_panel(args.smoothness_csv, args.output_dir, "gamma")
    _perturbation(args.stability_csv, args.output_dir, args.perturbation_metric)
    print(f"Saved score figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
