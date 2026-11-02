#!/usr/bin/env python3
"""Recreate the paper's 1D diffusion-TV and replace-one TV plots."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_style import (
    ERM, PRIVATE, THEORY_PRIVATE, THEORY_SAMPLING, draw_curve, log_panel,
    mean_ci, read_csv, require_columns, save_figure, scaled_reference,
    set_paper_style,
)


def _panel(private_path: Path, erm_path: Path, output_dir: Path,
           panel: str, metric: str) -> None:
    private = read_csv(private_path, ["n", "rep", "gamma", "alpha_total", metric])
    erm = read_csv(erm_path, ["n", "rep", "gamma", metric])
    fixed = "alpha_total" if panel == "alpha" else "gamma"
    values = sorted(private[fixed].unique())
    if len(values) != 5:
        raise ValueError(f"Expected five {fixed} values in {private_path}, found {values}")

    fig, axes = plt.subplots(1, 5, figsize=(10.8, 2.25), sharex=True)
    legend_handles = None
    for ax, value in zip(axes, values):
        subset = private[np.isclose(private[fixed], value)]
        gamma = float(subset["gamma"].iloc[0])
        erm_subset = erm if panel == "alpha" else erm[np.isclose(erm["gamma"], value)]
        p = mean_ci(subset, ["n"], metric)
        e = mean_ci(erm_subset, ["n"], metric)
        draw_curve(ax, p, color=PRIVATE, label="Private")
        draw_curve(ax, e, color=ERM, label="Non-private ERM", linestyle="--", marker="s")

        n = p.sort_values("n")["n"].to_numpy(float)
        anchor = float(p.sort_values("n")["mean"].iloc[0])
        sampling_exponent = -(gamma + 1) / (3 + 2 * gamma)
        private_exponent = -(gamma + 1) / (2 + gamma)
        sampling = scaled_reference(n, n ** sampling_exponent, anchor)
        total_alpha = float(value) if panel == "alpha" else float(subset["alpha_total"].iloc[0])
        privacy = scaled_reference(n, (n * total_alpha) ** private_exponent, anchor)
        ax.plot(n, sampling, color=THEORY_SAMPLING, linestyle=":",
                label=rf"$n^{{-(\gamma+1)/(3+2\gamma)}}$")
        ax.plot(n, privacy, color=THEORY_PRIVATE, linestyle="-.",
                label=rf"$(n\alpha)^{{-(\gamma+1)/(2+\gamma)}}$")
        title_name = "alpha" if panel == "alpha" else "gamma"
        ax.set_title(rf"$\{title_name}={value:g}$")
        log_panel(ax)
        if ax is axes[0]:
            ax.set_ylabel("TV error")
            legend_handles = ax.get_legend_handles_labels()
    if legend_handles:
        fig.legend(*legend_handles, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.23, top=0.76, wspace=0.30)
    save_figure(fig, output_dir, f"diffusion_{panel}_tv")


def _perturbation(path: Path, output_dir: Path, metric: str) -> None:
    private_col = f"private_{metric}"
    erm_col = f"nonprivate_{metric}"
    raw = read_csv(path, ["n", "rep", "alpha_total", private_col, erm_col])
    alphas = sorted(raw["alpha_total"].unique())
    colors = ["#2f6b93", "#218c74", "#b88a2b", "#dd6b20", "#b23a48"]
    fig, ax = plt.subplots(figsize=(4.55, 3.1))
    for alpha, color in zip(alphas, colors):
        subset = raw[np.isclose(raw["alpha_total"], alpha)]
        draw_curve(ax, mean_ci(subset, ["n"], private_col), color=color,
                   label=rf"$\alpha={alpha:g}$")
    # Non-private rows are repeated once for every alpha. Averaging them all is
    # valid but would artificially shrink the CI, so use one alpha slice.
    baseline = raw[np.isclose(raw["alpha_total"], alphas[0])]
    draw_curve(ax, mean_ci(baseline, ["n"], erm_col), color=ERM,
               label="Non-private ERM", linestyle="--", marker="s")
    log_panel(ax)
    ax.set_ylabel("Mean TV change")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.18, top=0.96)
    save_figure(fig, output_dir, "diffusion_perturbation_tv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-csv", type=Path, required=True,
                        help="completed alpha/raw.csv")
    parser.add_argument("--gamma-csv", type=Path, required=True,
                        help="completed gamma/raw.csv")
    parser.add_argument("--erm-alpha-csv", type=Path, required=True,
                        help="erm_baselines/alpha/raw.csv")
    parser.add_argument("--erm-gamma-csv", type=Path, required=True,
                        help="erm_baselines/gamma/raw.csv")
    parser.add_argument("--perturbation-csv", type=Path, required=True,
                        help="perturbation/raw.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--utility-metric", choices=("density_tv", "ou_tv", "generated_binned_tv"),
                        default="generated_binned_tv")
    parser.add_argument("--perturbation-metric",
                        choices=("ou_tv_change", "density_tv_change", "generated_binned_tv_change"),
                        default="ou_tv_change")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_paper_style()
    _panel(args.alpha_csv, args.erm_alpha_csv, args.output_dir,
           "alpha", args.utility_metric)
    _panel(args.gamma_csv, args.erm_gamma_csv, args.output_dir,
           "gamma", args.utility_metric)
    _perturbation(args.perturbation_csv, args.output_dir, args.perturbation_metric)
    print(f"Saved diffusion figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
