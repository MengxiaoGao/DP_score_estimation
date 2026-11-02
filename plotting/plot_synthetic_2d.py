#!/usr/bin/env python3
"""Plot the 2D synthetic score, diffusion, and replace-one experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_style import (
    ERM, PRIVATE, THEORY_PRIVATE, THEORY_SAMPLING, draw_curve, log_panel,
    mean_ci, read_csv, require_columns, save_figure, scaled_reference,
    set_paper_style,
)


def _read_panel(paths: list[Path], panel: str) -> pd.DataFrame:
    frames = [read_csv(path) for path in paths]
    raw = pd.concat(frames, ignore_index=True)
    required = ["method", "n", "rep", "gamma", "alpha_score", "alpha_total"]
    require_columns(raw, required, Path(f"{panel} inputs"))
    keys = ["method", "n", "rep", "gamma", "alpha_score"]
    if raw.duplicated(keys).any():
        raise ValueError(f"Duplicate rows across the supplied {panel} CSV files")
    parameter = "alpha_score" if panel == "alpha" else "gamma"
    values = sorted(raw[parameter].unique())
    if len(values) != 5:
        raise ValueError(f"Expected five {parameter} values, found {values}")
    return raw


def _reference_rates(n: np.ndarray, gamma: float, alpha: float,
                     target: str, anchor: float) -> tuple[np.ndarray, np.ndarray]:
    if target == "score":
        sampling = n ** (-2 * gamma / (2 * gamma + 4))
        private = (n * alpha) ** (-2 * gamma / (gamma + 3))
    else:
        sampling = n ** (-(gamma + 1) / (2 * gamma + 4))
        private = (n * alpha) ** (-(gamma + 1) / (gamma + 3))
    return (scaled_reference(n, sampling, anchor),
            scaled_reference(n, private, anchor))


def _panel(raw: pd.DataFrame, output_dir: Path, panel: str,
           target: str, metric: str) -> None:
    require_columns(raw, [metric], Path(f"2D {panel} CSV"))
    parameter = "alpha_score" if panel == "alpha" else "gamma"
    values = sorted(raw[parameter].unique())
    fig, axes = plt.subplots(1, 5, figsize=(10.8, 2.25), sharex=True)
    legend_handles = None

    for ax, value in zip(axes, values):
        part = raw[np.isclose(raw[parameter], value)]
        private = part[part["method"] == "private"]
        nonprivate = part[part["method"] == "nonprivate"]
        if private.empty or nonprivate.empty:
            raise ValueError(f"Missing private or non-private rows for {parameter}={value:g}")
        p = mean_ci(private, ["n"], metric)
        e = mean_ci(nonprivate, ["n"], metric)
        draw_curve(ax, p, color=PRIVATE, label="Private")
        draw_curve(ax, e, color=ERM, label="Non-private ERM",
                   linestyle="--", marker="s")

        gamma = float(private["gamma"].iloc[0])
        alpha_column = "alpha_score" if target == "score" else "alpha_total"
        alpha = float(private[alpha_column].iloc[0])
        ordered = p.sort_values("n")
        n = ordered["n"].to_numpy(float)
        anchor = float(ordered["mean"].iloc[0])
        sampling, privacy = _reference_rates(n, gamma, alpha, target, anchor)
        if target == "score":
            sampling_label = r"$n^{-2\gamma/(4+2\gamma)}$"
            private_label = r"$(n\alpha)^{-2\gamma/(3+\gamma)}$"
        else:
            sampling_label = r"$n^{-(\gamma+1)/(4+2\gamma)}$"
            private_label = r"$(n\alpha)^{-(\gamma+1)/(3+\gamma)}$"
        ax.plot(n, sampling, color=THEORY_SAMPLING, linestyle=":",
                label=sampling_label)
        ax.plot(n, privacy, color=THEORY_PRIVATE, linestyle="-.",
                label=private_label)

        if panel == "alpha":
            shown = float(private[alpha_column].iloc[0])
            ax.set_title(rf"$\alpha={shown:g}$")
        else:
            ax.set_title(rf"$\gamma={value:g}$")
        log_panel(ax)
        if ax is axes[0]:
            ax.set_ylabel("Score-matching risk" if target == "score" else "TV error")
            legend_handles = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(*legend_handles, loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(left=0.062, right=0.995, bottom=0.23,
                        top=0.76, wspace=0.30)
    save_figure(fig, output_dir, f"synthetic_2d_{target}_{panel}_panel")


def _perturbation(path: Path, output_dir: Path, metric: str) -> None:
    raw = read_csv(path, ["mode", "n", "rep", "alpha_total", metric])
    private_mode = "private_fixed3_crn"
    nonprivate_mode = "nonprivate_fixed3_crn"
    modes = set(raw["mode"])
    if not {private_mode, nonprivate_mode} <= modes:
        raise ValueError(f"Expected fixed-(3,3) modes, found {sorted(modes)}")
    alphas = sorted(raw["alpha_total"].unique())
    if len(alphas) != 5:
        raise ValueError(f"Expected five total privacy budgets, found {alphas}")
    colors = ["#2f6b93", "#218c74", "#b88a2b", "#dd6b20", "#b23a48"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharex=True)
    for ax, mode, title in (
        (axes[0], private_mode, "Private"),
        (axes[1], nonprivate_mode, "Non-private ERM"),
    ):
        for alpha, color in zip(alphas, colors):
            part = raw[(raw["mode"] == mode) & np.isclose(raw["alpha_total"], alpha)]
            draw_curve(ax, mean_ci(part, ["n"], metric), color=color,
                       label=rf"$\alpha={alpha:g}$")
        log_panel(ax)
        ax.set_title(title)
        ax.set_ylabel("Mean TV change")
    axes[0].legend(frameon=False, ncol=2, loc="best")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.19,
                        top=0.92, wspace=0.28)
    save_figure(fig, output_dir, "synthetic_2d_perturbation_tv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha-csv", type=Path, nargs="+", required=True,
                        help="one combined alpha CSV, or several batch raw.csv files")
    parser.add_argument("--gamma-csv", type=Path, nargs="+", required=True,
                        help="one combined gamma CSV, or several batch raw.csv files")
    parser.add_argument("--perturbation-csv", type=Path,
                        help="fixed-(3,3) perturbation raw.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diffusion-metric",
                        choices=("density_tv", "ou_tv", "generated_binned_tv"),
                        default="generated_binned_tv")
    parser.add_argument("--perturbation-metric",
                        choices=("density_tv_change", "ou_tv_change",
                                 "generated_binned_tv_change"),
                        default="generated_binned_tv_change")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_paper_style()
    alpha = _read_panel(args.alpha_csv, "alpha")
    gamma = _read_panel(args.gamma_csv, "gamma")
    _panel(alpha, args.output_dir, "alpha", "score", "score_risk")
    _panel(gamma, args.output_dir, "gamma", "score", "score_risk")
    _panel(alpha, args.output_dir, "alpha", "diffusion", args.diffusion_metric)
    _panel(gamma, args.output_dir, "gamma", "diffusion", args.diffusion_metric)
    if args.perturbation_csv is not None:
        _perturbation(args.perturbation_csv, args.output_dir,
                      args.perturbation_metric)
    print(f"Saved 2D synthetic figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
