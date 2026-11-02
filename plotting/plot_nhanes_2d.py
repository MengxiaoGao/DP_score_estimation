#!/usr/bin/env python3
"""Plot the formal 2D NHANES score, TV, downstream, and contour results."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from paper_style import (
    ERM, PRIVATE, PRIVATE_FILL, SECONDARY, SECONDARY_FILL,
    canonicalize_score_risk, mean_ci, read_csv, require_columns,
    save_figure, set_paper_style,
)


def _private_curve(ax, private: pd.DataFrame, metric: str, *, color: str,
                   label: str, marker: str = "o") -> None:
    summary = mean_ci(private, ["alpha"], metric).sort_values("alpha")
    x = summary["alpha"].to_numpy(float)
    ax.plot(x, summary["mean"], color=color, marker=marker, label=label,
            markerfacecolor="white", markeredgewidth=0.75)
    ax.fill_between(x, summary["low"], summary["high"], color=color,
                    alpha=0.15, linewidth=0)


def _nonprivate_band(ax, nonprivate: pd.DataFrame, metric: str, *, color: str,
                     label: str, xmin: float = 1, xmax: float = 16) -> None:
    summary = mean_ci(nonprivate.assign(_group=0), ["_group"], metric).iloc[0]
    ax.axhline(summary["mean"], color=color, linestyle="--", linewidth=1.05,
               label=label)
    ax.fill_between([xmin, xmax], [summary["low"]] * 2, [summary["high"]] * 2,
                    color=color, alpha=0.10, linewidth=0)


def _alpha_axis(ax, ylabel: str) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax.set_xlim(0.9, 17.5)
    ax.set_xlabel(r"Privacy budget $\alpha$")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)


def score_figure(private: pd.DataFrame, nonprivate: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("utility_score_matching_risk", "Score-matching risk"),
        ("perturbation_score_function_rms", "Replace-one score RMS"),
    ]
    for metric, _ in metrics:
        require_columns(private, ["alpha", metric], Path("private CSV"))
        require_columns(nonprivate, [metric], Path("non-private CSV"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    for ax, (metric, ylabel) in zip(axes, metrics):
        _private_curve(ax, private, metric, color=PRIVATE, label="Private")
        _nonprivate_band(ax, nonprivate, metric, color=ERM, label="Non-private ERM")
        _alpha_axis(ax, ylabel)
    axes[0].legend(frameon=False, loc="best")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.22, top=0.97, wspace=0.34)
    save_figure(fig, output, "nhanes_score")


def main_figure(private: pd.DataFrame, nonprivate: pd.DataFrame, output: Path) -> None:
    required = [
        "utility_binned_tv_2d", "utility_bmi_binned_tv",
        "utility_waist_binned_tv", "utility_correlation_error",
        "utility_obesity_error_pp", "utility_joint_tail_error_pp",
    ]
    require_columns(private, ["alpha", *required], Path("private CSV"))
    require_columns(nonprivate, required, Path("non-private CSV"))
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 2.75))

    _private_curve(axes[0], private, "utility_binned_tv_2d", color=PRIVATE,
                   label="Private")
    _nonprivate_band(axes[0], nonprivate, "utility_binned_tv_2d", color=ERM,
                     label="Non-private ERM")
    _alpha_axis(axes[0], "Joint binned TV")
    axes[0].legend(frameon=False, loc="best")

    _private_curve(axes[1], private, "utility_bmi_binned_tv", color=PRIVATE,
                   label="BMI")
    _private_curve(axes[1], private, "utility_waist_binned_tv", color=SECONDARY,
                   label="Waist", marker="s")
    _nonprivate_band(axes[1], nonprivate, "utility_bmi_binned_tv", color=PRIVATE,
                     label="BMI ERM")
    _nonprivate_band(axes[1], nonprivate, "utility_waist_binned_tv", color=SECONDARY,
                     label="Waist ERM")
    _alpha_axis(axes[1], "Marginal binned TV")
    axes[1].legend(frameon=False, ncol=2, loc="best", columnspacing=0.8)

    _private_curve(axes[2], private, "utility_correlation_error", color=PRIVATE,
                   label="Private")
    _nonprivate_band(axes[2], nonprivate, "utility_correlation_error", color=ERM,
                     label="Non-private ERM")
    _alpha_axis(axes[2], "Absolute correlation error")

    _private_curve(axes[3], private, "utility_obesity_error_pp", color=PRIVATE,
                   label="Obesity")
    _private_curve(axes[3], private, "utility_joint_tail_error_pp", color=SECONDARY,
                   label="Joint tail", marker="s")
    _nonprivate_band(axes[3], nonprivate, "utility_obesity_error_pp", color=PRIVATE,
                     label="Obesity ERM")
    _nonprivate_band(axes[3], nonprivate, "utility_joint_tail_error_pp", color=SECONDARY,
                     label="Joint-tail ERM")
    _alpha_axis(axes[3], "Absolute prevalence error (pp)")
    axes[3].legend(frameon=False, ncol=2, loc="best", columnspacing=0.8)

    fig.subplots_adjust(left=0.058, right=0.995, bottom=0.23, top=0.97, wspace=0.36)
    save_figure(fig, output, "nhanes_main")


def perturbation_figure(private: pd.DataFrame, nonprivate: pd.DataFrame,
                        output: Path) -> None:
    metric = "perturbation_generated_binned_tv_change"
    require_columns(private, ["alpha", metric], Path("private CSV"))
    require_columns(nonprivate, [metric], Path("non-private CSV"))
    fig, ax = plt.subplots(figsize=(4.15, 3.0))
    _private_curve(ax, private, metric, color=PRIVATE, label="Private")
    _nonprivate_band(ax, nonprivate, metric, color=ERM, label="Non-private ERM")
    _alpha_axis(ax, "Replace-one diffusion TV")
    ax.legend(frameon=False, loc="best")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.19, top=0.97)
    save_figure(fig, output, "nhanes_perturbation")


def _cohort(data_dir: Path) -> pd.DataFrame:
    bmx = pd.read_sas(data_dir / "P_BMX.xpt", format="xport")
    demo = pd.read_sas(data_dir / "P_DEMO.xpt", format="xport")
    frame = bmx.merge(
        demo[["SEQN", "RIDAGEYR", "RIAGENDR", "RIDEXPRG", "WTMECPRP"]],
        on="SEQN", how="inner", validate="one_to_one",
    )
    mask = (
        frame["RIDAGEYR"].between(20, 79)
        & frame["BMXBMI"].notna()
        & frame["BMXWAIST"].notna()
        & (frame["RIDEXPRG"].ne(1) | frame["RIDEXPRG"].isna())
    )
    return frame.loc[mask].sort_values("SEQN").reset_index(drop=True)


def _seed_int(base_seed: int, *keys: int) -> int:
    sequence = np.random.SeedSequence([int(base_seed), *[int(k) for k in keys]])
    return int(np.random.default_rng(sequence).integers(1, 2**31 - 1))


def _transformed_cohort(data_dir: Path, seed: int) -> np.ndarray:
    values = _cohort(data_dir)[["BMXBMI", "BMXWAIST"]].to_numpy(float)
    values[:, 0] = np.clip(values[:, 0], 10, 100)
    values[:, 1] = np.clip(values[:, 1], 40, 220)
    rng = np.random.default_rng(_seed_int(seed, 41_001))
    values += rng.uniform(-0.05, 0.05, size=values.shape)
    return np.column_stack((np.log(values[:, 0] / 25) / 0.35,
                            np.log(values[:, 1] / 90) / 0.30))


def _inverse(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return np.column_stack((
        np.clip(25 * np.exp(0.35 * values[:, 0]), 10, 100),
        np.clip(90 * np.exp(0.30 * values[:, 1]), 40, 220),
    ))


def _release(results_root: Path, split: int, alpha: int, nonprivate: bool = False) -> np.ndarray:
    root = results_root / f"split_{split:03d}" / f"alpha_{alpha:g}" / "paired"
    if nonprivate:
        root = root / "nonprivate"
    path = root / "releases" / "D.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as saved:
        generated = np.asarray(saved["generated"], float)
    if generated.ndim != 2 or generated.shape[1] != 2 or not np.isfinite(generated).all():
        raise ValueError(f"Invalid generated sample in {path}")
    return generated


def contour_figure(results_root: Path, data_dir: Path, output: Path,
                   split: int, seed: int) -> None:
    job = results_root / f"split_{split:03d}" / "alpha_1"
    indices_path = job / "split_indices.npz"
    if not indices_path.exists():
        raise FileNotFoundError(indices_path)
    with np.load(indices_path, allow_pickle=False) as saved:
        holdout_indices = saved["evaluation_holdout_indices"].astype(int)
    transformed = _transformed_cohort(data_dir, seed)
    holdout = _inverse(transformed[holdout_indices])
    generated = [
        _inverse(_release(results_root, split, 1)),
        _inverse(_release(results_root, split, 16)),
        _inverse(_release(results_root, split, 1, nonprivate=True)),
    ]
    labels = [r"Private, $\alpha=1$", r"Private, $\alpha=16$", "Non-private ERM"]

    # Frame the scientifically relevant held-out support. Generated tail
    # outliers are retained in the numerical TV metric but should not force
    # the central density comparison into a mostly empty plotting window.
    xlim = np.quantile(holdout[:, 0], [0.003, 0.997])
    ylim = np.quantile(holdout[:, 1], [0.003, 0.997])
    xpad, ypad = 0.05 * np.diff(xlim)[0], 0.05 * np.diff(ylim)[0]
    xlim += np.array([-xpad, xpad])
    ylim += np.array([-ypad, ypad])
    xedges = np.linspace(*xlim, 76)
    yedges = np.linspace(*ylim, 76)
    dx, dy = np.diff(xedges)[0], np.diff(yedges)[0]

    def density(sample: np.ndarray) -> np.ndarray:
        counts = np.histogram2d(sample[:, 0], sample[:, 1], bins=(xedges, yedges))[0]
        smoothed = gaussian_filter(counts, sigma=1.25, mode="nearest")
        return smoothed / (max(smoothed.sum(), 1) * dx * dy)

    reference = density(holdout)
    differences = [density(sample) - reference for sample in generated]
    vmax = max(float(np.max(np.abs(diff))) for diff in differences)
    vmax = max(vmax, np.finfo(float).eps)
    levels = np.linspace(-vmax, vmax, 17)
    centers_x = (xedges[:-1] + xedges[1:]) / 2
    centers_y = (yedges[:-1] + yedges[1:]) / 2
    contour_levels = np.array([0.18, 0.36, 0.58]) * float(reference.max())

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.15), sharex=True, sharey=True)
    image = None
    for ax, diff, label in zip(axes, differences, labels):
        image = ax.contourf(centers_x, centers_y, diff.T, levels=levels,
                            cmap="RdBu_r", extend="both")
        ax.contour(centers_x, centers_y, reference.T, levels=contour_levels,
                   colors="#333333", linestyles="--", linewidths=0.75, alpha=0.85)
        ax.set_title(label)
        ax.set_xlabel("BMI")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(False)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Waist circumference (cm)")
    cbar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.08,
                        pad=0.20, aspect=45)
    cbar.set_label("Generated density $-$ held-out density")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.29, top=0.90, wspace=0.10)
    save_figure(fig, output, "nhanes_contours")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-csv", type=Path, required=True,
                        help="combined_private_metrics.csv")
    parser.add_argument("--nonprivate-csv", type=Path, required=True,
                        help="combined_nonprivate_metrics.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results-root", type=Path,
                        help="formal results root; enables the contour figure")
    parser.add_argument("--data-dir", type=Path,
                        help="directory with P_BMX.xpt and P_DEMO.xpt; required for contours")
    parser.add_argument("--contour-split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_paper_style()
    private = canonicalize_score_risk(read_csv(args.private_csv, ["split_id", "alpha"]))
    nonprivate = canonicalize_score_risk(read_csv(args.nonprivate_csv, ["split_id"]))
    score_figure(private, nonprivate, args.output_dir)
    main_figure(private, nonprivate, args.output_dir)
    perturbation_figure(private, nonprivate, args.output_dir)
    if (args.results_root is None) != (args.data_dir is None):
        raise ValueError("--results-root and --data-dir must be supplied together")
    if args.results_root is not None:
        contour_figure(args.results_root.resolve(), args.data_dir.resolve(),
                       args.output_dir, args.contour_split, args.seed)
    print(f"Saved NHANES figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
