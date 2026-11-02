"""Shared paper plotting style and small validation helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


PRIVATE = "#2f6b93"
PRIVATE_FILL = "#b8d2e2"
ERM = "#3f3f3f"
SECONDARY = "#b86b25"
SECONDARY_FILL = "#ead2b8"
THEORY_SAMPLING = "#d28c24"
THEORY_PRIVATE = "#c24b35"
GRID = "#dfe7ec"


def set_paper_style() -> None:
    """Use Computer Modern math and a matching serif face without LaTeX."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.65,
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "lines.linewidth": 1.15,
        "lines.markersize": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        "grid.alpha": 0.9,
        "axes.axisbelow": True,
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def require_columns(frame: pd.DataFrame, columns: list[str] | tuple[str, ...], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def read_csv(path: Path, columns: list[str] | tuple[str, ...] = ()) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    require_columns(frame, columns, path)
    return frame


def canonicalize_score_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """Read legacy exports while keeping score_matching_risk canonical."""
    aliases = {
        "heldout_hyvarinen": "score_matching_risk",
        "utility_heldout_hyvarinen": "utility_score_matching_risk",
        "utility_Dprime_heldout_hyvarinen": "utility_Dprime_score_matching_risk",
    }
    rename = {}
    for old, new in aliases.items():
        if old in frame.columns and new not in frame.columns:
            rename[old] = new
    return frame.rename(columns=rename)


def mean_ci(frame: pd.DataFrame, groups: list[str], value: str) -> pd.DataFrame:
    require_columns(frame, [*groups, value], Path("data frame"))
    rows = []
    for keys, part in frame.groupby(groups, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        values = part[value].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite values in plotted metric {value}")
        if len(values) < 2:
            half = 0.0
        else:
            half = float(student_t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
        row = dict(zip(groups, keys))
        row.update(repetitions=len(values), mean=float(values.mean()),
                   low=float(values.mean() - half), high=float(values.mean() + half))
        rows.append(row)
    return pd.DataFrame(rows)


def positive(values: np.ndarray, label: str) -> None:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{label} must be finite and strictly positive for log axes")


def draw_curve(ax, summary: pd.DataFrame, *, color: str, label: str,
               linestyle: str = "-", marker: str = "o", fill: bool = True) -> None:
    summary = summary.sort_values("n")
    x = summary["n"].to_numpy(float)
    y = summary["mean"].to_numpy(float)
    positive(x, f"{label} sample sizes")
    positive(y, label)
    ax.plot(x, y, color=color, linestyle=linestyle, marker=marker, label=label,
            markerfacecolor="white", markeredgewidth=0.7)
    if fill and {"low", "high"} <= set(summary.columns):
        low = summary["low"].to_numpy(float)
        high = summary["high"].to_numpy(float)
        # A symmetric t interval can cross zero for a small positive metric.
        # Zero is not representable on a log axis; omit those parts of the
        # ribbon instead of clipping them to machine precision and distorting
        # the entire y scale.
        visible = np.isfinite(low) & np.isfinite(high) & (low > 0) & (high > 0)
        ax.fill_between(x, low, high, where=visible, color=color,
                        alpha=0.15, linewidth=0)


def log_panel(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = np.array([1000, 2000, 4000, 8000, 16000], dtype=float)
    ax.set_xticks(ticks, ["1k", "2k", "4k", "8k", "16k"])
    ax.set_xlim(900, 17500)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=2.5, pad=2)
    ax.set_xlabel(r"Training sample size $n$")


def save_figure(fig, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / f"{stem}.pdf"
    png = output_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf, png


def scaled_reference(n: np.ndarray, rate: np.ndarray, anchor: float) -> np.ndarray:
    n = np.asarray(n, dtype=float)
    rate = np.asarray(rate, dtype=float)
    positive(n, "reference sample sizes")
    positive(rate, "reference rate")
    return rate * (anchor / rate[0])
