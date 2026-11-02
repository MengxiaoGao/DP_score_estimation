#!/usr/bin/env python3
"""Combine completed 1D private panels with constrained-ERM baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parent

METRICS = {
    "density_tv": "Reconstructed-density TV",
    "ou_tv": "Early-stopped OU TV",
    "generated_binned_tv": "Generated binned TV",
    "generated_wasserstein": "Generated Wasserstein-1",
}


def summarize(frame: pd.DataFrame, groups, metrics=METRICS) -> pd.DataFrame:
    rows = []
    for keys, part in frame.groupby(groups):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(groups, keys))
        for metric in metrics:
            values = part[metric].to_numpy(float)
            mean = float(values.mean())
            se = float(values.std(ddof=1) / np.sqrt(len(values)))
            half = float(student_t.ppf(.975, len(values)-1) * se)
            rows.append({**base, "metric": metric, "repetitions": len(values),
                         "mean": mean, "se": se,
                         "ci95_low": mean-half, "ci95_high": mean+half})
    return pd.DataFrame(rows)


def save_summaries(panel: str, private: pd.DataFrame, erm: pd.DataFrame, out: Path) -> None:
    parameter = "alpha_total" if panel == "alpha" else "gamma"
    private_summary = summarize(private, [parameter, "n"])
    erm_summary = summarize(erm, ["gamma", "n"])
    private_summary.to_csv(out / "tables" / f"{panel}_private_summary.csv", index=False)
    erm_summary.to_csv(out / "tables" / f"{panel}_erm_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "erm_baselines")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    for folder in ("tables",):
        (out / folder).mkdir(parents=True, exist_ok=True)
    private_dirs = {
        "alpha": ROOT / "outputs" / "alpha" / "raw.csv",
        "gamma": ROOT / "outputs" / "gamma" / "raw.csv",
    }
    expected = {"alpha": 500, "gamma": 2500}
    for panel in ("alpha", "gamma"):
        erm_path = out / panel / "raw.csv"
        if not erm_path.exists():
            raise RuntimeError(f"Missing {erm_path}")
        erm = pd.read_csv(erm_path)
        if len(erm) != expected[panel]:
            raise RuntimeError(f"Incomplete {panel} ERM data: {len(erm)}/{expected[panel]}")
        save_summaries(panel, pd.read_csv(private_dirs[panel]), erm, out)
    print(f"Tables saved to {out / 'tables'}")


if __name__ == "__main__":
    main()
