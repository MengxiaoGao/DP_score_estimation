#!/usr/bin/env python3
"""Formal non-private ERM baselines for the paper-aligned 1D panels.

The alpha panel needs one ERM fit per (n, repetition), because the spline
geometry is identical across the five privacy budgets.  The gamma panel needs
one fit per (n, gamma, repetition).  Each job reconstructs an ordinary
histogram, evaluates density and early-stopped OU error, and runs the same
reverse-OU sampler used in the completed private panels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
ROOT = HERE
PARENT = ROOT.parent
sys.path[:0] = [str(ROOT), str(PARENT), str(PARENT / "numerics")]

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import diffusion_extension_lmc as d
import paper_numerics as pn
import run_paper as rp
from common import draw, reverse, target
from dp_score_sim.core import solve_constrained_erm
from runner import BASE, nested_training, rng

pn.install()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def job_tag(panel: str, n: int, rep: int, gamma: float) -> str:
    return f"{panel}_n{n}_r{rep:03d}_g{gamma:g}".replace(".", "p")


def one(job):
    panel, n, rep, gamma, source_hash, cdict, outdir = job
    c = SimpleNamespace(**cdict)
    c.gamma = float(gamma)
    c.alpha = float(c.gamma_panel_alpha_total)
    alpha_reference = float(c.gamma_panel_alpha_total)
    dest = Path(outdir) / panel / "rows" / f"{job_tag(panel, n, rep, gamma)}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    started = time.perf_counter()
    try:
        x = nested_training(c, n, rep, gamma)
        if array_hash(x) != source_hash:
            raise RuntimeError("Regenerated training sample does not match formal panel")

        basis_cfg = d.SimConfig(
            gamma=gamma, target=c.target, amplitude=c.amplitude, K=c.K,
            lam=c.lam, mixture_mu=c.mixture_mu,
        )
        model = d.build_model(
            x, gamma, alpha_reference * c.privacy_split,
            c.target, c.amplitude, c.K, c.lam, c.mixture_mu,
            basis_cfg.bandwidth_factor, basis_cfg.R_factor,
            basis_cfg.coef_bound_factor,
        )
        t = time.perf_counter()
        theta, erm_info = solve_constrained_erm(model)
        erm_seconds = time.perf_counter() - t

        edges = d.make_uniform_edges(model.R, c.h_factor * model.b)
        hist = d.histogram_masses(
            x, edges, None, rng(c, n, rep, gamma, 40, 0), True,
        )
        cells = d.affine_cells_from_hist(
            hist, d.score_from_theta(model, theta, hist.centers),
        )

        t = time.perf_counter()
        density_tv = rp.density_tv(cells, c, 192)
        density_tv_coarse = rp.density_tv(cells, c, 96)
        grid = np.linspace(-c.xmax, c.xmax, c.metric_grid)
        truth = target(grid, c)
        ptau, _ = pn.ou_affine_density_score(grid, cells, 1 / n)
        ou_tv = d.grid_tv(truth, ptau, grid)
        ou_tv_coarse = d.grid_tv(truth[::2], ptau[::2], grid[::2])
        true_bins = rp.bin_masses(lambda z: target(z, c), c.tv_bins, c.xmax, 128)
        ou_bins = rp.bin_masses(
            lambda z: pn.ou_affine_density_score(z, cells, 1 / n)[0],
            c.tv_bins, c.xmax, 128,
        )
        test = draw(c.test_particles, rng(c, n, rep, gamma, 41, 0), c)
        estimated_score = d.score_from_theta(model, theta, test)
        true_score = d.true_score(
            test, gamma, c.target, c.amplitude, c.K, c.lam, c.mixture_mu,
        )
        score_risk = float(np.mean((estimated_score - true_score) ** 2))
        metric_seconds = time.perf_counter() - t

        generated, N, kappa, reverse_seconds = reverse(
            cells, n, gamma, alpha_reference,
            rng(c, n, rep, gamma, 42, 0), c,
        )
        iid_ou = d.sample_ptau(
            cells, 1 / n, c.particles, rng(c, n, rep, gamma, 43, 0),
        )
        reference = draw(
            c.reference_particles, rng(c, 0, rep, gamma, 44, 0), c,
        )

        row = {
            "panel": panel,
            "method": "nonprivate_constrained_erm",
            "n": n,
            "rep": rep,
            "gamma": gamma,
            "alpha_total_reference_for_reverse_grid": alpha_reference,
            "alpha_panel_baseline_applies_to_all_budgets": panel == "alpha",
            "train_hash": source_hash,
            "K": c.K,
            "amplitude": c.amplitude,
            "basis_dim": len(model.idx),
            "b": model.b,
            "R": model.R,
            "coefficient_radius": model.coef_bound,
            "score_risk": score_risk,
            "density_tv": density_tv,
            "density_quad_delta": abs(density_tv - density_tv_coarse),
            "ou_tv": ou_tv,
            "ou_grid_delta": abs(ou_tv - ou_tv_coarse),
            "ou_mass": float(np.trapezoid(ptau, grid)),
            "generated_binned_tv": rp.btv(generated, true_bins, c),
            "generated_wasserstein": float(wasserstein_distance(generated, reference)),
            "reverse_to_ou_binned_tv": rp.btv(generated, ou_bins, c),
            "iid_ou_binned_tv": rp.btv(iid_ou, ou_bins, c),
            "particles": c.particles,
            "reference_particles": c.reference_particles,
            "test_particles": c.test_particles,
            "N": N,
            "kappa": kappa,
            "erm_iterations": int(erm_info["erm_nit"]),
            "erm_seconds": erm_seconds,
            "metric_seconds": metric_seconds,
            "reverse_seconds": reverse_seconds,
            "total_seconds": time.perf_counter() - started,
        }
        atomic_json(dest, row)
        return row
    except BaseException:
        atomic_json(
            Path(outdir) / "failures" / f"{job_tag(panel, n, rep, gamma)}.json",
            {"panel": panel, "n": n, "rep": rep, "gamma": gamma,
             "traceback": traceback.format_exc()},
        )
        raise


def aggregate(out: Path) -> dict:
    counts = {}
    for panel in ("alpha", "gamma"):
        rows = [json.loads(p.read_text()) for p in sorted((out / panel / "rows").glob("*.json"))]
        if rows:
            pd.DataFrame(rows).sort_values(["gamma", "n", "rep"]).to_csv(
                out / panel / "raw.csv", index=False,
            )
        counts[panel] = len(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--panel", choices=("all", "alpha", "gamma"), default="all")
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-stop", type=int)
    parser.add_argument("--n-values", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs" / "erm_baselines")
    args = parser.parse_args()

    c = dict(BASE)
    c["test_particles"] = 10000
    stop = c["repetitions"] if args.rep_stop is None else args.rep_stop
    if not (0 <= args.rep_start < stop <= c["repetitions"]):
        parser.error("Require 0 <= rep-start < rep-stop <= repetitions")
    ns = list(c["n_values"] if args.n_values is None else args.n_values)
    if not set(ns).issubset(c["n_values"]):
        parser.error("n-values must be drawn from the formal grid")

    out = args.output_dir.resolve()
    for panel in ("alpha", "gamma"):
        (out / panel / "rows").mkdir(parents=True, exist_ok=True)
    (out / "failures").mkdir(parents=True, exist_ok=True)
    source_dirs = {
        "alpha": ROOT / "outputs" / "alpha",
        "gamma": ROOT / "outputs" / "gamma",
    }
    sources = {}
    for panel, folder in source_dirs.items():
        lock = json.loads((folder / "settings.lock.json").read_text())
        if lock["config"] != BASE or lock.get("panel") != panel:
            raise RuntimeError(f"{panel} source settings do not match formal configuration")
        sources[panel] = pd.read_csv(folder / "raw.csv")

    jobs = []
    if args.panel in ("all", "alpha"):
        source = sources["alpha"]
        # Geometry and the ordinary histogram are identical across alpha. Use
        # total alpha=4 as the reference row and draw one ERM curve.
        ref = source[source.alpha_total == c["gamma_panel_alpha_total"]]
        lookup = {(int(r.n), int(r.rep)): str(r.train_hash)
                  for r in ref.itertuples(index=False)}
        for n in ns:
            for rep in range(args.rep_start, stop):
                jobs.append(("alpha", n, rep, c["alpha_panel_gamma"],
                             lookup[(n, rep)], c, str(out)))
    if args.panel in ("all", "gamma"):
        source = sources["gamma"]
        lookup = {(int(r.n), int(r.rep), float(r.gamma)): str(r.train_hash)
                  for r in source.itertuples(index=False)}
        for gamma in c["gamma_values"]:
            for n in ns:
                for rep in range(args.rep_start, stop):
                    jobs.append(("gamma", n, rep, gamma,
                                 lookup[(n, rep, gamma)], c, str(out)))

    lock_path = out / "settings.lock.json"
    lock = {
        "experiment": "nonprivate_constrained_erm_baselines_1d",
        "config": BASE,
        "alpha_jobs_expected": 500,
        "gamma_jobs_expected": 2500,
        "note": "Alpha ERM is shared across budgets because model geometry and the ordinary histogram do not depend on alpha.",
    }
    if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
        raise RuntimeError("Output settings lock mismatch")
    atomic_json(lock_path, lock)

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        future_map = {pool.submit(one, job): job for job in jobs}
        for future in as_completed(future_map):
            future.result()
            completed += 1
            if completed == 1 or completed % 25 == 0 or completed == len(jobs):
                counts = aggregate(out)
                print(f"progress invocation={completed}/{len(jobs)} saved={counts}", flush=True)
    counts = aggregate(out)
    print(f"complete saved={counts}", flush=True)


if __name__ == "__main__":
    main()
