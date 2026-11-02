#!/usr/bin/env python3
"""One-record perturbation experiment for the paper-aligned 1D diffusion model.

The original private fits are reused from the completed alpha panel.  For the
neighbouring dataset D'=(3,X_2,...,X_n), this script replays the original LMC,
histogram, and reverse-diffusion random streams.  It also fits constrained ERM
on D and D' and reconstructs ordinary (non-private) histograms.

The primary diffusion metric is the integrated TV distance between the two
early-stopped laws P_{1/n} rho_D and P_{1/n} rho_D'.  Reverse-sample binned TV
is saved as a numerical check, rather than treated as exact distributional TV.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(PARENT / "numerics"))

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import diffusion_extension_lmc as d
import paper_numerics as pn
import run_paper as rp
from common import draw, reverse, theory_grid
from dp_score_sim.core import solve_constrained_erm
from runner import BASE, atomic_json, nested_training, rng, tag, theoretical_ridge

pn.install()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def cells_from_npz(z, prefix: str = "cells_") -> d.AffineCells:
    return d.AffineCells(**{
        field: np.asarray(z[prefix + field])
        for field in ("left", "right", "centers", "widths", "masses",
                      "slopes", "intercept", "slopecoef")
    })


def same_geometry(a, b) -> bool:
    scalar = (a.b == b.b and a.R == b.R and a.coef_bound == b.coef_bound
              and a.diff_bound == b.diff_bound)
    return bool(scalar and np.array_equal(a.idx, b.idx))


def exact_affine_tv(a: d.AffineCells, b: d.AffineCells) -> float:
    """Exact TV for two cellwise-affine densities on a shared partition."""
    if not (np.array_equal(a.left, b.left) and np.array_equal(a.right, b.right)):
        raise RuntimeError("Density TV requires a shared histogram partition")
    dl = (a.intercept + a.slopecoef * a.left) - (b.intercept + b.slopecoef * b.left)
    dr = (a.intercept + a.slopecoef * a.right) - (b.intercept + b.slopecoef * b.right)
    width = a.widths
    same = dl * dr >= 0
    integral = np.empty_like(dl)
    integral[same] = width[same] * (np.abs(dl[same]) + np.abs(dr[same])) / 2
    cross = ~same
    denom = np.abs(dl[cross]) + np.abs(dr[cross])
    integral[cross] = width[cross] * (dl[cross] ** 2 + dr[cross] ** 2) / (2 * denom)
    return float(0.5 * integral.sum())


def empirical_binned_tv(a: np.ndarray, b: np.ndarray, c) -> float:
    edges = np.r_[-np.inf, np.linspace(-c.xmax, c.xmax, c.tv_bins + 1), np.inf]
    pa = np.histogram(a, bins=edges)[0] / len(a)
    pb = np.histogram(b, bins=edges)[0] / len(b)
    return float(0.5 * np.abs(pa - pb).sum())


def squared_displacements(model, theta_a, theta_b, test) -> dict:
    delta = np.asarray(theta_a) - np.asarray(theta_b)
    score_delta = d.score_from_theta(model, theta_a, test) - d.score_from_theta(model, theta_b, test)
    return {
        "coefficient_displacement": float(np.mean(delta**2)),
        "coefficient_max_change": float(np.max(np.abs(delta))),
        "score_displacement": float(np.mean(score_delta**2)),
        "score_max_change": float(np.max(np.abs(score_delta))),
    }


def output_metrics(cells_a, cells_b, generated_a, generated_b, n, c, seed_parts) -> dict:
    density_tv = exact_affine_tv(cells_a, cells_b)
    grid = np.linspace(-c.xmax, c.xmax, c.metric_grid)
    pa = pn.ou_affine_density_score(grid, cells_a, 1 / n)[0]
    pb = pn.ou_affine_density_score(grid, cells_b, 1 / n)[0]
    ou_tv = d.grid_tv(pa, pb, grid)
    ou_tv_coarse = d.grid_tv(pa[::2], pb[::2], grid[::2])
    bins_a = rp.bin_masses(lambda z: pn.ou_affine_density_score(z, cells_a, 1/n)[0],
                           c.tv_bins, c.xmax, 128)
    bins_b = rp.bin_masses(lambda z: pn.ou_affine_density_score(z, cells_b, 1/n)[0],
                           c.tv_bins, c.xmax, 128)
    exact_binned = float(0.5 * np.abs(bins_a - bins_b).sum())
    iid_a1 = d.sample_ptau(cells_a, 1/n, c.particles,
                           np.random.default_rng(np.random.SeedSequence([*seed_parts, 1])))
    iid_a2 = d.sample_ptau(cells_a, 1/n, c.particles,
                           np.random.default_rng(np.random.SeedSequence([*seed_parts, 2])))
    iid_b = d.sample_ptau(cells_b, 1/n, c.particles,
                          np.random.default_rng(np.random.SeedSequence([*seed_parts, 3])))
    if generated_a is None or generated_b is None:
        generated_btv = None
        generated_w1 = None
    else:
        generated_btv = empirical_binned_tv(generated_a, generated_b, c)
        generated_w1 = float(wasserstein_distance(generated_a, generated_b))
    return {
        "density_tv_change": density_tv,
        "ou_tv_change": ou_tv,
        "ou_tv_grid_delta": abs(ou_tv - ou_tv_coarse),
        "ou_mass_original": float(np.trapezoid(pa, grid)),
        "ou_mass_neighbor": float(np.trapezoid(pb, grid)),
        "ou_exact_binned_tv_change": exact_binned,
        "generated_binned_tv_change": generated_btv,
        "generated_wasserstein_change": generated_w1,
        "iid_same_ou_binned_tv_floor": empirical_binned_tv(iid_a1, iid_a2, c),
        "iid_neighbor_ou_binned_tv": empirical_binned_tv(iid_a1, iid_b, c),
    }


def make_private_neighbor(xp, model_neighbor, n, rep, gamma, alpha_total, c):
    alpha_score = alpha_total * c.privacy_split
    alpha_hist = alpha_total * (1 - c.privacy_split)
    alpha_code = round(alpha_total * 1000)
    cal = theoretical_ridge(n, alpha_score, gamma, model_neighbor, c.privacy_beta)
    started = time.perf_counter()
    theta, info = d.projected_lmc_draw(
        model_neighbor, rng(c, n, rep, gamma, 2, alpha_code), c.lmc_steps,
        cal["ridge"], c.lmc_step_factor, None, None, 0,
        c.projection_cycles, c.projection_tol)
    lmc_seconds = time.perf_counter() - started
    if info.max_projection_violation > 1e-8:
        raise RuntimeError(f"Neighbor projection violation {info.max_projection_violation}")
    edges = d.make_uniform_edges(model_neighbor.R, c.h_factor * model_neighbor.b)
    hist = d.histogram_masses(xp, edges, alpha_hist,
                              rng(c, n, rep, gamma, 3, alpha_code), True)
    cells = d.affine_cells_from_hist(hist, d.score_from_theta(model_neighbor, theta, hist.centers))
    kappa = c.kappa_multiplier * d.delta_theory(n, alpha_total, gamma) ** 2
    N = len(theory_grid(n, kappa)) - 1
    if c.run_reverse_validation:
        generated, N_check, kappa_check, reverse_seconds = reverse(
            cells, n, gamma, alpha_total, rng(c, n, rep, gamma, 4, alpha_code), c)
        if N_check != N or kappa_check != kappa:
            raise RuntimeError("Reverse-grid metadata mismatch")
    else:
        generated, reverse_seconds = None, 0.0
    return theta, cells, generated, {
        **cal,
        "neighbor_lmc_step_size": info.step_size,
        "neighbor_lmc_lipschitz": info.lipschitz,
        "neighbor_lmc_mean_projection_cycles": info.mean_projection_cycles,
        "neighbor_lmc_projection_violation": info.max_projection_violation,
        "neighbor_lmc_seconds": lmc_seconds,
        "neighbor_reverse_seconds": reverse_seconds,
        "N": N,
        "kappa": kappa,
    }


def make_nonprivate_pair(x, xp, model, model_neighbor, n, rep, gamma, alpha_total, c):
    theta, info = solve_constrained_erm(model)
    theta_p, info_p = solve_constrained_erm(model_neighbor)
    edges = d.make_uniform_edges(model.R, c.h_factor * model.b)
    h = d.histogram_masses(x, edges, None, rng(c, n, rep, gamma, 20), True)
    hp = d.histogram_masses(xp, edges, None, rng(c, n, rep, gamma, 20), True)
    cells = d.affine_cells_from_hist(h, d.score_from_theta(model, theta, h.centers))
    cells_p = d.affine_cells_from_hist(hp, d.score_from_theta(model_neighbor, theta_p, hp.centers))
    alpha_code = round(alpha_total * 1000)
    if c.run_reverse_validation:
        ya, _, _, _ = reverse(cells, n, gamma, alpha_total,
                              rng(c, n, rep, gamma, 21, alpha_code), c)
        yb, _, _, _ = reverse(cells_p, n, gamma, alpha_total,
                              rng(c, n, rep, gamma, 21, alpha_code), c)
    else:
        ya, yb = None, None
    return theta, theta_p, cells, cells_p, ya, yb, {
        "nonprivate_erm_iterations": info["erm_nit"],
        "nonprivate_neighbor_erm_iterations": info_p["erm_nit"],
    }


def one(job):
    n, rep, alpha_total, cdict, source_artifact, source_row, outdir = job
    c = SimpleNamespace(**cdict)
    c.gamma = float(c.alpha_panel_gamma)
    c.alpha = float(alpha_total)
    gamma = c.gamma
    out = Path(outdir)
    name = tag(n, rep, gamma, alpha_total)
    dest = out / "rows" / f"{name}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    started = time.perf_counter()
    x = nested_training(c, n, rep, gamma)
    if array_hash(x) != source_row["train_hash"]:
        raise RuntimeError("Regenerated training data do not match the source alpha panel")
    if not (0 <= c.replacement_index < n):
        raise ValueError("replacement index is outside the dataset")
    xp = x.copy()
    old_value = float(xp[c.replacement_index])
    xp[c.replacement_index] = c.replacement_value

    alpha_score = alpha_total * c.privacy_split
    basis_cfg = d.SimConfig(gamma=gamma, target=c.target, amplitude=c.amplitude,
                            K=c.K, lam=c.lam, mixture_mu=c.mixture_mu)
    model = d.build_model(x, gamma, alpha_score, c.target, c.amplitude,
                          c.K, c.lam, c.mixture_mu, basis_cfg.bandwidth_factor,
                          basis_cfg.R_factor, basis_cfg.coef_bound_factor)
    model_p = d.build_model(xp, gamma, alpha_score, c.target, c.amplitude,
                            c.K, c.lam, c.mixture_mu, basis_cfg.bandwidth_factor,
                            basis_cfg.R_factor, basis_cfg.coef_bound_factor)
    if not same_geometry(model, model_p):
        raise RuntimeError("Spline geometry changed across neighbouring datasets")
    if abs(c.replacement_value) > model.R:
        raise RuntimeError("Replacement lies outside the model localisation radius")
    test = draw(c.test_particles, rng(c, 0, rep, gamma, 7), c)

    with np.load(source_artifact) as z:
        theta = np.asarray(z["theta"])
        generated = np.asarray(z["generated"])
        cells = cells_from_npz(z)
    expected_edges = d.make_uniform_edges(model.R, c.h_factor * model.b)
    if not (np.allclose(cells.left, expected_edges[:-1]) and
            np.allclose(cells.right, expected_edges[1:])):
        raise RuntimeError("Source density partition does not match the rebuilt model")

    theta_p, cells_p, generated_p, private_info = make_private_neighbor(
        xp, model_p, n, rep, gamma, alpha_total, c)
    private_disp = squared_displacements(model, theta, theta_p, test)
    private_out = output_metrics(
        cells, cells_p, generated, generated_p, n, c,
        [c.seed, n, rep, round(alpha_total * 1000), 300])

    nt, ntp, nc, ncp, ny, nyp, np_info = make_nonprivate_pair(
        x, xp, model, model_p, n, rep, gamma, alpha_total, c)
    nonprivate_disp = squared_displacements(model, nt, ntp, test)
    nonprivate_out = output_metrics(
        nc, ncp, ny, nyp, n, c,
        [c.seed, n, rep, round(alpha_total * 1000), 400])

    row = {
        "n": n,
        "rep": rep,
        "gamma": gamma,
        "alpha_total": alpha_total,
        "alpha_score": alpha_score,
        "alpha_hist": alpha_total * (1 - c.privacy_split),
        "K": c.K,
        "amplitude": c.amplitude,
        "replacement_index": c.replacement_index,
        "replacement_value": c.replacement_value,
        "old_value": old_value,
        "train_hash": source_row["train_hash"],
        "neighbor_hash": array_hash(xp),
        "coupling": "shared_lmc_histogram_reverse_randomness",
        "lmc_start": "zero",
        "lmc_steps": c.lmc_steps,
        "lmc_step_factor": c.lmc_step_factor,
        "projection_cycle_cap": c.projection_cycles,
        "projection_tol": c.projection_tol,
        "basis_dim": len(model.idx),
        "b": model.b,
        "R": model.R,
        "Q": model.Q,
        **{f"private_{k}": v for k, v in private_disp.items()},
        **{f"private_{k}": v for k, v in private_out.items()},
        **{f"nonprivate_{k}": v for k, v in nonprivate_disp.items()},
        **{f"nonprivate_{k}": v for k, v in nonprivate_out.items()},
        **private_info,
        **np_info,
        "source_lmc_projection_violation": source_row["lmc_projection_violation"],
        "source_lmc_mean_projection_cycles": source_row["lmc_mean_projection_cycles"],
        "total_seconds": time.perf_counter() - started,
    }
    arrays = {
        "data": x,
        "neighbor_data": xp,
        "test": test,
        "private_theta_original": theta,
        "private_theta_neighbor": theta_p,
        "nonprivate_theta_original": nt,
        "nonprivate_theta_neighbor": ntp,
    }
    if generated_p is not None:
        arrays["private_generated_original"] = generated
        arrays["private_generated_neighbor"] = generated_p
    if ny is not None:
        arrays["nonprivate_generated_original"] = ny
        arrays["nonprivate_generated_neighbor"] = nyp
    for label, obj in (("private_original", cells), ("private_neighbor", cells_p),
                       ("nonprivate_original", nc), ("nonprivate_neighbor", ncp)):
        for key, value in asdict(obj).items():
            arrays[f"{label}_{key}"] = value
    np.savez_compressed(out / "replicates" / f"{name}.npz", **arrays)
    atomic_json(dest, row)
    return row


def aggregate(out: Path) -> int:
    rows = [json.loads(path.read_text()) for path in (out / "rows").glob("*.json")]
    if rows:
        pd.DataFrame(rows).sort_values(["n", "alpha_total", "rep"]).to_csv(
            out / "raw.csv", index=False)
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--rep-start", type=int, default=0)
    p.add_argument("--rep-stop", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--source-alpha-dir", type=Path,
                   default=HERE / "outputs" / "alpha")
    p.add_argument("--replacement-index", type=int, default=0)
    p.add_argument("--replacement-value", type=float, default=3.0)
    p.add_argument("--with-reverse", action="store_true",
                   help="Also rerun coupled reverse samplers; omitted in the formal run because OU TV is primary")
    args = p.parse_args()

    c = dict(BASE)
    c["test_particles"] = 10000
    c["replacement_index"] = args.replacement_index
    c["replacement_value"] = args.replacement_value
    c["run_reverse_validation"] = bool(args.with_reverse or args.smoke)
    stop = c["repetitions"] if args.rep_stop is None else args.rep_stop
    if not (0 <= args.rep_start < stop <= c["repetitions"]):
        p.error("Require 0 <= rep-start < rep-stop <= configured repetitions")
    source = args.source_alpha_dir.resolve()
    raw_path = source / "raw.csv"
    lock_path = source / "settings.lock.json"
    if not (raw_path.exists() and lock_path.exists()):
        p.error("Completed paper-aligned alpha outputs are required")
    source_lock = json.loads(lock_path.read_text())
    if source_lock["config"] != BASE or source_lock.get("panel") != "alpha":
        raise RuntimeError("Source alpha settings do not match the paper-aligned configuration")
    source_raw = pd.read_csv(raw_path)
    lookup = {
        (int(r.n), int(r.rep), float(r.alpha_total)): r._asdict()
        for r in source_raw.itertuples(index=False)
    }

    out = (args.output_dir or HERE / "outputs" / "perturbation").resolve()
    for folder in ("rows", "replicates", "logs", "tables"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), HERE / "runner.py", HERE / "config.json",
               PARENT / "common.py", PARENT / "numerics" / "diffusion_extension_lmc.py",
               PARENT / "numerics" / "paper_numerics.py"]
    lock = {
        "experiment": "one_record_replacement_diffusion_tv",
        "config": c,
        "rep_start": args.rep_start,
        "rep_stop": stop,
        "source_alpha_dir": str(source),
        "source_alpha_settings_sha256": file_hash(lock_path),
        "source_hashes": {str(path): file_hash(path) for path in sources},
        "primary_metric": "ou_tv_change",
        "replacement_rule": "replace X_1 by fixed value 3",
        "notes": [
            "Original private fits are reused from the completed alpha panel.",
            "Neighbour fits replay the same LMC, histogram, and reverse random streams.",
            "Coupled differences are empirical stability summaries, not DP certificates."
        ],
    }
    out_lock = out / "settings.lock.json"
    if out_lock.exists() and json.loads(out_lock.read_text()) != lock:
        raise RuntimeError(f"Settings differ from existing output: {out_lock}")
    if not out_lock.exists():
        atomic_json(out_lock, lock)

    jobs = []
    for rep in range(args.rep_start, stop):
        for n in c["n_values"]:
            for alpha in c["alpha_total_values"]:
                key = (n, rep, float(alpha))
                if key not in lookup:
                    raise RuntimeError(f"Missing source alpha row {key}")
                artifact = source / "replicates" / f"{tag(n, rep, c['alpha_panel_gamma'], alpha)}.npz"
                if not artifact.exists():
                    raise RuntimeError(f"Missing source artifact {artifact}")
                jobs.append((n, rep, alpha, c, str(artifact), lookup[key], str(out)))
    if args.smoke:
        jobs = [jobs[0], jobs[-1]]
    planned = len(jobs)
    pending = [job for job in jobs if not (out / "rows" /
               f"{tag(job[0], job[1], c['alpha_panel_gamma'], job[2])}.json").exists()]
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, job): job for job in pending}
        for done, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except BaseException:
                print("FAILED", futures[future][:3], traceback.format_exc(), flush=True)
                raise
            if done % 10 == 0 or done == len(pending):
                saved = aggregate(out)
                atomic_json(out / "logs" / "status.json", {
                    "saved": saved,
                    "planned_in_this_invocation": planned,
                    "complete_for_requested_range": saved == planned,
                    "rep_start": args.rep_start,
                    "rep_stop": stop,
                    "elapsed_seconds": time.time() - started,
                })
                print(f"perturbation: {saved}/{planned}; this run {done}/{len(pending)}", flush=True)
    aggregate(out)


if __name__ == "__main__":
    main()
