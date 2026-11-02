"""Appendix diagnostics aligned with the 2026-10-22 diffusion panels."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import runner
from runner import d, pn, rp
from common import reverse

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "appendix_checks"
CASES = [
    (0, 16000, 0.5, 0.2),
    (3, 16000, 8.0, 0.2),
    (6, 1000, 0.5, 0.8),
    (7, 1000, 8.0, 0.8),
    (8, 8000, 2.0, 0.4),
]
FACTORS = (0.01, 0.1, 0.5)
PREFIXES = (250000, 500000, 750000)
MULTIPLIERS = (0.5, 1, 2, 8, 32, 128)
REVERSE_REPS = (0, 1, 2)
REVERSE_INNER = 10
SEED = 20261023


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def cfg():
    return SimpleNamespace(**runner.BASE)


def case_model(case):
    case_id, n, alpha_score, gamma = case
    c = cfg()
    c.gamma = gamma
    c.alpha = 2 * alpha_score
    x = runner.nested_training(c, n, 0, gamma)
    bc = d.SimConfig(gamma=gamma, target=c.target, amplitude=c.amplitude,
                     K=c.K, lam=c.lam, mixture_mu=c.mixture_mu)
    model = d.build_model(x, gamma, alpha_score, c.target, c.amplitude,
                          c.K, c.lam, c.mixture_mu, bc.bandwidth_factor,
                          bc.R_factor, bc.coef_bound_factor)
    ridge = runner.theoretical_ridge(n, alpha_score, gamma, model,
                                     c.privacy_beta)["ridge"]
    hist_rng = np.random.default_rng(np.random.SeedSequence(
        [SEED, 98, case_id]))
    hist = d.histogram_masses(
        x, d.make_uniform_edges(model.R, c.h_factor * model.b),
        alpha_score, hist_rng, True)
    return c, x, model, ridge, hist


def lmc_prefixes(model, seed, factor, ridge, prefixes):
    """Same update as projected_lmc_draw, retaining three nested prefixes."""
    L = model.c * float(np.linalg.eigvalsh(model.A).max()) + ridge
    eta = factor / max(L, 1e-14)
    rng = np.random.default_rng(seed)
    theta = np.zeros(model.h.size)
    rows = []
    sum_cycles = 0
    cap_hits = 0
    max_violation = 0.0
    t0 = time.perf_counter()
    for step in range(1, max(prefixes) + 1):
        grad = model.c * (model.A @ theta + model.h) + ridge * theta
        proposal = theta - eta * grad + math.sqrt(2 * eta) * rng.normal(size=theta.size)
        theta, cycles, violation = d.dykstra_project(
            proposal, model.coef_bound, model.diff_bound,
            max_cycles=1000, tol=1e-10)
        sum_cycles += int(cycles)
        cap_hits += int(cycles == 1000)
        max_violation = max(max_violation, float(violation))
        if violation > 1e-8:
            raise RuntimeError(
                f"projection violation {violation:.6g} at step {step}; "
                f"cycles={cycles}, factor={factor}")
        if step in prefixes:
            rows.append((step, theta.copy(), sum_cycles / step,
                         cap_hits / step, max_violation,
                         time.perf_counter() - t0, eta, L))
    return rows


def density_job(job):
    case, factor, chain, steps = job
    case_id, n, alpha_score, gamma = case
    out = OUT / "density_lmc"
    key = f"case{case_id}_f{factor:g}_chain{chain}".replace(".", "p")
    dest = out / "rows" / f"{key}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    if (out / "failures" / f"{key}.json").exists():
        return []
    c, x, model, ridge, hist = case_model(case)
    seed = np.random.SeedSequence(
        [SEED, 97, case_id, int(round(factor * 1000)), chain])
    snapshots = lmc_prefixes(model, seed, factor, ridge, steps)
    train_hash = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    rows = []
    for step, theta, mean_cycles, cap_fraction, violation, seconds, eta, L in snapshots:
        if violation > 1e-8:
            raise RuntimeError(f"Projection violation {violation}: {key}")
        slopes = d.score_from_theta(model, theta, hist.centers)
        cells = d.affine_cells_from_hist(hist, slopes)
        tv = rp.density_tv(cells, c, 192)
        tv_coarse = rp.density_tv(cells, c, 96)
        rows.append(dict(case_id=case_id, n=n, alpha_score=alpha_score,
                         alpha_total=2 * alpha_score, gamma=gamma,
                         factor=factor, chain=chain, steps=step,
                         density_tv=tv, quadrature_delta=abs(tv - tv_coarse),
                         mean_projection_cycles=mean_cycles,
                         projection_cap_fraction=cap_fraction,
                         projection_violation=violation, elapsed_seconds=seconds,
                         ridge=ridge, step_size=eta, lipschitz=L,
                         basis_dim=len(model.h), train_hash=train_hash))
    atomic_json(dest, rows)
    return rows


def read_rows(folder):
    rows = []
    for f in sorted((folder / "rows").glob("*.json")):
        value = json.loads(f.read_text())
        rows.extend(value if isinstance(value, list) else [value])
    return rows


def run_density(pilot=False, workers=6):
    out = OUT / "density_lmc"
    (out / "rows").mkdir(parents=True, exist_ok=True)
    (out / "failures").mkdir(parents=True, exist_ok=True)
    if pilot:
        jobs = [(CASES[0], 0.5, 0, (250000, 500000, 750000))]
    else:
        jobs = [(case, factor, chain, PREFIXES)
                for case in CASES for factor in FACTORS for chain in range(4)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = {pool.submit(density_job, job): job for job in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            job = futures[fut]
            try:
                fut.result()
            except RuntimeError as exc:
                if "projection violation" not in str(exc):
                    raise
                case, factor, chain, _ = job
                key = f"case{case[0]}_f{factor:g}_chain{chain}".replace(".", "p")
                atomic_json(out / "failures" / f"{key}.json",
                            dict(case_id=case[0], n=case[1],
                                 alpha_score=case[2], gamma=case[3],
                                 factor=factor, chain=chain,
                                 error=str(exc)))
            if i % 5 == 0 or i == len(jobs):
                print(f"density jobs {i}/{len(jobs)}", flush=True)
    rows = read_rows(out)
    pd.DataFrame(rows).sort_values(["case_id", "factor", "chain", "steps"]).to_csv(
        out / "raw.csv", index=False)
    summary = pd.DataFrame(rows).groupby(
        ["case_id", "n", "alpha_score", "alpha_total", "gamma", "factor", "steps"],
        as_index=False).agg(density_tv_mean=("density_tv", "mean"),
                            density_tv_sd=("density_tv", "std"),
                            chains=("density_tv", "size"),
                            max_projection_violation=("projection_violation", "max"),
                            max_projection_cap_fraction=("projection_cap_fraction", "max"),
                            max_quadrature_delta=("quadrature_delta", "max"))
    summary.to_csv(out / "summary.csv", index=False)
    print(f"density completed: {len(rows)} rows; wall {time.perf_counter()-t0:.1f}s", flush=True)


def load_cells(n, rep, gamma, alpha):
    name = runner.tag(n, rep, gamma, alpha)
    path = ROOT / "outputs" / "alpha" / "replicates" / f"{name}.npz"
    with np.load(path) as z:
        cells = d.AffineCells(**{f.name: z["cells_" + f.name].copy()
                                 for f in fields(d.AffineCells)})
    return cells


def reverse_job(job):
    rep, multiplier, inner, particles = job
    out = OUT / "reverse_grid"
    key = f"rep{rep}_m{multiplier:g}_inner{inner}".replace(".", "p")
    dest = out / "rows" / f"{key}.json"
    if dest.exists():
        return json.loads(dest.read_text())
    n, gamma, alpha = 4000, 0.5, 4.0
    c = cfg()
    c.gamma = gamma
    c.alpha = alpha
    c.particles = particles
    c.kappa_multiplier = multiplier
    cells = load_cells(n, rep, gamma, alpha)
    ou_bins = rp.bin_masses(
        lambda z: pn.ou_affine_density_score(z, cells, 1 / n)[0],
        c.tv_bins, c.xmax, 128)
    y, N, kappa, seconds = reverse(
        cells, n, gamma, alpha,
        np.random.default_rng(np.random.SeedSequence(
            [SEED, 96, rep, int(round(multiplier * 1000)), inner])), c)
    iid = d.sample_ptau(
        cells, 1 / n, particles,
        np.random.default_rng(np.random.SeedSequence(
            [SEED, 95, rep, inner])))
    row = dict(rep=rep, multiplier=multiplier, inner=inner, n=n,
               gamma=gamma, alpha_total=alpha, particles=particles,
               N=N, kappa=kappa, reverse_seconds=seconds,
               reverse_to_ou_binned_tv=rp.btv(y, ou_bins, c),
               iid_ou_binned_tv=rp.btv(iid, ou_bins, c))
    atomic_json(dest, row)
    return row


def run_reverse(pilot=False, workers=6):
    out = OUT / "reverse_grid"
    (out / "rows").mkdir(parents=True, exist_ok=True)
    multipliers = (1, 32, 128) if pilot else MULTIPLIERS
    reps = (0,) if pilot else REVERSE_REPS
    inner_n = 1 if pilot else REVERSE_INNER
    jobs = [(rep, mult, inner, 20000)
            for rep in reps for mult in multipliers for inner in range(inner_n)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        futures = [pool.submit(reverse_job, job) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            if i % 10 == 0 or i == len(jobs):
                print(f"reverse jobs {i}/{len(jobs)}", flush=True)
    rows = read_rows(out)
    raw = pd.DataFrame(rows).sort_values(["rep", "multiplier", "inner"])
    raw.to_csv(out / "raw.csv", index=False)
    summary = raw.groupby("multiplier", as_index=False).agg(
        N=("N", "mean"), R=("N", "size"),
        reverse_tv_mean=("reverse_to_ou_binned_tv", "mean"),
        reverse_tv_sd=("reverse_to_ou_binned_tv", "std"),
        iid_tv_mean=("iid_ou_binned_tv", "mean"),
        iid_tv_sd=("iid_ou_binned_tv", "std"))
    summary.to_csv(out / "summary.csv", index=False)
    print(f"reverse completed: {len(rows)} rows; wall {time.perf_counter()-t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study", choices=["density", "reverse"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.study == "density":
        run_density(args.pilot, args.workers)
    else:
        run_reverse(args.pilot, args.workers)


if __name__ == "__main__":
    main()
