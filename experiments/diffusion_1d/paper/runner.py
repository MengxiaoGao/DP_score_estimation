"""Fresh 1D diffusion fits aligned with the score-estimation paper settings."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(PARENT / 'numerics'))

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import diffusion_extension_lmc as d
import paper_numerics as pn
import run_paper as rp
from common import reverse, target, draw

pn.install()
BASE = json.loads((ROOT / 'config.json').read_text())


def theoretical_ridge(n, alpha_score, gamma, model, beta):
    """Elly README ridge, with alpha equal to the score-stage budget."""
    m = len(model.h)
    radius = float(model.coef_bound)
    r_n = max(n ** (-2 * gamma / (3 + 2 * gamma)),
              (n * alpha_score) ** (-2 * gamma / (2 + gamma)))
    upsilon = beta / (1 + math.exp(alpha_score))
    tv_cap = 2 * math.log1p(upsilon / 2) / radius**2
    ridge = min(r_n, tv_cap) / m
    if not (np.isfinite(ridge) and ridge > 0):
        raise ValueError('Nonpositive theoretical ridge')
    return dict(ridge=ridge, ridge_r_n=r_n, ridge_upsilon=upsilon,
                ridge_tv_cap=tv_cap, coefficient_radius=radius,
                coefficient_count=m, privacy_beta=beta)


def rng(c, n, rep, gamma, stage, alpha_code=0):
    return np.random.default_rng(np.random.SeedSequence(
        [c.seed, n, rep, round(gamma * 1000), stage, alpha_code]))


def nested_training(c, n, rep, gamma):
    # Generate one maximum-size dataset per repetition and target, then take
    # prefixes. This matches Elly's across-n pairing without changing any
    # marginal training-sample distribution.
    maximum = max(c.n_values)
    return draw(maximum, rng(c, 0, rep, gamma, 1), c)[:n].copy()


def tag(n, rep, gamma, alpha_total):
    return f'n{n}_r{rep:03d}_g{gamma:g}_a{alpha_total:g}'.replace('.', 'p')


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def one(job):
    panel, n, rep, gamma, alpha_total, cdict, outdir = job
    c = SimpleNamespace(**cdict)
    out = Path(outdir)
    name = tag(n, rep, gamma, alpha_total)
    dest = out / 'rows' / f'{name}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    t0 = time.perf_counter()
    alpha_score = alpha_total * c.privacy_split
    alpha_hist = alpha_total * (1 - c.privacy_split)
    alpha_code = round(alpha_total * 1000)
    c.gamma = gamma
    c.alpha = alpha_total
    x = nested_training(c, n, rep, gamma)
    train_hash = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    t = time.perf_counter()
    basis_cfg = d.SimConfig(gamma=gamma, target=c.target, amplitude=c.amplitude,
                            K=c.K, lam=c.lam, mixture_mu=c.mixture_mu)
    model = d.build_model(x, gamma, alpha_score, c.target, c.amplitude,
                          c.K, c.lam, c.mixture_mu, basis_cfg.bandwidth_factor,
                          basis_cfg.R_factor, basis_cfg.coef_bound_factor)
    model_seconds = time.perf_counter() - t
    cal = theoretical_ridge(n, alpha_score, gamma, model, c.privacy_beta)
    t = time.perf_counter()
    theta, info = d.projected_lmc_draw(
        model, rng(c, n, rep, gamma, 2, alpha_code), c.lmc_steps,
        cal['ridge'], c.lmc_step_factor, None, None, 0,
        c.projection_cycles, c.projection_tol)
    lmc_seconds = time.perf_counter() - t
    if info.max_projection_violation > 1e-8:
        raise RuntimeError(f'Projection violation: {info.max_projection_violation}')
    hist = d.histogram_masses(x, d.make_uniform_edges(model.R, c.h_factor * model.b),
                              alpha_hist, rng(c, n, rep, gamma, 3, alpha_code), True)
    slopes = d.score_from_theta(model, theta, hist.centers)
    cells = d.affine_cells_from_hist(hist, slopes)
    t = time.perf_counter()
    density_tv = rp.density_tv(cells, c, 192)
    density_tv_coarse = rp.density_tv(cells, c, 96)
    grid = np.linspace(-c.xmax, c.xmax, c.metric_grid)
    truth = target(grid, c)
    ptau, _ = pn.ou_affine_density_score(grid, cells, 1 / n)
    ou_tv = d.grid_tv(truth, ptau, grid)
    ou_tv_coarse = d.grid_tv(truth[::2], ptau[::2], grid[::2])
    true_bins = rp.bin_masses(lambda z: target(z, c), c.tv_bins, c.xmax, 128)
    ou_bins = rp.bin_masses(lambda z: pn.ou_affine_density_score(z, cells, 1/n)[0],
                            c.tv_bins, c.xmax, 128)
    density_seconds = time.perf_counter() - t
    y, N, kappa, reverse_seconds = reverse(
        cells, n, gamma, alpha_total, rng(c, n, rep, gamma, 4, alpha_code), c)
    iid_ou = d.sample_ptau(cells, 1/n, c.particles,
                           rng(c, n, rep, gamma, 5, alpha_code))
    true_ref = draw(c.reference_particles, rng(c, 0, rep, gamma, 6), c)
    row = dict(panel=panel, n=n, rep=rep, gamma=gamma, alpha_total=alpha_total,
               alpha_score=alpha_score, alpha_hist=alpha_hist,
               K=c.K, amplitude=c.amplitude, train_hash=train_hash,
               data_nested_across_n=True,
               ridge_mode='theoretical', **cal,
               lmc_start='zero', lmc_steps=c.lmc_steps,
               lmc_step_factor=c.lmc_step_factor, lmc_step_size=info.step_size,
               lmc_lipschitz=info.lipschitz,
               lmc_mean_projection_cycles=info.mean_projection_cycles,
               lmc_projection_violation=info.max_projection_violation,
               projection_cycle_cap=c.projection_cycles,
               basis_dim=len(model.idx), b=model.b, b_raw=model.b_raw,
               b_capped=bool(model.b_capped), R=model.R, holder_L=model.holder_L,
               Q=model.Q, density_tv=density_tv,
               density_quad_delta=abs(density_tv-density_tv_coarse),
               ou_tv=ou_tv, ou_grid_delta=abs(ou_tv-ou_tv_coarse),
               ou_mass=float(np.trapezoid(ptau, grid)),
               generated_binned_tv=rp.btv(y, true_bins, c),
               generated_wasserstein=float(wasserstein_distance(y, true_ref)),
               reverse_to_ou_binned_tv=rp.btv(y, ou_bins, c),
               iid_ou_binned_tv=rp.btv(iid_ou, ou_bins, c),
               particles=c.particles, delta=d.delta_theory(n, alpha_total, gamma),
               kappa=kappa, N=N, model_seconds=model_seconds,
               lmc_seconds=lmc_seconds, density_seconds=density_seconds,
               reverse_seconds=reverse_seconds,
               total_seconds=time.perf_counter()-t0)
    np.savez_compressed(out / 'replicates' / f'{name}.npz', theta=theta,
                        generated=y, **{'cells_'+k:v for k,v in asdict(cells).items()})
    atomic_json(dest, row)
    return row


def aggregate(out):
    rows = [json.loads(f.read_text()) for f in (out / 'rows').glob('*.json')]
    if rows:
        pd.DataFrame(rows).sort_values(['n', 'gamma', 'alpha_total', 'rep']).to_csv(
            out / 'raw.csv', index=False)
    return len(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', choices=['alpha', 'gamma'], required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--steps', type=int, help='Only allowed with --smoke')
    p.add_argument('--particles', type=int, help='Only allowed with --smoke')
    p.add_argument('--output-dir', type=Path, help='New output directory for an independent run')
    args = p.parse_args()
    if (args.steps or args.particles) and not args.smoke:
        p.error('Step and particle overrides require --smoke')
    c = dict(BASE)
    if args.smoke:
        c['lmc_steps'] = args.steps or 1000
        c['particles'] = args.particles or 1000
        c['reference_particles'] = max(2000, c['particles'])
        c['score_grid'] = 801
        c['metric_grid'] = 4001
    out = args.output_dir or ROOT / 'outputs' / (args.panel + ('_smoke' if args.smoke else ''))
    for folder in ['rows', 'replicates', 'logs', 'tables']:
        (out / folder).mkdir(parents=True, exist_ok=True)
    sources = [ROOT/'runner.py', ROOT/'config.json', PARENT/'common.py',
               PARENT/'numerics'/'diffusion_extension_lmc.py',
               PARENT/'numerics'/'paper_numerics.py',
               PARENT/'numerics'/'suite'/'dp_score_sim'/'core.py',
               PARENT/'numerics'/'suite'/'dp_score_sim'/'density.py']
    source_hashes = {str(f.relative_to(PARENT)):hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in sources}
    lock = dict(panel=args.panel, smoke=args.smoke, config=c,
                source_hashes=source_hashes)
    lock_path = out / 'settings.lock.json'
    if lock_path.exists():
        if json.loads(lock_path.read_text()) != lock:
            raise RuntimeError(f'Settings differ from existing outputs: {lock_path}')
    else:
        atomic_json(lock_path, lock)
    if args.panel == 'alpha':
        jobs = [(args.panel,n,r,c['alpha_panel_gamma'],a,c,str(out))
                for r in range(c['repetitions']) for n in c['n_values']
                for a in c['alpha_total_values']]
        if args.smoke:
            jobs = [jobs[0], jobs[-1]]
    else:
        jobs = [(args.panel,n,r,g,c['gamma_panel_alpha_total'],c,str(out))
                for r in range(c['repetitions']) for n in c['n_values']
                for g in c['gamma_values']]
        if args.smoke:
            jobs = [jobs[0], jobs[-1]]
    planned = len(jobs)
    pending = [j for j in jobs if not (out/'rows'/f'{tag(j[1],j[2],j[3],j[4])}.json').exists()]
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, j): j for j in pending}
        for done, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except BaseException:
                print('FAILED', futures[fut], traceback.format_exc(), flush=True)
                raise
            if done % 25 == 0 or done == len(pending):
                saved = aggregate(out)
                atomic_json(out/'logs'/'status.json', dict(saved=saved, planned=planned,
                            complete=saved==planned, elapsed_seconds=time.time()-start))
                print(f'{args.panel}: {saved}/{planned}, completed this run {done}/{len(pending)}', flush=True)
    saved = aggregate(out)
    atomic_json(out/'logs'/'status.json', dict(saved=saved, planned=planned,
                complete=saved==planned, elapsed_seconds=time.time()-start))


if __name__ == '__main__':
    main()
