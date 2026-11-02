"""Coefficient and score stability across privacy budgets after replacing one point.

Uses production tuning: theoretical ridge beta=.01, factor=.1, 500000 steps,
zero initialisation, 1000 projection cycles, and 100 source repetitions.
The default perturbation replaces the first observation with the value 3.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import experiment_common as c


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_geometry(a, b):
    return (np.array_equal(a.idx, b.idx)
            and all(getattr(a, k) == getattr(b, k)
                    for k in ('b', 'R', 'coef_bound', 'diff_bound')))


def stability_case(job):
    module, source, i, j, rep, index, value, theta, erm_theta, cached = job
    p = c.load_module(module)
    n, alpha, gamma = c.NS[j], c.ALPHAS[i], c.PRIVACY_GAMMA
    x, test = c.data(source['seed'], rep, n, gamma, source['n_test'])
    model = c.build_model(p, x, gamma, alpha)
    calibration = c.resolved_ridge(p, model, source['lmc'])
    changed = x.copy()
    changed[index] = value
    other = c.build_model(p, changed, gamma, alpha)
    if c.resolved_ridge(p, other, source['lmc']) != calibration:
        raise ValueError('Neighbouring fits must use the same ridge calibration')
    if not same_geometry(model, other):
        raise ValueError('Neighbouring fits must use identical coefficient geometry')
    if abs(value) > model.R:
        raise ValueError(f'Replacement {value} lies outside truncation radius R={model.R}')
    if theta.shape != model.h.shape or not c.feasible(theta, model):
        raise ValueError('Invalid original coefficient vector')

    key = c.coefficient_key(i, j, rep)
    theta_changed = cached.get(key)
    if theta_changed is None:
        # Replaying the original seed gives the same standard Gaussian noise
        # sequence. Each dataset still uses its own prescribed eta=step_factor/L.
        random = c.release_rng(source['seed'], rep, n, gamma, alpha)
        theta_changed = c.lmc(p, other, random, source['steps'],
            label=f'stability: n={n}, alpha={alpha}, rep={rep + 1}', **source['lmc'])

    original_key = f'erm_original_{i}_{j}_{rep}'
    changed_key = f'erm_changed_{i}_{j}_{rep}'
    erm_original = cached.get(original_key)
    reused_erm = same_geometry(model, c.build_model(p, x, gamma, None))
    if erm_original is None:
        # Main-study ERM has its own bandwidth. Reuse only if geometry matches;
        # otherwise fit the cheap deterministic reference in the private basis.
        erm_original = erm_theta if reused_erm else c.erm(p, model)
    erm_changed = cached.get(changed_key)
    if erm_changed is None:
        erm_changed = c.erm(p, other)
    for vector in (theta_changed, erm_original, erm_changed):
        if vector.shape != model.h.shape or not c.feasible(vector, model):
            raise ValueError('Invalid cached or newly fitted coefficients')

    G, cross, constant = c.risk_quadratic(p, model, test, gamma)
    def risk(vector):
        result = float(vector @ G @ vector - 2 * vector @ cross + constant)
        if not np.isfinite(result) or result < -1e-8:
            raise ArithmeticError('Invalid quadratic risk')
        return max(0., result)
    def displacement(a, b):
        difference = a - b
        return float(np.mean(difference**2)), max(0., float(difference @ G @ difference))

    private_coef, private_score = displacement(theta_changed, theta)
    erm_coef, erm_score = displacement(erm_changed, erm_original)
    before, after = risk(theta), risk(theta_changed)
    before_erm, after_erm = risk(erm_original), risk(erm_changed)
    row = dict(i=i, j=j, rep=rep, n=n, alpha=alpha, gamma=gamma,
        coefficients=len(theta), replacement_index=index,
        original_value=float(x[index]), replacement_value=value,
        private_coefficient_mse=private_coef, erm_coefficient_mse=erm_coef,
        private_score_squared=private_score, erm_score_squared=erm_score,
        private_risk_before=before, private_risk_after=after,
        private_risk_change=after-before, erm_risk_before=before_erm,
        erm_risk_after=after_erm, erm_risk_change=after_erm-before_erm,
        original_chain_reused=True, original_erm_reused=reused_erm,
        holder_L=model.holder_L, b=model.b, R=model.R,
        steps=source['steps'], ridge=calibration['ridge'],
        ridge_mode=calibration['mode'], ridge_r_n=calibration['r_n'],
        ridge_upsilon=calibration['upsilon'], ridge_tv_cap=calibration['tv_cap'],
        coefficient_radius=calibration['coefficient_radius'],
        privacy_beta=calibration['privacy_beta'],
        step_factor=source['lmc']['step_factor'],
        projection_cycles=source['lmc']['projection_cycles'],
        initial_state=source['initial_state'],seed=source['seed'],n_test=source['n_test'])
    return row, {key:theta_changed, original_key:erm_original, changed_key:erm_changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', default=str(Path(__file__).resolve().parent/'results/privacy'))
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent/'results/stability'))
    parser.add_argument('--module', default=c.MODULE)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--ns', nargs='+', type=int, default=c.NS)
    parser.add_argument('--alphas', nargs='+', type=float, default=c.ALPHAS)
    parser.add_argument('--reps', type=int, help='Default: all saved repetitions')
    parser.add_argument('--replacement-value', type=float, default=3.)
    parser.add_argument('--replacement-index', type=int, default=0,
                        help='Zero-based replacement index; default 0 (first observation)')
    parser.add_argument('--export-only',dest='plot_only',action='store_true',help='Validate saved CSVs without new fits')
    args = parser.parse_args()
    source_dir, out = Path(args.source_dir), Path(args.output_dir)
    if source_dir.resolve() == out.resolve():
        parser.error('Source and stability outputs must be separate directories')
    if args.workers < 1 or not np.isfinite(args.replacement_value):
        parser.error('Positive worker count and finite replacement required')
    if Path(args.module).suffix == '.py':
        args.module = str(Path(args.module).resolve())
    if (source_dir/'running.lock').exists():
        raise ValueError('Finish the source privacy study before stability analysis')
    source = json.loads((source_dir/'settings.json').read_text())
    if source.get('study') != 'privacy' or source.get('coefficient_storage') != 'final_coefficients_npz_v1':
        raise ValueError('Source must be a privacy study with saved coefficients; risks alone cannot recover draws')
    config = SimpleNamespace(module=args.module, seed=source['seed'], n_test=source['n_test'],
        steps=source['steps'], reps=source['reps'], **source['lmc'])
    required=dict(steps=500000,reps=c.PRODUCTION_REPS,initial_state='zero')
    required_lmc=dict(ridge=None,ridge_rn=None,ridge_upsilon=None,privacy_beta=.01,
                      step_factor=.1,projection_cycles=1000,projection_tol=1e-10)
    if any(source.get(k)!=v for k,v in required.items()) or any(source['lmc'].get(k)!=v for k,v in required_lmc.items()):
        raise ValueError('Stability requires a source privacy study with 100 repetitions, zero start, 500000 steps, theoretical beta=0.01, factor=0.1, 1000 projection cycles and tolerance 1e-10. Run samples_and_privacy.py with the bundle defaults first.')
    # Source and current code/settings must match before reconstructing data
    # or replaying the Gaussian stream. Do not silently combine code versions.
    # Export-only validation does not execute data reconstruction or fitting.
    # Permit diagnostic edits while preserving strict checks for new fits.
    current=c.risk_settings(config,'privacy')
    # Diagnostics and stability scripts do not generate the source draws.
    # Permit edits to them while checking all production code and settings.
    def production_settings(value):
        value=json.loads(json.dumps(value))
        for name in ('convergence_common.py','coefficient_stability.py'):
            value.get('bundle_sha256',{}).pop(name,None)
        return value
    if not args.plot_only and production_settings(current) != production_settings(source):
        raise ValueError('Source settings/code do not match this bundle; use the code that generated the source')
    reps = source['reps'] if args.reps is None else args.reps
    if not 1 <= reps <= source['reps']:
        parser.error('Repetitions must be within the saved source range')
    if not args.ns or not set(args.ns) <= set(c.NS) or not args.alphas or not set(args.alphas) <= set(c.ALPHAS):
        parser.error('Select sample sizes and privacy levels present in the source study')
    index = args.replacement_index
    if not 0 <= index < min(c.NS):
        parser.error('Replacement index must lie in 0..999 for all sample sizes')
    original = c.load_coefficients(source_dir/'coefficients.npz')
    risks = np.load(source_dir/'privacy_risk.npy', allow_pickle=False)
    if risks.shape != (len(c.ALPHAS)+1,len(c.NS),source['reps']) or not np.isfinite(risks).all():
        raise ValueError('Finish the source privacy risk study first')
    cells = [(i,j,rep) for i,a in enumerate(c.ALPHAS) if a in args.alphas
             for j,n in enumerate(c.NS) if n in args.ns for rep in range(reps)]
    for i,j,rep in cells:
        if any(key not in original for key in (c.coefficient_key(i,j,rep), c.coefficient_key(len(c.ALPHAS),j,rep,True))):
            raise ValueError('Missing source coefficient vectors; original chains will not be rerun automatically')
    expected = dict(study='coefficient_stability', source_settings=source,
        source_coefficient_sha256=digest(source_dir/'coefficients.npz'),
        source_risk_sha256=digest(source_dir/'privacy_risk.npy'),
        ns=sorted(set(args.ns)), alphas=sorted(set(args.alphas)), reps=reps,
        replacement_index=index, replacement_value=args.replacement_value,
        replacement_selection='first observation of each randomly generated dataset' if index==0 else 'user-specified fixed index',
        replacement_type='fixed-value replacement; not a target-distribution draw',
        score_metric='mean squared score-function displacement on shared independent test observations',
        coupling='same Gaussian stream; each dataset uses prescribed step size',
        coefficient_metric='squared Euclidean displacement divided by coefficient count',
        erm='same geometry as each private fit; unregularised objective')
    with c.output_lock(out):
        c.check_settings(out, expected, plot_only=args.plot_only)
        file, cache_file = out/'coefficient_stability.csv', out/'modified_coefficients.npz'
        frame = pd.read_csv(file) if file.exists() else pd.DataFrame()
        cache = c.load_coefficients(cache_file)
        done = set()
        if not frame.empty:
            if frame.duplicated(['i','j','rep']).any():
                raise ValueError('Duplicate stored stability rows')
            done = set(zip(frame.i.astype(int),frame.j.astype(int),frame.rep.astype(int)))
            if not done <= set(cells):
                raise ValueError('Unexpected stored stability cells')
        jobs = []
        for i,j,rep in cells:
            if (i,j,rep) in done:
                continue
            jobs.append((args.module,source,i,j,rep,index,args.replacement_value,
                original[c.coefficient_key(i,j,rep)], original[c.coefficient_key(len(c.ALPHAS),j,rep,True)],
                {k:v for k,v in cache.items() if k in (c.coefficient_key(i,j,rep),
                    f'erm_original_{i}_{j}_{rep}',f'erm_changed_{i}_{j}_{rep}')}))
        if args.plot_only and jobs:
            raise ValueError('Stability study incomplete')
        if not args.plot_only:
            print(f'{len(jobs)} pending cells; original chains reused; replacement index={index}', flush=True)
            for row, saved in c.parallel_results(stability_case,jobs,args.workers):
                np.testing.assert_allclose(row['private_risk_before'], risks[row['i'],row['j'],row['rep']],
                                           rtol=1e-8, atol=1e-10)
                cache.update(saved)
                c.atomic_coefficients(cache,cache_file)
                frame = pd.concat((frame,pd.DataFrame([row])),ignore_index=True).sort_values(['i','j','rep'])
                c.atomic_csv(frame,file)
                print(f"Saved n={row['n']}, alpha={row['alpha']}, rep={row['rep']+1}", flush=True)
        print(f'Saved {file}', flush=True)


if __name__ == '__main__':
    main()
