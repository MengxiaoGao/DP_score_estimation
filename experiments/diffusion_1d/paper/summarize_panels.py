"""Numerical summaries and paired-bootstrap slopes for the 1D diffusion panels."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
METRICS = {
    'density_tv': 'Private density TV',
    'ou_tv': 'Early-stopped OU TV',
    'generated_binned_tv': 'Generated binned TV',
    'generated_wasserstein': 'Generated Wasserstein-1',
    'reverse_to_ou_binned_tv': 'Reverse-to-OU binned TV',
    'iid_ou_binned_tv': 'IID-to-OU binned TV',
}


def summarize(raw, group):
    rows = []
    for (n, value), df in raw.groupby(['n', group]):
        r = dict(n=n, **{group: value}, repetitions=len(df))
        for metric in METRICS:
            vals = df[metric].to_numpy()
            se = np.std(vals, ddof=1) / np.sqrt(len(vals))
            half = student_t.ppf(.975, len(vals)-1) * se
            r.update({metric+'_mean':np.mean(vals), metric+'_se':se,
                      metric+'_low':np.mean(vals)-half,
                      metric+'_high':np.mean(vals)+half})
        for metric in ['N', 'lmc_seconds', 'reverse_seconds', 'total_seconds',
                       'ridge', 'basis_dim', 'lmc_mean_projection_cycles']:
            r[metric+'_mean'] = df[metric].mean()
        rows.append(r)
    return pd.DataFrame(rows).sort_values(['n', group])


def slopes(raw, group, boot=2000):
    ns = sorted(raw.n.unique())
    groups = sorted(raw[group].unique())
    rng = np.random.default_rng(20261022)
    x = np.log(ns)
    weights = (x-x.mean()) / np.sum((x-x.mean())**2)
    rows = []
    for metric in ['density_tv', 'ou_tv', 'generated_binned_tv']:
        for value in groups:
            slices = [raw[(raw.n==n)&(raw[group]==value)].sort_values('rep')
                      for n in ns]
            rep_ids = slices[0].rep.to_numpy()
            if any(not np.array_equal(df.rep.to_numpy(), rep_ids) for df in slices[1:]):
                raise RuntimeError(f'Mismatched repetition IDs for {group}={value}')
            arr = np.stack([df[metric].to_numpy() for df in slices])
            mean = arr.mean(axis=1)
            slope = float(weights @ np.log(mean))
            draws = []
            for _ in range(boot):
                sampled = arr[:, rng.integers(len(rep_ids), size=len(rep_ids))].mean(axis=1)
                draws.append(float(weights @ np.log(sampled)))
            reference = (-(value+1)/(2*value+3) if group=='gamma'
                         else -3/8)
            lo, hi = np.quantile(draws, [.025, .975])
            rows.append(dict(metric=metric, **{group:value}, slope=slope,
                             ci95_low=lo, ci95_high=hi,
                             theory_reference=reference,
                             bootstrap_repetitions=boot))
    return pd.DataFrame(rows)




def main():
    p = argparse.ArgumentParser()
    p.add_argument('--panel', choices=['alpha','gamma'], required=True)
    p.add_argument('--output-dir', type=Path,
                   help='Output directory of a completed panel (default: outputs/<panel>)')
    args = p.parse_args()
    out = args.output_dir or ROOT/'outputs'/args.panel
    raw = pd.read_csv(out/'raw.csv')
    expected = 5*5*100
    group = 'alpha_total' if args.panel=='alpha' else 'gamma'
    if len(raw) != expected or not raw.groupby(['n',group]).size().eq(100).all():
        raise RuntimeError(f'Panel incomplete: {len(raw)}/{expected} runs')
    (out/'tables').mkdir(exist_ok=True)
    summary = summarize(raw, group)
    summary.to_csv(out/'tables'/'summary.csv', index=False)
    slope_table = slopes(raw, group)
    slope_table.to_csv(out/'tables'/'slopes.csv', index=False)
    pass  # Numerical outputs are saved above.
    print(f'Analyzed {args.panel}: {len(raw)} runs', flush=True)


if __name__ == '__main__':
    main()
