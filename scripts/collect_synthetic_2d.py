#!/usr/bin/env python3
"""Collect 50 completed synthetic batches without moving sample artifacts."""
from pathlib import Path
import argparse
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=['alpha', 'gamma'], required=True)
    parser.add_argument('--batches', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.batches.glob(f'batch_*/paper_results_2d/{args.panel}/rows/*.json'))
    if len(paths) != 2500:
        raise RuntimeError(f'Expected 2,500 completed fits, found {len(paths)}')
    rows = [record for path in paths for record in json.loads(path.read_text())]
    frame = pd.DataFrame(rows)
    keys = ['method', 'n', 'rep', 'gamma', 'alpha_score']
    if len(frame) != 5000 or frame.duplicated(keys).any():
        raise RuntimeError('Duplicate, missing, or malformed private/ERM rows')
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / f'{args.panel}.csv'
    frame.sort_values(keys).to_csv(target, index=False)
    print(f'Collected 2,500 fits into {target}')


if __name__ == '__main__':
    main()
