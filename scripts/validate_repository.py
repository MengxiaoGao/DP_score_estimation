#!/usr/bin/env python3
"""Check computation-only sources and formal settings."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_source(source, label):
    tree = ast.parse(source, filename=label)
    for prefix in ['/' + 'Users' + '/', '.' + 'codex' + '/']:
        assert prefix not in source, f'Personal path: {label}'
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(x.name.split('.')[0] in {'matplotlib', 'seaborn', 'plotly'} for x in node.names), label
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or '').split('.')[0] not in {'matplotlib', 'seaborn', 'plotly'}, label
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {'savefig', 'imshow', 'subplots'}, label
    # The standalone real-data driver stores two complete dependency sources.
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'_ENGINE_SOURCE', '_REAL_SOURCE'}:
                    check_source(node.value.value, label + ':' + target.id)


def main():
    paths = [p for base in ('experiments', 'scripts') for p in (ROOT / base).rglob('*.py')]
    for path in paths:
        check_source(path.read_text(), str(path.relative_to(ROOT)))
    config = json.loads((ROOT / 'experiments/diffusion_1d/paper/config.json').read_text())
    assert config['n_values'] == [1000, 2000, 4000, 8000, 16000]
    assert config['alpha_total_values'] == [1, 2, 4, 8, 16]
    assert config['gamma_values'] == [.2, .4, .6, .8, 1]
    assert (config['K'], config['amplitude'], config['repetitions'], config['lmc_steps']) == (40, .25, 100, 500000)
    assert config['alpha_panel_gamma'] == .5 and config['gamma_panel_alpha_total'] == 4
    assert config['lmc_step_factor'] == .1 and config['privacy_split'] == .5
    records = json.loads((ROOT / 'configs/nhanes_sources.json').read_text())['files']
    assert {r['name'] for r in records} == {'P_BMX.xpt', 'P_DEMO.xpt'}
    assert all(len(r['sha256']) == 64 and r['url'].startswith('https://wwwn.cdc.gov/') for r in records)
    print(f'PASS: {len(paths)} Python sources, embedded sources, formal configuration and input metadata')


if __name__ == '__main__':
    main()
