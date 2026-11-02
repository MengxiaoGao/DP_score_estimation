#!/usr/bin/env python3
"""Restore the exact public NHANES input files identified in configs/nhanes_sources.json."""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data/nhanes')
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads((ROOT / 'configs/nhanes_sources.json').read_text())['files']
    expected = {item['name']: item['sha256'] for item in records}
    for name, checksum in expected.items():
        target = args.data_dir / name
        if not target.exists():
            temporary = target.with_suffix('.download')
            urllib.request.urlretrieve(URL + name.upper(), temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != checksum:
                raise RuntimeError(f'Unexpected content for {name}; downloaded file retained for inspection')
            temporary.replace(target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f'Input checksum differs: {target}')
        print(f'Verified {name}')


if __name__ == '__main__':
    main()
