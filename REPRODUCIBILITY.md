# Reproducibility

The recorded environment used Python 3.13 and the versions pinned in
`requirements.txt`.

## Rebuild all paper tables and figures from committed data

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python alpha_panel/analyze.py
python analyze_gamma.py
python analyze_stability.py
```

These commands use the committed aggregate CSVs and recreate each metric in a
separate PNG/PDF figure. Bootstrap seeds are fixed.

## Rerun the theory-aligned alpha reverse sampler

```bash
python alpha_panel/driver.py
```

The 25 files in `alpha_panel/private_densities/` contain the 30 projected-LMC
private-density releases for every `(n, alpha)` cell. They omit old generated
particles but retain everything required to rerun the revised reverse sampler,
its interpolation validation, and the discretization experiment. Target
reference samples are deterministically regenerated from the recorded seeds.

## Rerun the gamma panel and stability pilot

```bash
python driver.py
```

Full runs take hours and write resume checkpoints and per-run arrays to ignored
paths. Reissuing the command after interruption skips completed checkpoints.
Putting a laptop to sleep pauses computation at the operating-system level;
the saved checkpoints remain valid.
