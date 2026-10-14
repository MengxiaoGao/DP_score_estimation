# Artifact manifest

## Included

- Alpha panel: 750 density rows, 750 revised sampler rows, 210 discretization
  rows, tables, figures, validation diagnostics, and 25 grouped private-density
  snapshots containing 750 releases.
- Gamma panel: 750 complete observations, summaries, empirical/theoretical
  slopes, runtimes, and figures.
- Stability pilot: 450 observations, dataset-level summaries, ratios, and
  separate coefficient/output-function figures.
- Self-contained analysis and simulation code, configurations, package
  versions, methods notes, and SHA-256 checksums.

## Omitted

- Generated-particle NPZ files and redundant per-run JSON checkpoints.
- Logs, caches, macOS metadata, background-process state, and HTML reports with
  base64-embedded image copies.
- SVG duplicates; PNG previews and vector PDF figures are retained.
- The superseded Euler alpha-panel output and clean-score-risk plots.

The aggregate CSVs are the row-level data used for the included paper tables
and figures. `CHECKSUMS.sha256` covers every included file except itself.
