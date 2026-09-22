# Depth-22 saved run records

This directory contains the saved evaluation summaries and compact run metadata used by the Section 4.5 analysis.

The three `floor15_direct9_tsa0700` treatment runs are under
`d22_alpha70/nanochat-causal-speedrun-v2/results/`, in directories ending in
`_s42`, `_s43`, and `_s44`. Each contains:

- `tsa_core.json`: the saved CORE score, validation BPB, training time, and model identifier.
- `DIGEST.txt`: the compact run summary, including `tsa_core`.
- `run_config.txt`, `candidate.csv`, `snapshot_manifest.json`, and `meta_010172.json`: configuration, candidate, snapshot, and checkpoint metadata.

The control observations are in `existing_controls_n3.csv`, with additional run metadata under `d22_controls/`.

The analysis reads the per-seed treatment CORE scores from `tsa_core.json` and computes their mean and 95% Student-t interval. All three treatment observations are also included in the per-seed export.

These are saved summaries, not complete evaluation transcripts. The `log` fields identify the evaluation log paths used by the runs; the corresponding `tsa_core.log` files are not included in this snapshot.
