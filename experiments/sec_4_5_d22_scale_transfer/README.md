# Section 4.5; A.4 — Depth-22 matched-endpoint scale transfer

**Paper items:** Table 2. Numbering follows the manuscript.

| Item | Location |
|---|---|
| Paper-facing data and tables | [results/sec_4_5_d22_scale_transfer/](../../results/sec_4_5_d22_scale_transfer/) |
| Source records | [results/_sources/d22_runs/](../../results/_sources/d22_runs/) |
| Original experiment entrypoint | [alpha70_confirmation/README.md](../../experiments/sec_4_5_d22_scale_transfer/archived_training_packages/alpha70_confirmation/README.md) |
| CPU analysis implementation | [reproduce.py](../_tools/reproduce.py) |

## Reproduce this section’s exports

From the repository root, after installing `experiments/_tools/requirements-analysis.txt`:

```bash
python experiments/sec_4_5_d22_scale_transfer/analyze.py
```

This writes only derived CSVs/tables and replots. It does **not** run training,
load model weights, submit cluster jobs or provision cloud machines. Existing derived
exports are replaced; source records under `results/_sources/` are left unchanged.
Use `--output-root /tmp/tsa-rebuild` to write outside the repository, or `--no-plots`
for numerical exports only.
