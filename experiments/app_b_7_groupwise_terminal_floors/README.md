# Section B.7 — Groupwise terminal-floor mechanism probe

**Paper items:** Table 7. Numbering follows the manuscript.

| Item | Location |
|---|---|
| Paper-facing data and tables | [results/app_b_7_groupwise_terminal_floors/](../../results/app_b_7_groupwise_terminal_floors/) |
| Source records | [results/_sources/app_b_7_groupwise_floor/](../../results/_sources/app_b_7_groupwise_floor/) |
| Original experiment entrypoint | [nanochat_meta/groupwise_floor_d12.py](../../experiments/_shared/d12_training/nanochat_meta/groupwise_floor_d12.py) |
| CPU analysis implementation | [reproduce.py](../_tools/reproduce.py) |

## Reproduce this section’s exports

From the repository root, after installing `experiments/_tools/requirements-analysis.txt`:

```bash
python experiments/app_b_7_groupwise_terminal_floors/analyze.py
```

This writes only derived CSVs/tables and replots. It does **not** run training,
load model weights, submit cluster jobs or provision cloud machines. Existing derived
exports are replaced; source records under `results/_sources/` are left unchanged.
Use `--output-root /tmp/tsa-rebuild` to write outside the repository, or `--no-plots`
for numerical exports only.

## Evidence and reproduction status

The protocol, manifest, and all 32 run records (eight streams by four schedule arms) are included. Table 7 is recomputed within stream from mapped minus matched-uniform BPB, with the interaction equal to raw effect minus structured-estimator effect. Do not replace the matched-uniform arm by the 10% arm. The per-run protocol hashes and training permutations are checked. The original B.7 critical value 2.364624 is retained.

[Integrated training source and environment notes](../_shared/d12_training/README.md). Training and model evaluation were not rerun.
