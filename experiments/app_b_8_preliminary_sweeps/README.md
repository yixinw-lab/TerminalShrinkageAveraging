# Section B.8 — Preliminary single-trajectory scalar and floor sweeps

**Paper items:** Figure 12; Table 8; Table 9. Numbering follows the manuscript.

| Item | Location |
|---|---|
| Paper-facing data and tables | [results/app_b_8_preliminary_sweeps/](../../results/app_b_8_preliminary_sweeps/) |
| Source records | [results/_sources/d12_preliminary/](../../results/_sources/d12_preliminary/) |
| Original experiment entrypoint | [nanochat_meta/paper_main_d12.py](../../experiments/_shared/d12_training/nanochat_meta/paper_main_d12.py) |
| CPU analysis implementation | [reproduce.py](../_tools/reproduce.py) |

## Reproduce this section’s exports

From the repository root, after installing `experiments/_tools/requirements-analysis.txt`:

```bash
python experiments/app_b_8_preliminary_sweeps/analyze.py
```

This writes only derived CSVs/tables and replots. It does **not** run training,
load model weights, submit cluster jobs or provision cloud machines. Existing derived
exports are replaced; source records under `results/_sources/` are left unchanged.
Use `--output-root /tmp/tsa-rebuild` to write outside the repository, or `--no-plots`
for numerical exports only.

## Evidence and reproduction status

All five floor-arm records and their evaluation-batch arrays are included. Tables 8/9 and Figure 12 use the 32-batch **curve** block at step 3000, not the coefficient-selection or 96-batch holdout blocks. To match the reported tables, intervals are 1.96 × paired evaluation-batch SE; they are not across-training-run intervals.

## Integrated source

The previously missing local D12 modules are now in the [integrated harness](../_shared/d12_training/README.md). Exact upstream/environment and large-artifact setup remain separate release requirements.
