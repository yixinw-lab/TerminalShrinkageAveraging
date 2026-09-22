# Section B.5 — Initial two-group shrinkage and parameter localization

**Paper items:** Figure 8; Figure 9; Table 4; Table 5. Numbering follows the manuscript.

| Item | Location |
|---|---|
| Paper-facing data and tables | [results/app_b_5_two_group_shrinkage/](../../results/app_b_5_two_group_shrinkage/) |
| Source records | [results/_sources/d12_replicated/](../../results/_sources/d12_replicated/) |
| Original experiment entrypoint | [nanochat_meta/structured_tsa_d12.py](../../experiments/_shared/d12_training/nanochat_meta/structured_tsa_d12.py) |
| CPU analysis implementation | [reproduce.py](../_tools/reproduce.py) |

## Reproduce this section’s exports

From the repository root, after installing `experiments/_tools/requirements-analysis.txt`:

```bash
python experiments/app_b_5_two_group_shrinkage/analyze.py
```

This writes only derived CSVs/tables and replots. It does **not** run training,
load model weights, submit cluster jobs or provision cloud machines. Existing derived
exports are replaced; source records under `results/_sources/` are left unchanged.
Use `--output-root /tmp/tsa-rebuild` to write outside the repository, or `--no-plots`
for numerical exports only.

## Evidence and reproduction status

Per-run development and confirmation records are included. Current-paper confirmation uses seeds 11103, 12203, 13303, 14403, 15503 and evaluation batches 160–255. A full training rerun additionally requires the compatible upstream code and model/data-pack setup.

## Integrated source

The previously missing local D12 modules are now in the [integrated harness](../_shared/d12_training/README.md). Exact upstream/environment and large-artifact setup remain separate release requirements.
