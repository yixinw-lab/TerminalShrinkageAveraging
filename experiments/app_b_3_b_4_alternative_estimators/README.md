# Section B.3; B.4 — Alternative estimators and tensorwise adaptive TSA

**Paper items:** Figure 7. Numbering follows the manuscript.

| Item | Location |
|---|---|
| Paper-facing data and tables | [results/app_b_3_b_4_alternative_estimators/](../../results/app_b_3_b_4_alternative_estimators/) |
| Source records | [results/_sources/d12_window_estimators/](../../results/_sources/d12_window_estimators/) |
| Original experiment entrypoint | [nanochat_meta/appendix_averaging_d12.py](../../experiments/_shared/d12_training/nanochat_meta/appendix_averaging_d12.py) |
| CPU analysis implementation | [reproduce.py](../_tools/reproduce.py) |

## Reproduce this section’s exports

From the repository root, after installing `experiments/_tools/requirements-analysis.txt`:

```bash
python experiments/app_b_3_b_4_alternative_estimators/analyze.py
```

This writes only derived CSVs/tables and replots. It does **not** run training,
load model weights, submit cluster jobs or provision cloud machines. Existing derived
exports are replaced; source records under `results/_sources/` are left unchanged.
Use `--output-root /tmp/tsa-rebuild` to write outside the repository, or `--no-plots`
for numerical exports only.

## Evidence and reproduction status

All five per-run endpoint records are included (seeds 1103, 2203, 3303, 4403, 5503). These are a separate trajectory suite, not the five main-confirmation streams. The shared implementation modules are included in the integrated D12 harness; the upstream environment and large-artifact setup remain separate requirements.

## Integrated source

The previously missing local D12 modules are now in the [integrated harness](../_shared/d12_training/README.md). Exact upstream/environment and large-artifact setup remain separate release requirements.
