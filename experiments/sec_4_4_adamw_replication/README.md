# Section 4.4 — Pure-AdamW replication

**Paper items:** Figure 4. Numbering follows the manuscript.

| Item | Location |
|---|---|
| Paper-facing data and tables | [results/sec_4_4_adamw_replication/](../../results/sec_4_4_adamw_replication/) |
| Source records | [results/_sources/d12_replicated/](../../results/_sources/d12_replicated/) |
| Original experiment entrypoint | [nanochat_meta/structured_tsa_d12.py](../../experiments/_shared/d12_training/nanochat_meta/structured_tsa_d12.py) |
| CPU analysis implementation | [reproduce.py](../_tools/reproduce.py) |

## Reproduce this section’s exports

From the repository root, after installing `experiments/_tools/requirements-analysis.txt`:

```bash
python experiments/sec_4_4_adamw_replication/analyze.py
```

This writes only derived CSVs/tables and replots. It does **not** run training,
load model weights, submit cluster jobs or provision cloud machines. Existing derived
exports are replaced; source records under `results/_sources/` are left unchanged.
Use `--output-root /tmp/tsa-rebuild` to write outside the repository, or `--no-plots`
for numerical exports only.

## Evidence and reproduction status

Per-run development and confirmation records are included. Current-paper confirmation uses seeds 11103, 12203, 13303, 14403, 15503 and evaluation batches 160–255. A full training rerun additionally requires the compatible upstream code and model/data-pack setup.

The main-paper control uses **alpha=0.55** in both optimizer regimes. Do not substitute the pure-AdamW development-selected alpha=0.50 used in Appendix B.5.

## Integrated source

The previously missing local D12 modules are now in the [integrated harness](../_shared/d12_training/README.md). Exact upstream/environment and large-artifact setup remain separate release requirements.
