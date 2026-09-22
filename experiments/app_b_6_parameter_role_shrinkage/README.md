# Section B.6 — Parameter-role group search and frozen confirmation

**Paper items:** Figure 10; Figure 11; Table 6. Numbering follows the manuscript.

| Item | Location |
|---|---|
| Paper-facing data and tables | [results/app_b_6_parameter_role_shrinkage/](../../results/app_b_6_parameter_role_shrinkage/) |
| Source records | [results/_sources/app_b_6_parameter_roles/](../../results/_sources/app_b_6_parameter_roles/) |
| Original experiment entrypoint | [nanochat_meta/tsa_group_search_d12.py](../../experiments/_shared/d12_training/nanochat_meta/tsa_group_search_d12.py) |
| CPU analysis implementation | [reproduce.py](../_tools/reproduce.py) |

## Reproduce this section’s exports

From the repository root, after installing `experiments/_tools/requirements-analysis.txt`:

```bash
python experiments/app_b_6_parameter_role_shrinkage/analyze.py
```

This writes only derived CSVs/tables and replots. It does **not** run training,
load model weights, submit cluster jobs or provision cloud machines. Existing derived
exports are replaced; source records under `results/_sources/` are left unchanged.
Use `--output-root /tmp/tsa-rebuild` to write outside the repository, or `--no-plots`
for numerical exports only.

## Evidence and reproduction status

Eight development runs (290 candidates each), sixteen confirmation runs (eight paired streams under two optimizers), the complete frozen-rule JSON, and both manifests are included. Analysis now replays development selection in a temporary directory and requires the complete frozen object to match before rebuilding Figure 10, Figure 11 and Table 6. Confirmation never selects or refits a rule. The original B.6 critical value 2.365 is retained.

[Integrated training source and environment notes](../_shared/d12_training/README.md). The implementation imports and CPU self-tests pass; training and model evaluation were not rerun.
