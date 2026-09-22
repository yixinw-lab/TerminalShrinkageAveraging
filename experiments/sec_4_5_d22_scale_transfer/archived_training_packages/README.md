# Depth-22 training packages

Paper mapping: Section 4.5, Appendix A.4, Table 2.

These folders contain archived launchers and support files for the depth-22 experiments. The CPU analysis entrypoint for the repository is `experiments/sec_4_5_d22_scale_transfer/analyze.py`; the scripts here are retained as training/operations reference code and can request real compute resources.

| Folder | Contents |
|---|---|
| `alpha70_confirmation/` | Direct ratio-9.0 schedule, 15% floor, alpha=0.70; treatment seeds 42/43/44 and comparison records |
| `control_replication_n3/` | Replication scripts for the raw-control configurations |
| `alpha0587_development/` | Earlier alpha=0.587 development/screening configuration |
| `delta_replication_historical/` | Earlier two-arm cluster replication |
| `base_speedrun_reference/` | Expanded common speedrun base used by the launchers |

## Table 2 configuration

All three paper-facing configurations stop at step 10,172. The early-stop raw control follows the ratio-9.4 schedule; the direct raw control is calibrated to 9.0. The treatment uses the direct-9.0 schedule, a 15% terminal floor, and TSA with alpha=0.70. The eight checkpoints are 9381, 9494, 9607, 9720, 9833, 9946, 10059, and 10172. Reported time is accumulated training-iteration time rather than wall-clock time.

The paper names NanoChat PR #830 commit `e09bc164162f35da0b5b8315be791e9e974a4c3a` as the starting recipe.

The nested `runpod_d22_causal_speedrun_v2.tar.gz` archive is retained in the relevant launcher folders because its SHA-256 is checked by those launchers. The expanded reference omits Python caches, so checksum manifests inside the original packages can mention omitted bytecode.

The paper-facing Table 2 export is under [`results/sec_4_5_d22_scale_transfer/`](../../../results/sec_4_5_d22_scale_transfer/).
