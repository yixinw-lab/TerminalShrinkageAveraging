# D12 appendix averaging / sensitivity suite

Purpose: close the highest-value appendix gaps without rerunning controls that already exist.

New compute: five D12 Muon+AdamW trajectories (stream seeds 1103..5503), all using the frozen 10% floor, T=H=3000. Every estimator is evaluated post hoc from the same trajectory.

New sensitivity grid:
- K = 4, 8, 16
- spacing = 16, 32, 64
- TSA alpha = .55 fixed
- uniform LAWA for every K/s pair

Additional controls evaluated on those same trajectories:
- matched-window finite EWA, K=8,s=32, beta in {.50,.75,.90,.95}
- historical checkpoint EMA beta=.95, K=16,s=16
- SWA-style uniform late average K=32,s=16
- tensorwise adaptive hybrid TSA on all tensors
- tensorwise adaptive hybrid TSA on Muon-managed tensors only

The merge also copies the existing Muon-vs-AdamW/adaptive result tables from `averaging_curves_d12_s5_v1` when present and fits the theorem's quadratic alpha response to the clean baseline holdout sweep. No additional AdamW training is performed.

## Install on DeltaAI

Copy `d12_appendix_averaging_suite_v1.tar.gz` to `${PROJECT_ROOT}`, then:

```bash
cd ${PROJECT_ROOT}
tar -xzf d12_appendix_averaging_suite_v1.tar.gz
./verify_appendix_averaging_suite.sh "$PWD"
```

Submit:

```bash
ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
MAX_PARALLEL=4 \
GROUP_NAME=d12_appendix_averaging_v1 \
bash slurm/submit_appendix_averaging.sh
```

Monitor:

```bash
squeue -u "$USER" -o '%.18i %.9P %.32j %.2t %.10M %.10l %R'
```

When complete:

```bash
ROOT=outputs/optimization_paper/d12_appendix_averaging_v1
cat "$ROOT/figures/DIGEST.txt"
column -s, -t < "$ROOT/figures/ks_summary.csv"
column -s, -t < "$ROOT/figures/averaging_family_summary.csv"
```

Package:

```bash
GROUP_NAME=d12_appendix_averaging_v1 bash pack_appendix_averaging_results.sh "$PWD"
```
