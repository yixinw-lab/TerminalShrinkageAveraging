# D22 DeltaAI paired replication suite v1

Purpose: quality/uncertainty replication of the D22 schedule-estimator interaction on DeltaAI. This is **not** an official speedrun timing experiment because 8 GPUs span two DeltaAI nodes rather than one 8xH100 node.

Design: 3 paired experiment seeds (42,43,44) x 2 terminal schedule arms. Both arms retain the exact PR #830 ratio-9.4 scaling/schedule horizon and physically stop at step 10,172 (ratio 9.0). Control leaves the schedule unchanged; treatment clamps the LR multiplier at 15%. Both save the same eight terminal snapshots. Raw, TSA alpha .55, TSA alpha .587, LAWA, and finite-window EWA beta .75 are evaluated post hoc on every trained trajectory.

The exact upstream base is commit `e09bc164162f35da0b5b8315be791e9e974a4c3a`. Instrumentation is written to a new script; upstream `scripts/base_train.py` is not edited.

## On DeltaAI

```bash
cd ${PROJECT_ROOT}
# unpack the downloaded suite here
./d22_delta_replication_suite_v1/verify_d22_delta_suite.sh
./d22_delta_replication_suite_v1/discover_d22_delta_assets.sh "$PWD"

export WORK_ROOT=${EXTRAP_ROOT}/d22_delta_replication_v1
bash d22_delta_replication_suite_v1/prepare_d22_delta_checkout.sh

ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
WORK_ROOT="$WORK_ROOT" \
MAX_PARALLEL=3 \
bash d22_delta_replication_suite_v1/slurm/submit_d22_delta.sh
```

`MAX_PARALLEL=3` requests at most 6 DeltaAI nodes simultaneously. Set `MAX_PARALLEL=6` only if you intentionally want all six 2-node jobs queued to run concurrently (up to 12 nodes).

Monitor:

```bash
squeue -u "$USER" -o '%.18i %.9P %.28j %.2t %.10M %.10l %R'
```

When done:

```bash
cat "$WORK_ROOT/results/analysis/DIGEST.txt"
column -s, -t < "$WORK_ROOT/results/analysis/arm_summary.csv"
column -s, -t < "$WORK_ROOT/results/analysis/paired_effects.csv"
```

Compact package:

```bash
WORK_ROOT="$WORK_ROOT" bash d22_delta_replication_suite_v1/pack_d22_delta_results.sh
```

## Interpretation

Use paired BPB contrasts across the three experiment seeds. DeltaAI training time is hardware-specific and is not comparable to the official single-node NanoChat leaderboard. Seed 42 treatment is compared descriptively to the historical RunPod BPB 0.720050 as an environment-consistency diagnostic only.
