# D12 TSA parameter/tensor group search v1

This suite is a direct follow-up to the structured-TSA result. Its purpose is to
find out whether the useful averaging heterogeneity can be localized more sharply
than the existing **hidden matrix block vs. complement** split.

## Design

The primary experiment is deliberately large enough to discover a real structural
pattern without using confirmation holdout data for rule selection.

1. **8 fresh native Muon+AdamW development trajectories**, all at a 10% terminal floor.
2. Each development run saves only the final `K=8`, `s=32` float32 CPU snapshots.
3. On 32 calibration batches, each run evaluates **290 predeclared TSA candidates**.
4. A freeze job chooses the best scalar, one winner per structural family, the
   overall development winner, and several near-winners.
5. **8 fresh native confirmation trajectories** evaluate only those frozen rules
   on the untouched final 96 validation batches.
6. The same frozen rules are also evaluated on **8 paired pure-AdamW confirmation
   trajectories** to test whether any new split reflects tensor role/geometry
   rather than Muon specifically.

The primary contrast is frozen overall structured rule vs. frozen scalar TSA on
the fresh native holdout. A second key contrast is frozen overall rule vs. the
best development-selected `hidden_rest` rule, which is the direct analogue of the
existing matrix-vs-complement split.

## Candidate families

The development library contains:

- dense scalar alpha grid;
- hidden matrices vs. complement;
- attention vs. MLP vs. complement;
- input/feature-building projections vs. residual-writing projections;
- square-ish vs. rectangular hidden matrices;
- early vs. late depth halves;
- a curated early/middle/late thirds design;
- four functional hidden-matrix roles: QKV, attention output, MLP up, MLP down;
- complement decomposition into embeddings, unembedding, scalars, and other;
- trajectory-statistic high/low tensor buckets;
- affine tensorwise rules based on the trajectory noise/persistence score;
- per-layer perturbation scans;
- per-layer attention/MLP perturbation scans.

The trajectory score is computed only from the eight saved checkpoints. It uses
the same broad idea as the prior exploratory adaptive TSA: residual update energy
plus lack of consecutive-update directional persistence.

## Resource choices

Training stages request one GPU and **180 GB host memory**. Only eight full
float32 snapshots are retained, rather than the 27-snapshot pattern in the
forecast suite. The freeze and merge jobs also request one GPU because DeltaAI's
GPU partition rejects jobs that request no GPU resource.

## Install

Copy the archive and checksum into `${PROJECT_ROOT}`, then on DeltaAI:

```bash
cd ${PROJECT_ROOT}
sha256sum -c d12_tsa_group_search_v1.tar.gz.sha256
tar -xzf d12_tsa_group_search_v1.tar.gz
./verify_tsa_group_search_d12_suite.sh "$PWD"
```

The verifier expects the existing repository to already contain:

```text
nanochat_meta/structured_tsa_d12.py
nanochat_meta/paper_main_d12.py
nanochat_meta/filter_suite.py
nanochat_meta/skip_suite.py
slurm/_skip_env.sh
sub.sh
```

It also checks both existing D12 pack files.

## Submit

Use a new result root:

```bash
cd ${PROJECT_ROOT}

export RESULT_ROOT=/work/nvme/bhji/$USER/ExtrapProj/outputs/optimization_paper/d12_tsa_group_search_v1

ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
GROUP_ROOT="$RESULT_ROOT" \
MAX_PARALLEL=8 \
GROUP_NAME=d12_tsa_group_search_v1 \
bash slurm/submit_tsa_group_search_d12.sh
```

The dependency graph is:

```text
GPU smoke
  -> 8 native development jobs
  -> GPU freeze job
  -> 16 confirmation jobs (8 native + 8 pure AdamW)
  -> GPU merge/analyze job
```

## Monitor

```bash
squeue -u "$USER" -o '%.18i %.9P %.30j %.2t %.10M %.10l %R'
cat "$RESULT_ROOT/SMOKE.txt"
cat "$RESULT_ROOT/SUBMISSION.txt"
```

For completed/failed tasks:

```bash
sacct -X -S today -u "$USER" \
  -o JobID,JobName%24,State,ExitCode,Elapsed,NodeList%20
```

After the merge job:

```bash
cat "$RESULT_ROOT/figures/DIGEST.txt"
column -s, -t < "$RESULT_ROOT/DEV_FAMILY_WINNERS.csv"
column -s, -t < "$RESULT_ROOT/figures/confirmation_summary.csv"
column -s, -t < "$RESULT_ROOT/figures/key_contrasts.csv"
column -s, -t < "$RESULT_ROOT/figures/trajectory_role_stats.csv"
```

To make a small result archive:

```bash
./pack_tsa_group_search_d12_results.sh \
  "$RESULT_ROOT" \
  d12_tsa_group_search_v1_summary.tar.gz
```

## Primary verdict

The digest prints the frozen development winner and two native confirmation
contrasts:

```text
overall structured rule - frozen scalar

overall structured rule - hidden/rest family winner
```

A genuinely better split should have a positive 95% interval for the first, and
ideally also for the second. The pure-AdamW block is a transfer diagnostic: if the
same frozen structural rule survives there, that is evidence that the split is
tracking tensor role/geometry rather than merely the Muon assignment.
