# D12 predeclared groupwise terminal-floor test v1

This suite tests one follow-up hypothesis from the structured-TSA result: **if
parameter roles prefer different terminal averaging strengths, do they also prefer
different amounts of terminal optimizer activity?**

The experiment intentionally contains **no new hyperparameter search**.  Every
schedule arm, stream seed, estimator, and primary contrast is frozen in
`configs/groupwise_floor_protocol.json` before any new trajectory is trained.

## Frozen hypothesis

The preceding structured-TSA confirmation selected the following coefficients:

- embedding: `alpha = 0.00`
- hidden: `alpha = 0.65`
- unembedding: `alpha = 0.45`
- rest (scalar + other): `alpha = 0.25`

We map those coefficients monotonically onto the already studied 5%-15% terminal
learning-rate-floor range:

```text
rho_g = 0.05 + 0.10 * alpha_g / 0.65
```

so the **mapped** schedule is fixed at approximately:

```text
embedding       5.0000%
hidden matrices 15.0000%
unembedding    11.9231%
rest             8.8462%
```

For LR routing, `hidden` means matrix-valued tensors inside transformer layers;
1D layer scales are placed in `rest`, matching the optimizer tensor groups. The
returned structured-TSA estimator itself stays exactly frozen from the preceding
search (where the tiny set of layer-indexed 1D scales were included in `hidden`).
No values are selected from the new runs.

## Four paired schedule arms

Eight fresh training stream seeds are crossed with four arms (32 D12 trajectories):

1. `uniform05` — uniform 5% terminal floor.
2. `uniform10` — uniform 10% terminal floor.
3. `matched_uniform` — one uniform floor equal to the **parameter-count-weighted
   mean** of the mapped per-parameter floors.  It is computed from the restored
   model before training and is therefore fixed independently of validation loss.
4. `mapped` — the frozen per-role floors above.

The parameter-count-matched arm is the primary schedule control.  It asks whether
**allocating** terminal activity by parameter role is useful, rather than merely
changing the average floor.

The matched control does not claim to match update energy exactly: NanoChat's
parameter groups use different base learning rates and optimizer geometry.

## Returned estimators

Every trained trajectory saves the final `K=8` checkpoints at spacing `s=32` and
is evaluated three ways on the same final 96 validation batches:

```text
raw          final optimizer iterate
scalar055    scalar TSA with alpha=0.55
structured   frozen role-wise TSA:
             embedding=0.00, hidden=0.65, unembedding=0.45, rest=0.25
```

The structured estimator is not re-fit for any schedule arm.

## Predeclared primary tests

The main comparison is

```text
structured(mapped) - structured(matched_uniform)
```

reported as `BPB(reference) - BPB(candidate)`, so positive values favor the
mapped groupwise schedule.

The mechanistic interaction is

```text
[raw(mapped) - raw(matched_uniform)]
  - [structured(mapped) - structured(matched_uniform)]
```

A positive interaction means redistributing terminal activity is more favorable
when the frozen structured estimator is returned than when the raw endpoint is
returned.

Secondary outputs include mapped vs uniform 10%, the full mapped+structured
recipe vs uniform-5%-raw, structured-vs-raw gains within every schedule arm, and
a fresh uniform-10%-vs-uniform-5% schedule-estimator interaction.

## Implementation of per-tensor learning-rate floors

The smoke job first checks whether each optimizer parameter group is homogeneous
with respect to the four frozen tensor buckets.

- If it is, the suite uses **direct optimizer param-group LR routing**.  At each
  step it asks the repository's existing `paper_main_d12._apply_floor_lr` helper
  for the LR produced by each requested floor and routes the corresponding LR to
  each optimizer group.  This preserves the repository's warmup/cooldown logic.
- The code contains a **post-step tensor update routing** fallback for mixed
  optimizer groups, but the paper protocol does **not** enable it by default. If
  the mapped schedule cannot be expressed by direct optimizer param-group LRs,
  the smoke job writes `SMOKE_OPTIMIZER_GROUPS.csv` and deliberately fails before
  any of the 32 training jobs start. This keeps the appendix intervention exact
  and easy to describe. The fallback can be enabled manually later only after
  inspecting that diagnostic.

The smoke test also checks that repeated calls to the repository floor scheduler
are idempotent before the training array can start. `SMOKE.txt` records which
routing mode is available.

## Fresh paired streams

The eight predeclared stream seeds are:

```text
329303 340403 351503 362603 373703 384803 395903 407003
```

Within a seed, all four arms use the same initialization and the same explicit
3000-step data-order permutation.  The merge job verifies the permutation
fingerprints before computing paired effects.

## Install

Copy the archive and checksum into `${PROJECT_ROOT}`, then on DeltaAI:

```bash
cd ${PROJECT_ROOT}
sha256sum -c d12_tsa_groupwise_floor_v1.tar.gz.sha256
tar -xzf d12_tsa_groupwise_floor_v1.tar.gz
./verify_groupwise_floor_d12_suite.sh "$PWD"
```

The verifier expects the repository to already contain the same base D12 harness
used by the earlier TSA suites, including:

```text
nanochat_meta/structured_tsa_d12.py
nanochat_meta/paper_main_d12.py
nanochat_meta/filter_suite.py
nanochat_meta/skip_suite.py
slurm/_skip_env.sh
sub.sh
```

It also checks the existing native D12 pack.

## Submit

Use a new result root:

```bash
cd ${PROJECT_ROOT}

export RESULT_ROOT=/work/nvme/bhji/$USER/ExtrapProj/outputs/optimization_paper/d12_tsa_groupwise_floor_v1

ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
GROUP_ROOT="$RESULT_ROOT" \
MAX_PARALLEL=8 \
GROUP_NAME=d12_tsa_groupwise_floor_v1 \
bash slurm/submit_groupwise_floor_d12.sh
```

Dependency graph:

```text
GPU smoke / scheduler-routing validation
  -> 32 training jobs (8 paired streams x 4 fixed arms)
  -> GPU merge + paired analysis + appendix draft
```

All stages request one GPU because DeltaAI rejects jobs on the GPU partition that
request no GPU.  Training jobs request 180 GB host memory and retain only eight
float32 CPU snapshots.

## Monitor

```bash
squeue -u "$USER" -o '%.18i %.9P %.30j %.2t %.10M %.10l %R'
cat "$RESULT_ROOT/SMOKE.txt"
cat "$RESULT_ROOT/SUBMISSION.txt"
```

The smoke outputs that are most useful if routing fails are:

```bash
column -s, -t < "$RESULT_ROOT/SMOKE_OPTIMIZER_GROUPS.csv" | less -S
column -s, -t < "$RESULT_ROOT/SMOKE_SCHEDULE.csv"
```

For completed or failed jobs:

```bash
sacct -X -S today -u "$USER" \
  -o JobID,JobName%24,State,ExitCode,Elapsed,NodeList%20
```

## Results

After the merge job:

```bash
cat "$RESULT_ROOT/figures/DIGEST.txt"
column -s, -t < "$RESULT_ROOT/figures/paired_contrasts.csv"
column -s, -t < "$RESULT_ROOT/figures/arm_summary.csv"
cat "$RESULT_ROOT/figures/APPENDIX_DRAFT.md"
```

`APPENDIX_DRAFT.md` is generated from the frozen protocol and measured numbers;
it is meant as a starting point for the paper appendix rather than an automatic
claim of significance.
