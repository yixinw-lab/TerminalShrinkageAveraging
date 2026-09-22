# NanoChat terminal-risk + averaging record attempt

This package turns the D12 result into a disciplined NanoChat leaderboard attempt.
It does **not** claim that a record is guaranteed. It is designed to maximize the
chance of finding a real speed win without tuning on the final confirmation runs.

## Frozen empirical hypothesis

The D12 experiments support the following intervention:

1. follow NanoChat's ordinary warmdown;
2. once the learning-rate multiplier reaches **15%**, stop cooling and hold it there;
3. collect a tail window of **8 model snapshots** with spacing scaled from the D12
   winner (`32/3000` of the schedule horizon);
4. average the output model post hoc;
5. spend the resulting quality cushion on a lower data:parameter ratio.

The training recipe is pinned to NanoChat PR #830 at commit
`e09bc164162f35da0b5b8315be791e9e974a4c3a`:

- depth 22;
- ratio 9.4 reference schedule;
- 49,152-token tokenizer;
- device batch 32 and total batch 524,288;
- FP8;
- Liger fused cross entropy;
- selective learnable RMSNorm scales.

## Why the package has exploration and confirmation phases

The archived D12 results motivate the intervention, while this historical package
does not contain the original tensorwise `trajectory_groups.py` implementation.
It therefore screens a **small, declared** portable family from the same snapshot
window:

- raw endpoint;
- uniform average;
- fixed blends, including `blend:0.587`, the observed mean D12 all-tensor
  averaging strength;
- variance-based, autocorrelation-based, and hybrid tensorwise averages.

This is done **once** on one exploratory D22 trajectory. One endpoint and one
recipe are then written to `config/frozen_candidate.env`. The confirmation array
must use that frozen file unchanged.

## Hardware choice

### Official timing: PSC Bridges-2

Use one Bridges-2 node with 8 H100-SXM5-80GB GPUs. This matches the NanoChat
leaderboard's one-node 8xH100 hardware class. The included Slurm scripts request:

```text
--partition=GPU
--nodes=1
--gpus=h100-80:8
```

### NCSA DeltaAI

DeltaAI nodes contain 4 GH200/H100 GPUs and ARM CPUs. The included DeltaAI job is
only a functional/transfer probe. It is not directly comparable to the official
one-node 8xH100 time, and the Liger dependency may require an ARM-compatible
build.

Do not use two DeltaAI nodes to make an official timing claim. Multi-node network
and CPU differences make that a different hardware result.

## What the patch changes

`tools/apply_record_patch.py` modifies only two upstream files and fails closed if
its source anchors no longer match.

### `scripts/base_train.py`

It adds:

```text
--record-schedule-param-data-ratio
--record-terminal-lr-clamp-frac
--record-snapshot-ratios
--record-snapshot-k
--record-snapshot-spacing-frac
--record-snapshot-dir
```

The schedule ratio is separated from the actual stop ratio. A confirmation run
can therefore stop at, for example, ratio 9.2 while retaining the exact LR,
Muon-momentum, and weight-decay prefix of the ratio-9.4 recipe.

Model-only snapshots are captured by rank zero after the optimizer step and
outside NanoChat's timed training interval. The manifest records both the
official `total_training_time` and snapshot-copy overhead. Report both.

### `scripts/base_eval.py`

The CORE CSV slug includes the model tag, preventing candidate checkpoints at the
same step from overwriting each other's output.

## Workflow

### 1. Obtain an 8xH100 Bridges-2 allocation

A reservation is strongly recommended for the final six-run confirmation so
queue variability does not fragment the campaign.

### 2. Install the pinned repository

Extract this package on Bridges-2, then:

```bash
cd ~/nanochat_record_attempt_v1
./verify_record_package.sh

bash tools/setup_pr830.sh "$HOME/nanochat-record"
```

The setup script:

- clones `karpathy/nanochat` if necessary;
- fetches PR #830;
- checks out the exact measured commit;
- installs the record snapshot module;
- applies the fail-closed patch;
- runs `uv sync --extra gpu --extra liger`;
- runs syntax and merge self-tests.

### 3. Prepare shared data and the 49k tokenizer

Choose a persistent shared path, not node-local scratch:

```bash
export NANOCHAT_BASE_DIR=/path/on/shared/storage/nanochat-record-data
bash tools/prepare_data.sh "$HOME/nanochat-record"
```

This downloads 170 training shards and trains/evaluates the 49,152-token
speedrun tokenizer.

### 4. Configure Bridges-2

```bash
cp config/explore.env.example config/explore.env
cp config/confirm.env.example config/confirm.env
cp config/site_bridges2.env.example config/site_bridges2.env
```

Edit the three files. At minimum set:

```text
REPO
NANOCHAT_BASE_DIR
RECORD_PACKAGE_DIR
BRIDGES2_ACCOUNT
```

Keep the initial frozen intervention unchanged:

```text
SCHEDULE_RATIO=9.4
TERMINAL_CLAMP_FRAC=0.15
SNAPSHOT_K=8
SNAPSHOT_SPACING_FRAC=0.010666666666666666
```

### 5. Run the optional 250-step hardware preflight

```bash
bash slurm/submit_bridges2_preflight.sh
```

Check that all eight GPUs are H100-80GB, Liger imports, and steady-state step time is plausible. This is only a wiring/performance check.

### 6. Run one exploration trajectory

```bash
bash slurm/submit_bridges2_explore.sh
```

The job trains once to ratio 9.4 and captures windows ending at ratios
9.0, 9.1, 9.2, 9.3, and 9.4. It then:

- constructs all declared output candidates;
- runs a cheap BPB screen;
- writes `top_candidates.csv`, one candidate per ratio.

The job prints the exact path to `top_candidates.csv`.

### 7. Run the full CORE ladder

```bash
bash slurm/submit_bridges2_core.sh \
  "$NANOCHAT_BASE_DIR/record_attempts/<RUN_TAG>/bpb_screen/top_candidates.csv"
```

Each array task evaluates one ratio's chosen model with canonical BPB and full
22-task CORE.

### 8. Freeze the earliest qualified candidate

After the CORE array finishes:

```bash
python record_tools/freeze_candidate.py \
  --results-dir "$NANOCHAT_BASE_DIR/record_attempts/<RUN_TAG>/core_results" \
  --output config/frozen_candidate.env \
  --threshold 0.256525 \
  --safety-margin 0.0015
```

Selection policy:

1. choose the lowest ratio whose CORE exceeds threshold + safety margin;
2. if none has the margin, choose the lowest ratio that exceeds the threshold
   and print a warning;
3. if none qualifies, stop. Do not massage the test set or launch confirmations.

Inspect `config/frozen_candidate.env`. Do not alter it based on later outcomes.

### 9. Run six frozen confirmations

```bash
bash slurm/submit_bridges2_confirm.sh
```

Every confirmation run:

- trains only to the frozen lower ratio;
- uses the ratio-9.4 schedule calibration;
- clamps terminal LR at 15%;
- captures exactly one 8-snapshot window;
- applies exactly the frozen recipe;
- runs canonical BPB and full CORE.

### 10. Summarize

The submit script prints the confirmation result directory. Then:

```bash
python record_tools/summarize_confirmations.py \
  --results-dir "$NANOCHAT_BASE_DIR/record_attempts/confirm_<JOBID>" \
  --output "$NANOCHAT_BASE_DIR/record_attempts/confirm_<JOBID>/DIGEST.txt"
```

A credible leaderboard claim should report:

- mean ± SEM `total_training_time`;
- mean ± SEM CORE;
- mean ± SEM validation BPB;
- all individual CORE values;
- minimum CORE;
- official timed training and true end-to-end wall time;
- exact git commit and package checksum.

## Optional DeltaAI probe

After installing the repo and preparing data on DeltaAI:

```bash
cp config/explore.env.example config/explore.env
cp config/site_deltaai.env.example config/site_deltaai.env
# edit paths/account/qos
bash slurm/submit_deltaai_probe.sh
```

This runs four GPUs with gradient accumulation 2. It can validate the clamp,
snapshot, merge, and BPB path. Its timing is not a leaderboard result.

## Stopping rules

- Do not tune on the six confirmation runs.
- Do not keep lowering the ratio until one stochastic run barely passes.
- Do not claim the snapshot-copy time as zero: NanoChat's official timer excludes
  it, but the package records the overhead so it can be disclosed.
- Do not compare a DeltaAI multi-node time to a single-node 8xH100 leaderboard.
- If the exploration run finds no CORE-qualified candidate, preserve the negative
  result and return to the conservative ratio-9.4 recipe.

## Files

- `nanochat/record_snapshots.py`: exact rank-zero state capture and manifest.
- `tools/apply_record_patch.py`: fail-closed upstream patcher.
- `record_tools/merge_record_snapshots.py`: output candidates.
- `record_tools/screen_candidates.py`: cheap BPB screen.
- `record_tools/eval_candidate.py`: canonical BPB+CORE evaluator.
- `record_tools/freeze_candidate.py`: predeclared freeze policy.
- `record_tools/summarize_confirmations.py`: final digest.
- `slurm/bridges2_*.sbatch`: official 8xH100 workflow.
- `slurm/deltaai_probe.sbatch`: non-official four-GPU probe.
