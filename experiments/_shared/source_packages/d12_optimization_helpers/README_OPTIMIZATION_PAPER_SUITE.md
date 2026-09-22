# Optimization Paper Suite v1

This package runs two programs in parallel:

1. **Paper program:** explain and improve the trajectory-averaging effect with
   Muon-aware group attribution, causal tensorwise averaging, a live
   state-consistent filter, and current-batch verified fast-forward proposals.
2. **Leaderboard program:** tune LR × terminal averaging on canonical NanoChat,
   then add lightweight late snapshots to the current speedrun recipe.

It depends on the already-installed `nanochat_meta` foundation from the earlier
suites (`skip_core.py`, `skip_suite.py`, `filter_suite.py`, and
`causal_fastforward.py`). Extraction only adds new modules and scripts.

## What each experiment answers

### `muon_aware_averaging`

One exact trajectory is trained per learning-rate/schedule configuration. Many
post-hoc recipes are evaluated on the *same* trajectory:

- uniform LAWA,
- EMA and recency-weighted temporal baselines,
- Muon-only versus AdamW-only averaging,
- attention versus MLP and early/middle/late layers,
- causal tensorwise strengths inferred from update autocorrelation and
  noise-to-drift estimates.

This is the cheapest and cleanest mechanism/attribution experiment.

### `muon_selective_filter`

A small causal pull toward a recent average is applied to selected parameter
groups during live training. First-moment state can be preserved, reset, or
damped. The filter turns off before the run ends, so the report distinguishes
transient smoothing from durable optimization progress.

### `forward_probe_fastforward`

A history model proposes a multi-step displacement. A current-batch forward
probe either rejects it or accepts it without a backward pass. `qf` additionally
fits a one-dimensional quadratic from a few observed forward losses. Every
forward probe is charged in the FLOP ledger.

### `speedrun/`

Creates a copy of NanoChat's base trainer with lightweight model-only late
snapshots, and averages those snapshots into a standard NanoChat checkpoint.
The upstream `base_train.py` and `runs/speedrun.sh` are left untouched.

---

# Install

Upload the tarball to `${PROJECT_ROOT}`, then:

```bash
cd ${PROJECT_ROOT}
tar -xzf optimization_paper_suite_v1.tar.gz
./verify_optimization_paper_install.sh "$PWD"
```

Expected ending:

```text
[selftest] muon-aware averaging PASS
[selftest] muon selective filter PASS
[selftest] forward-probe fast-forward PASS
optimization paper suite verification PASS
```

# Put heavy outputs on `bhji` NVMe

```bash
cd ${PROJECT_ROOT}

ACTIVE=/work/nvme/bhji/$USER/ExtrapProj/outputs/optimization_paper
mkdir -p "$ACTIVE" "$PWD/outputs"

if [[ -e "$PWD/outputs/optimization_paper" && ! -L "$PWD/outputs/optimization_paper" ]]; then
  mv "$PWD/outputs/optimization_paper" "$ACTIVE.previous.$(date +%s)"
fi

ln -sfn "$ACTIVE" "$PWD/outputs/optimization_paper"
ls -ld "$PWD/outputs/optimization_paper"
```

---

# Smoke tests

## 1. Offline Muon-aware averaging smoke

```bash
cd ${PROJECT_ROOT}

ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
RUN_NAME=muavg_smoke \
TRAJECTORIES_FILE="$PWD/configs/trajectory_configs_smoke.txt" \
RECIPES_FILE="$PWD/configs/averaging_recipes_smoke.txt" \
OPTIMIZER=native \
DEPTH=6 \
PREFIX_STEPS=80 \
PREFIX_LR_SCALE=0.30 \
COMPUTE_BUDGET=30 \
STREAM_EQUIV=80 \
TOTAL_ITERATIONS=300 \
RECIPE_EVAL_EQUIVS=20,30 \
EVAL_BATCHES=8 \
EVAL_EVERY_EQUIV=5 \
SNAPSHOT_EVERY=1 \
MAX_PARALLEL=1 \
bash slurm/submit_muon_avg_suite.sh
```

## 2. Live selective-filter smoke

After `outputs/optimization_paper/muavg_smoke/pack.pt` exists:

```bash
ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
RUN_NAME=mufilter_smoke \
REUSE_PACK="$PWD/outputs/optimization_paper/muavg_smoke/pack.pt" \
ARMS_FILE="$PWD/configs/muon_filter_arms_smoke.txt" \
COMPUTE_BUDGET=30 \
BASE_LR_SCALE=0.30 \
FILTER_START_EQUIV=5 \
FILTER_STOP_EQUIV=20 \
EVAL_EVERY_EQUIV=5 \
SNAPSHOT_EVERY=1 \
MAX_PARALLEL=4 \
bash slurm/submit_muon_filter_suite.sh
```

## 3. Forward-probe smoke

```bash
ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
RUN_NAME=fprobe_smoke \
REUSE_PACK="$PWD/outputs/optimization_paper/muavg_smoke/pack.pt" \
ARMS_FILE="$PWD/configs/forward_probe_arms_smoke.txt" \
COMPUTE_BUDGET=30 \
BASE_LR_SCALE=0.30 \
JUMP_START_EQUIV=5 \
JUMP_STOP_EQUIV=20 \
EVAL_EVERY_EQUIV=5 \
LAWA_WINDOW=32 \
MAX_PARALLEL=4 \
bash slurm/submit_forward_probe_suite.sh
```

Inspect:

```bash
for run in muavg_smoke mufilter_smoke fprobe_smoke; do
  echo "===== $run ====="
  find "outputs/optimization_paper/$run" -name result.json | wc -l
  cat "outputs/optimization_paper/$run/validation.json" 2>/dev/null || true
  column -s, -t < "outputs/optimization_paper/$run/leaderboard.csv" 2>/dev/null | head -20 || true
done
```

Do not launch the scientific arrays if a smoke arm crashes or equal-compute
spread is unexpectedly large.

---

# Parallel scientific wave

The recommended first wave runs the paper experiments and leaderboard proxy in
parallel. The native optimizer is the main paper setting; pure AdamW is the
control.

## A. Offline mechanism and attribution — native

This starts from initialization so each schedule is internally coherent.

```bash
cd ${PROJECT_ROOT}

ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
RUN_NAME=paper_muavg_native_d6 \
TRAJECTORIES_FILE="$PWD/configs/trajectory_configs_mechanism.txt" \
RECIPES_FILE="$PWD/configs/averaging_recipes_mechanism.txt" \
OPTIMIZER=native \
DEPTH=6 \
PREFIX_STEPS=0 \
COMPUTE_BUDGET=3600 \
STREAM_EQUIV=3800 \
TOTAL_ITERATIONS=3600 \
RECIPE_EVAL_EQUIVS=2400,3000,3600 \
EVAL_BATCHES=256 \
EVAL_EVERY_EQUIV=200 \
SNAPSHOT_EVERY=4 \
MAX_PARALLEL=4 \
bash slurm/submit_muon_avg_suite.sh
```

## B. Offline mechanism control — pure AdamW

```bash
ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
RUN_NAME=paper_muavg_adam_d6 \
TRAJECTORIES_FILE="$PWD/configs/trajectory_configs_mechanism.txt" \
RECIPES_FILE="$PWD/configs/averaging_recipes_mechanism.txt" \
OPTIMIZER=pure_adamw \
DEPTH=6 \
PREFIX_STEPS=0 \
COMPUTE_BUDGET=3600 \
STREAM_EQUIV=3800 \
TOTAL_ITERATIONS=3600 \
RECIPE_EVAL_EQUIVS=2400,3000,3600 \
EVAL_BATCHES=256 \
EVAL_EVERY_EQUIV=200 \
SNAPSHOT_EVERY=4 \
MAX_PARALLEL=4 \
bash slurm/submit_muon_avg_suite.sh
```

The comparison to look for is not merely "native averages better." It is
whether the marginal gain is concentrated in `group:muon`, whether attention or
MLP matrices dominate, and whether the causal adaptive recipes beat the best
uniform/EMA recipe on a frozen trajectory.

## C. Live Muon-aware optimizer experiment

This uses a shared production-like prefix at LR ×0.30, applies the filter in the
middle of the branch, then gives every arm 600 exact equivalents of
continuation after the filter turns off.

```bash
ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
RUN_NAME=paper_mufilter_native_d6 \
ARMS_FILE="$PWD/configs/muon_filter_arms_mechanism.txt" \
OPTIMIZER=native \
DEPTH=6 \
PREFIX_STEPS=1800 \
PREFIX_LR_SCALE=0.30 \
TOTAL_ITERATIONS=3600 \
PACK_WARMDOWN_RATIO=0.65 \
PACK_FINAL_LR_FRAC=0.05 \
COMPUTE_BUDGET=1800 \
BASE_LR_SCALE=0.30 \
BRANCH_WARMDOWN_RATIO=0.65 \
BRANCH_FINAL_LR_FRAC=0.05 \
FILTER_START_EQUIV=100 \
FILTER_STOP_EQUIV=1200 \
STREAM_EQUIV=2200 \
EVAL_BATCHES=256 \
EVAL_EVERY_EQUIV=100 \
SNAPSHOT_EVERY=4 \
MAX_PARALLEL=8 \
bash slurm/submit_muon_filter_suite.sh
```

Run the AdamW control with the same command and:

```text
RUN_NAME=paper_mufilter_adam_d6
OPTIMIZER=pure_adamw
```

## D. Forward-verified descendant of fast-forwarding

Reuse the matched native prefix from C once its pack is complete:

```bash
ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
RUN_NAME=paper_fprobe_native_d6 \
REUSE_PACK="$PWD/outputs/optimization_paper/paper_mufilter_native_d6/pack.pt" \
ARMS_FILE="$PWD/configs/forward_probe_arms_screen.txt" \
COMPUTE_BUDGET=1800 \
BASE_LR_SCALE=0.30 \
BRANCH_WARMDOWN_RATIO=0.65 \
BRANCH_FINAL_LR_FRAC=0.05 \
JUMP_START_EQUIV=100 \
JUMP_STOP_EQUIV=1200 \
EVAL_EVERY_EQUIV=100 \
LAWA_WINDOW=225 \
MAX_PARALLEL=8 \
bash slurm/submit_forward_probe_suite.sh
```

Run the pure-Adam control by reusing the Adam filter pack.

---

# Leaderboard program in parallel

## D12 production-schedule proxy

This is a scientific proxy on the GH200 setup, not an official leaderboard run.
It searches LR, warmdown, uniform LAWA, Muon-only averaging, and causal adaptive
averaging on canonical d12.

```bash
ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
RUN_NAME=leaderboard_muavg_d12 \
TRAJECTORIES_FILE="$PWD/configs/trajectory_configs_d12.txt" \
RECIPES_FILE="$PWD/configs/averaging_recipes_leaderboard.txt" \
OPTIMIZER=native \
DEPTH=12 \
MAX_SEQ_LEN=512 \
DEVICE_BATCH_SIZE=4 \
GRAD_ACCUM=16 \
PREFIX_STEPS=0 \
COMPUTE_BUDGET=3000 \
STREAM_EQUIV=3200 \
TOTAL_ITERATIONS=3000 \
RECIPE_EVAL_EQUIVS=2400,2700,3000 \
EVAL_BATCHES=128 \
EVAL_EVERY_EQUIV=200 \
SNAPSHOT_EVERY=4 \
MAX_PARALLEL=4 \
bash slurm/submit_muon_avg_suite.sh
```

If d12 confirms that LR+LAWA and/or a Muon-aware recipe transfers, freeze the
best recipe and replicate it across seeds before spending on the official d24
run.

## Lightweight snapshots in the actual NanoChat speedrun

Run this in the NanoChat checkout. It creates copies and leaves upstream files
unchanged.

```bash
cd "$NANOCHAT_ROOT"

python ${PROJECT_ROOT}/speedrun/install_speedrun_lawa_hook.py \
  --repo "$NANOCHAT_ROOT"

SNAP=/work/nvme/bhji/$USER/nanochat_speedrun_lawa/snapshots
mkdir -p "$SNAP"

python ${PROJECT_ROOT}/speedrun/create_speedrun_lawa_script.py \
  --repo "$NANOCHAT_ROOT" \
  --snapshot-every 16 \
  --snapshot-start-frac 0.80 \
  --snapshot-keep 64 \
  --snapshot-dtype bfloat16 \
  --snapshot-dir "$SNAP"

python -m py_compile scripts/base_train_lawa.py
bash -n runs/speedrun_lawa.sh
```

On the official 8×H100 environment:

```bash
WANDB_RUN=speedrun_lawa bash runs/speedrun_lawa.sh
```

One training run can produce many terminal models without retraining:

```bash
BASE=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}

python ${PROJECT_ROOT}/speedrun/average_lawa_snapshots.py \
  --snapshot-dir "$SNAP" --window 8 --spacing 1 --selector all \
  --output-dir "$BASE/base_checkpoints/d24_lawa_all_s16" --output-step 999901

python ${PROJECT_ROOT}/speedrun/average_lawa_snapshots.py \
  --snapshot-dir "$SNAP" --window 8 --spacing 2 --selector all \
  --output-dir "$BASE/base_checkpoints/d24_lawa_all_s32" --output-step 999902

python ${PROJECT_ROOT}/speedrun/average_lawa_snapshots.py \
  --snapshot-dir "$SNAP" --window 8 --spacing 2 --selector muon \
  --output-dir "$BASE/base_checkpoints/d24_lawa_muon_s32" --output-step 999903
```

Evaluate BPB first, then full CORE for the finalists:

```bash
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
  --model-tag d24_lawa_all_s32 --step 999902 --device-batch-size 16 --eval bpb

torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- \
  --model-tag d24_lawa_all_s32 --step 999902 --device-batch-size 16 --eval core,bpb
```

Report both NanoChat's official measured `total_training_time` and true
end-to-end snapshot/merge overhead.

---

# Read results while arrays run

```bash
for root in outputs/optimization_paper/*; do
  [[ -d "$root" ]] || continue
  printf '%-45s %4d results\n' "$(basename "$root")" \
    "$(find "$root" -name result.json 2>/dev/null | wc -l)"
done | sort
```

After merge:

```bash
for run in \
  paper_muavg_native_d6 paper_muavg_adam_d6 \
  paper_mufilter_native_d6 paper_fprobe_native_d6 \
  leaderboard_muavg_d12; do
  root="outputs/optimization_paper/$run"
  echo "===== $run ====="
  cat "$root/DIGEST.txt" 2>/dev/null || true
  cat "$root/validation.json" 2>/dev/null || true
  column -s, -t < "$root/leaderboard.csv" 2>/dev/null | head -30 || true
done
```

Package only small outputs for analysis:

```bash
bash pack_optimization_paper_results.sh "$PWD"
```

---

# Go/no-go rules

## Muon-aware averaging survives if

- Muon-only or a causal adaptive recipe beats the best uniform LAWA/EMA recipe
  on frozen configurations and independent seeds, or
- it matches uniform LAWA while averaging materially fewer parameters and
  reduces snapshot/merge overhead, with a stable mechanism across depths.

## Live filtering survives if

- it beats tuned exact after the exact-continuation interval, not only while the
  filter is active.

## Forward-verified fast-forward survives if

- accepted proposals reduce exact backward calls,
- `advance` beats matched `hold`,
- and the result beats tuned LR + LAWA at equal charged FLOPs and wall time.

Otherwise, retain uniform LAWA for the leaderboard and treat the paper tracks
as negative mechanistic results rather than forcing a claim.
