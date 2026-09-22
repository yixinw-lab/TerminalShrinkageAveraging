# D12 structured TSA + AdamW confirmation suite

This package runs the next decision-critical experiment for the paper. It does **not** replace the completed checkpoint-window, EWA, EMA, LAWA, SWA-style, or adaptive-TSA appendix suite. It adds a clean development/confirmation test of low-dimensional structured shrinkage and reruns the AdamW diagnostic with genuinely different data-order streams.

## Scientific design

The terminal estimator is

\[
\hat\theta
=
\theta_T
+
\alpha_G P_G(\bar\theta_T-\theta_T)
+
\alpha_O P_O(\bar\theta_T-\theta_T),
\]

where `G` is the coordinate set managed by Muon in NanoChat's native optimizer and `O` is its complement. The same structural partition is reused in pure-AdamW runs, even though every parameter is then trained with AdamW. This avoids defining an empty "Muon group" in the AdamW condition and separates tensor role from the optimizer actually used.

The package runs:

- **Development:** 2 optimizers × 5 stream seeds at the 10% terminal floor = 10 trajectories.
- **Confirmation:** 2 optimizers × 3 floors `{5%, 10%, 15%}` × 5 paired stream seeds = 30 trajectories.
- **Total:** 40 independent training jobs, each on one GPU.

Development trajectories evaluate:

- a structured `9 × 9` grid, with each alpha in `{0, .125, .25, .375, .5, .625, .75, .875, 1}`;
- a scalar grid from `0` to `1` in increments of `.05`.

A freeze job selects one structured rule and one scalar rule for each optimizer using **only** the development trajectories and the calibration validation block. Confirmation trajectories are trained in parallel but save only the terminal endpoint and checkpoint mean. Their holdout evaluation cannot begin until `FROZEN_RULES.json` exists.

The primary decision gate is the paired native-optimizer contrast at the 10% floor:

\[
\operatorname{BPB}(\text{development-selected scalar TSA})
-
\operatorname{BPB}(\text{development-selected structured TSA}).
\]

Positive values favor structured TSA. The final digest recommends a paper pivot only when the development optimum is off the scalar diagonal and the frozen structured rule confirms across the separate streams.

All runs use `T=3000`, `K=8`, checkpoint spacing `s=32`, model seed `1337`, and explicit stream-seed permutations. The five confirmation stream seeds are paired across optimizer and floor arms.

## Expected existing assets

By default, the suite uses the packs already produced by the earlier project:

```text
outputs/optimization_paper/averaging_curves_d12_s5_v1/native_seed1337/pack.pt
outputs/optimization_paper/averaging_curves_d12_s5_v1/pure_adamw_seed1337/pack.pt
```

The smoke job verifies that the packs have the same model dimensionality, sampled initial parameters, and sampled training/evaluation data fingerprints. It also reconstructs the native Muon-managed flattened-coordinate ranges and writes an immutable `GROUP_SCHEMA.json` before any training begins. A mismatch stops the suite before the arrays launch, because the cross-optimizer comparison is intended to be paired in initialization and data.

The AdamW experiment is a within-recipe estimator and schedule diagnostic. The package records the exact pure-AdamW pack metadata; do not use absolute native-versus-AdamW BPB as an optimizer leaderboard claim unless that AdamW recipe's tuning provenance is independently documented.

## Install on DeltaAI

Copy the archive and checksum into `${PROJECT_ROOT}`, then run:

```bash
cd ${PROJECT_ROOT}
sha256sum -c d12_structured_tsa_suite_v1.tar.gz.sha256
tar -xzf d12_structured_tsa_suite_v1.tar.gz

./discover_structured_tsa_assets.sh "$PWD"
./verify_structured_tsa_d12_suite.sh "$PWD"
```

The package uses the existing project environment helpers:

```text
slurm/_skip_env.sh
slurm/_paper_python.sh
sub.sh
```

If either pack is in a different location, pass its path when submitting:

```bash
PACK_NATIVE=/path/to/native/pack.pt \
PACK_ADAMW=/path/to/pure_adamw/pack.pt \
...submission command...
```

## Submit all D12 work

This launches development and confirmation training arrays concurrently after one short smoke job:

```bash
cd ${PROJECT_ROOT}

ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
GROUP_ROOT=/work/nvme/bhji/$USER/ExtrapProj/outputs/optimization_paper/d12_structured_tsa_v1 \
DEV_MAX_PARALLEL=10 \
CONFIRM_MAX_PARALLEL=30 \
EVAL_MAX_PARALLEL=30 \
GROUP_NAME=d12_structured_tsa_v1 \
bash slurm/submit_structured_tsa_d12.sh
```

The array limits express the desired parallelism. Slurm and the account QOS may run fewer jobs concurrently; no scientific assumption depends on simultaneous execution.
The training tasks request two hours each; prior D12 jobs were substantially shorter, while two hours stays within the current documented production-job limit.

The explicit `GROUP_ROOT` keeps the roughly 90--100 GB of terminal artifacts off the home filesystem. Change only the allocation code/path if your work allocation is mounted elsewhere.

The dependency graph is:

```text
smoke
  ├── development training array ──> freeze rules ──┐
  └── confirmation training array ──────────────────┼──> confirmation evaluation array ──> merge
                                                    ┘
```

Thus, expensive training runs in parallel, while the confirmation estimators remain genuinely frozen.

## Monitor

```bash
squeue -u "$USER" -o '%.18i %.9P %.32j %.2t %.10M %.10l %R'
```

Check the smoke output before assuming the main arrays are healthy:

```bash
ROOT=/work/nvme/bhji/$USER/ExtrapProj/outputs/optimization_paper/d12_structured_tsa_v1
cat "$ROOT/SMOKE.txt"
cat "$ROOT/SUBMISSION.txt"
```

If the smoke job cannot identify the native group partition, it fails before any training and writes:

```text
GROUP_PROBE_infos.json
logs/d12stsmoke_<jobid>.err
logs/d12stsmoke_<jobid>.out
```

Those files are the right debugging artifacts to inspect rather than modifying the experiment blindly.

## Results

After the dependent merge job finishes:

```bash
ROOT=/work/nvme/bhji/$USER/ExtrapProj/outputs/optimization_paper/d12_structured_tsa_v1
cat "$ROOT/FROZEN_RULES.txt"
cat "$ROOT/figures/DIGEST.txt"

column -s, -t < "$ROOT/figures/paired_contrasts_summary.csv"
column -s, -t < "$ROOT/figures/schedule_effects_summary.csv"
column -s, -t < "$ROOT/figures/optimizer_interactions_summary.csv"
```

The main files are:

```text
GROUP_SCHEMA.json
FROZEN_RULES.json
FROZEN_RULES.sha256
development_surface_summary.csv
figures/DIGEST.txt
figures/primary_summary.csv
figures/paired_contrasts_summary.csv
figures/schedule_effects_summary.csv
figures/optimizer_interactions_summary.csv
figures/confirmation_surface_summary.csv
figures/development_surface_native.pdf
figures/confirmation_surface_native_floor10.pdf
figures/structured_vs_scalar_per_seed.pdf
figures/native_schedule_effects.pdf
```

Each training directory also contains `result.json`, `training_trace.csv`, and `terminal_artifact.pt`. The artifact contains two float32 flattened vectors—the final endpoint and the `K=8, s=32` checkpoint mean—so storage is approximately `8 × parameter_count` bytes per trajectory, plus small metadata. Keep the output root on project/work storage rather than relying on home-directory I/O.

## Package compact results for review

The default results archive omits the large terminal vectors while retaining all statistics, manifests, traces, frozen rules, and figures:

```bash
cd ${PROJECT_ROOT}
GROUP_NAME=d12_structured_tsa_v1 \
GROUP_ROOT=/work/nvme/bhji/$USER/ExtrapProj/outputs/optimization_paper/d12_structured_tsa_v1 \
bash pack_structured_tsa_d12_results.sh "$PWD"
```

This creates:

```text
d12_structured_tsa_v1_compact_results.tar.gz
d12_structured_tsa_v1_compact_results.tar.gz.sha256
```

Set `INCLUDE_ARTIFACTS=1` only when a full archival copy of the endpoint vectors is needed.

## How the result changes the paper

- **Strong confirmation:** the native development optimum is off-diagonal, the frozen structured rule beats the frozen scalar rule on at least four of five 10%-floor streams, and the paired 95% interval is above zero. Recenter the method around structured TSA and transfer the frozen native pair to D22 without retuning.
- **Suggestive confirmation:** the mean and signs favor structure but the interval includes zero. Keep scalar TSA as the main method unless D22 gives a clean structured advantage; report groupwise shrinkage as a promising extension.
- **No confirmation:** retain the current scalar-TSA paper and use the structured surfaces as a negative diagnostic. Do not select a favorable confirmation-grid point after the fact.

The 15% confirmation arm is included to bridge directly to the present D22 recipe. The AdamW arms test whether group heterogeneity persists when the same tensor-role partition is trained entirely with AdamW.
