# D12 main-paper suite: scalar TSA × terminal floor


## DeltaAI Python compatibility

This revision selects a Python >=3.9 explicitly. The DeltaAI login-node `python3` may be an older system interpreter, while the existing NanoChat Slurm environment uses a uv-managed CPython. `verify_paper_main_d12_suite.sh` and all worker/merge jobs now use the same selected interpreter. You may override it with `PAPER_PYTHON=/path/to/python`.

This suite replaces the exploratory D12 plotting with one clean, predeclared experiment under NanoChat's native **Muon+AdamW** optimizer. It is designed to generate the first three experimental blocks of the main paper:

1. **Averaging alone:** Raw vs scalar TSA vs uniform LAWA on the baseline 5% terminal schedule.
2. **Clamp alone:** Raw 5% schedule vs Raw selected hotter terminal floor.
3. **Interaction:** 5%/selected-floor × Raw/TSA, showing whether averaging changes the preferred schedule.

The final D22 speedrun remains a separate frozen RunPod experiment after the D12 recipe is frozen.

## What is selected, and on which data

The cached D12 validation batches are partitioned once into four disjoint roles:

- `curve`: 32 batches. Used only for full and terminal-zoom figures.
- `alpha_select`: 64 batches. Used only to select scalar `alpha` on the baseline 5% schedule.
- `floor_select`: 64 batches. Used only to select the terminal LR floor after alpha has been frozen.
- `holdout`: 96 batches. Used only for headline endpoint tables and post-selection comparisons.

Selection is sequential:

1. Train all five schedule arms using the exact same D12 pack and data order.
2. On the **5% baseline only**, choose alpha from the predeclared grid using `alpha_select` BPB.
3. Freeze alpha.
4. Choose the floor from `{5, 10, 12.5, 15, 17.5}%` using `floor_select` BPB with that frozen alpha.
5. Report all headline endpoint results on untouched `holdout` batches.

This cleanly answers where both hyperparameters came from without selecting on the final reported data.

## Averaging estimators

The main method uses `K=8` checkpoints with `s=32` optimizer steps between checkpoints.

- **Raw:** alpha = 0.
- **TSA:** scalar alpha selected from the grid in `configs/paper_main_alpha_grid.txt`.
- **LAWA:** alpha = 1 on the same K=8, s=32 checkpoint mean.

Endpoint-only comparison rows also include:

- **Checkpoint EMA:** beta=0.95 over 16 checkpoints spaced 16 steps apart.
- **SWA-style late average:** equal average over 32 late checkpoints spaced 16 steps apart. We call this *SWA-style* because the training schedule itself is not the classical constant/cyclical SWA schedule.
- **Tensorwise adaptive TSA:** the historical `adaptive:hybrid:8:32:1.0:all` rule. It is evaluated for the appendix and is never allowed to influence the frozen scalar method.

The main plots remain visually simple: Raw, TSA, LAWA for averaging; two raw curves for clamp-only; four curves for the 2×2 interaction. EMA/SWA/adaptive appear in endpoint tables or appendix analyses rather than cluttering the main trajectories.

## Main figures produced

Every figure is emitted as both PDF and PNG with compact single-column dimensions, bold titles, and bold subtitles.

- `fig1_averaging_full.*`
- `fig1_averaging_zoom.*`
- `fig2_clamp_full.*`
- `fig2_clamp_zoom.*`
- `fig3_interaction_full.*`
- `fig3_interaction_zoom.*`

Hyperparameter provenance / appendix plots:

- `figS_alpha_selection.*`
- `figS_floor_selection.*`
- `figA_alpha_floor_interaction.*`

Tables:

- `table_averaging.csv/.tex`
- `table_clamp.csv/.tex`
- `table_interaction.csv/.tex`
- `appendix_adaptive_vs_scalar.csv`
- `endpoint_matrix.csv`

The merge writes `FROZEN.env` containing the selected alpha and terminal floor.

## Upload to DeltaAI

From your Mac, assuming the archive is in `${DOWNLOAD_DIR}`:

```bash
cd ${DOWNLOAD_DIR}
scp paper_main_d12_suite_v1.tar.gz \
    paper_main_d12_suite_v1.tar.gz.sha256 \
    ${USER}@dtai-login.delta.ncsa.illinois.edu:${PROJECT_ROOT}/
```

Log in:

```bash
ssh ${USER}@dtai-login.delta.ncsa.illinois.edu
cd ${PROJECT_ROOT}
```

Verify the transfer:

```bash
sha256sum -c paper_main_d12_suite_v1.tar.gz.sha256
```

Extract directly into the existing repository. The archive adds the new module/configs/scripts; it does not replace the existing NanoChat training core.

```bash
tar -xzf paper_main_d12_suite_v1.tar.gz
```

Run the preflight:

```bash
./verify_paper_main_d12_suite.sh "$PWD"
```

Expected ending:

```text
paper-main D12 selftest PASS
config preflight PASS floors=5 ...
paper-main D12 suite verification PASS
```

## Stage A: select alpha and floor, generate paper plots

Submit the five paired schedule arms:

```bash
ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
MAX_PARALLEL=4 \
GROUP_NAME=paper_main_d12_v1 \
bash slurm/submit_paper_main_d12.sh
```

Monitor:

```bash
squeue -u "$USER" -o '%.18i %.9P %.32j %.2t %.10M %.10l %R'
```

The array contains exactly five one-GPU trajectories, one per floor. The merge job runs after the array and performs the sequential selection and plotting.

If anything fails, inspect:

```bash
ls -lt logs/papd12* | head
cat logs/papd12_<ARRAY_JOB>_<TASK>.err
cat logs/papd12merge_<MERGE_JOB>.err
```

When the jobs finish:

```bash
ROOT=outputs/optimization_paper/paper_main_d12_v1
cat "$ROOT/figures/DIGEST.txt"
cat "$ROOT/FROZEN.env"
column -s, -t < "$ROOT/figures/table_averaging.csv"
column -s, -t < "$ROOT/figures/table_clamp.csv"
column -s, -t < "$ROOT/figures/table_interaction.csv"
ls -lh "$ROOT/figures"/*.pdf
```

The plots you want for Overleaf are already in:

```text
outputs/optimization_paper/paper_main_d12_v1/figures/
```

## Stage B: optional but recommended frozen confirmation

After Stage A has produced `FROZEN.env`, run five paired data-order repetitions of the baseline 5% schedule and the selected floor. Each repetition uses the same initialization and finite batch multiset, but a different permutation of optimizer-step batch blocks; the same permutation is used for both schedules within a repetition.

```bash
ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
MAX_PARALLEL=4 \
SOURCE_GROUP=paper_main_d12_v1 \
GROUP_NAME=paper_main_d12_confirm_v1 \
bash slurm/submit_paper_main_d12_confirmation.sh
```

When complete:

```bash
ROOT=outputs/optimization_paper/paper_main_d12_confirm_v1
cat "$ROOT/figures/DIGEST.txt"
column -s, -t < "$ROOT/figures/confirmation_summary.csv"
ls -lh "$ROOT/figures"/confirmed_*.pdf
```

These confirmation plots contain means and across-repetition standard-error bands:

- `confirmed_fig1_averaging_full/zoom.*`
- `confirmed_fig3_interaction_full/zoom.*`

The paper should call these **paired data-order repetitions**, not independent initialization seeds.

## Package results to send back

For Stage A:

```bash
cd ${PROJECT_ROOT}
GROUP_NAME=paper_main_d12_v1 \
OUT=paper_main_d12_v1_results.tar.gz \
bash pack_paper_main_d12_results.sh "$PWD"
```

Then copy back to your Mac:

```bash
scp ${USER}@dtai-login.delta.ncsa.illinois.edu:${PROJECT_ROOT}/paper_main_d12_v1_results.tar.gz* ${DOWNLOAD_DIR}/
```

For confirmation results, reuse the same packer with overrides:

```bash
GROUP_NAME=paper_main_d12_confirm_v1 \
OUT=paper_main_d12_confirm_v1_results.tar.gz \
bash pack_paper_main_d12_results.sh "$PWD"
```

## Interpretation gates

Do not force the desired narrative. Read the selected/holdout outputs literally:

- If baseline alpha selection chooses alpha=0, D12 does not support averaging on the new clean protocol.
- If it chooses alpha=1, ordinary LAWA is sufficient on D12; do not claim shrinkage is required there.
- If floor selection chooses 5%, the new clean protocol does not support a hotter D12 tail.
- The strongest intended result is an interior alpha plus a floor >5% where the raw endpoint worsens but TSA improves.
- `appendix_adaptive_vs_scalar.csv` determines whether tensorwise adaptive TSA is actually better than the scalar method. Keep it in the appendix unless the gain is both meaningful and reproducible.

## Why this suite is preferable to the old figures

The old exploratory plots mixed adaptive Muon-only estimators, adaptive all-tensor estimators, multiple checkpoint-window labels, and methods that were not part of the final paper definition. This suite uses the paper's actual scalar TSA definition in the main experiment, while still preserving EMA, SWA-style averaging, LAWA, and adaptive TSA as explicit comparators.

### DeltaAI interpreter note (v1.2)
The suite sources `slurm/_skip_env.sh` and then selects a Python interpreter that
actually imports `numpy`, `torch`, and `matplotlib`. It deliberately rejects bare
uv-managed CPython installations without the NanoChat environment packages.
