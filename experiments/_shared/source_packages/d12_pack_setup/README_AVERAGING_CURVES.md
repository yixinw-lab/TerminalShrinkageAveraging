# Optimizer × Averaging Loss-Curve Suite

This is the narrow experiment requested for the paper:

- optimizers: pure AdamW and native Muon+AdamW;
- one exact trajectory per optimizer/seed;
- post-hoc output estimators evaluated on the same trajectory;
- validation BPB curves rather than misleading duplicate training-loss curves.

Because all averaging methods reuse one exact trajectory, the experiment isolates the
output estimator and does not multiply training cost by the number of methods.

## Primary matrix

Main mode submits ten one-GPU runs:

| Optimizer | Seeds | Depth | Schedule |
|---|---:|---:|---|
| pure AdamW | 1337, 2337, 3337, 4337, 5337 | 12 | LR ×0.40, 65% warmdown, final LR 5% |
| native Muon+AdamW | 1337, 2337, 3337, 4337, 5337 | 12 | LR ×0.40, 65% warmdown, final LR 5% |

Every run evaluates:

- no averaging (`raw`);
- short uniform LAWA (`4 × 16`);
- long uniform LAWA (`8 × 32`);
- checkpoint EMA (`beta=0.95`, 16 snapshots spaced 4 steps);
- uniform Muon-only averaging (`8 × 32`);
- adaptive hybrid averaging over all tensors;
- adaptive hybrid averaging over Muon-managed tensors.

The two main figures contain five curves per optimizer. The Muon-only control and the
unused adaptive selector remain in the endpoint tables.

Intermediate curve anchors use 32 fixed validation batches. The final endpoint is
re-evaluated with 256 fixed validation batches. All seeds and methods share the same
batches within each run.

## Install

Upload `averaging_curve_suite_v1.tar.gz` to `${PROJECT_ROOT}`, then:

```bash
cd ${PROJECT_ROOT}

# The reference source uses this exact hash. Stop if yours differs unexpectedly.
sha256sum nanochat_meta/muon_aware_averaging.py
# fd80d7bcf17dcc9fa9e6cc270cb9d3bb2bd767876b59233947cd2792b9c14536

tar -xzf averaging_curve_suite_v1.tar.gz
./verify_averaging_curve_suite.sh "$PWD"
```

The package adds new submission/plotting files and makes two backward-compatible
changes:

1. intermediate and final evaluation batch counts can differ;
2. the small-results packer follows the NVMe output symlink.

## Confirm that the old result packer issue is fixed

`outputs/optimization_paper` is a symlink to the NVMe filesystem. The old script used
plain `find`, which did not traverse a command-line symlink. The replacement uses
`find -L`.

```bash
cd ${PROJECT_ROOT}
readlink -f outputs/optimization_paper
find -L outputs/optimization_paper -name result.json | head

OUT=optimization_paper_small_results.tar.gz \
  bash pack_optimization_paper_results.sh "$PWD"
```

This command does not include `.pt`, `.pth`, `.bin`, or `.safetensors` files.

## Smoke test first

This submits one D6 seed for each optimizer. It is only a wiring test.

```bash
cd ${PROJECT_ROOT}

ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
MODE=smoke \
GROUP_NAME=averaging_curves_smoke_v1 \
bash slurm/submit_averaging_curve_matrix.sh
```

Monitor:

```bash
squeue -u "$USER" \
  -o '%.18i %.9P %.32j %.2t %.10M %.10l %R'

tail -f logs/avgcurve_*.out
```

After the final plotting job finishes:

```bash
cat outputs/optimization_paper/averaging_curves_smoke_v1/figures/DIGEST.txt
ls -lh outputs/optimization_paper/averaging_curves_smoke_v1/figures/
```

Expected figure families:

```text
fig_adamw_averaging_loss_curves.*
fig_adamw_averaging_late_zoom.*
fig_adamw_averaging_gain_curves.*
fig_muon_averaging_loss_curves.*
fig_muon_averaging_late_zoom.*
fig_muon_averaging_gain_curves.*
```

The absolute loss curves establish the full optimization trajectory. The late-stage
zoom and paired gain curves make the small but potentially important differences
visible without distorting the full-range plot.

Do not submit main mode until both smoke trajectories contain `result.json` and the
plotting job succeeds.

## Main D12 five-seed experiment

```bash
cd ${PROJECT_ROOT}

ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
MODE=main \
GROUP_NAME=averaging_curves_d12_s5_v1 \
bash slurm/submit_averaging_curve_matrix.sh
```

This submits ten independent one-GPU trajectories and one final CPU plotting job.
The submission manifest is written immediately to:

```text
outputs/optimization_paper/averaging_curves_d12_s5_v1/SUBMISSION.txt
```

Check progress without restarting anything:

```bash
for root in outputs/optimization_paper/averaging_curves_d12_s5_v1/*_seed*; do
  [[ -d "$root" ]] || continue
  printf '%-58s %s\n' "$root" \
    "$(find -L "$root" -name result.json 2>/dev/null | wc -l) result.json"
done

sacct -S "$(date +%F)" \
  -o JobIDRaw,JobName%24,State,ExitCode,Elapsed,NodeList%24 \
  -X | tail -40
```

## Read the results

```bash
ROOT=outputs/optimization_paper/averaging_curves_d12_s5_v1
cat "$ROOT/figures/DIGEST.txt"
column -s, -t < "$ROOT/figures/final_summary.csv"
ls -lh "$ROOT/figures/"
```

Files produced:

- `curve_points.csv`: every optimizer/seed/method/anchor;
- `final_seed_results.csv`: final result for every individual seed;
- `final_summary.csv`: across-seed means and standard errors;
- `DIGEST.txt`: compact ranking;
- separate AdamW and Muon validation-BPB loss curves in PNG and PDF.

## Package only this experiment for download

```bash
cd ${PROJECT_ROOT}

RESULT_ROOT=outputs/optimization_paper/averaging_curves_d12_s5_v1 \
OUT=averaging_curves_d12_s5_v1_small_results.tar.gz \
  bash pack_optimization_paper_results.sh "$PWD"

ls -lh averaging_curves_d12_s5_v1_small_results.tar.gz
```

## Rerun protection

The matrix submitter refuses to use a nonempty group directory. This is intentional.
Do not set `RESUME=1` merely to bypass the check. First inspect `SUBMISSION.txt`,
`squeue`, `sacct`, and the existing result directories.

## Interpretation

The decisive comparisons are within optimizer and seed:

```text
raw vs LAWA
raw vs EMA
raw vs adaptive
adaptive vs the strongest scalar average
```

The optimizer interaction is secondary:

```text
(adaptive gain under Muon) - (adaptive gain under AdamW)
```

The plotted curves are validation BPB curves. Post-hoc averaging methods do not have
separate training-loss curves because they share exactly the same training trajectory.
