# D22 causal speedrun suite v2

This suite implements the three-run causal design for the paper.

The scientific comparison is:

1. `og_full_control`: unmodified PR #830 recipe at ratio 9.4, no clamp, raw output. This calibrates the machine and reproduces the reference speedrun.
2. `baseline_x`: direct run to ratio X with the schedule calibrated to X, no clamp, raw output.
3. `tsa_clamp_x`: direct run to the same ratio X with the schedule calibrated to X, a 15% terminal LR floor, and TSA alpha 0.60. The treatment run evaluates BOTH the raw endpoint and the TSA output, so the raw output is a same-trajectory control at no additional training cost.

The main causal target is an X for which baseline_x raw is below CORE 0.256525 while tsa_clamp_x TSA is above it. The OG 9.4 control shows that the local machine reproduces the public recipe and gives the reference wall-clock time.

## Recommended exploration protocol

Start at X=9.0 because the earlier exploratory D22 trajectory showed a qualifying partially averaged output there, while the directly calibrated schedule remains untested. Run the treatment first. If TSA fails, increase X to 9.2. If TSA passes, run the baseline at the same X. If the baseline also passes, decrease X to 8.8. Once a bracket is found, refine by 0.1 if useful. Do not call an exploratory X the final causal result. Freeze X, then rerun baseline_x and tsa_clamp_x fresh for the paper. CORE is noisy, so repeat a borderline pair.

## On the pod

After uploading/extracting the suite:

```bash
./verify_suite.sh
bash runpod/setup_once.sh
```

Run the full calibration control once:

```bash
bash runpod/run_full_control.sh
```

Run a candidate treatment, e.g. X=9.0:

```bash
bash runpod/run_treatment_x.sh 9.0
```

If TSA qualifies, run the matched direct baseline:

```bash
bash runpod/run_baseline_x.sh 9.0
```

Inspect all results:

```bash
bash runpod/show_results.sh
```

Each run writes a small `<RUN_TAG>_results.tar.gz` and checksum under `${D22_ROOT}/results/`.

By default large model checkpoints and treatment snapshots are deleted after canonical evaluation. Set `KEEP_CHECKPOINTS=1` only if you explicitly need to retain them.
