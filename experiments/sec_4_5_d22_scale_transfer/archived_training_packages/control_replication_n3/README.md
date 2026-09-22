# D22 RunPod N=3 replication bundle (v2)

This bundle completes the current depth-22 table with **two fresh RunPod seeds (43 and 44)** while retaining the existing RunPod result as seed 42 / replicate 1.

**v2 fixes the Phase-4 nounset failure in v1** (`tag: unbound variable`) by separating Bash local declarations from dependent assignments. It also adds a pre-training path-construction preflight and makes supervisor failures report `FAILED` and return nonzero rather than printing `DONE`. The v1 failure occurred before any optimizer step began.

The supervisor acquires one **8 x NVIDIA H100 80GB HBM3** pod and, on that same pod, runs all six fresh cells unconditionally:

| Fresh seed | Configuration | Physical stop | Schedule calibration | Returned model |
|---|---|---:|---:|---|
| 43 | PR #830 early stop | ratio 9.0 / step 10,172 | ratio 9.4 | raw |
| 43 | PR #830 direct 9.0 | ratio 9.0 / step 10,172 | ratio 9.0 | raw |
| 43 | 15% floor + TSA | ratio 9.0 / step 10,172 | ratio 9.4 | TSA alpha 0.587 |
| 44 | same three cells | | | |

For the TSA cells it uses K=8 snapshots with exact 113-step spacing:

`9381, 9494, 9607, 9720, 9833, 9946, 10059, 10172`.

For **each returned model**, the job records:

- NanoChat accumulated training-iteration time (minutes), not Slurm/wall-clock duration;
- canonical validation BPB;
- full DCLM CORE (`max_per_task=-1`).

The treatment trajectory is merged to TSA and **only the returned TSA model gets the expensive canonical BPB+CORE evaluation**. The raw treatment endpoint is not CORE-evaluated because it is not a row in the paper table.

## Historical seed 42

`historical_seed42.csv` contains the three existing RunPod rows currently used in the paper:

- early stop: `77.09, 0.721556, 0.2522`
- direct 9.0: `77.10, 0.720258, 0.2500`
- 15% floor + TSA: `77.37, 0.720050, 0.2601`

These are stored at the precision currently printed in the paper. If higher-precision JSON for the historical run is available, replace the numeric values in `historical_seed42.csv` **before launch**; keep the seed and configuration names unchanged.

The pinned NanoChat commit uses initialization seed 42 by default. The fresh runs add an explicit `--experiment-seed` hook after NanoChat's compute initialization and audit the training logs for seed 43 or 44 before accepting a result.

## Start once and leave it alone

Unpack this directory somewhere convenient (Downloads is fine). In the shell where `RUNPOD_API_KEY` is exported:

```bash
cd ${DOWNLOAD_DIR}/d22_runpod_n3_bundle_v2
chmod +x *.sh
./start_d22_runpod_n3_background.sh
```

The launcher runs the supervisor under `nohup caffeinate -i`, so you may close the Terminal window after it starts.

Watch it with:

```bash
tail -f ${DOWNLOAD_DIR}/d22_runpod_n3_supervisor.log
```

or:

```bash
cd ${DOWNLOAD_DIR}/d22_runpod_n3_bundle_v2
./status_d22_runpod_n3.sh
```

The supervisor searches Secure and Community RunPod capacity. It refuses to start if an older D22 RunPod waiter is still running, to avoid accidentally acquiring two expensive pods.

## Cancellation

To cancel:

```bash
cd ${DOWNLOAD_DIR}/d22_runpod_n3_bundle_v2
./stop_d22_runpod_n3.sh
```

Cancellation **stops rather than deletes** an acquired pod, preserving its disk for inspection while capping GPU billing. Delete the stopped pod manually once you are sure you do not need its state.

On normal successful completion, the supervisor downloads the results and **deletes the pod automatically**.

## Outputs

The latest state is symlinked at:

```text
${DOWNLOAD_DIR}/d22_runpod_n3_latest
```

On success it contains at least:

- `FRESH_RESULTS.csv` — six new rows;
- `N3_ALL_RUNS.csv` — historical seed 42 + fresh 43/44 = nine rows;
- `N3_SUMMARY.csv` — mean, SD, SE, and 95% Student-t CI for Time, Val BPB, and CORE;
- `N3_TABLE.tex` — ready-to-edit LaTeX replacement for the current D22 table;
- `FINAL_SUMMARY.txt` — readable summary plus CORE qualification counts;
- `evidence.tar.gz` — compact audit evidence, logs, configuration diffs, model hashes and metadata.

The final N=3 interval uses the two-sided Student-t critical value for df=2.

## Failure behavior

The pod has a **16-hour server-side stop failsafe**. The six-run worker is detached from the local SSH connection. If a failure happens after training has begun, the supervisor stops the pod and preserves its disk rather than deleting available checkpoints. If the 16-hour failsafe itself is reached before completion, the supervisor does not silently restart the experiment; it retrieves whatever partial evidence it can and leaves the pod stopped.

All six cells are run regardless of CORE qualification. There is no conditional "skip the baseline" gate in this replication bundle.

## Trusted base package

The embedded `runpod_d22_causal_speedrun_v2.tar.gz` is checked before renting GPUs. Expected SHA-256:

```text
501afed8dacbaaf93117d3efe6e54d2836adca9423fdee0bc1b70646a37d9bb1
```
