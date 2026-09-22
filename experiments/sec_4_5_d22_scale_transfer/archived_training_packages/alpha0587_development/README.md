# D22 missing-row RunPod bundle v1

Runs the missing matched-endpoint cell on one acquired 8xH100 RunPod node:

- seeds 42, 43, 44
- physical endpoint ratio 9.0 / step 10,172
- schedule calibration ratio 9.0
- terminal LR floor 15%
- primary returned estimator frozen at TSA alpha 0.587
- K=8 with the same physical checkpoint window as the existing treatment: 9381, 9494, 9607, 9720, 9833, 9946, 10059, 10172
- canonical validation BPB and full DCLM CORE on the primary alpha=.587 output

## Alpha diagnostic

The same three trajectories also screen this predeclared alpha grid on 4,194,304 BPB tokens:

`0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.587, 0.60, 0.65, 0.70, 0.75, 0.80, 1.00`

This screen is descriptive only. It does not change the primary missing row. The bundle reports the pooled discrete screen minimizer, per-seed minimizers, and leave-one-seed-out choices.

Launch:

```bash
cd ${DOWNLOAD_DIR}
tar -xzf d22_runpod_missing_row_bundle_v2.tar.gz
cd d22_runpod_missing_row_bundle_v2
./start_d22_missing_row_background.sh
```

Watch with `tail -f ${DOWNLOAD_DIR}/d22_runpod_missing_row_supervisor.log`.


## v2 fix

v1 could self-abort immediately after launch because the supervisor's duplicate-waiter check matched its own `caffeinate ... runpod_d22_missing_row_supervisor...` parent process. v2 removes the self-matching pattern from the in-supervisor check; the launcher still performs the full duplicate-controller check before spawning the supervisor. No experiment or pod was started by that v1 failure.
