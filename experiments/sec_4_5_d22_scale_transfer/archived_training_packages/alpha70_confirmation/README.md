# D22 final alpha=0.70 confirmation bundle

This reruns **only the third row** needed for the intended main-text D22 table. The two raw-control rows are already complete on seeds 42/43/44 and are included as `existing_controls_n3.csv`.

The new runs are fixed before launch:

- seeds 42, 43, 44
- single acquired 8x NVIDIA H100 80GB HBM3 RunPod pod
- physical endpoint ratio 9.0 / step 10,172
- schedule calibration ratio 9.0
- terminal LR floor 15%
- TSA alpha **0.70**, frozen from the previous BPB-only alpha screen
- K=8, fixed physical checkpoints 9381, 9494, 9607, 9720, 9833, 9946, 10059, 10172
- canonical validation BPB and full DCLM CORE at alpha=.70
- **no new alpha sweep and no CORE-based model selection**

The supervisor automatically assembles the intended final paper comparison:

1. PR #830 early stop — calibration 9.4, no floor, raw
2. PR #830 direct 9.0 — calibration 9.0, no floor, raw
3. 15% floor + TSA — calibration 9.0, floor 15%, TSA alpha=.70

Outputs include `FRESH_RESULTS.csv`, `N3_SUMMARY.csv`, `FINAL_THREE_ROW_ALL_RUNS.csv`, `FINAL_THREE_ROW_SUMMARY.csv`, `FINAL_THREE_ROW_TABLE.tex`, and `evidence.tar.gz`.

## Launch

```bash
cd ${DOWNLOAD_DIR}
tar -xzf d22_runpod_alpha70_confirm_bundle_v1.tar.gz
cd d22_runpod_alpha70_confirm_bundle_v1
./start_d22_alpha70_background.sh
```

Monitor:

```bash
tail -f ${DOWNLOAD_DIR}/d22_runpod_alpha70_supervisor.log
```

On successful completion the pod is deleted. If a post-training failure occurs, it is stopped rather than deleted so the disk can be inspected.
