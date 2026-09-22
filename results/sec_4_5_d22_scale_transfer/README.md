# Section 4.5; A.4 — Depth-22 matched-endpoint scale transfer

**Paper items:** Table 2.

The saved depth-22 run summaries and the paper-facing numerical exports are included here.

| File | Contents |
|---|---|
| [table_02_sec_4_5_d22_scale_transfer.csv](table_02_sec_4_5_d22_scale_transfer.csv) | Means and 95% Student-t interval half-widths across seeds |
| [table_02_sec_4_5_time_bpb_and_control_core_per_seed.csv](table_02_sec_4_5_time_bpb_and_control_core_per_seed.csv) | Training time, validation BPB, and CORE for every configuration and seed |
| [table_02_sec_4_5_d22_scale_transfer.tex](table_02_sec_4_5_d22_scale_transfer.tex) | LaTeX table |

[Experiment and analysis instructions](../../experiments/sec_4_5_d22_scale_transfer/) · [Source records](../../results/_sources/d22_runs/)

These files are derived exports, not raw logs. All three configurations include observed CORE scores for seeds 42, 43, and 44. Treatment scores are read from the saved `tsa_core.json` files; control scores are read from `existing_controls_n3.csv`.

Table 2 reports means and 95% Student-t intervals across the three seeds separately for each configuration. These are intervals for configuration means, not paired contrasts.
