# Terminal Shrinkage Averaging

**Companion code and results for “Terminal Shrinkage Averaging Reveals a Schedule-Estimator Interaction in LLM Pretraining.”**

This repository is organized around the paper: experiment folders, result folders, and figure folders use the paper section or figure number in their names.

## Repository layout

```text
TerminalShrinkageAveraging/
├── experiments/
│   ├── sec_4_1_replicated_shrinkage/
│   ├── sec_4_3_schedule_estimator_interaction/
│   ├── sec_4_5_d22_scale_transfer/
│   ├── app_b_5_two_group_shrinkage/
│   ├── ...
│   ├── _shared/                  # shared analysis/training support code
│   └── _tools/                   # CPU result reconstruction and checks
├── figures/                      # manuscript excerpts and numerical replots
├── results/
│   ├── sec_*/ and app_*/         # paper-facing numerical exports
│   ├── _sources/                 # saved numerical records used by the analyses
│   └── _historical/              # earlier experiments not used as current-paper evidence
├── README.md
└── .gitignore
```

`sec_4_3` denotes Section 4.3, `app_b_5` denotes Appendix B.5, and names such as `fig_04` and `table_02` follow the manuscript numbering.

## Paper → experiment → results → figure

| Paper section and experiment | Experiment / CPU entrypoint | Results and tables | Figures |
|---|---|---|---|
| **4.1** — Replicated interior shrinkage optimum | [sec_4_1_replicated_shrinkage](experiments/sec_4_1_replicated_shrinkage/) | [Numerical exports](results/sec_4_1_replicated_shrinkage/) | [Fig. 2](figures/fig_02_sec_4_1_replicated_shrinkage/) |
| **4.2** — Raw final-iterate response to terminal activity | [sec_4_2_raw_terminal_activity](experiments/sec_4_2_raw_terminal_activity/) | [Numerical exports](results/sec_4_2_raw_terminal_activity/) | [Fig. 3](figures/fig_03_sec_4_2_raw_terminal_activity/) |
| **1; 4.3** — Schedule-estimator interaction | [sec_4_3_schedule_estimator_interaction](experiments/sec_4_3_schedule_estimator_interaction/) | [Table 1](results/sec_4_3_schedule_estimator_interaction/) | [Fig. 1](figures/fig_01_sec_1_sec_4_3_schedule_estimator_interaction/); [Fig. 4](figures/fig_04_sec_4_3_sec_4_4_optimizer_interactions/) |
| **4.4** — Pure-AdamW replication | [sec_4_4_adamw_replication](experiments/sec_4_4_adamw_replication/) | [Numerical exports](results/sec_4_4_adamw_replication/) | [Fig. 4](figures/fig_04_sec_4_3_sec_4_4_optimizer_interactions/) |
| **4.5; A.4** — Depth-22 matched-endpoint scale transfer | [sec_4_5_d22_scale_transfer](experiments/sec_4_5_d22_scale_transfer/) | [Table 2](results/sec_4_5_d22_scale_transfer/) | — |
| **B.1** — Scalar shrinkage across floors and optimizers | [app_b_1_shrinkage_across_floors](experiments/app_b_1_shrinkage_across_floors/) | [Table 3](results/app_b_1_shrinkage_across_floors/) | [Fig. 5](figures/fig_05_app_b_1_shrinkage_across_floors/) |
| **B.2** — Checkpoint-count and spacing sensitivity | [app_b_2_checkpoint_windows](experiments/app_b_2_checkpoint_windows/) | [Numerical exports](results/app_b_2_checkpoint_windows/) | [Fig. 6](figures/fig_06_app_b_2_checkpoint_windows/) |
| **B.3; B.4** — Alternative estimators and tensorwise adaptive TSA | [app_b_3_b_4_alternative_estimators](experiments/app_b_3_b_4_alternative_estimators/) | [Numerical exports](results/app_b_3_b_4_alternative_estimators/) | [Fig. 7](figures/fig_07_app_b_3_b_4_alternative_estimators/) |
| **B.5** — Initial two-group shrinkage and parameter localization | [app_b_5_two_group_shrinkage](experiments/app_b_5_two_group_shrinkage/) | [Table 4; Table 5](results/app_b_5_two_group_shrinkage/) | [Fig. 8](figures/fig_08_app_b_5_two_group_development/); [Fig. 9](figures/fig_09_app_b_5_parameter_localization/) |
| **B.6** — Parameter-role group search and frozen confirmation | [app_b_6_parameter_role_shrinkage](experiments/app_b_6_parameter_role_shrinkage/) | [Table 6](results/app_b_6_parameter_role_shrinkage/) | [Fig. 10](figures/fig_10_app_b_6_group_search/); [Fig. 11](figures/fig_11_app_b_6_group_confirmation/) |
| **B.7** — Groupwise terminal-floor mechanism probe | [app_b_7_groupwise_terminal_floors](experiments/app_b_7_groupwise_terminal_floors/) | [Table 7](results/app_b_7_groupwise_terminal_floors/) | — |
| **B.8** — Preliminary single-trajectory scalar and floor sweeps | [app_b_8_preliminary_sweeps](experiments/app_b_8_preliminary_sweeps/) | [Table 8; Table 9](results/app_b_8_preliminary_sweeps/) | [Fig. 12](figures/fig_12_app_b_8_preliminary_floor_sweep/) |
| **B.9** — Quadratic response of replicated scalar shrinkage | [app_b_9_quadratic_response](experiments/app_b_9_quadratic_response/) | [Table 10](results/app_b_9_quadratic_response/) | [Fig. 13](figures/fig_13_app_b_9_quadratic_response/) |

## Reproduce the saved-data analyses

From the repository root, using Python 3.11 or later:

```bash
python -m pip install -r experiments/_tools/requirements-analysis.txt
python experiments/_tools/reproduce.py
python experiments/_tools/audit_repository.py
python experiments/_tools/verify_groupwise_records.py
```

`reproduce.py` rebuilds the section-indexed CSV tables, Table 2 LaTeX, and standalone chart panels from the saved records under `results/_sources/`. It overwrites derived exports only. Use `--output-root /tmp/tsa-rebuild` to rebuild elsewhere or `--no-plots` for numerical exports only.

The verification commands are CPU-only and write local check reports under `.checks-local/`, which is ignored by Git. They do not train models, load model weights, submit scheduler jobs, or provision cloud resources.

For a single experiment, run its section entrypoint, for example:

```bash
python experiments/sec_4_3_schedule_estimator_interaction/analyze.py
python experiments/sec_4_5_d22_scale_transfer/analyze.py
python experiments/app_b_9_quadratic_response/analyze.py
```

Numerical replots reproduce the plotted quantities; they are not intended to reproduce every manuscript styling detail.
