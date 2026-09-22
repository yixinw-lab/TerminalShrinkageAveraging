# Integrated D12 training implementation

This folder contains the local D12 implementation modules, configs, launchers, and analysis helpers used by the experiments in this repository. A complete from-scratch training environment additionally requires the matching upstream NanoChat checkout, tokenizer/data assets, and model or stream packs.

## Paper mapping

| Paper sections | Main module | Experiment-specific configs/launchers |
|---|---|---|
| 4.1–4.4; B.1; B.5; B.9 | `nanochat_meta/structured_tsa_d12.py` | `structured_tsa_*` |
| B.2–B.4 | `nanochat_meta/appendix_averaging_d12.py` | `appendix_averaging_*` |
| B.6 | `nanochat_meta/tsa_group_search_d12.py` | `tsa_group_search_*` |
| B.7 | `nanochat_meta/groupwise_floor_d12.py` | `groupwise_floor_*` |
| B.8 / earlier pack setup | `nanochat_meta/paper_main_d12.py`; `muon_aware_averaging.py` | `paper_main_d12_*`; `averaging_curve_*` |

`filter_suite`, `skip_core`, `skip_suite`, `skip_predictors`, `meta_experiment`, `online_meta`, and `trajectory_groups` provide local dependencies. Legacy module names are retained for import compatibility.

## Local source checks

After installing PyTorch and the analysis dependencies in a suitable environment, run from the repository root:

```bash
python experiments/_tools/check_training_sources.py
```

This imports the local implementation modules and runs the CPU self-tests included with the D12 code. It does not instantiate the full NanoChat model, load model packs, evaluate CORE, run optimizer steps, or submit jobs.

## Before a training rerun

`meta_experiment.build` lazily imports upstream NanoChat modules including `nanochat.common`, `gpt`, `tokenizer`, and `dataloader`; `skip_suite` also uses NanoChat evaluation and optimizer APIs. A full training rerun therefore requires a compatible NanoChat checkout and environment plus the corresponding tokenizer/data and pack inputs.

The saved run metadata under `results/_sources/` records model dimensions, snapshot steps, batch counts, and other experiment settings used by the analyses. Large model-weight files are intentionally not part of this repository.

For cluster use, copy `env.example.sh` to the Git-ignored `env.sh` and set the account, partition, NanoChat path, and output locations for your environment. Submission scripts request real compute jobs; they are not invoked by the CPU reproduction workflow.
