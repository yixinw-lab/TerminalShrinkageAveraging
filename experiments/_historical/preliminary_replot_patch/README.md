# D12 joint-selection + publication replot patch

This patch does **no training**. It reads the five completed Stage-A `result.json` files under
`outputs/optimization_paper/paper_main_d12_v1/runs/`.

It:
- preserves Experiment 1 alpha selection on the dedicated baseline `alpha_select` split;
- jointly selects `(alpha, terminal floor)` using only the separate `floor_select` split;
- never selects on holdout;
- writes `JOINT_FROZEN.env` for the confirmation suite;
- exports actual curve data to CSV;
- regenerates compact full/terminal-zoom publication plots with bold titles/subtitles;
- writes an improved averaging table and joint-selection diagnostics.

Run from the repository root:

```bash
source slurm/_skip_env.sh
source slurm/_paper_python.sh
"$PAPER_PYTHON" replot_and_joint_select_d12.py \
  --root outputs/optimization_paper/paper_main_d12_v1
```

Results appear in:

```text
outputs/optimization_paper/paper_main_d12_v1/paper_figures_v2/
```

The expected joint optimum from the already-reported selection matrix is `10% floor, alpha=0.55`; the script recomputes it from the original `result.json` files and does not hard-code it.

To start the paired confirmation using the joint recipe:

```bash
ACCOUNT_OVERRIDE=bhji-dtai-gh \
QOS_OVERRIDE=bhji-dtai-gh \
MAX_PARALLEL=4 \
SOURCE_GROUP=paper_main_d12_v1 \
FROZEN="$PWD/outputs/optimization_paper/paper_main_d12_v1/paper_figures_v2/JOINT_FROZEN.env" \
GROUP_NAME=paper_main_d12_joint_confirm_v1 \
bash slurm/submit_paper_main_d12_confirmation.sh
```
