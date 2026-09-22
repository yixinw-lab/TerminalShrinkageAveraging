#!/usr/bin/env python3
"""Create scripts/base_train_lawa.py with lightweight late model-only snapshots.

The patch is deliberately text-marker based and leaves scripts/base_train.py
untouched. It targets current NanoChat master, where parser --save-every,
checkpoint_dir, and the '# save checkpoint:' comment are stable markers.
"""
from __future__ import annotations
from pathlib import Path
import argparse

PARSER_MARK = 'parser.add_argument("--save-every", type=int, default=-1, help="save checkpoints every N steps (-1 = only at end)")'
CHECKPOINT_MARK = 'checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)'
SAVE_MARK = '# save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step'

ARGS_BLOCK = '''\n# Lightweight model-only snapshots for post-hoc LAWA / groupwise averaging.\nparser.add_argument("--lawa-snapshot-every", type=int, default=-1, help="save rank-0 model-only snapshots every N late steps")\nparser.add_argument("--lawa-snapshot-start-frac", type=float, default=0.85, help="fraction of training after which lightweight snapshots begin")\nparser.add_argument("--lawa-snapshot-keep", type=int, default=24, help="maximum lightweight snapshots retained")\nparser.add_argument("--lawa-snapshot-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")\nparser.add_argument("--lawa-snapshot-dir", type=str, default=None, help="override lightweight snapshot directory")\n'''

INIT_BLOCK = '''\n# Lightweight LAWA snapshot state. These files contain model weights only, not optimizer state.\nlawa_snapshot_dir = args.lawa_snapshot_dir or os.path.join(base_dir, "lawa_snapshots", output_dirname)\nif ddp_rank == 0 and args.lawa_snapshot_every > 0:\n    os.makedirs(lawa_snapshot_dir, exist_ok=True)\n\ndef _lawa_cpu_state_dict(module, dtype_name):\n    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]\n    out = {}\n    for key, value in module.state_dict().items():\n        value = value.detach().cpu()\n        out[key] = value.to(dtype) if torch.is_floating_point(value) else value\n    return out\n'''

SAVE_BLOCK = '''# Lightweight rank-0 snapshots for terminal averaging. This block is outside the measured training-step timer.\nif ddp_rank == 0 and args.lawa_snapshot_every > 0:\n    lawa_start_step = int(args.lawa_snapshot_start_frac * num_iterations)\n    lawa_due = step >= lawa_start_step and ((step - lawa_start_step) % args.lawa_snapshot_every == 0 or last_step)\n    if lawa_due:\n        lawa_path = os.path.join(lawa_snapshot_dir, f"step_{step:08d}.pt")\n        torch.save({\n            "step": step,\n            "model": _lawa_cpu_state_dict(orig_model, args.lawa_snapshot_dtype),\n            "model_config": model_config_kwargs,\n            "user_config": user_config,\n            "total_training_time": total_training_time,\n        }, lawa_path)\n        lawa_paths = sorted(Path(lawa_snapshot_dir).glob("step_*.pt"))\n        while len(lawa_paths) > args.lawa_snapshot_keep:\n            lawa_paths.pop(0).unlink(missing_ok=True)\n        print0(f"Saved lightweight LAWA snapshot: {lawa_path}")\n\n'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path.cwd())
    ap.add_argument('--source', default='scripts/base_train.py')
    ap.add_argument('--dest', default='scripts/base_train_lawa.py')
    args = ap.parse_args()
    src = args.repo / args.source; dst = args.repo / args.dest
    text = src.read_text()
    if 'lawa-snapshot-every' in text:
        raise SystemExit(f'{src} already appears patched; use a clean source')
    for marker in (PARSER_MARK, CHECKPOINT_MARK, SAVE_MARK):
        if marker not in text:
            raise SystemExit(f'marker not found in {src}: {marker[:80]}')
    text = text.replace(PARSER_MARK, PARSER_MARK + ARGS_BLOCK, 1)
    text = text.replace(CHECKPOINT_MARK, CHECKPOINT_MARK + INIT_BLOCK, 1)
    text = text.replace(SAVE_MARK, SAVE_BLOCK + SAVE_MARK, 1)
    # Path is needed by the inserted pruning block.
    if 'from pathlib import Path' not in text:
        text = text.replace('import argparse\n', 'import argparse\nfrom pathlib import Path\n', 1)
    dst.write_text(text)
    print(f'wrote {dst}')
    print('validate with: python -m py_compile', dst)

if __name__ == '__main__': main()
