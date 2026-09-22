#!/usr/bin/env python3
"""Patch nanochat PR #830's base_train/base_eval with record-attempt hooks.

The patch is deliberately anchor-based and fail-closed: if upstream code moves in
an unexpected way, it refuses to write rather than silently producing a subtly
wrong training schedule.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

MARKER = "# RECORD_ATTEMPT_V1"


def require_once(text: str, needle: str, label: str) -> None:
    n = text.count(needle)
    if n != 1:
        raise RuntimeError(f"expected exactly one {label} anchor, found {n}: {needle!r}")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    require_once(text, old, label)
    return text.replace(old, new, 1)


def patch_base_train(path: Path) -> None:
    text = path.read_text()
    if MARKER in text:
        print(f"already patched: {path}")
        return

    text = replace_once(
        text,
        "from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint\n",
        "from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint\n"
        "from nanochat.record_snapshots import RecordSnapshotManager\n",
        "checkpoint import",
    )

    final_lr_anchor = re.search(
        r'^parser\.add_argument\("--final-lr-frac"[^\n]*\)\n', text, flags=re.MULTILINE
    )
    if not final_lr_anchor:
        raise RuntimeError("could not find --final-lr-frac parser line")
    cli = '''parser.add_argument("--record-schedule-param-data-ratio", type=float, default=-1.0, help="schedule/weight-decay calibration ratio; -1 uses actual target ratio")
parser.add_argument("--record-terminal-lr-clamp-frac", type=float, default=-1.0, help="during warmdown, clamp LR multiplier at this fraction; -1 disables")
parser.add_argument("--record-snapshot-ratios", type=str, default="", help="comma-separated data:param endpoints whose trailing model windows are captured")
parser.add_argument("--record-snapshot-k", type=int, default=8, help="number of model snapshots per endpoint")
parser.add_argument("--record-snapshot-spacing-frac", type=float, default=0.010666666666666666, help="snapshot spacing as fraction of schedule iterations (32/3000 reproduces D12)")
parser.add_argument("--record-snapshot-dir", type=str, default="", help="directory for model-only snapshots; required when snapshot ratios are set")
'''
    pos = final_lr_anchor.end()
    text = text[:pos] + cli + text[pos:]

    target_anchor = re.search(
        r'^(target_tokens\s*=\s*int\(args\.target_param_data_ratio\s*\*\s*num_scaling_params\)[^\n]*\n)',
        text,
        flags=re.MULTILINE,
    )
    if not target_anchor:
        raise RuntimeError("could not find target_tokens assignment")
    schedule_block = '''record_schedule_ratio = args.target_param_data_ratio if args.record_schedule_param_data_ratio <= 0 else args.record_schedule_param_data_ratio
if record_schedule_ratio < args.target_param_data_ratio:
    raise ValueError("record schedule ratio must be >= actual target ratio")
record_schedule_target_tokens = int(record_schedule_ratio * num_scaling_params)
'''
    pos = target_anchor.end()
    text = text[:pos] + schedule_block + text[pos:]

    text = re.sub(
        r'D_REF\s*=\s*args\.target_param_data_ratio\s*\*\s*get_scaling_params\(d12_ref\)',
        'D_REF = record_schedule_ratio * get_scaling_params(d12_ref)',
        text,
        count=1,
    )
    if 'D_REF = record_schedule_ratio * get_scaling_params(d12_ref)' not in text:
        raise RuntimeError("failed to patch D_REF")
    text = replace_once(
        text,
        "batch_size_ratio = target_tokens / D_REF",
        "batch_size_ratio = record_schedule_target_tokens / D_REF",
        "auto batch ratio",
    )
    text = replace_once(
        text,
        "weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)",
        "weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / record_schedule_target_tokens)",
        "weight decay scaling",
    )

    horizon_anchor = '''else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
'''
    horizon_new = '''else:
    raise ValueError("No training horizon specified")
schedule_num_iterations = record_schedule_target_tokens // total_batch_size
if schedule_num_iterations < num_iterations:
    raise ValueError(f"schedule horizon {schedule_num_iterations} is shorter than train horizon {num_iterations}")
print0(f"Record schedule ratio: {record_schedule_ratio:.4f} => {schedule_num_iterations:,} schedule steps")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
'''
    text = replace_once(text, horizon_anchor, horizon_new, "training horizon tail")

    scheduler_pattern = re.compile(
        r'# Learning rate schedule \(linear warmup, constant, linear warmdown\)\n.*?'
        r'(?=# -----------------------------------------------------------------------------\n# Training loop)',
        flags=re.DOTALL,
    )
    matches = list(scheduler_pattern.finditer(text))
    if len(matches) != 1:
        raise RuntimeError(f"expected one scheduler block, found {len(matches)}")
    scheduler = '''# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * schedule_num_iterations)
    warmdown_start = schedule_num_iterations - warmdown_iters
    if it < warmup_iters:
        base_multiplier = (it + 1) / warmup_iters
    elif it <= warmdown_start:
        base_multiplier = 1.0
    else:
        progress = max(0.0, (schedule_num_iterations - it) / max(warmdown_iters, 1))
        base_multiplier = progress * 1.0 + (1 - progress) * args.final_lr_frac
    if args.record_terminal_lr_clamp_frac > 0 and it >= warmdown_start:
        base_multiplier = max(base_multiplier, args.record_terminal_lr_clamp_frac)
    return base_multiplier
# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * schedule_num_iterations)
    warmdown_start = schedule_num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = min(1.0, (it - warmdown_start) / max(warmdown_iters, 1))
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97
# Weight decay scheduler for Muon optimizer (cosine decay to zero over the schedule horizon)
def get_weight_decay(it):
    progress = min(max(it / schedule_num_iterations, 0.0), 1.0)
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * progress))

'''
    text = scheduler_pattern.sub(scheduler, text, count=1)

    # PR #830 has seen tiny formatting changes around the training-loop prologue.
    # Anchor on the unique semantic marker instead of an exact neighboring print line.
    go_pattern = re.compile(r'(?m)^# Go!\s*$')
    go_matches = list(go_pattern.finditer(text))
    if len(go_matches) != 1:
        raise RuntimeError(f"expected exactly one training loop # Go! marker, found {len(go_matches)}")
    go_init = '''if args.record_snapshot_ratios and not args.record_snapshot_dir:
    raise ValueError("--record-snapshot-dir is required when --record-snapshot-ratios is nonempty")
record_snapshot_manager = RecordSnapshotManager.create(
    enabled=bool(args.record_snapshot_ratios.strip()),
    writer=master_process,
    snapshot_dir=args.record_snapshot_dir or os.path.join(checkpoint_dir, "record_snapshots"),
    ratios=args.record_snapshot_ratios,
    num_scaling_params=num_scaling_params,
    total_batch_size=total_batch_size,
    schedule_num_iterations=schedule_num_iterations,
    train_num_iterations=num_iterations,
    k=args.record_snapshot_k,
    spacing_fraction=args.record_snapshot_spacing_frac,
    model_tag=output_dirname,
    user_config=user_config,
    model_config=model_config_kwargs,
)
'''
    text = go_pattern.sub(go_init + "# Go!", text, count=1)

    # Capture immediately after the canonical state-update increment. Allow blank
    # lines/formatting variation but still fail closed if the semantic anchor moves.
    step_pattern = re.compile(
        r'(?m)^(?P<indent>[ \t]*)first_step_of_run = \(step == 0\) or \(resuming and step == args\.resume_from_step\)\s*\n'
        r'(?P=indent)step \+= 1\s*$',
    )
    step_matches = list(step_pattern.finditer(text))
    if len(step_matches) != 1:
        raise RuntimeError(f"expected exactly one step increment anchor, found {len(step_matches)}")
    def _step_repl(m):
        indent = m.group('indent')
        return (
            f"{indent}first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)\n"
            f"{indent}step += 1\n"
            f"{indent}record_snapshot_manager.capture(\n"
            f"{indent}    step=step,\n"
            f"{indent}    model=orig_model,\n"
            f"{indent}    total_training_time=total_training_time,\n"
            f"{indent})"
        )
    text = step_pattern.sub(_step_repl, text, count=1)

    # Finalize before the unique post-loop statistics section.
    cleanup_pattern = re.compile(r'(?m)^# print a few more stats\s*$')
    cleanup_matches = list(cleanup_pattern.finditer(text))
    if len(cleanup_matches) != 1:
        raise RuntimeError(f"expected exactly one post-loop stats marker, found {len(cleanup_matches)}")
    text = cleanup_pattern.sub('record_snapshot_manager.finalize()\n# print a few more stats', text, count=1)

    text = text.replace('"record-snapshot-spacing-frac"', '"record-snapshot-spacing-frac"')
    # Attribute name generated by argparse replaces hyphens with underscores.
    text = text.replace('args.record_snapshot_spacing_frac', 'args.record_snapshot_spacing_frac')

    # Add a marker after the docstring/import region.
    text = text.replace('import os\n', f'import os\n{MARKER}\n', 1)

    backup = path.with_suffix(path.suffix + ".record_backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text)
    print(f"patched: {path}")


def patch_base_eval(path: Path) -> None:
    text = path.read_text()
    if "RECORD_ATTEMPT_BASE_EVAL_V1" in text:
        print(f"already patched: {path}")
        return
    old = 'model_slug = f"base_model_{meta[\'step\']:06d}"'
    if old not in text:
        # Accept double-quoted key variation.
        old = 'model_slug = f"base_model_{meta[\"step\"]:06d}"'
    if old not in text:
        raise RuntimeError("could not find base_eval model_slug assignment")
    new = '''safe_model_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model_tag or "base_model")
    model_slug = f"{safe_model_tag}_{meta['step']:06d}" # RECORD_ATTEMPT_BASE_EVAL_V1'''
    text = text.replace(old, new, 1)
    if "import re\n" not in text:
        # base_eval imports os near the top in current nanochat.
        if "import os\n" not in text:
            raise RuntimeError("could not find import os in base_eval")
        text = text.replace("import os\n", "import os\nimport re\n", 1)
    backup = path.with_suffix(path.suffix + ".record_backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text)
    print(f"patched: {path}")


def selftest() -> None:
    fixture = '''import os
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="x")
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # x
d12_ref = build_model_meta(12)
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref)
batch_size_ratio = target_tokens / D_REF
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if args.num_iterations > 0:
    num_iterations = args.num_iterations
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # the actual number of tokens we will train for
# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    return 1
# Momentum scheduler for Muon optimizer (warms up to 0.97, warms down to 0.90 during LR warmdown)
def get_muon_momentum(it):
    return 0.9
# Weight decay scheduler for Muon optimizer (cosine decay to zero over the course of training)
def get_weight_decay(it):
    return 0

# -----------------------------------------------------------------------------
# Training loop
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Go!
while True:
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1
# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
'''
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "base_train.py"
        p.write_text(fixture)
        patch_base_train(p)
        out = p.read_text()
        assert MARKER in out
        assert "schedule_num_iterations" in out
        assert "record_snapshot_manager.capture" in out
    print("apply_record_patch selftest PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", nargs="?", type=Path)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if args.repo is None:
        ap.error("repo is required unless --selftest is used")
    repo = args.repo.resolve()
    patch_base_train(repo / "scripts" / "base_train.py")
    patch_base_eval(repo / "scripts" / "base_eval.py")


if __name__ == "__main__":
    main()
