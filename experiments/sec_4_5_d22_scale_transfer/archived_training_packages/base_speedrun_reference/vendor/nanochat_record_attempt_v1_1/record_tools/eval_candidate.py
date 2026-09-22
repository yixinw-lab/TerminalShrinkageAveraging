#!/usr/bin/env python3
"""Run canonical BPB+CORE evaluation for one candidate and write JSON."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess


def last_float(pattern: str, text: str, label: str) -> float:
    hits = re.findall(pattern, text)
    if not hits:
        raise RuntimeError(f"could not parse {label}")
    return float(hits[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--split-tokens", type=int, default=40 * 524288)
    ap.add_argument("--device-batch-size", type=int, default=32)
    ap.add_argument("--max-per-task", type=int, default=-1)
    args = ap.parse_args()

    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={args.nproc}",
        "-m", "scripts.base_eval", "--",
        "--eval=core,bpb",
        f"--model-tag={args.model_tag}",
        f"--step={args.step}",
        f"--device-batch-size={args.device_batch_size}",
        f"--split-tokens={args.split_tokens}",
        f"--max-per-task={args.max_per_task}",
    ]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=args.repo.resolve(),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    args.log.write_text(proc.stdout)
    if proc.returncode != 0:
        raise SystemExit(f"base_eval failed ({proc.returncode}); see {args.log}")

    train_bpb = last_float(r"train bpb:\s*([0-9.]+)", proc.stdout, "train bpb")
    val_bpb = last_float(r"val bpb:\s*([0-9.]+)", proc.stdout, "val bpb")
    core = last_float(r"CORE metric:\s*([0-9.]+)", proc.stdout, "CORE")

    base_dir = Path(os.environ["NANOCHAT_BASE_DIR"])
    meta_path = base_dir / "base_checkpoints" / args.model_tag / f"meta_{args.step:06d}.json"
    meta = json.loads(meta_path.read_text())
    training_time = float(meta["loop_state"]["total_training_time"])
    payload = {
        "model_tag": args.model_tag,
        "step": args.step,
        "ratio": args.ratio,
        "recipe": args.recipe,
        "train_bpb": train_bpb,
        "val_bpb": val_bpb,
        "core_metric": core,
        "total_training_time": training_time,
        "training_minutes": training_time / 60.0,
        "log": str(args.log),
        "meta": str(meta_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
