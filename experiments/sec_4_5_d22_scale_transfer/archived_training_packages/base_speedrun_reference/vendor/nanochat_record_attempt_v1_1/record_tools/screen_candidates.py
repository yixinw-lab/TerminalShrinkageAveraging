#!/usr/bin/env python3
"""Quick BPB screen of generated nanochat candidate checkpoints."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess


def parse_bpb(text: str) -> tuple[float, float]:
    train = re.findall(r"train bpb:\s*([0-9.]+)", text)
    val = re.findall(r"val bpb:\s*([0-9.]+)", text)
    if not train or not val:
        raise RuntimeError("could not parse train/val bpb from base_eval output")
    return float(train[-1]), float(val[-1])


def run(args: argparse.Namespace) -> None:
    repo = args.repo.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.candidates.open()))
    results: list[dict[str, str]] = []
    env = os.environ.copy()
    for i, row in enumerate(rows):
        tag = row["model_tag"]
        step = row["endpoint_step"]
        log_path = out_dir / f"{i:03d}_{tag}.log"
        result_path = out_dir / f"{i:03d}_{tag}.result"
        if result_path.exists():
            train_bpb, val_bpb = map(float, result_path.read_text().strip().split(","))
        else:
            cmd = [
                "torchrun", "--standalone", f"--nproc_per_node={args.nproc}",
                "-m", "scripts.base_eval", "--",
                "--eval=bpb",
                f"--model-tag={tag}",
                f"--step={step}",
                f"--device-batch-size={args.device_batch_size}",
                f"--split-tokens={args.split_tokens}",
            ]
            print(f"[{i+1}/{len(rows)}] {' '.join(cmd)}", flush=True)
            proc = subprocess.run(
                cmd,
                cwd=repo,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            log_path.write_text(proc.stdout)
            if proc.returncode != 0:
                raise RuntimeError(f"base_eval failed for {tag}; see {log_path}")
            train_bpb, val_bpb = parse_bpb(proc.stdout)
            result_path.write_text(f"{train_bpb},{val_bpb}\n")
        out = dict(row)
        out["screen_train_bpb"] = f"{train_bpb:.9f}"
        out["screen_val_bpb"] = f"{val_bpb:.9f}"
        results.append(out)

    screen_csv = out_dir / "bpb_screen.csv"
    fields = list(results[0])
    with screen_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(results)

    best: list[dict[str, str]] = []
    ratios = sorted({float(r["ratio"]) for r in results})
    for ratio in ratios:
        candidates = [r for r in results if float(r["ratio"]) == ratio]
        winner = min(candidates, key=lambda r: float(r["screen_val_bpb"]))
        best.append(winner)
    top_csv = out_dir / "top_candidates.csv"
    with top_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(best)

    print(f"wrote {screen_csv}")
    print(f"wrote {top_csv}")
    print("\nTop candidate per ratio:")
    for r in best:
        print(
            f"ratio={float(r['ratio']):.3f} val_bpb={float(r['screen_val_bpb']):.6f} "
            f"recipe={r['recipe']} tag={r['model_tag']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--split-tokens", type=int, default=4 * 524288)
    ap.add_argument("--device-batch-size", type=int, default=32)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
