#!/usr/bin/env python3
"""Summarize frozen confirmation runs into a compact record digest."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def mean_sem(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.256525)
    ap.add_argument("--reference-minutes", type=float, default=81.835, help="current PR #830 candidate mean time")
    args = ap.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted(args.results_dir.glob("*.json"))]
    if not rows:
        raise SystemExit("no confirmation result JSONs")
    times = [float(r["total_training_time"]) for r in rows]
    cores = [float(r["core_metric"]) for r in rows]
    bpbs = [float(r["val_bpb"]) for r in rows]
    mt, st = mean_sem(times)
    mc, sc = mean_sem(cores)
    mb, sb = mean_sem(bpbs)
    passed = [c > args.threshold for c in cores]
    digest = f'''Frozen nanochat record confirmation
{'='*72}
runs: {len(rows)}
ratio: {rows[0]['ratio']}
recipe: {rows[0]['recipe']}
training time: {mt/60:.3f} +/- {st/60:.3f} min (SEM)
CORE: {mc:.6f} +/- {sc:.6f} (SEM)
validation BPB: {mb:.6f} +/- {sb:.6f} (SEM)
all runs CORE > {args.threshold:.6f}: {'YES' if all(passed) else 'NO'} ({sum(passed)}/{len(passed)})
fastest run: {min(times)/60:.3f} min
slowest run: {max(times)/60:.3f} min
minimum CORE: {min(cores):.6f}
mean time below {args.reference_minutes:.3f} min reference: {'YES' if mt/60 < args.reference_minutes else 'NO'}
estimated margin vs reference: {args.reference_minutes - mt/60:+.3f} min
'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(digest)
    print(digest)


if __name__ == "__main__":
    main()
