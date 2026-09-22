#!/usr/bin/env python3
"""Freeze the earliest CORE-qualified exploratory candidate for confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.256525)
    ap.add_argument("--safety-margin", type=float, default=0.0015)
    ap.add_argument("--schedule-ratio", type=float, default=9.4)
    ap.add_argument("--clamp-frac", type=float, default=0.15)
    ap.add_argument("--snapshot-k", type=int, default=8)
    ap.add_argument("--spacing-frac", type=float, default=32 / 3000)
    args = ap.parse_args()

    rows = [json.loads(p.read_text()) for p in sorted(args.results_dir.glob("*.json"))]
    if not rows:
        raise SystemExit(f"no result JSONs under {args.results_dir}")
    target = args.threshold + args.safety_margin
    safe = [r for r in rows if float(r["core_metric"]) >= target]
    qualified = [r for r in rows if float(r["core_metric"]) > args.threshold]
    if safe:
        selected = min(safe, key=lambda r: (float(r["ratio"]), float(r["total_training_time"])))
        policy = f"CORE >= threshold+safety ({target:.6f})"
    elif qualified:
        selected = min(qualified, key=lambda r: (float(r["ratio"]), float(r["total_training_time"])))
        policy = "WARNING: no safety-margin candidate; selected earliest threshold-only candidate"
    else:
        best = max(rows, key=lambda r: float(r["core_metric"]))
        raise SystemExit(
            f"no candidate beats CORE {args.threshold:.6f}; best was {best['core_metric']:.6f} "
            f"at ratio {best['ratio']} recipe {best['recipe']}"
        )

    env = {
        "TARGET_RATIO": selected["ratio"],
        "RECIPE": selected["recipe"],
        "SCHEDULE_RATIO": args.schedule_ratio,
        "TERMINAL_CLAMP_FRAC": args.clamp_frac,
        "SNAPSHOT_K": args.snapshot_k,
        "SNAPSHOT_SPACING_FRAC": args.spacing_frac,
        "EXPLORATORY_CORE": selected["core_metric"],
        "EXPLORATORY_VAL_BPB": selected["val_bpb"],
        "EXPLORATORY_TRAINING_TIME": selected["total_training_time"],
        "FREEZE_POLICY": policy,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Frozen from one exploratory trajectory. Do not retune on confirmation outcomes."]
    for key, value in env.items():
        lines.append(f"{key}={shlex.quote(str(value))}")
    args.output.write_text("\n".join(lines) + "\n")
    print(f"selected ratio={selected['ratio']} recipe={selected['recipe']} core={selected['core_metric']}")
    print(f"policy: {policy}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
