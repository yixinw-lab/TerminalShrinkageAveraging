#!/usr/bin/env python3
"""Freeze structured and scalar TSA rules using development trajectories only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SUITE = "d12_structured_tsa_suite_v1"
EXPECTED_OPTIMIZERS = ("native", "pure_adamw")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: Sequence[float]) -> float:
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return 0.0
    try:
        from scipy.stats import t

        crit = float(t.ppf(0.975, len(a) - 1))
    except Exception:
        crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(len(a), 1.96)
    return float(crit * a.std(ddof=1) / math.sqrt(len(a)))


def canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def load_development(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted((root / "development").glob("*/floor_*/seed*/calibration_results.csv"))
    if not paths:
        raise RuntimeError(f"no development calibration_results.csv under {root/'development'}")
    for path in paths:
        for raw in read_csv(path):
            row: dict[str, Any] = dict(raw)
            for key in ("alpha_group", "alpha_other", "bpb", "method_minus_raw", "paired_eval_se", "floor"):
                if row.get(key, "") != "":
                    row[key] = float(row[key])
            row["stream_seed"] = int(float(row["stream_seed"]))
            row["source_path"] = str(path)
            rows.append(row)
    return rows


def summarize_family(rows: Sequence[Mapping[str, Any]], family: str) -> list[dict[str, Any]]:
    by: dict[tuple[str, float, float], list[dict[str, Any]]] = defaultdict(list)
    raw_by: dict[tuple[str, int], float] = {}
    for row in rows:
        if row["family"] == "structured_grid" and float(row["alpha_group"]) == 0.0 and float(row["alpha_other"]) == 0.0:
            raw_by[(str(row["optimizer"]), int(row["stream_seed"]))] = float(row["bpb"])
    for row in rows:
        if row["family"] != family:
            continue
        key = (str(row["optimizer"]), float(row["alpha_group"]), float(row["alpha_other"]))
        by[key].append(dict(row))

    out: list[dict[str, Any]] = []
    for (optimizer, alpha_group, alpha_other), group in sorted(by.items()):
        gains = [raw_by[(optimizer, int(r["stream_seed"]))] - float(r["bpb"]) for r in group]
        bpbs = [float(r["bpb"]) for r in group]
        out.append(
            {
                "optimizer": optimizer,
                "family": family,
                "alpha_group": alpha_group,
                "alpha_other": alpha_other,
                "n": len(group),
                "mean_bpb": float(np.mean(bpbs)),
                "mean_gain_vs_raw": float(np.mean(gains)),
                "gain_ci95_half": ci95(gains),
                "positive_seed_count": int(sum(x > 0 for x in gains)),
                "min_gain": float(np.min(gains)),
                "max_gain": float(np.max(gains)),
            }
        )
    return out


def select_rule(summary: Sequence[Mapping[str, Any]], optimizer: str, family: str) -> dict[str, Any]:
    candidates = [dict(row) for row in summary if row["optimizer"] == optimizer and row["family"] == family]
    if not candidates:
        raise RuntimeError(f"no {family} candidates for {optimizer}")
    # Primary criterion: highest paired mean gain over raw across development streams.
    # Deterministic ties prefer the less extreme point and then lexicographic order.
    candidates.sort(
        key=lambda row: (
            -float(row["mean_gain_vs_raw"]),
            abs(float(row["alpha_group"]) - 0.5) + abs(float(row["alpha_other"]) - 0.5),
            abs(float(row["alpha_group"]) - float(row["alpha_other"])),
            float(row["alpha_group"]),
            float(row["alpha_other"]),
        )
    )
    return candidates[0]


def freeze(root: Path) -> Path:
    rows = load_development(root)
    summary = summarize_family(rows, "structured_grid") + summarize_family(rows, "scalar_grid")
    write_csv(root / "development_surface_summary.csv", summary)

    rules: dict[str, Any] = {}
    for optimizer in EXPECTED_OPTIMIZERS:
        structured = select_rule(summary, optimizer, "structured_grid")
        scalar = select_rule(summary, optimizer, "scalar_grid")
        rules[optimizer] = {
            "alpha_group": float(structured["alpha_group"]),
            "alpha_other": float(structured["alpha_other"]),
            "structured_development_mean_gain": float(structured["mean_gain_vs_raw"]),
            "structured_development_ci95_half": float(structured["gain_ci95_half"]),
            "structured_positive_seed_count": int(structured["positive_seed_count"]),
            "scalar_alpha": float(scalar["alpha_group"]),
            "scalar_development_mean_gain": float(scalar["mean_gain_vs_raw"]),
            "scalar_development_ci95_half": float(scalar["gain_ci95_half"]),
            "structured_minus_scalar_development": float(
                structured["mean_gain_vs_raw"] - scalar["mean_gain_vs_raw"]
            ),
            "off_diagonal": bool(
                abs(float(structured["alpha_group"]) - float(structured["alpha_other"])) > 1e-12
            ),
            "n_development_streams": int(structured["n"]),
        }

    schema_path = root / "GROUP_SCHEMA.json"
    if not schema_path.exists():
        raise RuntimeError(f"missing {schema_path}")
    schema = json.loads(schema_path.read_text())
    payload: dict[str, Any] = {
        "suite": SUITE,
        "selection_data": "development trajectories only; calibration validation block only",
        "selection_rule": (
            "maximize mean paired BPB gain versus raw across development streams separately for the "
            "predeclared structured 9x9 grid and scalar 0.05 grid; deterministic ties favor less-extreme points"
        ),
        "group_schema_sha256": schema.get("schema_sha256"),
        "rules": rules,
    }
    payload["freeze_sha256"] = canonical_hash(payload)
    out = root / "FROZEN_RULES.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (root / "FROZEN_RULES.sha256").write_text(
        hashlib.sha256(out.read_bytes()).hexdigest() + "  FROZEN_RULES.json\n"
    )

    lines = [
        "D12 structured TSA rule freeze",
        "================================",
        "Selection used development trajectories and calibration batches only.",
    ]
    for optimizer in EXPECTED_OPTIMIZERS:
        rule = rules[optimizer]
        lines.extend(
            [
                "",
                f"optimizer={optimizer}",
                f"structured=(group={rule['alpha_group']:.3f}, other={rule['alpha_other']:.3f})",
                f"scalar={rule['scalar_alpha']:.3f}",
                f"structured dev gain={rule['structured_development_mean_gain']:+.6f} +/- {rule['structured_development_ci95_half']:.6f}",
                f"scalar dev gain={rule['scalar_development_mean_gain']:+.6f} +/- {rule['scalar_development_ci95_half']:.6f}",
                f"structured-minus-scalar dev={rule['structured_minus_scalar_development']:+.6f}",
                f"off_diagonal={rule['off_diagonal']}",
            ]
        )
    lines.extend(["", f"freeze_sha256={payload['freeze_sha256']}", "FREEZE COMPLETE"])
    (root / "FROZEN_RULES.txt").write_text("\n".join(lines) + "\n")
    print((root / "FROZEN_RULES.txt").read_text())
    return out


def selftest() -> None:
    rows: list[dict[str, Any]] = []
    for seed in (1, 2, 3):
        raw = 1.0 + seed * 0.01
        for ag in (0.0, 0.5, 1.0):
            for ao in (0.0, 0.5, 1.0):
                gain = 0.01 - (ag - 0.5) ** 2 * 0.02 - (ao - 1.0) ** 2 * 0.02
                rows.append(
                    {
                        "family": "structured_grid",
                        "optimizer": "native",
                        "alpha_group": ag,
                        "alpha_other": ao,
                        "bpb": raw - gain,
                        "stream_seed": seed,
                    }
                )
        for alpha in (0.0, 0.5, 1.0):
            gain = 0.005 - (alpha - 0.5) ** 2 * 0.01
            rows.append(
                {
                    "family": "scalar_grid",
                    "optimizer": "native",
                    "alpha_group": alpha,
                    "alpha_other": alpha,
                    "bpb": raw - gain,
                    "stream_seed": seed,
                }
            )
    summary = summarize_family(rows, "structured_grid") + summarize_family(rows, "scalar_grid")
    structured = select_rule(summary, "native", "structured_grid")
    scalar = select_rule(summary, "native", "scalar_grid")
    assert structured["alpha_group"] == 0.5 and structured["alpha_other"] == 1.0
    assert scalar["alpha_group"] == 0.5
    print("freeze rules selftest PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    if args.root is None:
        parser.error("--root is required unless --selftest is used")
    freeze(args.root.resolve())


if __name__ == "__main__":
    main()
