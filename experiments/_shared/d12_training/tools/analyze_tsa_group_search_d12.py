#!/usr/bin/env python3
"""Freeze and analyze the D12 TSA group-search suite."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SUITE = "d12_tsa_group_search_v1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _canonical_hash(obj: Mapping[str, Any]) -> str:
    import hashlib
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _tcrit95(n: int) -> float:
    # Two-sided 95%, df=n-1. Exact enough for the suite's n=8 groups.
    table = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
        8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179,
        14: 2.160, 15: 2.145, 16: 2.131, 17: 2.120, 18: 2.110, 19: 2.101,
        20: 2.093, 21: 2.086, 22: 2.080, 23: 2.074, 24: 2.069, 25: 2.064,
        26: 2.060, 27: 2.056, 28: 2.052, 29: 2.048, 30: 2.045,
    }
    if n <= 1:
        return math.nan
    if n in table:
        return table[n]
    return 1.96


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else math.nan
    if n <= 1:
        return {"n": n, "mean": mean, "se": math.nan, "halfwidth95": math.nan, "lo95": math.nan, "hi95": math.nan}
    se = float(arr.std(ddof=1) / math.sqrt(n))
    hw = _tcrit95(n) * se
    return {"n": n, "mean": mean, "se": se, "halfwidth95": hw, "lo95": mean - hw, "hi95": mean + hw}


def _dev_files(root: Path) -> list[Path]:
    return sorted(root.glob("dev/**/candidate_results.csv"))


def freeze(root: Path, expected_dev: int = 8) -> Path:
    files = _dev_files(root)
    if len(files) != expected_dev:
        raise RuntimeError(f"expected {expected_dev} development candidate files, found {len(files)}")

    by_run: list[dict[str, dict[str, Any]]] = []
    spec_by_id: dict[str, dict[str, Any]] = {}
    family_by_id: dict[str, str] = {}
    for path in files:
        rows = _read_csv(path)
        run: dict[str, dict[str, Any]] = {}
        for row in rows:
            cid = row["candidate_id"]
            spec = json.loads(row["spec_json"])
            run[cid] = {
                "gain_vs_raw": float(row["gain_vs_raw"]),
                "bpb": float(row["bpb"]),
            }
            spec_by_id[cid] = spec
            family_by_id[cid] = row["family"]
        by_run.append(run)

    common = set(by_run[0])
    for run in by_run[1:]:
        common &= set(run)
    if not common:
        raise RuntimeError("no candidate IDs are common across development runs")

    candidate_rows: list[dict[str, Any]] = []
    for cid in sorted(common):
        gains = [run[cid]["gain_vs_raw"] for run in by_run]
        s = _summary(gains)
        candidate_rows.append({
            "candidate_id": cid,
            "family": family_by_id[cid],
            "mean_gain_vs_raw": s["mean"],
            "se_gain_vs_raw": s["se"],
            "lo95_gain_vs_raw": s["lo95"],
            "hi95_gain_vs_raw": s["hi95"],
            "spec": spec_by_id[cid],
        })

    scalar_rows = [r for r in candidate_rows if r["family"] == "scalar"]
    if not scalar_rows:
        raise RuntimeError("scalar candidate family missing")
    scalar_best = max(scalar_rows, key=lambda r: (float(r["mean_gain_vs_raw"]), r["candidate_id"]))
    scalar_id = scalar_best["candidate_id"]

    # Compute paired development improvement relative to the frozen scalar.
    for row in candidate_rows:
        cid = row["candidate_id"]
        diffs = [run[cid]["gain_vs_raw"] - run[scalar_id]["gain_vs_raw"] for run in by_run]
        s = _summary(diffs)
        row["mean_gain_vs_scalar"] = s["mean"]
        row["se_gain_vs_scalar"] = s["se"]
        row["lo95_gain_vs_scalar"] = s["lo95"]
        row["hi95_gain_vs_scalar"] = s["hi95"]

    family_winners: dict[str, dict[str, Any]] = {}
    for family in sorted({r["family"] for r in candidate_rows}):
        rows = [r for r in candidate_rows if r["family"] == family]
        family_winners[family] = max(rows, key=lambda r: (float(r["mean_gain_vs_scalar"]), r["candidate_id"]))

    non_scalar = [r for r in candidate_rows if r["family"] != "scalar"]
    overall = max(non_scalar, key=lambda r: (float(r["mean_gain_vs_scalar"]), r["candidate_id"]))
    top_overall = sorted(non_scalar, key=lambda r: (float(r["mean_gain_vs_scalar"]), r["candidate_id"]), reverse=True)[:5]

    selected_ids: list[str] = [scalar_id]
    for family in sorted(family_winners):
        cid = family_winners[family]["candidate_id"]
        if cid not in selected_ids:
            selected_ids.append(cid)
    for row in top_overall:
        if row["candidate_id"] not in selected_ids:
            selected_ids.append(row["candidate_id"])

    selected_rules: list[dict[str, Any]] = []
    row_by_id = {r["candidate_id"]: r for r in candidate_rows}
    for cid in selected_ids:
        r = row_by_id[cid]
        selected_rules.append({
            "candidate_id": cid,
            "family": r["family"],
            "spec": r["spec"],
            "development_mean_gain_vs_raw": r["mean_gain_vs_raw"],
            "development_mean_gain_vs_scalar": r["mean_gain_vs_scalar"],
        })

    frozen = {
        "suite": SUITE,
        "stage": "frozen_rules",
        "development_run_count": len(files),
        "selection_metric": "mean paired BPB gain versus raw across native 10pct development streams",
        "frozen_scalar": {
            "candidate_id": scalar_id,
            "spec": scalar_best["spec"],
            "mean_gain_vs_raw": scalar_best["mean_gain_vs_raw"],
        },
        "overall_best": {
            "candidate_id": overall["candidate_id"],
            "family": overall["family"],
            "spec": overall["spec"],
            "mean_gain_vs_scalar": overall["mean_gain_vs_scalar"],
        },
        "family_winners": {
            family: {
                "candidate_id": row["candidate_id"],
                "spec": row["spec"],
                "mean_gain_vs_scalar": row["mean_gain_vs_scalar"],
            }
            for family, row in family_winners.items()
        },
        "selected_rules": selected_rules,
        "external_transfer_scalar": {"pure_adamw_10pct_alpha": 0.50},
    }
    frozen["freeze_sha256"] = _canonical_hash(frozen)

    out = root / "FROZEN_GROUP_RULES.json"
    out.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    freeze_rows = []
    for row in sorted(candidate_rows, key=lambda r: float(r["mean_gain_vs_scalar"]), reverse=True):
        freeze_rows.append({k: v for k, v in row.items() if k != "spec"})
    _write_csv(root / "DEV_CANDIDATE_SUMMARY.csv", freeze_rows)
    fam_rows = []
    for family, row in family_winners.items():
        fam_rows.append({
            "family": family,
            "candidate_id": row["candidate_id"],
            "mean_gain_vs_scalar": row["mean_gain_vs_scalar"],
            "lo95_gain_vs_scalar": row["lo95_gain_vs_scalar"],
            "hi95_gain_vs_scalar": row["hi95_gain_vs_scalar"],
        })
    _write_csv(root / "DEV_FAMILY_WINNERS.csv", sorted(fam_rows, key=lambda r: r["mean_gain_vs_scalar"], reverse=True))

    print(f"frozen_rules={out}")
    print(f"freeze_sha256={frozen['freeze_sha256']}")
    print(f"frozen_scalar={scalar_id}")
    print(f"overall_best={overall['candidate_id']}")
    print(f"selected_rule_count={len(selected_rules)}")
    return out


def _load_frozen(root: Path) -> dict[str, Any]:
    path = root / "FROZEN_GROUP_RULES.json"
    obj = json.loads(path.read_text())
    claimed = obj.get("freeze_sha256")
    payload = dict(obj); payload.pop("freeze_sha256", None)
    if claimed != _canonical_hash(payload):
        raise RuntimeError("frozen-rule hash mismatch")
    return obj


def analyze(root: Path, expected_confirm: int = 16) -> Path:
    frozen = _load_frozen(root)
    result_files = sorted(root.glob("confirm/**/confirmation_results.csv"))
    if len(result_files) != expected_confirm:
        raise RuntimeError(f"expected {expected_confirm} confirmation files, found {len(result_files)}")

    # Infer optimizer/seed from neighboring result.json.
    runs: list[dict[str, Any]] = []
    for csv_path in result_files:
        meta = json.loads((csv_path.parent / "result.json").read_text())
        rows = _read_csv(csv_path)
        by_id = {r["candidate_id"]: r for r in rows}
        runs.append({
            "optimizer": meta["optimizer_label"],
            "floor": float(meta["terminal_floor"]),
            "stream_seed": int(meta["stream_seed"]),
            "rows": by_id,
        })

    selected = frozen["selected_rules"]
    overall_id = frozen["overall_best"]["candidate_id"]
    frozen_scalar = frozen["frozen_scalar"]["candidate_id"]
    pure_scalar = "scalar|alpha=0.500"
    hidden_rest_id = frozen.get("family_winners", {}).get("hidden_rest", {}).get("candidate_id")

    summary_rows: list[dict[str, Any]] = []
    runlevel_rows: list[dict[str, Any]] = []
    for optimizer in ("native", "pure_adamw"):
        subset = [r for r in runs if r["optimizer"] == optimizer]
        baseline_id = frozen_scalar if optimizer == "native" else pure_scalar
        if not subset:
            continue
        candidate_ids = [x["candidate_id"] for x in selected]
        if overall_id not in candidate_ids:
            candidate_ids.append(overall_id)
        for cid in candidate_ids:
            diffs: list[float] = []
            for run in subset:
                rows = run["rows"]
                if cid not in rows or baseline_id not in rows:
                    continue
                gain = float(rows[baseline_id]["bpb"]) - float(rows[cid]["bpb"])
                diffs.append(gain)
                runlevel_rows.append({
                    "optimizer": optimizer,
                    "stream_seed": run["stream_seed"],
                    "baseline_id": baseline_id,
                    "candidate_id": cid,
                    "gain_vs_baseline": gain,
                })
            if diffs:
                s = _summary(diffs)
                family = next((x["family"] for x in selected if x["candidate_id"] == cid), "unknown")
                summary_rows.append({
                    "optimizer": optimizer,
                    "baseline_id": baseline_id,
                    "candidate_id": cid,
                    "family": family,
                    "n": s["n"],
                    "mean_gain_vs_baseline": s["mean"],
                    "se": s["se"],
                    "halfwidth95": s["halfwidth95"],
                    "lo95": s["lo95"],
                    "hi95": s["hi95"],
                })

    # Directly compare overall winner to the prior hidden/rest family winner.
    special_rows: list[dict[str, Any]] = []
    if hidden_rest_id:
        for optimizer in ("native", "pure_adamw"):
            subset = [r for r in runs if r["optimizer"] == optimizer]
            vals = []
            for run in subset:
                rows = run["rows"]
                if overall_id in rows and hidden_rest_id in rows:
                    vals.append(float(rows[hidden_rest_id]["bpb"]) - float(rows[overall_id]["bpb"]))
            if vals:
                s = _summary(vals)
                special_rows.append({
                    "optimizer": optimizer,
                    "contrast": "overall_best_minus_hidden_rest_winner",
                    "candidate_id": overall_id,
                    "reference_id": hidden_rest_id,
                    "n": s["n"], "mean_gain": s["mean"], "se": s["se"],
                    "halfwidth95": s["halfwidth95"], "lo95": s["lo95"], "hi95": s["hi95"],
                })

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _write_csv(figures / "confirmation_summary.csv", summary_rows)
    _write_csv(figures / "confirmation_run_level.csv", runlevel_rows)
    _write_csv(figures / "key_contrasts.csv", special_rows)

    # Aggregate development tensor statistics by role and normalized depth bin.
    stat_files = sorted(root.glob("dev/**/trajectory_tensor_stats.csv"))
    role_scores: dict[str, list[float]] = defaultdict(list)
    role_noise: dict[str, list[float]] = defaultdict(list)
    role_cos: dict[str, list[float]] = defaultdict(list)
    for path in stat_files:
        for row in _read_csv(path):
            role = row["role"]
            role_scores[role].append(float(row["trajectory_score"]))
            role_noise[role].append(float(row["noise_fraction"]))
            role_cos[role].append(float(row["mean_update_cosine"]))
    stat_summary = []
    for role in sorted(role_scores):
        stat_summary.append({
            "role": role,
            "tensor_observations": len(role_scores[role]),
            "mean_trajectory_score": float(np.mean(role_scores[role])),
            "mean_noise_fraction": float(np.mean(role_noise[role])),
            "mean_update_cosine": float(np.mean(role_cos[role])),
        })
    _write_csv(figures / "trajectory_role_stats.csv", stat_summary)

    def find_summary(opt: str, cid: str) -> dict[str, Any] | None:
        for row in summary_rows:
            if row["optimizer"] == opt and row["candidate_id"] == cid:
                return row
        return None

    native = find_summary("native", overall_id)
    adamw = find_summary("pure_adamw", overall_id)
    native_hidden = None
    if hidden_rest_id:
        native_hidden = find_summary("native", hidden_rest_id)
    direct_native = next((r for r in special_rows if r["optimizer"] == "native"), None)
    direct_adamw = next((r for r in special_rows if r["optimizer"] == "pure_adamw"), None)

    lines = [
        "D12 TSA GROUP SEARCH DIGEST",
        "============================",
        f"frozen_sha256={frozen['freeze_sha256']}",
        f"frozen_scalar={frozen_scalar}",
        f"overall_best={overall_id}",
        f"overall_family={frozen['overall_best']['family']}",
        f"hidden_rest_winner={hidden_rest_id}",
        "",
    ]
    if native:
        lines += [
            "PRIMARY: fresh native Muon+AdamW confirmation",
            f"overall vs frozen scalar: {native['mean_gain_vs_baseline']:+.6f} ± {native['halfwidth95']:.6f} BPB (95% t)",
            f"95% interval: [{native['lo95']:+.6f}, {native['hi95']:+.6f}]",
        ]
        if direct_native:
            lines.append(
                f"overall vs hidden/rest winner: {direct_native['mean_gain']:+.6f} ± {direct_native['halfwidth95']:.6f} BPB"
            )
        if native["lo95"] > 0 and (not direct_native or direct_native["lo95"] > 0):
            verdict = "NEW PARAMETER SPLIT CONFIRMED BEYOND SCALAR AND HIDDEN/REST"
        elif native["lo95"] > 0:
            verdict = "NEW RULE BEATS SCALAR, BUT NOT RESOLVED BEYOND HIDDEN/REST"
        elif native["mean_gain_vs_baseline"] > 0:
            verdict = "WEAK POSITIVE; CONFIRMATION INTERVAL INCLUDES ZERO"
        else:
            verdict = "NO CONFIRMED IMPROVEMENT OVER FROZEN SCALAR"
        lines += [f"verdict={verdict}", ""]
    if adamw:
        lines += [
            "TRANSFER: fresh pure-AdamW confirmation (baseline scalar alpha=0.50)",
            f"overall vs scalar0.50: {adamw['mean_gain_vs_baseline']:+.6f} ± {adamw['halfwidth95']:.6f} BPB",
            f"95% interval: [{adamw['lo95']:+.6f}, {adamw['hi95']:+.6f}]",
        ]
        if direct_adamw:
            lines.append(
                f"overall vs hidden/rest winner: {direct_adamw['mean_gain']:+.6f} ± {direct_adamw['halfwidth95']:.6f} BPB"
            )
        lines.append("")
    lines += [
        "See:",
        "  figures/confirmation_summary.csv",
        "  figures/key_contrasts.csv",
        "  DEV_FAMILY_WINNERS.csv",
        "  figures/trajectory_role_stats.csv",
    ]
    digest = figures / "DIGEST.txt"
    digest.write_text("\n".join(lines) + "\n")
    print(digest.read_text())
    return digest


def selftest() -> None:
    s = _summary([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert s["n"] == 8 and abs(s["mean"] - 0.45) < 1e-12
    assert s["lo95"] < s["mean"] < s["hi95"]
    print("analyze_tsa_group_search_d12 selftest PASS")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--expected-dev", type=int, default=8)
    ap.add_argument("--expected-confirm", type=int, default=16)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    if args.root is None:
        raise SystemExit("--root is required")
    if args.freeze:
        freeze(args.root.resolve(), expected_dev=args.expected_dev)
    else:
        analyze(args.root.resolve(), expected_confirm=args.expected_confirm)


if __name__ == "__main__":
    main()
