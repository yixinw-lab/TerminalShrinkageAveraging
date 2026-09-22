#!/usr/bin/env python3
"""Merge/analyze the predeclared D12 groupwise terminal-floor experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

SUITE = "d12_tsa_groupwise_floor_v1"
ARMS = ("uniform05", "uniform10", "matched_uniform", "mapped")
ESTIMATORS = ("raw", "scalar055", "structured")

# Two-sided 95% Student-t critical values, df=1..30.  The protocol uses n=8
# paired streams (df=7), but keeping a small table makes diagnostics graceful.
T975 = {
    1: 12.706205, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
    6: 2.446912, 7: 2.364624, 8: 2.306004, 9: 2.262157, 10: 2.228139,
    11: 2.200985, 12: 2.178813, 13: 2.160369, 14: 2.144787, 15: 2.131450,
    16: 2.119905, 17: 2.109816, 18: 2.100922, 19: 2.093024, 20: 2.085963,
    21: 2.079614, 22: 2.073873, 23: 2.068658, 24: 2.063899, 25: 2.059539,
    26: 2.055529, 27: 2.051831, 28: 2.048407, 29: 2.045230, 30: 2.042272,
}


def _tcrit(n: int) -> float:
    if n <= 1:
        return math.nan
    return T975.get(n - 1, 1.959964)


def _stats(values: Sequence[float]) -> dict[str, float]:
    xs = [float(x) for x in values]
    n = len(xs)
    mean = statistics.mean(xs) if xs else math.nan
    if n <= 1:
        se = hw = math.nan
    else:
        se = statistics.stdev(xs) / math.sqrt(n)
        hw = _tcrit(n) * se
    return {
        "n": n, "mean": mean, "se": se, "halfwidth95": hw,
        "lo95": mean - hw if math.isfinite(hw) else math.nan,
        "hi95": mean + hw if math.isfinite(hw) else math.nan,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _load(root: Path):
    protocol = json.loads((root / "PROTOCOL.json").read_text())
    expected_seeds = [int(x) for x in protocol["stream_seeds"]]
    data: dict[tuple[int, str], dict[str, Any]] = {}
    for p in sorted((root / "runs").glob("*/seed*/result.json")):
        obj = json.loads(p.read_text())
        if obj.get("suite") != SUITE:
            raise RuntimeError(f"wrong suite in {p}: {obj.get('suite')}")
        seed = int(obj["stream_seed"])
        arm = str(obj["arm"])
        key = (seed, arm)
        if key in data:
            raise RuntimeError(f"duplicate result for {key}")
        data[key] = obj

    missing = [(s, a) for s in expected_seeds for a in ARMS if (s, a) not in data]
    extra = [k for k in data if k[0] not in expected_seeds or k[1] not in ARMS]
    if missing:
        raise RuntimeError(f"missing {len(missing)} results, first={missing[:8]}")
    if extra:
        raise RuntimeError(f"unexpected results: {extra[:8]}")

    for seed in expected_seeds:
        fps = {data[(seed, a)]["permutation_fingerprint"] for a in ARMS}
        if len(fps) != 1:
            raise RuntimeError(f"training permutation mismatch within seed {seed}: {fps}")
        shas = {data[(seed, a)]["protocol_sha256"] for a in ARMS}
        if shas != {protocol["protocol_sha256"]}:
            raise RuntimeError(f"protocol hash mismatch within seed {seed}: {shas}")
        matched = [float(data[(seed, a)]["matched_uniform_floor"]) for a in ARMS]
        if max(matched) - min(matched) > 1e-12:
            raise RuntimeError(f"matched floor differs across arms for seed {seed}: {matched}")
        mapped = [json.dumps(data[(seed, a)]["mapped_floors"], sort_keys=True) for a in ARMS]
        if len(set(mapped)) != 1:
            raise RuntimeError(f"mapped floor rule differs across arms for seed {seed}")
    return protocol, expected_seeds, data


def _bpb(data, seed: int, arm: str, estimator: str) -> float:
    return float(data[(seed, arm)]["evaluations"][estimator]["bpb"])


def _contrast(
    seeds: Sequence[int], data,
    *, name: str,
    candidate_arm: str, candidate_estimator: str,
    reference_arm: str, reference_estimator: str,
) -> tuple[dict[str, Any], list[float]]:
    vals = [
        _bpb(data, s, reference_arm, reference_estimator)
        - _bpb(data, s, candidate_arm, candidate_estimator)
        for s in seeds
    ]
    row: dict[str, Any] = {
        "contrast": name,
        "candidate_arm": candidate_arm,
        "candidate_estimator": candidate_estimator,
        "reference_arm": reference_arm,
        "reference_estimator": reference_estimator,
    }
    row.update(_stats(vals))
    return row, vals


def _interaction(
    seeds: Sequence[int], data,
    *, name: str, candidate_arm: str, reference_arm: str,
    estimator: str = "structured",
) -> tuple[dict[str, Any], list[float]]:
    # Positive means the candidate schedule is more favorable under the specified
    # estimator than under raw:
    #   [raw(cand)-raw(ref)] - [est(cand)-est(ref)].
    vals = [
        (_bpb(data, s, candidate_arm, "raw") - _bpb(data, s, reference_arm, "raw"))
        - (_bpb(data, s, candidate_arm, estimator) - _bpb(data, s, reference_arm, estimator))
        for s in seeds
    ]
    row: dict[str, Any] = {
        "contrast": name,
        "candidate_arm": candidate_arm,
        "candidate_estimator": estimator,
        "reference_arm": reference_arm,
        "reference_estimator": "raw/interaction",
    }
    row.update(_stats(vals))
    return row, vals


def _fmt(row: Mapping[str, Any], digits: int = 6) -> str:
    return f"{float(row['mean']):+.{digits}f} ± {float(row['halfwidth95']):.{digits}f}"


def analyze(root: Path) -> None:
    protocol, seeds, data = _load(root)
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    for s in seeds:
        for arm in ARMS:
            obj = data[(s, arm)]
            row = {
                "stream_seed": s,
                "arm": arm,
                "routing_mode": obj["routing_mode"],
                "matched_uniform_floor": obj["matched_uniform_floor"],
            }
            for est in ESTIMATORS:
                row[f"bpb_{est}"] = _bpb(data, s, arm, est)
            row["gain_structured_vs_raw"] = row["bpb_raw"] - row["bpb_structured"]
            row["gain_scalar055_vs_raw"] = row["bpb_raw"] - row["bpb_scalar055"]
            per_seed_rows.append(row)
    _write_csv(out / "per_seed.csv", per_seed_rows)

    arm_rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for est in ESTIMATORS:
            vals = [_bpb(data, s, arm, est) for s in seeds]
            row = {"arm": arm, "estimator": est}
            row.update(_stats(vals))
            arm_rows.append(row)
        gain = [_bpb(data, s, arm, "raw") - _bpb(data, s, arm, "structured") for s in seeds]
        row = {"arm": arm, "estimator": "structured_gain_vs_raw"}
        row.update(_stats(gain))
        arm_rows.append(row)
    _write_csv(out / "arm_summary.csv", arm_rows)

    contrasts: list[dict[str, Any]] = []
    primary, primary_vals = _contrast(
        seeds, data,
        name="PRIMARY_mapped_structured_minus_matched_uniform_structured",
        candidate_arm="mapped", candidate_estimator="structured",
        reference_arm="matched_uniform", reference_estimator="structured",
    )
    contrasts.append(primary)
    primary_i, primary_i_vals = _interaction(
        seeds, data,
        name="PRIMARY_INTERACTION_mapped_vs_matched_uniform",
        candidate_arm="mapped", reference_arm="matched_uniform", estimator="structured",
    )
    contrasts.append(primary_i)

    specs = [
        ("mapped_structured_minus_uniform10_structured", "mapped", "structured", "uniform10", "structured"),
        ("mapped_raw_minus_matched_uniform_raw", "mapped", "raw", "matched_uniform", "raw"),
        ("matched_uniform_structured_minus_uniform10_structured", "matched_uniform", "structured", "uniform10", "structured"),
        ("FULL_RECIPE_mapped_structured_minus_uniform05_raw", "mapped", "structured", "uniform05", "raw"),
        ("mapped_structured_minus_mapped_scalar055", "mapped", "structured", "mapped", "scalar055"),
        ("uniform10_raw_minus_uniform05_raw", "uniform10", "raw", "uniform05", "raw"),
        ("uniform10_structured_minus_uniform05_structured", "uniform10", "structured", "uniform05", "structured"),
    ]
    for name, ca, ce, ra, re in specs:
        row, _ = _contrast(seeds, data, name=name, candidate_arm=ca, candidate_estimator=ce, reference_arm=ra, reference_estimator=re)
        contrasts.append(row)

    u10i, _ = _interaction(
        seeds, data,
        name="INTERACTION_uniform10_vs_uniform05",
        candidate_arm="uniform10", reference_arm="uniform05", estimator="structured",
    )
    contrasts.append(u10i)
    for arm in ARMS:
        row, _ = _contrast(
            seeds, data,
            name=f"structured_minus_raw_within_{arm}",
            candidate_arm=arm, candidate_estimator="structured",
            reference_arm=arm, reference_estimator="raw",
        )
        contrasts.append(row)
    _write_csv(out / "paired_contrasts.csv", contrasts)

    matched_floor = float(data[(seeds[0], "mapped")]["matched_uniform_floor"])
    mapped_floors = data[(seeds[0], "mapped")]["mapped_floors"]
    routing_modes = sorted({data[(s, "mapped")]["routing_mode"] for s in seeds})

    primary_ok = float(primary["lo95"]) > 0
    interaction_ok = float(primary_i["lo95"]) > 0
    verdict = (
        "GROUPWISE FLOOR CONFIRMED VS PARAMETER-COUNT-MATCHED UNIFORM"
        if primary_ok else
        "NO RESOLVED GROUPWISE-FLOOR ADVANTAGE VS MATCHED UNIFORM"
    )
    interaction_verdict = (
        "GROUPWISE SCHEDULE-ESTIMATOR INTERACTION RESOLVED"
        if interaction_ok else
        "GROUPWISE SCHEDULE-ESTIMATOR INTERACTION NOT RESOLVED"
    )

    contrast_by_name = {r["contrast"]: r for r in contrasts}
    digest = [
        "D12 GROUPWISE TERMINAL-FLOOR DIGEST",
        "===================================",
        f"protocol_sha256={protocol['protocol_sha256']}",
        f"streams={len(seeds)} fresh paired native trajectories",
        f"mapped_floors={json.dumps(mapped_floors, sort_keys=True)}",
        f"parameter_count_matched_uniform_floor={matched_floor:.9f}",
        f"mapped_routing_modes={','.join(routing_modes)}",
        "",
        "PRIMARY: structured TSA, mapped groupwise floor vs parameter-count-matched uniform floor",
        f"gain={_fmt(primary, 9)} BPB",
        f"95% interval=[{float(primary['lo95']):+.9f}, {float(primary['hi95']):+.9f}]",
        f"verdict={verdict}",
        "",
        "PRIMARY INTERACTION: does mapped allocation help more under structured TSA than raw?",
        f"interaction={_fmt(primary_i, 9)} BPB",
        f"95% interval=[{float(primary_i['lo95']):+.9f}, {float(primary_i['hi95']):+.9f}]",
        f"interaction_verdict={interaction_verdict}",
        "",
        f"mapped structured vs uniform10 structured: {_fmt(contrast_by_name['mapped_structured_minus_uniform10_structured'], 9)} BPB",
        f"mapped raw vs matched-uniform raw: {_fmt(contrast_by_name['mapped_raw_minus_matched_uniform_raw'], 9)} BPB",
        f"full recipe (mapped structured) vs uniform05 raw: {_fmt(contrast_by_name['FULL_RECIPE_mapped_structured_minus_uniform05_raw'], 9)} BPB",
        f"structured increment over scalar055 on mapped schedule: {_fmt(contrast_by_name['mapped_structured_minus_mapped_scalar055'], 9)} BPB",
        f"uniform10-vs-uniform05 interaction on new streams: {_fmt(contrast_by_name['INTERACTION_uniform10_vs_uniform05'], 9)} BPB",
        "",
        "See:",
        "  figures/paired_contrasts.csv",
        "  figures/arm_summary.csv",
        "  figures/per_seed.csv",
        "  figures/groupwise_floor_summary.png",
        "  figures/APPENDIX_DRAFT.md",
    ]
    (out / "DIGEST.txt").write_text("\n".join(digest) + "\n")

    # A compact figure: mean BPB relative to the paired uniform05 raw endpoint.
    labels = ["Uniform 5%", "Uniform 10%", f"Uniform {100*matched_floor:.2f}%\n(mean-matched)", "Mapped floors"]
    raw_means, raw_hw, struct_means, struct_hw = [], [], [], []
    for arm in ARMS:
        raw_d = [_bpb(data, s, arm, "raw") - _bpb(data, s, "uniform05", "raw") for s in seeds]
        st_d = [_bpb(data, s, arm, "structured") - _bpb(data, s, "uniform05", "raw") for s in seeds]
        rs, ss = _stats(raw_d), _stats(st_d)
        raw_means.append(rs["mean"]); raw_hw.append(rs["halfwidth95"])
        struct_means.append(ss["mean"]); struct_hw.append(ss["halfwidth95"])
    x = np.arange(len(ARMS), dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.errorbar(x - 0.08, raw_means, yerr=raw_hw, fmt="o", capsize=3, label="Raw endpoint")
    ax.errorbar(x + 0.08, struct_means, yerr=struct_hw, fmt="s", capsize=3, label="Frozen structured TSA")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("BPB difference vs paired uniform-5% raw (lower is better)")
    ax.set_title("Predeclared groupwise terminal-floor test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "groupwise_floor_summary.png", dpi=180)
    plt.close(fig)

    appendix = f"""# Appendix draft: parameterwise terminal activity

The structured-TSA experiment indicated that different parameter roles benefit from different amounts of terminal averaging. We therefore asked whether the same heterogeneity predicts the terminal learning-rate activity preferred during training. This experiment was fully predeclared: no new-run hyperparameter search or coefficient fitting was performed. We retained the frozen structured-TSA coefficients `embedding=0`, `hidden=0.65`, `unembedding=0.45`, and `rest=0.25`, and mapped them monotonically onto the previously studied 5%-15% terminal-floor interval according to

`rho_g = 0.05 + 0.10 * alpha_g / 0.65`.

This gives mapped floors {json.dumps(mapped_floors, sort_keys=True)}. For LR routing, the hidden schedule bucket contains matrix-valued tensors inside transformer layers, while 1D layer scales are assigned to the rest bucket; the returned structured-TSA estimator itself remains exactly frozen from the preceding search. As a control for the average amount of terminal activity, we also trained a uniform-floor arm at the parameter-count-weighted mean of these coefficients ({100*matched_floor:.3f}%), together with uniform 5% and 10% controls. All four arms used the same eight fresh stream seeds, initialization, data-order permutation within seed, checkpoint window (`K=8`, `s=32`), and 96-batch validation holdout. The structured estimator was frozen before these trajectories were trained.

The predeclared primary contrast, mapped groupwise activity versus the parameter-count-matched uniform floor under structured TSA, was **{_fmt(primary, 9)} BPB** (95% CI [{float(primary['lo95']):+.9f}, {float(primary['hi95']):+.9f}]; positive favors the mapped schedule). The corresponding schedule-estimator interaction was **{_fmt(primary_i, 9)} BPB** (95% CI [{float(primary_i['lo95']):+.9f}, {float(primary_i['hi95']):+.9f}]). The full mapped-schedule + structured-TSA recipe improved over the paired uniform-5% raw endpoint by **{_fmt(contrast_by_name['FULL_RECIPE_mapped_structured_minus_uniform05_raw'], 9)} BPB**.

These results should be interpreted as a test of a frozen structural hypothesis rather than an optimization over groupwise learning-rate schedules. The matched-uniform control equalizes the arithmetic mean floor across parameter coordinates, but it does not claim to exactly match update energy because the underlying parameter groups use different base learning rates and optimizer geometry.
"""
    (out / "APPENDIX_DRAFT.md").write_text(appendix)

    print((out / "DIGEST.txt").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()
    analyze(args.root.resolve())


if __name__ == "__main__":
    main()
