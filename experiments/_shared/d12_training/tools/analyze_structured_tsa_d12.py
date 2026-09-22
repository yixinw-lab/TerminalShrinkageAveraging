#!/usr/bin/env python3
"""Aggregate the frozen-rule D12 structured-TSA confirmation suite."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SUITE = "d12_structured_tsa_suite_v1"
OPTIMIZERS = ("native", "pure_adamw")
FLOORS = (0.05, 0.10, 0.15)
PRIMARY_RECIPES = (
    "raw",
    "scalar_fixed_050",
    "scalar_fixed_055",
    "scalar_dev_selected",
    "structured_dev_selected",
    "group_only_dev_selected",
    "other_only_dev_selected",
    "lawa",
)


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


def tcrit(n: int) -> float:
    if n < 2:
        return 0.0
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, n - 1))
    except Exception:
        return {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}.get(n, 1.96)


def stats(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    if len(a) == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "se": float("nan"), "ci95_half": float("nan")}
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    se = sd / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "sd": sd,
        "se": se,
        "ci95_half": float(tcrit(len(a)) * se),
        "min": float(a.min()),
        "max": float(a.max()),
        "positive_count": int(np.sum(a > 0)),
        "negative_count": int(np.sum(a < 0)),
    }


def parse_rows(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "confirmation").glob("*/floor_*/seed*/holdout_results.csv"))
    if not paths:
        raise RuntimeError(f"no holdout_results.csv found under {root/'confirmation'}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        for raw in read_csv(path):
            row: dict[str, Any] = dict(raw)
            for key in (
                "alpha_group",
                "alpha_other",
                "bpb",
                "gain_vs_raw",
                "method_minus_raw",
                "paired_eval_se",
                "eval_seconds",
                "floor",
            ):
                if row.get(key, "") not in ("", None):
                    row[key] = float(row[key])
            row["stream_seed"] = int(float(row["stream_seed"]))
            row["source_path"] = str(path)
            rows.append(row)
    return rows


def index_primary(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, float, int, str], dict[str, Any]]:
    out: dict[tuple[str, float, int, str], dict[str, Any]] = {}
    for raw in rows:
        if raw["eval_kind"] != "primary":
            continue
        row = dict(raw)
        key = (str(row["optimizer"]), round(float(row["floor"]), 6), int(row["stream_seed"]), str(row["recipe"]))
        if key in out:
            raise RuntimeError(f"duplicate primary result {key}")
        out[key] = row
    return out


def primary_summary(index: Mapping[tuple[str, float, int, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    methods: dict[tuple[str, float, str], str] = {}
    for (optimizer, floor, seed, recipe), row in index.items():
        groups[(optimizer, floor, recipe)].append(float(row["bpb"]))
        methods[(optimizer, floor, recipe)] = str(row["method"])
    out: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        optimizer, floor, recipe = key
        s = stats(values)
        out.append(
            {
                "optimizer": optimizer,
                "floor": floor,
                "recipe": recipe,
                "method": methods[key],
                "n": s["n"],
                "mean_bpb": s["mean"],
                "bpb_sd": s["sd"],
                "bpb_ci95_half": s["ci95_half"],
                "min_bpb": s["min"],
                "max_bpb": s["max"],
            }
        )
    return out


def paired_contrast(
    index: Mapping[tuple[str, float, int, str], Mapping[str, Any]],
    optimizer: str,
    floor: float,
    left_recipe: str,
    right_recipe: str,
    direction: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seeds = sorted(
        seed
        for opt, fl, seed, rec in index
        if opt == optimizer and abs(fl - floor) < 1e-9 and rec == left_recipe
    )
    per_seed: list[dict[str, Any]] = []
    values: list[float] = []
    for seed in seeds:
        left = float(index[(optimizer, round(floor, 6), seed, left_recipe)]["bpb"])
        right = float(index[(optimizer, round(floor, 6), seed, right_recipe)]["bpb"])
        if direction == "right_better":
            value = left - right
        elif direction == "left_better":
            value = right - left
        else:
            raise ValueError(direction)
        values.append(value)
        per_seed.append(
            {
                "optimizer": optimizer,
                "floor": floor,
                "stream_seed": seed,
                "left_recipe": left_recipe,
                "right_recipe": right_recipe,
                "contrast": value,
            }
        )
    s = stats(values)
    summary = {
        "optimizer": optimizer,
        "floor": floor,
        "left_recipe": left_recipe,
        "right_recipe": right_recipe,
        "positive_means": f"{right_recipe} has lower BPB" if direction == "right_better" else f"{left_recipe} has lower BPB",
        "n": s["n"],
        "mean_contrast": s["mean"],
        "contrast_sd": s["sd"],
        "contrast_ci95_half": s["ci95_half"],
        "positive_seed_count": s["positive_count"],
        "min_contrast": s["min"],
        "max_contrast": s["max"],
    }
    return per_seed, summary


def schedule_effects(index: Mapping[tuple[str, float, int, str], Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for optimizer in OPTIMIZERS:
        seeds = sorted(
            seed for opt, floor, seed, recipe in index if opt == optimizer and floor == 0.05 and recipe == "raw"
        )
        for active_floor in (0.10, 0.15):
            effects: dict[str, list[float]] = {r: [] for r in ("raw", "scalar_dev_selected", "structured_dev_selected")}
            interactions_raw_structured: list[float] = []
            interactions_scalar_structured: list[float] = []
            for seed in seeds:
                seed_effects: dict[str, float] = {}
                for recipe in effects:
                    base = float(index[(optimizer, 0.05, seed, recipe)]["bpb"])
                    active = float(index[(optimizer, active_floor, seed, recipe)]["bpb"])
                    effect = active - base  # positive: active floor worsens BPB
                    effects[recipe].append(effect)
                    seed_effects[recipe] = effect
                    per_seed.append(
                        {
                            "optimizer": optimizer,
                            "active_floor": active_floor,
                            "stream_seed": seed,
                            "quantity": f"schedule_effect_{recipe}",
                            "value": effect,
                        }
                    )
                irs = seed_effects["raw"] - seed_effects["structured_dev_selected"]
                iss = seed_effects["scalar_dev_selected"] - seed_effects["structured_dev_selected"]
                interactions_raw_structured.append(irs)
                interactions_scalar_structured.append(iss)
                per_seed.extend(
                    [
                        {
                            "optimizer": optimizer,
                            "active_floor": active_floor,
                            "stream_seed": seed,
                            "quantity": "interaction_raw_minus_structured",
                            "value": irs,
                        },
                        {
                            "optimizer": optimizer,
                            "active_floor": active_floor,
                            "stream_seed": seed,
                            "quantity": "interaction_scalar_minus_structured",
                            "value": iss,
                        },
                    ]
                )
            for recipe, values in effects.items():
                s = stats(values)
                summary.append(
                    {
                        "optimizer": optimizer,
                        "active_floor": active_floor,
                        "quantity": f"schedule_effect_{recipe}",
                        "positive_means": "active floor worsens BPB",
                        "n": s["n"],
                        "mean": s["mean"],
                        "ci95_half": s["ci95_half"],
                        "positive_seed_count": s["positive_count"],
                    }
                )
            for name, values in (
                ("interaction_raw_minus_structured", interactions_raw_structured),
                ("interaction_scalar_minus_structured", interactions_scalar_structured),
            ):
                s = stats(values)
                summary.append(
                    {
                        "optimizer": optimizer,
                        "active_floor": active_floor,
                        "quantity": name,
                        "positive_means": "active schedule is more favorable under structured TSA",
                        "n": s["n"],
                        "mean": s["mean"],
                        "ci95_half": s["ci95_half"],
                        "positive_seed_count": s["positive_count"],
                    }
                )
    return per_seed, summary


def optimizer_interactions(index: Mapping[tuple[str, float, int, str], Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for floor in FLOORS:
        common = sorted(
            set(seed for opt, fl, seed, rec in index if opt == "native" and fl == floor and rec == "structured_dev_selected")
            & set(seed for opt, fl, seed, rec in index if opt == "pure_adamw" and fl == floor and rec == "structured_dev_selected")
        )
        values: list[float] = []
        for seed in common:
            native_gain = (
                float(index[("native", floor, seed, "scalar_dev_selected")]["bpb"])
                - float(index[("native", floor, seed, "structured_dev_selected")]["bpb"])
            )
            adam_gain = (
                float(index[("pure_adamw", floor, seed, "scalar_dev_selected")]["bpb"])
                - float(index[("pure_adamw", floor, seed, "structured_dev_selected")]["bpb"])
            )
            value = native_gain - adam_gain
            values.append(value)
            per_seed.append(
                {
                    "floor": floor,
                    "stream_seed": seed,
                    "native_structured_over_scalar_gain": native_gain,
                    "adamw_structured_over_scalar_gain": adam_gain,
                    "optimizer_by_estimator_interaction": value,
                }
            )
        s = stats(values)
        summary.append(
            {
                "floor": floor,
                "n": s["n"],
                "mean_optimizer_by_estimator_interaction": s["mean"],
                "ci95_half": s["ci95_half"],
                "positive_seed_count": s["positive_count"],
                "positive_means": "structured-over-scalar advantage is larger for native Muon+AdamW",
            }
        )
    return per_seed, summary


def surface_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float, str, float, float], list[float]] = defaultdict(list)
    for row in rows:
        if row["eval_kind"] not in ("structured_grid_diagnostic", "scalar_grid_diagnostic"):
            continue
        key = (
            str(row["optimizer"]),
            float(row["floor"]),
            str(row["eval_kind"]),
            float(row["alpha_group"]),
            float(row["alpha_other"]),
        )
        grouped[key].append(float(row["gain_vs_raw"]))
    out: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        optimizer, floor, kind, ag, ao = key
        s = stats(values)
        out.append(
            {
                "optimizer": optimizer,
                "floor": floor,
                "eval_kind": kind,
                "alpha_group": ag,
                "alpha_other": ao,
                "n": s["n"],
                "mean_gain_vs_raw": s["mean"],
                "gain_ci95_half": s["ci95_half"],
                "positive_seed_count": s["positive_count"],
            }
        )
    return out


def make_figures(root: Path, dev_summary: Sequence[Mapping[str, Any]], surf: Sequence[Mapping[str, Any]], contrast_seed_rows: Sequence[Mapping[str, Any]], sched_summary: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (root / "figures" / "FIGURE_ERROR.txt").parent.mkdir(parents=True, exist_ok=True)
        (root / "figures" / "FIGURE_ERROR.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        return

    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    def heatmap(records: Sequence[Mapping[str, Any]], optimizer: str, floor: float | None, prefix: str, title: str) -> None:
        selected = [
            r
            for r in records
            if r["optimizer"] == optimizer
            and r.get("family", "structured_grid") in ("structured_grid", None)
            and (floor is None or abs(float(r["floor"]) - floor) < 1e-9)
            and ("eval_kind" not in r or r["eval_kind"] == "structured_grid_diagnostic")
        ]
        if not selected:
            return
        xs = sorted(set(float(r["alpha_other"]) for r in selected))
        ys = sorted(set(float(r["alpha_group"]) for r in selected))
        z = np.full((len(ys), len(xs)), np.nan)
        lookup = {(float(r["alpha_group"]), float(r["alpha_other"])): float(r["mean_gain_vs_raw"]) for r in selected}
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                z[i, j] = lookup.get((y, x), np.nan)
        fig, ax = plt.subplots(figsize=(5.2, 4.3))
        image = ax.imshow(z, origin="lower", aspect="auto", extent=[min(xs), max(xs), min(ys), max(ys)])
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="scalar diagonal")
        ax.set_xlabel("Group-B shrinkage")
        ax.set_ylabel("Group-A shrinkage")
        ax.set_title(title)
        ax.legend(loc="best", frameon=False)
        fig.colorbar(image, ax=ax, label="Mean BPB gain over raw")
        fig.tight_layout()
        fig.savefig(figdir / f"{prefix}.pdf")
        fig.savefig(figdir / f"{prefix}.png", dpi=180)
        plt.close(fig)

    heatmap(dev_summary, "native", None, "development_surface_native", "Native development surface")
    heatmap(dev_summary, "pure_adamw", None, "development_surface_adamw", "AdamW development surface")
    heatmap(surf, "native", 0.10, "confirmation_surface_native_floor10", "Native confirmation diagnostic surface (10% floor)")
    heatmap(surf, "pure_adamw", 0.10, "confirmation_surface_adamw_floor10", "AdamW confirmation diagnostic surface (10% floor)")

    relevant = [
        r
        for r in contrast_seed_rows
        if r["left_recipe"] == "scalar_dev_selected" and r["right_recipe"] == "structured_dev_selected"
    ]
    if relevant:
        labels: list[str] = []
        values: list[float] = []
        for r in sorted(relevant, key=lambda x: (x["optimizer"], x["floor"], x["stream_seed"])):
            labels.append(f"{r['optimizer']}\n{int(round(100*r['floor']))}%\n{r['stream_seed']}")
            values.append(float(r["contrast"]))
        fig, ax = plt.subplots(figsize=(max(7.0, 0.38 * len(values)), 4.0))
        ax.axhline(0.0, linewidth=1.0)
        ax.plot(range(len(values)), values, marker="o", linestyle="none")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_ylabel("Scalar BPB - structured BPB")
        ax.set_title("Frozen structured TSA versus frozen scalar TSA")
        fig.tight_layout()
        fig.savefig(figdir / "structured_vs_scalar_per_seed.pdf")
        fig.savefig(figdir / "structured_vs_scalar_per_seed.png", dpi=180)
        plt.close(fig)

    records = [
        r
        for r in sched_summary
        if r["optimizer"] == "native" and r["quantity"] in (
            "schedule_effect_raw",
            "schedule_effect_scalar_dev_selected",
            "schedule_effect_structured_dev_selected",
        )
    ]
    if records:
        labels = [f"{int(round(100*r['active_floor']))}% {r['quantity'].replace('schedule_effect_', '')}" for r in records]
        means = [float(r["mean"]) for r in records]
        errors = [float(r["ci95_half"]) for r in records]
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.axhline(0.0, linewidth=1.0)
        ax.errorbar(range(len(means)), means, yerr=errors, fmt="o", capsize=3)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Active-floor BPB - 5%-floor BPB")
        ax.set_title("Native schedule effects under frozen estimators")
        fig.tight_layout()
        fig.savefig(figdir / "native_schedule_effects.pdf")
        fig.savefig(figdir / "native_schedule_effects.png", dpi=180)
        plt.close(fig)


def analyze(root: Path) -> Path:
    frozen_path = root / "FROZEN_RULES.json"
    schema_path = root / "GROUP_SCHEMA.json"
    if not frozen_path.exists() or not schema_path.exists():
        raise RuntimeError("missing FROZEN_RULES.json or GROUP_SCHEMA.json")
    frozen = json.loads(frozen_path.read_text())
    schema = json.loads(schema_path.read_text())
    if frozen.get("suite") != SUITE:
        raise RuntimeError("wrong frozen-rule suite")

    rows = parse_rows(root)
    index = index_primary(rows)
    expected = len(OPTIMIZERS) * len(FLOORS) * 5
    run_keys = set((opt, floor, seed) for opt, floor, seed, recipe in index if recipe == "raw")
    if len(run_keys) != expected:
        raise RuntimeError(f"expected {expected} confirmation runs, found {len(run_keys)}")

    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    psummary = primary_summary(index)
    write_csv(figdir / "primary_summary.csv", psummary)

    contrast_seed_rows: list[dict[str, Any]] = []
    contrast_summary_rows: list[dict[str, Any]] = []
    contrasts = (
        ("raw", "scalar_dev_selected", "right_better"),
        ("raw", "structured_dev_selected", "right_better"),
        ("scalar_dev_selected", "structured_dev_selected", "right_better"),
        ("raw", "group_only_dev_selected", "right_better"),
        ("raw", "other_only_dev_selected", "right_better"),
        ("structured_dev_selected", "lawa", "left_better"),
    )
    for optimizer in OPTIMIZERS:
        for floor in FLOORS:
            for left, right, direction in contrasts:
                seed_rows, summary = paired_contrast(index, optimizer, floor, left, right, direction)
                contrast_seed_rows.extend(seed_rows)
                contrast_summary_rows.append(summary)
    write_csv(figdir / "paired_contrasts_per_seed.csv", contrast_seed_rows)
    write_csv(figdir / "paired_contrasts_summary.csv", contrast_summary_rows)

    sched_seed, sched_summary = schedule_effects(index)
    write_csv(figdir / "schedule_effects_per_seed.csv", sched_seed)
    write_csv(figdir / "schedule_effects_summary.csv", sched_summary)

    opt_seed, opt_summary = optimizer_interactions(index)
    write_csv(figdir / "optimizer_interactions_per_seed.csv", opt_seed)
    write_csv(figdir / "optimizer_interactions_summary.csv", opt_summary)

    surf = surface_summary(rows)
    write_csv(figdir / "confirmation_surface_summary.csv", surf)

    dev_summary_path = root / "development_surface_summary.csv"
    dev_summary: list[dict[str, Any]] = []
    if dev_summary_path.exists():
        for raw in read_csv(dev_summary_path):
            row: dict[str, Any] = dict(raw)
            for key in ("alpha_group", "alpha_other", "mean_gain_vs_raw", "gain_ci95_half"):
                if row.get(key, "") != "":
                    row[key] = float(row[key])
            dev_summary.append(row)

    make_figures(root, dev_summary, surf, contrast_seed_rows, sched_summary)

    def find_contrast(optimizer: str, floor: float, left: str, right: str) -> Mapping[str, Any]:
        return next(
            r
            for r in contrast_summary_rows
            if r["optimizer"] == optimizer
            and abs(float(r["floor"]) - floor) < 1e-9
            and r["left_recipe"] == left
            and r["right_recipe"] == right
        )

    primary = find_contrast("native", 0.10, "scalar_dev_selected", "structured_dev_selected")
    native_rule = frozen["rules"]["native"]
    lower = float(primary["mean_contrast"]) - float(primary["contrast_ci95_half"])
    off_diagonal = bool(native_rule["off_diagonal"])
    signs = int(primary["positive_seed_count"])
    if off_diagonal and lower > 0.0 and signs >= 4:
        recommendation = "YES — recenter the paper on structured TSA"
    elif off_diagonal and float(primary["mean_contrast"]) > 0.0 and signs >= 4:
        recommendation = "MAYBE — structured TSA is promising but keep scalar TSA primary unless D22 also confirms"
    else:
        recommendation = "NO — retain scalar TSA as the main method and report structure as a diagnostic"

    lines = [
        "D12 structured TSA confirmation",
        "================================",
        f"group-A fraction of parameters: {float(schema['group_a']['fraction']):.4%}",
        f"native frozen rule: group={native_rule['alpha_group']:.3f}, other={native_rule['alpha_other']:.3f}",
        f"native frozen scalar: alpha={native_rule['scalar_alpha']:.3f}",
        "",
        "PRIMARY DECISION GATE (native, 10% floor)",
        f"structured advantage over selected scalar: {float(primary['mean_contrast']):+.6f} +/- {float(primary['contrast_ci95_half']):.6f}",
        f"positive seeds: {int(primary['positive_seed_count'])}/{int(primary['n'])}",
        f"95% interval lower endpoint: {lower:+.6f}",
        f"development optimum off diagonal: {off_diagonal}",
        f"PIVOT_RECOMMENDATION={recommendation}",
    ]

    for optimizer in OPTIMIZERS:
        rule = frozen["rules"][optimizer]
        lines.extend(
            [
                "",
                f"[{optimizer}] frozen structured=({rule['alpha_group']:.3f},{rule['alpha_other']:.3f}); scalar={rule['scalar_alpha']:.3f}",
            ]
        )
        for floor in FLOORS:
            ss = find_contrast(optimizer, floor, "scalar_dev_selected", "structured_dev_selected")
            rr = find_contrast(optimizer, floor, "raw", "structured_dev_selected")
            lines.append(
                f"floor={floor:.2f}: structured-vs-scalar={float(ss['mean_contrast']):+.6f} +/- {float(ss['contrast_ci95_half']):.6f}; "
                f"structured-vs-raw={float(rr['mean_contrast']):+.6f} +/- {float(rr['contrast_ci95_half']):.6f}"
            )

    for optimizer in OPTIMIZERS:
        lines.append("")
        lines.append(f"[{optimizer}] schedule effects (positive means active floor is worse)")
        for active in (0.10, 0.15):
            pieces = []
            for quantity in (
                "schedule_effect_raw",
                "schedule_effect_scalar_dev_selected",
                "schedule_effect_structured_dev_selected",
                "interaction_raw_minus_structured",
            ):
                row = next(
                    r
                    for r in sched_summary
                    if r["optimizer"] == optimizer
                    and abs(float(r["active_floor"]) - active) < 1e-9
                    and r["quantity"] == quantity
                )
                pieces.append(f"{quantity}={float(row['mean']):+.6f}+/-{float(row['ci95_half']):.6f}")
            lines.append(f"5% -> {int(round(100*active))}%: " + "; ".join(pieces))

    lines.extend(
        [
            "",
            "Files:",
            f"  {figdir/'primary_summary.csv'}",
            f"  {figdir/'paired_contrasts_summary.csv'}",
            f"  {figdir/'schedule_effects_summary.csv'}",
            f"  {figdir/'optimizer_interactions_summary.csv'}",
            f"  {figdir/'confirmation_surface_summary.csv'}",
        ]
    )
    digest = figdir / "DIGEST.txt"
    digest.write_text("\n".join(lines) + "\n")

    readme = figdir / "RESULTS_README.md"
    readme.write_text(
        "# D12 structured TSA result conventions\n\n"
        "The primary decision gate is the paired confirmation contrast at the native 10% floor: "
        "selected-scalar BPB minus selected-structured BPB. Positive values favor structured TSA. "
        "The structured and scalar rules were frozen using only the separate development trajectories "
        "and the calibration validation block. Diagnostic confirmation grids were evaluated only after "
        "the freeze file existed.\n\n"
        "Schedule effects are active-floor BPB minus 5%-floor BPB, so positive values mean the more "
        "active terminal floor worsened the returned model. The raw-minus-structured interaction is "
        "positive when the active schedule is more favorable under structured TSA than under the raw endpoint.\n"
    )
    print(digest.read_text())
    return digest


def selftest() -> None:
    s = stats([1.0, 2.0, 3.0])
    assert s["n"] == 3 and abs(s["mean"] - 2.0) < 1e-12
    assert s["positive_count"] == 3
    print("structured TSA analysis selftest PASS")


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
    analyze(args.root.resolve())


if __name__ == "__main__":
    main()
