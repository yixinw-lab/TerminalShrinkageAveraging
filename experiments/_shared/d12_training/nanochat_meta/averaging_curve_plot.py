#!/usr/bin/env python3
"""Aggregate multiseed optimizer x averaging experiments into clean loss curves."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "raw": "No averaging",
    "uniform:4:16": "LAWA (4 x 16)",
    "uniform:8:32": "LAWA (8 x 32)",
    "ema:0.95:16:4": "EMA (beta=0.95)",
    "group:muon:8:32": "Muon-only LAWA",
    "adaptive:hybrid:8:32:1.0:all": "Adaptive (all tensors)",
    "adaptive:hybrid:8:32:1.0:muon": "Adaptive (Muon tensors)",
}

PRIMARY = {
    "pure_adamw": [
        "raw",
        "uniform:4:16",
        "uniform:8:32",
        "ema:0.95:16:4",
        "adaptive:hybrid:8:32:1.0:all",
    ],
    "native": [
        "raw",
        "uniform:4:16",
        "uniform:8:32",
        "ema:0.95:16:4",
        "adaptive:hybrid:8:32:1.0:muon",
    ],
}

OPT_LABELS = {
    "pure_adamw": "AdamW",
    "native": "Muon + AdamW",
}


def _number(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("result.json")):
        try:
            result = json.loads(path.read_text())
        except Exception as exc:  # pragma: no cover - cluster diagnostics
            print(f"skip unreadable {path}: {exc}")
            continue
        evals = result.get("recipe_evaluations")
        if not evals:
            continue
        runtime = result.get("runtime_args", {})
        seed = runtime.get("seed", result.get("seed", result.get("build_args", {}).get("seed")))
        optimizer = str(result.get("optimizer", result.get("build_args", {}).get("optimizer", "unknown")))
        trajectory = str(result.get("trajectory_id", "unknown"))
        for row in evals:
            rows.append({
                "optimizer": optimizer,
                "seed": int(seed),
                "trajectory_id": trajectory,
                "recipe": str(row["recipe"]),
                "step": int(row["step"]),
                "charged_flop_equiv": _number(row.get("charged_flop_equiv")),
                "bpb": _number(row.get("bpb")),
                "bpb_se_within_run": _number(row.get("bpb_se")),
                "eval_kind": str(row.get("eval_kind", "legacy")),
                "eval_batches_count": int(row.get("eval_batches_count", 0) or 0),
                "result_path": str(path),
            })
    if not rows:
        raise SystemExit(f"no averaging recipe evaluations under {root}")

    # Remove accidental duplicates while retaining both the fixed-batch curve
    # row and the higher-precision final row at the same optimizer step.
    best: dict[tuple[str, int, str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["optimizer"], row["seed"], row["recipe"], row["step"], row["eval_kind"]
        )
        old = best.get(key)
        if old is None or row["eval_batches_count"] > old["eval_batches_count"]:
            best[key] = row
    return sorted(
        best.values(),
        key=lambda r: (r["optimizer"], r["seed"], r["recipe"], r["step"], r["eval_kind"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def curve_subset(rows: list[dict[str, Any]], optimizer: str, recipe: str) -> list[dict[str, Any]]:
    subset = [r for r in rows if r["optimizer"] == optimizer and r["recipe"] == recipe]
    curve = [r for r in subset if r["eval_kind"] == "curve"]
    if curve:
        return curve
    # Backward compatibility for results produced before eval_kind existed.
    return [r for r in subset if r["eval_kind"] in {"legacy", "final"}]


def aggregate_curve(rows: list[dict[str, Any]], optimizer: str, recipe: str) -> list[dict[str, float]]:
    subset = curve_subset(rows, optimizer, recipe)
    steps = sorted({int(r["step"]) for r in subset})
    out: list[dict[str, float]] = []
    for step in steps:
        step_rows = [r for r in subset if int(r["step"]) == step]
        vals = np.asarray([r["bpb"] for r in step_rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        sem = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        charged = np.asarray([
            r["charged_flop_equiv"] for r in step_rows if np.isfinite(r["charged_flop_equiv"])
        ], dtype=float)
        out.append({
            "step": float(step),
            "charged_flop_equiv": float(charged.mean()) if charged.size else float(step),
            "mean_bpb": float(vals.mean()),
            "seed_sem": sem,
            "n_seeds": float(vals.size),
        })
    return out


def aggregate_gain(rows: list[dict[str, Any]], optimizer: str, recipe: str) -> list[dict[str, float]]:
    if recipe == "raw":
        return []
    method = curve_subset(rows, optimizer, recipe)
    raw = curve_subset(rows, optimizer, "raw")
    raw_by_key = {(r["seed"], r["step"]): r for r in raw}
    paired: list[dict[str, float]] = []
    for row in method:
        base = raw_by_key.get((row["seed"], row["step"]))
        if base is None:
            continue
        paired.append({
            "seed": float(row["seed"]),
            "step": float(row["step"]),
            "charged_flop_equiv": float(row["charged_flop_equiv"]),
            "gain": float(base["bpb"] - row["bpb"]),
        })
    out: list[dict[str, float]] = []
    for step in sorted({int(r["step"]) for r in paired}):
        step_rows = [r for r in paired if int(r["step"]) == step]
        vals = np.asarray([r["gain"] for r in step_rows], dtype=float)
        sem = float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0
        charged = np.asarray([r["charged_flop_equiv"] for r in step_rows], dtype=float)
        out.append({
            "step": float(step),
            "charged_flop_equiv": float(charged.mean()),
            "mean_gain": float(vals.mean()),
            "seed_sem": sem,
            "n_seeds": float(vals.size),
        })
    return out


def _primary_recipes(rows: list[dict[str, Any]], optimizer: str) -> list[str]:
    return [r for r in PRIMARY[optimizer] if any(
        x["optimizer"] == optimizer and x["recipe"] == r for x in rows
    )]


def plot_absolute(rows: list[dict[str, Any]], optimizer: str, out: Path) -> None:
    recipes = _primary_recipes(rows, optimizer)
    if not recipes:
        print(f"no primary rows for {optimizer}")
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for recipe in recipes:
        agg = aggregate_curve(rows, optimizer, recipe)
        x = np.asarray([r["charged_flop_equiv"] for r in agg], dtype=float)
        y = np.asarray([r["mean_bpb"] for r in agg], dtype=float)
        err = np.asarray([r["seed_sem"] for r in agg], dtype=float)
        ax.errorbar(
            x, y, yerr=err,
            marker="o", markersize=3.5, linewidth=1.5, capsize=2,
            label=LABELS.get(recipe, recipe),
        )
    ax.set_xlabel("Training-equivalent optimizer steps")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.set_title(f"Validation loss on {OPT_LABELS.get(optimizer, optimizer)}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = "adamw" if optimizer == "pure_adamw" else "muon"
    fig.savefig(out / f"fig_{stem}_averaging_loss_curves.png", dpi=220)
    fig.savefig(out / f"fig_{stem}_averaging_loss_curves.pdf")
    plt.close(fig)


def plot_late_zoom(rows: list[dict[str, Any]], optimizer: str, out: Path) -> None:
    recipes = _primary_recipes(rows, optimizer)
    series = {recipe: aggregate_curve(rows, optimizer, recipe) for recipe in recipes}
    all_steps = sorted({int(p["step"]) for points in series.values() for p in points})
    if not all_steps:
        return
    keep_steps = set(all_steps[-5:])
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for recipe, agg in series.items():
        agg = [p for p in agg if int(p["step"]) in keep_steps]
        x = np.asarray([r["charged_flop_equiv"] for r in agg], dtype=float)
        y = np.asarray([r["mean_bpb"] for r in agg], dtype=float)
        err = np.asarray([r["seed_sem"] for r in agg], dtype=float)
        ax.errorbar(
            x, y, yerr=err,
            marker="o", markersize=4, linewidth=1.6, capsize=2,
            label=LABELS.get(recipe, recipe),
        )
    ax.set_xlabel("Training-equivalent optimizer steps")
    ax.set_ylabel("Validation BPB (lower is better)")
    ax.set_title(f"Late-stage validation loss on {OPT_LABELS.get(optimizer, optimizer)}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = "adamw" if optimizer == "pure_adamw" else "muon"
    fig.savefig(out / f"fig_{stem}_averaging_late_zoom.png", dpi=220)
    fig.savefig(out / f"fig_{stem}_averaging_late_zoom.pdf")
    plt.close(fig)


def plot_gain(rows: list[dict[str, Any]], optimizer: str, out: Path) -> None:
    recipes = [r for r in _primary_recipes(rows, optimizer) if r != "raw"]
    if not recipes:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    for recipe in recipes:
        agg = aggregate_gain(rows, optimizer, recipe)
        x = np.asarray([r["charged_flop_equiv"] for r in agg], dtype=float)
        y = np.asarray([r["mean_gain"] for r in agg], dtype=float)
        err = np.asarray([r["seed_sem"] for r in agg], dtype=float)
        ax.errorbar(
            x, y, yerr=err,
            marker="o", markersize=4, linewidth=1.6, capsize=2,
            label=LABELS.get(recipe, recipe),
        )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Training-equivalent optimizer steps")
    ax.set_ylabel("BPB improvement over no averaging (higher is better)")
    ax.set_title(f"Averaging gain on {OPT_LABELS.get(optimizer, optimizer)}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    stem = "adamw" if optimizer == "pure_adamw" else "muon"
    fig.savefig(out / f"fig_{stem}_averaging_gain_curves.png", dpi=220)
    fig.savefig(out / f"fig_{stem}_averaging_gain_curves.pdf")
    plt.close(fig)


def final_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    final: list[dict[str, Any]] = []
    for optimizer in sorted({r["optimizer"] for r in rows}):
        for seed in sorted({r["seed"] for r in rows if r["optimizer"] == optimizer}):
            sr = [r for r in rows if r["optimizer"] == optimizer and r["seed"] == seed]
            if not sr:
                continue
            final_step = max(int(r["step"]) for r in sr)
            preferred = [
                r for r in sr if int(r["step"]) == final_step and r["eval_kind"] == "final"
            ]
            fr = preferred or [r for r in sr if int(r["step"]) == final_step]
            # If legacy data contain duplicate rows, retain the one with the
            # largest evaluation set for each recipe.
            by_recipe: dict[str, dict[str, Any]] = {}
            for row in fr:
                old = by_recipe.get(row["recipe"])
                if old is None or row["eval_batches_count"] > old["eval_batches_count"]:
                    by_recipe[row["recipe"]] = row
            fr = list(by_recipe.values())
            raw = next((r for r in fr if r["recipe"] == "raw"), None)
            if raw is None:
                continue
            for r in fr:
                final.append({
                    **r,
                    "delta_vs_raw": float(r["bpb"] - raw["bpb"]),
                    "improvement_vs_raw": float(raw["bpb"] - r["bpb"]),
                })

    summary: list[dict[str, Any]] = []
    keys = sorted({(r["optimizer"], r["recipe"]) for r in final})
    for optimizer, recipe in keys:
        vals = np.asarray([
            r["bpb"] for r in final if r["optimizer"] == optimizer and r["recipe"] == recipe
        ], dtype=float)
        gains = np.asarray([
            r["improvement_vs_raw"] for r in final
            if r["optimizer"] == optimizer and r["recipe"] == recipe
        ], dtype=float)
        if vals.size == 0:
            continue
        summary.append({
            "optimizer": optimizer,
            "optimizer_label": OPT_LABELS.get(optimizer, optimizer),
            "recipe": recipe,
            "method": LABELS.get(recipe, recipe),
            "n_seeds": int(vals.size),
            "mean_final_bpb": float(vals.mean()),
            "final_bpb_seed_sem": float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0,
            "mean_improvement_vs_raw": float(gains.mean()),
            "improvement_seed_sem": float(gains.std(ddof=1) / math.sqrt(gains.size)) if gains.size > 1 else 0.0,
        })
    return final, summary


def write_digest(out: Path, summary: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for optimizer in ("pure_adamw", "native"):
        opt_rows = [r for r in summary if r["optimizer"] == optimizer]
        if not opt_rows:
            continue
        opt_rows.sort(key=lambda r: r["mean_final_bpb"])
        lines.append(OPT_LABELS.get(optimizer, optimizer))
        lines.append("-" * 78)
        for row in opt_rows:
            lines.append(
                f"{row['method']:<28} BPB={row['mean_final_bpb']:.6f} "
                f"gain={row['mean_improvement_vs_raw']:+.6f} "
                f"seed_SE={row['improvement_seed_sem']:.6f} n={row['n_seeds']}"
            )
        lines.append("")
    (out / "DIGEST.txt").write_text("\n".join(lines).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.root)
    write_csv(args.out / "curve_points.csv", rows)
    for optimizer in ("pure_adamw", "native"):
        plot_absolute(rows, optimizer, args.out)
        plot_late_zoom(rows, optimizer, args.out)
        plot_gain(rows, optimizer, args.out)
    final, summary = final_tables(rows)
    write_csv(args.out / "final_seed_results.csv", final)
    write_csv(args.out / "final_summary.csv", summary)
    write_digest(args.out, summary)

    print(f"loaded {len(rows)} evaluation rows from {args.root}")
    print(f"wrote figures and summaries to {args.out}")


if __name__ == "__main__":
    main()
