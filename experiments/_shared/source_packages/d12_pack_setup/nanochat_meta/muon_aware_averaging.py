#!/usr/bin/env python3
"""Muon-aware and adaptive checkpoint averaging experiments.

One exact optimizer trajectory is trained once per learning-rate/schedule config.
Many evaluation-only averaging recipes are then evaluated on exactly the same
trajectory. This makes the group attribution experiment cheap and statistically
clean, and avoids pretending that uniform LAWA is a novel optimizer.

Recipe grammar
--------------
raw
uniform:<window>:<spacing_steps>
group:<selector>:<window>:<spacing_steps>
adaptive:<rule>:<window>:<spacing_steps>:<strength>[:<selector>]
ema:<beta>:<window>:<spacing_steps>
recency:<power>:<window>:<spacing_steps>

Selectors include all, muon, adamw, attention, mlp, embedding, value_embed,
lm_head, scalar, early, middle, late, muon_attention, muon_mlp.
Adaptive rules include snr, autocorr, hybrid, max, product.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from .skip_core import Timer, apply_lr_schedule, seed_everything
from .skip_suite import PACK_VERSION, exact_gradient, restore_branch
from .filter_suite import (
    BudgetLedger,
    OPTIMIZER_OPS_PER_PARAMETER,
    evaluate_detailed,
    _install_canonical_build,
    _stabilize_nanochat_adamw_kernel,
)
from .trajectory_groups import (
    SparseSnapshotBank,
    average_selected,
    build_slice_infos,
    make_averaged_candidate,
    selector_matches,
    tensor_stats,
    weighted_group_aggregate,
)


def parse_recipe(line: str) -> tuple[str, list[str]]:
    parts = line.strip().split(":")
    return parts[0], parts[1:]


def load_recipes(path: Path) -> list[str]:
    rows = [x.strip() for x in path.read_text().splitlines() if x.strip() and not x.lstrip().startswith("#")]
    if "raw" not in rows:
        rows.insert(0, "raw")
    return rows


def recipe_window_spacing(recipe: str) -> tuple[int, int] | None:
    name, x = parse_recipe(recipe)
    if name == "raw":
        return None
    if name == "uniform":
        return int(x[0]), int(x[1])
    if name == "group":
        return int(x[1]), int(x[2])
    if name == "adaptive":
        return int(x[1]), int(x[2])
    if name in {"ema", "recency"}:
        return int(x[1]), int(x[2])
    raise ValueError(f"unknown recipe {recipe!r}")


def paired_delta(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape or aa.size == 0:
        return float("nan"), float("nan"), float("nan")
    d = aa - bb
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else 0.0
    z = mean / se if se > 0 else (float("inf") if mean > 0 else -float("inf") if mean < 0 else 0.0)
    return mean, se, z


def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _apply_lr(optimizer: torch.optim.Optimizer, step: int, args: argparse.Namespace,
              runtime: argparse.Namespace) -> float:
    warmdown = args.warmdown_ratio if runtime.warmdown_ratio is None else runtime.warmdown_ratio
    final_frac = args.final_lr_frac if runtime.final_lr_frac is None else runtime.final_lr_frac
    mult = apply_lr_schedule(
        optimizer, step, args.warmup_steps, args.total_iterations,
        warmdown, final_frac,
    )
    for group in optimizer.param_groups:
        group["lr"] *= float(runtime.lr_scale)
    return mult * float(runtime.lr_scale)


def _micro_tokens(args: argparse.Namespace) -> int:
    return int(args.device_batch_size) * int(args.max_seq_len)


def evaluate_recipes(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: Any,
    infos: Sequence[Any],
    bank: SparseSnapshotBank,
    recipes: Sequence[str],
    eval_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    token_bytes: Any,
    device: torch.device,
    *,
    step: int,
    n_layers: int,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    current = spec.flatten_parameters(dtype=torch.float32)
    with Timer() as rt:
        raw = evaluate_detailed(model, eval_batches, token_bytes, device)
    rows: list[dict[str, Any]] = [{"recipe": "raw", "step": step, **raw, "eval_seconds": rt.seconds}]
    stats_by_key: dict[str, list[dict[str, Any]]] = {}
    cache: dict[tuple[int, int], tuple[torch.Tensor, list[dict[str, Any]]]] = {}

    for recipe in recipes:
        name, x = parse_recipe(recipe)
        if name == "raw":
            continue
        ws = recipe_window_spacing(recipe)
        assert ws is not None
        window, spacing = ws
        key = (window, spacing)
        if key not in cache:
            selected = bank.select(window, spacing, end_step=step)
            avg = average_selected(selected, dtype=torch.float32)
            stats = tensor_stats(selected, infos)
            cache[key] = (avg, stats)
            stats_by_key[f"{window}x{spacing}"] = stats
        avg, stats = cache[key]
        if name in {"ema", "recency"}:
            selected = bank.select(window, spacing, end_step=step)
            n = len(selected)
            if name == "ema":
                beta = float(x[0])
                weights = np.asarray([beta ** (n - 1 - i) for i in range(n)], dtype=float)
            else:
                power = float(x[0])
                weights = np.asarray([(i + 1) ** power for i in range(n)], dtype=float)
            weights /= weights.sum()
            weighted = torch.zeros_like(selected[0][1], dtype=torch.float32, device="cpu")
            for weight, (_, vec) in zip(weights, selected):
                weighted.add_(vec.float(), alpha=float(weight))
            avg = weighted

        if name in {"uniform", "ema", "recency"}:
            candidate, alphas = make_averaged_candidate(
                current, avg, infos, selector="all", n_layers=n_layers,
            )
        elif name == "group":
            selector = x[0]
            candidate, alphas = make_averaged_candidate(
                current, avg, infos, selector=selector, n_layers=n_layers,
            )
        elif name == "adaptive":
            rule = x[0]
            strength = float(x[3])
            selector = x[4] if len(x) > 4 else "all"
            candidate, alphas = make_averaged_candidate(
                current, avg, infos, selector="all", n_layers=n_layers,
                adaptive_rule=rule, adaptive_strength=strength,
                stats_rows=stats, adaptive_selector=selector,
            )
        else:
            raise ValueError(recipe)

        alpha_num = sum(float(a["alpha"]) * int(a["numel"]) for a in alphas)
        alpha_den = max(sum(int(a["numel"]) for a in alphas), 1)
        try:
            spec.assign_parameters(candidate)
            with Timer() as et:
                ev = evaluate_detailed(model, eval_batches, token_bytes, device)
        finally:
            spec.assign_parameters(current)
        delta, se, z = paired_delta(ev["bpb_values"], raw["bpb_values"])
        rows.append({
            "recipe": recipe,
            "step": step,
            **ev,
            "eval_seconds": et.seconds,
            "paired_delta_vs_raw": delta,
            "paired_se_vs_raw": se,
            "paired_z_vs_raw": z,
            "mean_alpha_by_numel": alpha_num / alpha_den,
            "displacement_norm": float((candidate - current).norm()),
        })
    return rows, stats_by_key


def run(runtime: argparse.Namespace) -> Path:
    _stabilize_nanochat_adamw_kernel()
    pack = torch.load(runtime.pack, map_location="cpu", weights_only=False)
    if int(pack.get("pack_version", -1)) != PACK_VERSION:
        raise RuntimeError("pack version mismatch")
    args, model, optimizer, token_bytes, device, spec, _history, _bank = restore_branch(pack, runtime)
    seed_everything(int(pack.get("seed", 1337)) if runtime.seed is None else int(runtime.seed))
    recipes = load_recipes(runtime.recipes_file)
    infos = build_slice_infos(spec, optimizer)
    n_layers = int(getattr(model.config, "n_layer", getattr(model.config, "depth", args.depth)))

    max_span = 1
    for recipe in recipes:
        ws = recipe_window_spacing(recipe)
        if ws:
            max_span = max(max_span, 1 + (ws[0] - 1) * ws[1])
    max_snaps = max(4, math.ceil(max_span / runtime.snapshot_every) + 8)
    bank = SparseSnapshotBank(max_snaps, runtime.snapshot_every, _dtype(runtime.snapshot_dtype))

    current = spec.flatten_parameters(dtype=torch.float32)
    bank.append(0, current, force=True)
    stream = pack["branch_batches"]
    cursor = 0
    micro_tokens = _micro_tokens(args)
    full_tokens = micro_tokens * int(args.grad_accum)
    ledger = BudgetLedger(spec.numel, full_tokens)
    raw_trace: list[dict[str, Any]] = []
    recipe_evals: list[dict[str, Any]] = []
    tensor_stats_out: dict[str, list[dict[str, Any]]] = {}
    eval_points = sorted({float(x) for x in runtime.recipe_eval_equivs.split(",") if x.strip()})

    # The full evaluation set can be expensive when every averaging recipe is
    # reconstructed at many curve anchors. For curve experiments we use a
    # deterministic prefix of the cached evaluation batches at intermediate
    # anchors, then re-evaluate the final endpoint on the full requested set.
    all_eval_batches = pack["eval_batches"]
    if not all_eval_batches:
        raise RuntimeError("pack contains no evaluation batches")
    curve_n = len(all_eval_batches) if int(runtime.curve_eval_batches) <= 0 else min(
        int(runtime.curve_eval_batches), len(all_eval_batches)
    )
    final_n = len(all_eval_batches) if int(runtime.final_eval_batches) <= 0 else min(
        int(runtime.final_eval_batches), len(all_eval_batches)
    )
    curve_eval_batches = all_eval_batches[:curve_n]
    final_eval_batches = all_eval_batches[:final_n]
    next_raw = 0.0
    eval_idx = 0

    def raw_record() -> None:
        nonlocal next_raw
        with Timer() as et:
            ev = evaluate_detailed(model, curve_eval_batches, token_bytes, device)
        ledger.eval_seconds += et.seconds
        raw_trace.append({
            "step": ledger.optimizer_steps,
            "eval_kind": "curve",
            "eval_batches_count": curve_n,
            **ev,
            **ledger.as_dict(),
        })
        print(
            f"[traj {runtime.trajectory_id}] step={ledger.optimizer_steps:5d} "
            f"F={ledger.charged_flop_equiv:8.2f} bpb={ev['bpb']:.6f}+/-{ev['bpb_se']:.6f} "
            f"eval_batches={curve_n}"
        )
        next_raw += runtime.eval_every_equiv

    raw_record()
    termination = "budget"
    while ledger.charged_flop_equiv < runtime.compute_budget:
        k = int(args.grad_accum)
        if cursor + k > len(stream):
            termination = "stream_exhausted"
            break
        batches = stream[cursor:cursor + k]
        cursor += k
        global_step = int(pack["prefix_steps"]) + ledger.optimizer_steps
        _apply_lr(optimizer, global_step, args, runtime)
        with Timer() as tt:
            _g, train_loss, _, _ = exact_gradient(
                model, optimizer, spec, batches, device,
                need_forward_features=False,
                token_sample_max=args.token_sample_max,
            )
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        ledger.add_exact(full_tokens, k, tt.seconds, charged=True)
        ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
        ledger.optimizer_steps += 1
        theta = spec.flatten_parameters(dtype=torch.float32)
        bank.append(ledger.optimizer_steps, theta)

        while eval_idx < len(eval_points) and ledger.charged_flop_equiv >= eval_points[eval_idx]:
            bank.append(ledger.optimizer_steps, theta, force=True)
            rows, stats = evaluate_recipes(
                model, optimizer, spec, infos, bank, recipes, curve_eval_batches,
                token_bytes, device, step=ledger.optimizer_steps, n_layers=n_layers,
            )
            for r in rows:
                r.update({
                    "charged_flop_equiv": ledger.charged_flop_equiv,
                    "trajectory_id": runtime.trajectory_id,
                    "eval_kind": "curve",
                    "eval_batches_count": curve_n,
                })
            recipe_evals.extend(rows)
            tensor_stats_out.update({f"step{ledger.optimizer_steps}_{k}": v for k, v in stats.items()})
            eval_idx += 1

        if ledger.charged_flop_equiv >= next_raw:
            raw_record()

    theta = spec.flatten_parameters(dtype=torch.float32)
    bank.append(ledger.optimizer_steps, theta, force=True)

    # Always add a higher precision final evaluation. If the final step was also
    # a curve anchor, retain both rows: the curve row keeps a consistent fixed
    # evaluation set for plotting, while the final row supports the endpoint
    # table with more validation batches.
    rows, stats = evaluate_recipes(
        model, optimizer, spec, infos, bank, recipes, final_eval_batches,
        token_bytes, device, step=ledger.optimizer_steps, n_layers=n_layers,
    )
    for r in rows:
        r.update({
            "charged_flop_equiv": ledger.charged_flop_equiv,
            "trajectory_id": runtime.trajectory_id,
            "eval_kind": "final",
            "eval_batches_count": final_n,
        })
    recipe_evals.extend(rows)
    tensor_stats_out.update({f"step{ledger.optimizer_steps}_{k}": v for k, v in stats.items()})

    runtime.out.mkdir(parents=True, exist_ok=True)
    result = {
        "trajectory_id": runtime.trajectory_id,
        "optimizer": pack["build_args"].get("optimizer", "unknown"),
        "prefix_steps": int(pack["prefix_steps"]),
        "runtime_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(runtime).items()},
        "build_args": pack["build_args"],
        "raw_trace": raw_trace,
        "recipe_evaluations": recipe_evals,
        "tensor_stats": tensor_stats_out,
        "group_aggregates": {
            key: {
                metric: weighted_group_aggregate(rows, metric)
                for metric in ("noise_fraction", "autocorr", "average_displacement_rms")
            }
            for key, rows in tensor_stats_out.items()
        },
        "final_ledger": ledger.as_dict(),
        "termination": termination,
        "snapshot_steps": bank.steps(),
        "snapshot_memory_bytes": bank.memory_bytes,
        "curve_eval_batches": curve_n,
        "final_eval_batches": final_n,
    }
    path = runtime.out / "result.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=True))
    # Convenience CSV for the final recipe sweep.
    final_step = ledger.optimizer_steps
    final_rows = [
        r for r in recipe_evals
        if int(r["step"]) == final_step and r.get("eval_kind") == "final"
    ]
    if not final_rows:
        final_rows = [r for r in recipe_evals if int(r["step"]) == final_step]
    if final_rows:
        fields = sorted({k for r in final_rows for k, v in r.items() if not isinstance(v, (list, dict))})
        with (runtime.out / "final_recipes.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            w.writerows([{k: r.get(k) for k in fields} for r in final_rows])
    print(f"[result] wrote {path}")
    return path


def selftest() -> None:
    from .trajectory_groups import selftest as group_selftest
    group_selftest()
    a = [1.0, 2.0, 3.0]; b = [1.1, 2.2, 3.3]
    d, se, z = paired_delta(a, b)
    assert d < 0 and se > 0 and z < 0
    print("[selftest] muon-aware averaging PASS")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    bp = sub.add_parser("branch")
    bp.add_argument("--pack", type=Path, required=True)
    bp.add_argument("--out", type=Path, required=True)
    bp.add_argument("--trajectory-id", required=True)
    bp.add_argument("--recipes-file", type=Path, required=True)
    bp.add_argument("--device", default="cuda")
    bp.add_argument("--history-device", default=None)
    bp.add_argument("--history-dtype", default=None)
    bp.add_argument("--compute-budget", type=float, default=1800)
    bp.add_argument("--lr-scale", type=float, default=0.4)
    bp.add_argument("--warmdown-ratio", type=float, default=None)
    bp.add_argument("--final-lr-frac", type=float, default=None)
    bp.add_argument("--snapshot-every", type=int, default=4)
    bp.add_argument("--snapshot-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    bp.add_argument("--recipe-eval-equivs", default="1200,1500,1800")
    bp.add_argument("--eval-every-equiv", type=float, default=100)
    bp.add_argument(
        "--curve-eval-batches", type=int, default=0,
        help="number of cached evaluation batches at intermediate curve anchors; 0 uses all",
    )
    bp.add_argument(
        "--final-eval-batches", type=int, default=0,
        help="number of cached evaluation batches at the final endpoint; 0 uses all",
    )
    bp.add_argument("--seed", type=int, default=None)
    return ap


def main() -> None:
    p = make_parser(); args = p.parse_args()
    if args.command == "selftest":
        selftest(); return
    _install_canonical_build()
    run(args)


if __name__ == "__main__":
    main()
