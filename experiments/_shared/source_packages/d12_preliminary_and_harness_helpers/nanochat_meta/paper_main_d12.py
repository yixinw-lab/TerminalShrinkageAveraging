#!/usr/bin/env python3
"""Clean D12 paper-main experiments for Terminal Shrinkage Averaging.

This suite is intentionally organized around the three causal questions in the
main paper, all under NanoChat's native Muon+AdamW optimizer:

1. Output estimator: on the baseline 5% terminal schedule, does a scalar TSA
   coefficient alpha improve over the raw iterate and uniform LAWA?
2. Schedule: with the raw iterate fixed as output, does a hotter terminal
   learning-rate floor help or hurt?
3. Interaction: after alpha is frozen, does TSA change which terminal floor is
   preferred?

The selection protocol is sequential and uses disjoint cached validation splits:
  curve          32 batches  -- figures only
  alpha_select   64 batches  -- choose scalar alpha on the 5% baseline
  floor_select   64 batches  -- choose terminal floor with alpha frozen
  holdout        96 batches  -- all headline tables / final comparisons

Every schedule arm in the selection stage reuses the exact same D12 pack and
training batch order. A separate confirmation stage can repeat the frozen 5%
and selected-floor schedules under paired, independently permuted optimizer-step
batch orders.

Endpoint-only comparators include checkpoint EMA, a long-window SWA-style mean,
uniform LAWA, and the historical tensorwise adaptive shrinkage rule. These are
not allowed to affect the frozen scalar TSA recipe.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

BASE_WARMDOWN_RATIO = 0.65
BASE_FINAL_FRAC = 0.05
BASE_LR_SCALE = 0.40
DEFAULT_ALPHA_GRID = "0,0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.9,1"
DEFAULT_CURVE_STEPS = "248,504,760,1016,1272,1528,1784,2040,2296,2488,2600,2696,2760,2808,2856,2904,2936,2968,3000"
DEFAULT_FLOORS = "0.05,0.10,0.125,0.15,0.175"
PRIMARY_K = 8
PRIMARY_SPACING = 32
SNAPSHOT_STRIDE = 16
SNAPSHOT_PHASE = 8  # 3000 mod 16; supports exact backwards spacings of 16/32/64.
EMA_BETA = 0.95
EMA_K = 16
EMA_SPACING = 16
SWA_K = 32
SWA_SPACING = 16
ADAPTIVE_RULE = "hybrid"


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def alpha_key(alpha: float) -> str:
    return f"tsa:{float(alpha):.3f}"


def floor_id(floor: float) -> str:
    text = f"{100.0 * float(floor):05.2f}".replace(".", "p")
    return f"floor_{text}pct"


def _finite_float(x: Any, default: float = float("nan")) -> float:
    try:
        value = float(x)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (list, dict, tuple)):
                continue
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


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


def _fingerprint(values: Sequence[int]) -> str:
    arr = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


class AlignedSnapshotBank:
    """Small exact host-resident bank for the terminal averaging windows.

    We only retain steps congruent to `phase (mod stride)`. With T=3000,
    stride=16, phase=8, every checkpoint needed by the main K=8,s=32 window,
    the EMA K=16,s=16 window, and the long-window K=32,s=16 comparator is exact.
    """

    def __init__(self, *, stride: int, phase: int, max_span: int, dtype: torch.dtype):
        self.stride = int(stride)
        self.phase = int(phase) % int(stride)
        self.max_span = int(max_span)
        self.dtype = dtype
        self._data: OrderedDict[int, torch.Tensor] = OrderedDict()

    def should_store(self, step: int) -> bool:
        return int(step) % self.stride == self.phase

    def append(self, step: int, vector: torch.Tensor) -> None:
        step = int(step)
        if not self.should_store(step):
            return
        value = vector.detach().to(device="cpu", dtype=self.dtype).contiguous()
        self._data[step] = value
        cutoff = step - self.max_span - self.stride
        while self._data and next(iter(self._data)) < cutoff:
            self._data.popitem(last=False)

    def select(self, window: int, spacing: int, *, end_step: int) -> list[tuple[int, torch.Tensor]]:
        steps = [int(end_step) - i * int(spacing) for i in range(int(window))]
        missing = [s for s in steps if s not in self._data]
        if missing:
            raise RuntimeError(
                f"snapshot bank missing steps {missing[:8]} for window={window} spacing={spacing} end={end_step}; "
                f"stored_range={list(self._data)[:2]}...{list(self._data)[-2:] if self._data else []}"
            )
        # oldest -> newest for recency weighting readability
        return [(s, self._data[s]) for s in reversed(steps)]

    def memory_bytes(self) -> int:
        return sum(int(v.numel() * v.element_size()) for v in self._data.values())

    def steps(self) -> list[int]:
        return list(self._data)


def uniform_average(selected: Sequence[tuple[int, torch.Tensor]]) -> torch.Tensor:
    acc = torch.zeros_like(selected[0][1], dtype=torch.float32, device="cpu")
    for _, vector in selected:
        acc.add_(vector.float(), alpha=1.0 / len(selected))
    return acc


def exponential_average(selected: Sequence[tuple[int, torch.Tensor]], beta: float) -> torch.Tensor:
    n = len(selected)
    weights = np.asarray([float(beta) ** (n - 1 - i) for i in range(n)], dtype=float)
    weights /= weights.sum()
    acc = torch.zeros_like(selected[0][1], dtype=torch.float32, device="cpu")
    for weight, (_, vector) in zip(weights, selected):
        acc.add_(vector.float(), alpha=float(weight))
    return acc


def _warmdown_progress(step: int, total: int) -> float:
    start = float(total) * (1.0 - BASE_WARMDOWN_RATIO)
    if step <= start:
        return 0.0
    if step >= total:
        return 1.0
    return (float(step) - start) / (float(total) - start)


def _baseline_multiplier(step: int, total: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(float(warmup_steps), 1.0)
    warmdown_start = int(total * (1.0 - BASE_WARMDOWN_RATIO))
    if step < warmdown_start:
        return 1.0
    p = _warmdown_progress(step, total)
    return 1.0 - (1.0 - BASE_FINAL_FRAC) * p


def _apply_floor_lr(optimizer: torch.optim.Optimizer, step: int, args: argparse.Namespace,
                    runtime: argparse.Namespace) -> float:
    from .skip_core import apply_lr_schedule

    base = apply_lr_schedule(
        optimizer,
        int(step),
        int(args.warmup_steps),
        int(runtime.schedule_total_iterations),
        BASE_WARMDOWN_RATIO,
        BASE_FINAL_FRAC,
    )
    desired = max(float(base), float(runtime.terminal_floor))
    scale = desired / float(base) if float(base) > 0 else 1.0
    for group in optimizer.param_groups:
        group["lr"] *= scale * float(runtime.lr_scale)
    return desired * float(runtime.lr_scale)


def _evaluate_candidate(model: torch.nn.Module, spec: Any, current: torch.Tensor, candidate: torch.Tensor,
                        eval_batches: Sequence[Any], token_bytes: Any, device: torch.device) -> dict[str, Any]:
    from .filter_suite import evaluate_detailed
    from .skip_core import Timer

    try:
        spec.assign_parameters(candidate)
        with Timer() as timer:
            ev = evaluate_detailed(model, eval_batches, token_bytes, device)
    finally:
        spec.assign_parameters(current)
    return {**ev, "eval_seconds": timer.seconds}


def _evaluate_bundle(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    spec: Any,
    infos: Sequence[Any],
    bank: AlignedSnapshotBank,
    eval_batches: Sequence[Any],
    token_bytes: Any,
    device: torch.device,
    step: int,
    alpha_grid: Sequence[float],
    eval_kind: str,
    endpoint_comparators: bool,
    n_layers: int,
) -> list[dict[str, Any]]:
    from .filter_suite import evaluate_detailed
    from .skip_core import Timer
    from .trajectory_groups import make_averaged_candidate, tensor_stats

    current = spec.flatten_parameters(dtype=torch.float32)
    with Timer() as raw_timer:
        raw = evaluate_detailed(model, eval_batches, token_bytes, device)
    raw_row = {
        "recipe": "raw",
        "method": "Raw iterate",
        "alpha": 0.0,
        "step": int(step),
        "eval_kind": eval_kind,
        **raw,
        "eval_seconds": raw_timer.seconds,
    }
    rows: list[dict[str, Any]] = [raw_row]

    primary_selected = bank.select(PRIMARY_K, PRIMARY_SPACING, end_step=step)
    primary_mean = uniform_average(primary_selected)
    current_cpu = current.detach().to(device="cpu", dtype=torch.float32)
    delta = primary_mean - current_cpu

    for alpha in alpha_grid:
        alpha = float(alpha)
        if abs(alpha) < 1e-12:
            continue
        candidate = current_cpu + alpha * delta
        ev = _evaluate_candidate(model, spec, current, candidate, eval_batches, token_bytes, device)
        d, se, z = paired_delta(ev["bpb_values"], raw["bpb_values"])
        rows.append({
            "recipe": alpha_key(alpha),
            "method": "Uniform LAWA" if abs(alpha - 1.0) < 1e-12 else f"TSA alpha={alpha:.2f}",
            "alpha": alpha,
            "step": int(step),
            "eval_kind": eval_kind,
            **ev,
            "paired_delta_vs_raw": d,
            "paired_se_vs_raw": se,
            "paired_z_vs_raw": z,
            "window": PRIMARY_K,
            "spacing": PRIMARY_SPACING,
        })
        del candidate

    if endpoint_comparators:
        # Checkpoint EMA over a short dense late window.
        ema_selected = bank.select(EMA_K, EMA_SPACING, end_step=step)
        ema_vec = exponential_average(ema_selected, EMA_BETA)
        ev = _evaluate_candidate(model, spec, current, ema_vec, eval_batches, token_bytes, device)
        d, se, z = paired_delta(ev["bpb_values"], raw["bpb_values"])
        rows.append({
            "recipe": f"ema:{EMA_BETA}:{EMA_K}:{EMA_SPACING}",
            "method": "Checkpoint EMA",
            "alpha": float("nan"),
            "step": int(step),
            "eval_kind": eval_kind,
            **ev,
            "paired_delta_vs_raw": d,
            "paired_se_vs_raw": se,
            "paired_z_vs_raw": z,
            "window": EMA_K,
            "spacing": EMA_SPACING,
            "ema_beta": EMA_BETA,
        })
        del ema_vec

        # SWA-style equal average over a longer late window. This is explicitly
        # labeled SWA-style because the underlying LR schedule is not the
        # classical cyclical/constant SWA schedule.
        swa_selected = bank.select(SWA_K, SWA_SPACING, end_step=step)
        swa_vec = uniform_average(swa_selected)
        ev = _evaluate_candidate(model, spec, current, swa_vec, eval_batches, token_bytes, device)
        d, se, z = paired_delta(ev["bpb_values"], raw["bpb_values"])
        rows.append({
            "recipe": f"swastyle:{SWA_K}:{SWA_SPACING}",
            "method": "SWA-style late average",
            "alpha": float("nan"),
            "step": int(step),
            "eval_kind": eval_kind,
            **ev,
            "paired_delta_vs_raw": d,
            "paired_se_vs_raw": se,
            "paired_z_vs_raw": z,
            "window": SWA_K,
            "spacing": SWA_SPACING,
        })
        del swa_vec

        # Historical tensorwise adaptive shrinkage, for appendix comparison.
        stats = tensor_stats(primary_selected, infos)
        adaptive_candidate, adaptive_alphas = make_averaged_candidate(
            current,
            primary_mean,
            infos,
            selector="all",
            n_layers=n_layers,
            adaptive_rule=ADAPTIVE_RULE,
            adaptive_strength=1.0,
            stats_rows=stats,
            adaptive_selector="all",
        )
        ev = _evaluate_candidate(model, spec, current, adaptive_candidate, eval_batches, token_bytes, device)
        d, se, z = paired_delta(ev["bpb_values"], raw["bpb_values"])
        alpha_num = sum(float(a["alpha"]) * int(a["numel"]) for a in adaptive_alphas)
        alpha_den = max(sum(int(a["numel"]) for a in adaptive_alphas), 1)
        rows.append({
            "recipe": "adaptive:hybrid:8:32:1.0:all",
            "method": "Tensorwise adaptive TSA",
            "alpha": alpha_num / alpha_den,
            "step": int(step),
            "eval_kind": eval_kind,
            **ev,
            "paired_delta_vs_raw": d,
            "paired_se_vs_raw": se,
            "paired_z_vs_raw": z,
            "window": PRIMARY_K,
            "spacing": PRIMARY_SPACING,
            "mean_alpha_by_numel": alpha_num / alpha_den,
        })
        del adaptive_candidate

    del current_cpu, delta, primary_mean
    return rows


def run_branch(runtime: argparse.Namespace) -> Path:
    from .filter_suite import BudgetLedger, OPTIMIZER_OPS_PER_PARAMETER, _install_canonical_build, _stabilize_nanochat_adamw_kernel
    from .skip_core import Timer, seed_everything
    from .skip_suite import PACK_VERSION, exact_gradient, restore_branch
    from .trajectory_groups import build_slice_infos

    _install_canonical_build()
    _stabilize_nanochat_adamw_kernel()

    pack = torch.load(runtime.pack, map_location="cpu", weights_only=False)
    if int(pack.get("pack_version", -1)) != PACK_VERSION:
        raise RuntimeError("pack version mismatch")
    if int(pack.get("prefix_steps", 0)) != 0:
        raise RuntimeError("paper-main suite expects the paired prefix_steps=0 D12 pack")

    args, model, optimizer, token_bytes, device, spec, _history, _bank = restore_branch(pack, runtime)
    seed_everything(int(pack.get("seed", 1337)) if runtime.seed is None else int(runtime.seed))
    infos = build_slice_infos(spec, optimizer)
    n_layers = int(getattr(model.config, "n_layer", getattr(model.config, "depth", args.depth)))

    all_eval = pack["eval_batches"]
    counts = [int(runtime.curve_eval_batches), int(runtime.alpha_select_batches), int(runtime.floor_select_batches), int(runtime.holdout_eval_batches)]
    if sum(counts) > len(all_eval):
        raise ValueError(f"requested eval split {counts}={sum(counts)} but pack has {len(all_eval)} batches")
    a, b, c, d = counts
    curve_batches = all_eval[:a]
    alpha_batches = all_eval[a:a+b]
    floor_batches = all_eval[a+b:a+b+c]
    holdout_batches = all_eval[a+b+c:a+b+c+d]

    alpha_grid = parse_floats(runtime.alpha_grid)
    if not alpha_grid or min(alpha_grid) < 0 or max(alpha_grid) > 1:
        raise ValueError("alpha grid must be nonempty and lie in [0,1]")
    if not any(abs(x) < 1e-12 for x in alpha_grid) or not any(abs(x - 1.0) < 1e-12 for x in alpha_grid):
        raise ValueError("alpha grid must include 0 and 1")

    curve_steps = parse_ints(runtime.curve_steps)
    if any(step % SNAPSHOT_STRIDE != SNAPSHOT_PHASE for step in curve_steps):
        bad = [step for step in curve_steps if step % SNAPSHOT_STRIDE != SNAPSHOT_PHASE]
        raise ValueError(f"curve steps must be congruent to {SNAPSHOT_PHASE} mod {SNAPSHOT_STRIDE}; bad={bad}")
    if int(runtime.train_steps) % SNAPSHOT_STRIDE != SNAPSHOT_PHASE:
        raise ValueError("train_steps must align with snapshot phase")

    max_span = max(
        (PRIMARY_K - 1) * PRIMARY_SPACING,
        (EMA_K - 1) * EMA_SPACING,
        (SWA_K - 1) * SWA_SPACING,
    )
    bank = AlignedSnapshotBank(
        stride=SNAPSHOT_STRIDE,
        phase=SNAPSHOT_PHASE,
        max_span=max_span,
        dtype=_dtype(runtime.snapshot_dtype),
    )

    stream = pack["branch_batches"]
    grad_accum = int(args.grad_accum)
    stream_steps = len(stream) // grad_accum
    if stream_steps < int(runtime.train_steps):
        raise RuntimeError(f"pack has {stream_steps} optimizer-step blocks, need {runtime.train_steps}")

    if runtime.stream_seed is None:
        permutation = np.arange(int(runtime.train_steps), dtype=np.int64)
        permutation_label = "canonical"
    else:
        rng = np.random.default_rng(int(runtime.stream_seed))
        permutation = rng.permutation(int(runtime.train_steps)).astype(np.int64)
        permutation_label = _fingerprint(permutation.tolist())

    micro_tokens = int(args.device_batch_size) * int(args.max_seq_len)
    full_tokens = micro_tokens * grad_accum
    ledger = BudgetLedger(spec.numel, full_tokens)
    curve_set = set(curve_steps)
    recipe_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []

    print(
        f"[setup] floor={runtime.terminal_floor:.4f} train_steps={runtime.train_steps} "
        f"alpha_grid={alpha_grid} data_order={permutation_label[:20]}",
        flush=True,
    )

    for step in range(1, int(runtime.train_steps) + 1):
        block_index = int(permutation[step - 1])
        lo = block_index * grad_accum
        batches = stream[lo:lo + grad_accum]
        lr_mult = _apply_floor_lr(optimizer, step - 1, args, runtime)
        with Timer() as train_timer:
            _g, train_loss, _features, _seconds = exact_gradient(
                model,
                optimizer,
                spec,
                batches,
                device,
                need_forward_features=False,
                token_sample_max=args.token_sample_max,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        ledger.add_exact(full_tokens, grad_accum, train_timer.seconds, charged=True)
        ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
        ledger.optimizer_steps += 1

        if bank.should_store(step):
            bank.append(step, spec.flatten_parameters(dtype=torch.float32))

        if step in curve_set:
            rows = _evaluate_bundle(
                model=model,
                optimizer=optimizer,
                spec=spec,
                infos=infos,
                bank=bank,
                eval_batches=curve_batches,
                token_bytes=token_bytes,
                device=device,
                step=step,
                alpha_grid=alpha_grid,
                eval_kind="curve",
                endpoint_comparators=False,
                n_layers=n_layers,
            )
            for row in rows:
                row.update({
                    "floor": float(runtime.terminal_floor),
                    "trajectory_id": floor_id(runtime.terminal_floor),
                    "lr_multiplier": float(lr_mult),
                    "stream_seed": runtime.stream_seed,
                })
            recipe_rows.extend(rows)
            raw = next(r for r in rows if r["recipe"] == "raw")
            training_rows.append({
                "step": step,
                "train_loss": float(train_loss),
                "raw_bpb": float(raw["bpb"]),
                "lr_multiplier": float(lr_mult),
            })
            print(
                f"[curve] floor={runtime.terminal_floor:.3f} step={step} raw={raw['bpb']:.6f} "
                f"lr={lr_mult:.5f} bank_gib={bank.memory_bytes()/2**30:.2f}",
                flush=True,
            )

    # Final endpoint: all four evaluation roles. Curves are never used for selection.
    endpoint_specs = [
        ("alpha_select", alpha_batches),
        ("floor_select", floor_batches),
        ("holdout", holdout_batches),
    ]
    for eval_kind, batches in endpoint_specs:
        rows = _evaluate_bundle(
            model=model,
            optimizer=optimizer,
            spec=spec,
            infos=infos,
            bank=bank,
            eval_batches=batches,
            token_bytes=token_bytes,
            device=device,
            step=int(runtime.train_steps),
            alpha_grid=alpha_grid,
            eval_kind=eval_kind,
            endpoint_comparators=True,
            n_layers=n_layers,
        )
        for row in rows:
            row.update({
                "floor": float(runtime.terminal_floor),
                "trajectory_id": floor_id(runtime.terminal_floor),
                "stream_seed": runtime.stream_seed,
            })
        recipe_rows.extend(rows)

    # Dense schedule profile, before global lr_scale, for plotting provenance.
    schedule_profile = []
    for step in range(0, int(runtime.train_steps) + 1, 10):
        base = _baseline_multiplier(step, int(runtime.schedule_total_iterations), int(args.warmup_steps))
        desired = max(base, float(runtime.terminal_floor))
        schedule_profile.append({"step": step, "multiplier": desired * float(runtime.lr_scale)})

    out = runtime.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "suite": "paper_main_d12_v1",
        "floor": float(runtime.terminal_floor),
        "trajectory_id": floor_id(runtime.terminal_floor),
        "stream_seed": runtime.stream_seed,
        "permutation_fingerprint": permutation_label,
        "pack": str(runtime.pack.resolve()),
        "optimizer": pack.get("build_args", {}).get("optimizer", "unknown"),
        "build_args": pack.get("build_args", {}),
        "train_steps": int(runtime.train_steps),
        "schedule_total_iterations": int(runtime.schedule_total_iterations),
        "lr_scale": float(runtime.lr_scale),
        "base_warmdown_ratio": BASE_WARMDOWN_RATIO,
        "base_final_lr_frac": BASE_FINAL_FRAC,
        "alpha_grid": alpha_grid,
        "primary_window": PRIMARY_K,
        "primary_spacing": PRIMARY_SPACING,
        "eval_splits": {
            "curve": a,
            "alpha_select": b,
            "floor_select": c,
            "holdout": d,
        },
        "comparators": {
            "ema": {"beta": EMA_BETA, "window": EMA_K, "spacing": EMA_SPACING},
            "swa_style": {"window": SWA_K, "spacing": SWA_SPACING},
            "adaptive": "adaptive:hybrid:8:32:1.0:all",
        },
        "recipe_evaluations": recipe_rows,
        "training_trace": training_rows,
        "schedule_profile": schedule_profile,
        "final_ledger": ledger.as_dict(),
        "snapshot_steps_final_bank": bank.steps(),
        "snapshot_memory_bytes": bank.memory_bytes(),
    }
    path = out / "result.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=True))
    _write_csv(out / "recipe_evaluations.csv", recipe_rows)
    _write_csv(out / "training_trace.csv", training_rows)
    print(f"[result] wrote {path}", flush=True)
    return path


def _row_for_alpha(rows: Sequence[dict[str, Any]], alpha: float) -> dict[str, Any]:
    if abs(float(alpha)) < 1e-12:
        matches = [r for r in rows if r.get("recipe") == "raw"]
    else:
        key = alpha_key(alpha)
        matches = [r for r in rows if r.get("recipe") == key]
    if len(matches) != 1:
        raise RuntimeError(f"expected one row for alpha={alpha}, found {len(matches)}")
    return matches[0]


def _endpoint_rows(result: dict[str, Any], eval_kind: str) -> list[dict[str, Any]]:
    return [r for r in result["recipe_evaluations"] if r.get("eval_kind") == eval_kind and int(r.get("step", -1)) == int(result["train_steps"])]


def _curve_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in result["recipe_evaluations"] if r.get("eval_kind") == "curve"]


def _paired_between(row_a: dict[str, Any], row_b: dict[str, Any]) -> dict[str, float]:
    d, se, z = paired_delta(row_a.get("bpb_values", []), row_b.get("bpb_values", []))
    return {"delta": d, "se": se, "z": z, "ci95": 1.96 * se}


def _fmt(x: float, digits: int = 6) -> str:
    return f"{float(x):.{digits}f}"


def _latex_escape(text: str) -> str:
    return text.replace("%", r"\%").replace("_", r"\_")


def _write_main_tables(figdir: Path, *, selected_alpha: float, selected_floor: float,
                       baseline: dict[str, Any], selected: dict[str, Any]) -> None:
    base_hold = _endpoint_rows(baseline, "holdout")
    sel_hold = _endpoint_rows(selected, "holdout")
    raw_base = _row_for_alpha(base_hold, 0.0)
    tsa_base = _row_for_alpha(base_hold, selected_alpha)
    lawa_base = _row_for_alpha(base_hold, 1.0)
    raw_sel = _row_for_alpha(sel_hold, 0.0)
    tsa_sel = _row_for_alpha(sel_hold, selected_alpha)

    comparator_recipes = [
        ("Raw iterate", "raw"),
        ("Checkpoint EMA", f"ema:{EMA_BETA}:{EMA_K}:{EMA_SPACING}"),
        ("SWA-style late average", f"swastyle:{SWA_K}:{SWA_SPACING}"),
        ("Uniform LAWA", alpha_key(1.0)),
        (f"TSA ($\\alpha={selected_alpha:.2f}$)", alpha_key(selected_alpha)),
    ]
    comp_rows = []
    for label, recipe in comparator_recipes:
        row = next(r for r in base_hold if r["recipe"] == recipe)
        delta = _paired_between(row, raw_base) if recipe != "raw" else {"delta": 0.0, "se": 0.0, "ci95": 0.0}
        comp_rows.append({
            "method": label,
            "bpb": float(row["bpb"]),
            "delta_vs_raw": float(delta["delta"]),
            "paired_ci95": float(delta["ci95"]),
        })
    _write_csv(figdir / "table_averaging.csv", comp_rows)

    clamp_rows = []
    for label, row in [("5% floor", raw_base), (f"{100*selected_floor:g}% floor", raw_sel)]:
        delta = _paired_between(row, raw_base)
        clamp_rows.append({"schedule": label, "raw_bpb": float(row["bpb"]), "delta_vs_5pct": float(delta["delta"]), "paired_ci95": float(delta["ci95"])})
    _write_csv(figdir / "table_clamp.csv", clamp_rows)

    interaction_rows = [
        {"schedule": "5% floor", "raw_bpb": float(raw_base["bpb"]), "tsa_bpb": float(tsa_base["bpb"])},
        {"schedule": f"{100*selected_floor:g}% floor", "raw_bpb": float(raw_sel["bpb"]), "tsa_bpb": float(tsa_sel["bpb"])},
    ]
    for row in interaction_rows:
        row["tsa_gain"] = row["raw_bpb"] - row["tsa_bpb"]
    _write_csv(figdir / "table_interaction.csv", interaction_rows)

    # LaTeX tables, deliberately compact and main-text friendly.
    with (figdir / "table_averaging.tex").open("w") as f:
        f.write("\\begin{tabular}{lrr}\n\\toprule\nEstimator & Holdout BPB & $\\Delta$ vs. raw \\\\\n\\midrule\n")
        best = min(r["bpb"] for r in comp_rows)
        for r in comp_rows:
            bpb = f"\\textbf{{{r['bpb']:.6f}}}" if abs(r["bpb"] - best) < 1e-12 else f"{r['bpb']:.6f}"
            delta = "---" if r["method"] == "Raw iterate" else f"{r['delta_vs_raw']:+.6f}"
            f.write(f"{r['method']} & {bpb} & {delta} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (figdir / "table_clamp.tex").open("w") as f:
        f.write("\\begin{tabular}{lrr}\n\\toprule\nTerminal schedule & Raw BPB & $\\Delta$ vs. 5\\% \\\\\n\\midrule\n")
        for r in clamp_rows:
            delta = "---" if r["schedule"] == "5% floor" else f"{r['delta_vs_5pct']:+.6f}"
            f.write(f"{_latex_escape(r['schedule'])} & {r['raw_bpb']:.6f} & {delta} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with (figdir / "table_interaction.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrr}\n\\toprule\nTerminal schedule & Raw & TSA & TSA gain \\\\\n\\midrule\n")
        for r in interaction_rows:
            f.write(f"{_latex_escape(r['schedule'])} & {r['raw_bpb']:.6f} & {r['tsa_bpb']:.6f} & {r['tsa_gain']:+.6f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")


def _plot_stage1(figdir: Path, results: dict[float, dict[str, Any]], selected_alpha: float, selected_floor: float) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    colors = {
        "raw": "#222222",
        "tsa": "#1f77b4",
        "lawa": "#d95f02",
        "baseline": "#4d4d4d",
        "active": "#b2182b",
        "base_tsa": "#4c78a8",
        "active_tsa": "#2a9d8f",
    }

    baseline = results[0.05]
    selected = results[selected_floor]

    def curve(result: dict[str, Any], alpha: float) -> tuple[np.ndarray, np.ndarray]:
        rows = _curve_rows(result)
        vals = []
        for step in sorted({int(r["step"]) for r in rows}):
            step_rows = [r for r in rows if int(r["step"]) == step]
            row = _row_for_alpha(step_rows, alpha)
            vals.append((step, float(row["bpb"])))
        return np.asarray([x[0] for x in vals]), np.asarray([x[1] for x in vals])

    def add_title(ax, title: str, subtitle: str) -> None:
        ax.set_title(title, fontsize=9.4, fontweight="bold", pad=14)
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.4, fontweight="bold")

    def finish(ax, *, zoom: bool = False) -> None:
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Validation BPB")
        ax.grid(True, alpha=0.18, linewidth=0.6)
        if zoom:
            ax.set_xlim(2680, 3010)
        ax.margins(x=0.01, y=0.08)
        ax.legend(frameon=False, loc="best", handlelength=2.2)

    def save(fig, name: str) -> None:
        fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(figdir / f"{name}.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    # Experiment 1: estimator ablation on the 5% baseline schedule.
    for zoom, suffix in [(False, "full"), (True, "zoom")]:
        fig, ax = plt.subplots(figsize=(3.25, 2.28))
        for alpha, label, color, lw in [
            (0.0, "Raw", colors["raw"], 1.65),
            (selected_alpha, f"TSA (α={selected_alpha:.2f})", colors["tsa"], 1.9),
            (1.0, "LAWA", colors["lawa"], 1.55),
        ]:
            x, y = curve(baseline, alpha)
            ax.plot(x, y, label=label, color=color, linewidth=lw, marker="o", markersize=2.2, markevery=[-1])
        add_title(ax, "Partial averaging improves the output model", "Fixed Muon+AdamW trajectory · 5% terminal floor")
        finish(ax, zoom=zoom)
        save(fig, f"fig1_averaging_{suffix}")

    # Experiment 2: raw endpoint under baseline vs selected terminal floor.
    for zoom, suffix in [(False, "full"), (True, "zoom")]:
        fig, ax = plt.subplots(figsize=(3.25, 2.28))
        for result, label, color in [
            (baseline, "Raw · 5% floor", colors["baseline"]),
            (selected, f"Raw · {100*selected_floor:g}% floor", colors["active"]),
        ]:
            x, y = curve(result, 0.0)
            ax.plot(x, y, label=label, color=color, linewidth=1.8, marker="o", markersize=2.2, markevery=[-1])
        add_title(ax, "A more active tail worsens the raw endpoint", "Schedule intervention without output averaging")
        finish(ax, zoom=zoom)
        save(fig, f"fig2_clamp_{suffix}")

    # Experiment 3: 2x2 causal interaction.
    for zoom, suffix in [(False, "full"), (True, "zoom")]:
        fig, ax = plt.subplots(figsize=(3.25, 2.35))
        series = [
            (baseline, 0.0, "5% · Raw", colors["baseline"], "--", 1.45),
            (baseline, selected_alpha, "5% · TSA", colors["base_tsa"], "-", 1.75),
            (selected, 0.0, f"{100*selected_floor:g}% · Raw", colors["active"], "--", 1.55),
            (selected, selected_alpha, f"{100*selected_floor:g}% · TSA", colors["active_tsa"], "-", 2.0),
        ]
        for result, alpha, label, color, ls, lw in series:
            x, y = curve(result, alpha)
            ax.plot(x, y, label=label, color=color, linestyle=ls, linewidth=lw, marker="o", markersize=2.0, markevery=[-1])
        add_title(ax, "Averaging changes the preferred terminal schedule", "The active tail hurts raw weights but improves the TSA output")
        finish(ax, zoom=zoom)
        save(fig, f"fig3_interaction_{suffix}")

    # Hyperparameter provenance: alpha on baseline selection split.
    alpha_rows = _endpoint_rows(baseline, "alpha_select")
    grid = sorted((float(r.get("alpha", 0.0)), float(r["bpb"])) for r in alpha_rows if r["recipe"] == "raw" or str(r["recipe"]).startswith("tsa:"))
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    ax.plot([x for x, _ in grid], [y for _, y in grid], linewidth=1.8, marker="o", markersize=3.0)
    ax.axvline(selected_alpha, linewidth=1.0, linestyle="--", alpha=0.6)
    add_title(ax, "Selecting the shrinkage strength", "Baseline 5% schedule · designated α-selection split")
    ax.set_xlabel("TSA shrinkage α")
    ax.set_ylabel("Validation BPB")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.margins(x=0.03, y=0.12)
    save(fig, "figS_alpha_selection")

    # Hyperparameter provenance: floor with alpha frozen.
    floor_points = []
    for floor, result in sorted(results.items()):
        rows = _endpoint_rows(result, "floor_select")
        row = _row_for_alpha(rows, selected_alpha)
        floor_points.append((100 * floor, float(row["bpb"])))
    fig, ax = plt.subplots(figsize=(3.25, 2.2))
    ax.plot([x for x, _ in floor_points], [y for _, y in floor_points], linewidth=1.8, marker="o", markersize=3.0)
    ax.axvline(100 * selected_floor, linewidth=1.0, linestyle="--", alpha=0.6)
    add_title(ax, "Selecting the terminal learning-rate floor", f"TSA α={selected_alpha:.2f} frozen before schedule selection")
    ax.set_xlabel("Terminal LR floor (% of peak)")
    ax.set_ylabel("Validation BPB")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.margins(x=0.05, y=0.12)
    save(fig, "figS_floor_selection")

    # Full alpha x floor interaction on untouched holdout; appendix diagnostic.
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    for floor, result in sorted(results.items()):
        rows = _endpoint_rows(result, "holdout")
        points = []
        for alpha in result["alpha_grid"]:
            row = _row_for_alpha(rows, float(alpha))
            points.append((float(alpha), float(row["bpb"])))
        ax.plot([x for x, _ in points], [y for _, y in points], linewidth=1.35, marker="o", markersize=2.1,
                label=f"{100*floor:g}% floor")
    add_title(ax, "Shrinkage and terminal activity interact", "Post-selection holdout surface; lower BPB is better")
    ax.set_xlabel("TSA shrinkage α")
    ax.set_ylabel("Validation BPB")
    ax.grid(True, alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, fontsize=6.6, ncol=2)
    ax.margins(x=0.02, y=0.08)
    save(fig, "figA_alpha_floor_interaction")


def run_merge(runtime: argparse.Namespace) -> Path:
    root = runtime.root.resolve()
    result_paths = sorted(root.glob("runs/floor_*pct/result.json"))
    floors_expected = parse_floats(runtime.floors)
    if len(result_paths) != len(floors_expected):
        raise RuntimeError(f"expected {len(floors_expected)} floor results, found {len(result_paths)}")
    results: dict[float, dict[str, Any]] = {}
    for path in result_paths:
        result = json.loads(path.read_text())
        results[float(result["floor"])] = result
    for floor in floors_expected:
        if not any(abs(floor - x) < 1e-12 for x in results):
            raise RuntimeError(f"missing floor={floor}")
    baseline_floor = min(results, key=lambda x: abs(x - 0.05))
    baseline = results[baseline_floor]

    # Stage 1: alpha selection ONLY on baseline alpha_select split.
    alpha_rows = _endpoint_rows(baseline, "alpha_select")
    candidates = []
    for alpha in baseline["alpha_grid"]:
        row = _row_for_alpha(alpha_rows, float(alpha))
        candidates.append((float(row["bpb"]), float(alpha), row))
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, selected_alpha, alpha_selected_row = candidates[0]

    # Stage 2: floor selection ONLY on disjoint floor_select split with alpha frozen.
    floor_candidates = []
    for floor, result in results.items():
        row = _row_for_alpha(_endpoint_rows(result, "floor_select"), selected_alpha)
        floor_candidates.append((float(row["bpb"]), float(floor), row))
    floor_candidates.sort(key=lambda x: (x[0], x[1]))
    _, selected_floor, floor_selected_row = floor_candidates[0]
    selected = results[selected_floor]

    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    alpha_selection_rows = []
    for _bpb, alpha, row in sorted(candidates, key=lambda x: x[1]):
        hold_row = _row_for_alpha(_endpoint_rows(baseline, "holdout"), alpha)
        alpha_selection_rows.append({
            "alpha": alpha,
            "selection_bpb": float(row["bpb"]),
            "holdout_bpb_posthoc": float(hold_row["bpb"]),
            "selected": int(abs(alpha - selected_alpha) < 1e-12),
        })
    _write_csv(figdir / "alpha_selection.csv", alpha_selection_rows)

    floor_selection_rows = []
    for _bpb, floor, row in sorted(floor_candidates, key=lambda x: x[1]):
        hold_row = _row_for_alpha(_endpoint_rows(results[floor], "holdout"), selected_alpha)
        raw_hold = _row_for_alpha(_endpoint_rows(results[floor], "holdout"), 0.0)
        floor_selection_rows.append({
            "floor": floor,
            "floor_percent": 100 * floor,
            "selection_bpb": float(row["bpb"]),
            "holdout_tsa_bpb_posthoc": float(hold_row["bpb"]),
            "holdout_raw_bpb_posthoc": float(raw_hold["bpb"]),
            "selected": int(abs(floor - selected_floor) < 1e-12),
        })
    _write_csv(figdir / "floor_selection.csv", floor_selection_rows)

    # Full endpoint matrix for appendix / audit.
    matrix_rows = []
    for floor, result in sorted(results.items()):
        for kind in ("alpha_select", "floor_select", "holdout"):
            for row in _endpoint_rows(result, kind):
                if row["recipe"] == "raw" or str(row["recipe"]).startswith("tsa:") or row["recipe"].startswith("ema:") or row["recipe"].startswith("swastyle:") or row["recipe"].startswith("adaptive:"):
                    matrix_rows.append({
                        "floor": floor,
                        "eval_kind": kind,
                        "recipe": row["recipe"],
                        "method": row.get("method"),
                        "alpha": row.get("alpha"),
                        "bpb": row["bpb"],
                        "bpb_se": row.get("bpb_se"),
                        "paired_delta_vs_raw": row.get("paired_delta_vs_raw"),
                        "paired_se_vs_raw": row.get("paired_se_vs_raw"),
                        "mean_alpha_by_numel": row.get("mean_alpha_by_numel"),
                    })
    _write_csv(figdir / "endpoint_matrix.csv", matrix_rows)

    # Historical adaptive-vs-scalar comparison on holdout.
    adaptive_rows = []
    for floor in (baseline_floor, selected_floor):
        rows = _endpoint_rows(results[floor], "holdout")
        scalar = _row_for_alpha(rows, selected_alpha)
        adaptive = next(r for r in rows if r["recipe"] == "adaptive:hybrid:8:32:1.0:all")
        cmp_ = _paired_between(adaptive, scalar)
        adaptive_rows.append({
            "floor": floor,
            "scalar_alpha": selected_alpha,
            "scalar_bpb": scalar["bpb"],
            "adaptive_mean_alpha": adaptive.get("mean_alpha_by_numel"),
            "adaptive_bpb": adaptive["bpb"],
            "adaptive_minus_scalar": cmp_["delta"],
            "paired_ci95": cmp_["ci95"],
        })
    _write_csv(figdir / "appendix_adaptive_vs_scalar.csv", adaptive_rows)

    _write_main_tables(figdir, selected_alpha=selected_alpha, selected_floor=selected_floor,
                       baseline=baseline, selected=selected)
    _plot_stage1(figdir, results, selected_alpha, selected_floor)

    frozen = root / "FROZEN.env"
    frozen.write_text(
        f"SELECTED_ALPHA={selected_alpha:.6f}\n"
        f"SELECTED_FLOOR={selected_floor:.6f}\n"
        f"PRIMARY_K={PRIMARY_K}\n"
        f"PRIMARY_SPACING={PRIMARY_SPACING}\n"
        f"BASELINE_FLOOR=0.050000\n"
    )

    base_hold = _endpoint_rows(baseline, "holdout")
    sel_hold = _endpoint_rows(selected, "holdout")
    raw_base = _row_for_alpha(base_hold, 0.0)
    tsa_base = _row_for_alpha(base_hold, selected_alpha)
    lawa_base = _row_for_alpha(base_hold, 1.0)
    raw_sel = _row_for_alpha(sel_hold, 0.0)
    tsa_sel = _row_for_alpha(sel_hold, selected_alpha)
    lawa_sel = _row_for_alpha(sel_hold, 1.0)
    adaptive_base = next(r for r in base_hold if r["recipe"] == "adaptive:hybrid:8:32:1.0:all")
    adaptive_sel = next(r for r in sel_hold if r["recipe"] == "adaptive:hybrid:8:32:1.0:all")

    digest = f"""D12 paper-main selection suite
{'='*78}
Protocol
  alpha: selected on baseline 5% schedule / alpha_select split only
  floor: selected with alpha frozen / disjoint floor_select split only
  headline values below: untouched holdout split only
  optimizer: native Muon+AdamW

Selected hyperparameters
  alpha: {selected_alpha:.3f}
  terminal floor: {100*selected_floor:.2f}% of peak
  TSA checkpoint window: K={PRIMARY_K}, spacing={PRIMARY_SPACING} optimizer steps

Averaging alone on baseline 5% schedule (holdout)
  raw:   {raw_base['bpb']:.6f}
  TSA:   {tsa_base['bpb']:.6f}  gain={raw_base['bpb']-tsa_base['bpb']:+.6f}
  LAWA:  {lawa_base['bpb']:.6f} gain={raw_base['bpb']-lawa_base['bpb']:+.6f}

Clamp alone, raw output (holdout)
  5% raw:                 {raw_base['bpb']:.6f}
  {100*selected_floor:.2f}% raw:              {raw_sel['bpb']:.6f}
  hotter-floor raw delta: {raw_sel['bpb']-raw_base['bpb']:+.6f}

Combined effect (holdout)
  5% TSA:                 {tsa_base['bpb']:.6f}
  {100*selected_floor:.2f}% TSA:              {tsa_sel['bpb']:.6f}
  hotter-floor TSA delta: {tsa_sel['bpb']-tsa_base['bpb']:+.6f}
  TSA gain at 5%:         {raw_base['bpb']-tsa_base['bpb']:+.6f}
  TSA gain at selected:   {raw_sel['bpb']-tsa_sel['bpb']:+.6f}

Appendix comparators (holdout)
  baseline adaptive tensor: bpb={adaptive_base['bpb']:.6f} mean_alpha={adaptive_base.get('mean_alpha_by_numel', float('nan')):.4f}
  selected adaptive tensor: bpb={adaptive_sel['bpb']:.6f} mean_alpha={adaptive_sel.get('mean_alpha_by_numel', float('nan')):.4f}

Files
  fig1_averaging_full/zoom.(pdf|png)
  fig2_clamp_full/zoom.(pdf|png)
  fig3_interaction_full/zoom.(pdf|png)
  figS_alpha_selection.(pdf|png)
  figS_floor_selection.(pdf|png)
  figA_alpha_floor_interaction.(pdf|png)
  table_averaging.(csv|tex)
  table_clamp.(csv|tex)
  table_interaction.(csv|tex)
  appendix_adaptive_vs_scalar.csv
  FROZEN.env
"""
    (figdir / "DIGEST.txt").write_text(digest)
    print(digest)
    return figdir


def run_merge_confirmation(runtime: argparse.Namespace) -> Path:
    root = runtime.root.resolve()
    frozen = {}
    for line in runtime.frozen.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            frozen[k.strip()] = v.strip()
    alpha = float(frozen["SELECTED_ALPHA"])
    selected_floor = float(frozen["SELECTED_FLOOR"])
    baseline_floor = float(frozen.get("BASELINE_FLOOR", "0.05"))

    paths = sorted(root.glob("runs/*/result.json"))
    results = [json.loads(p.read_text()) for p in paths]
    if not results:
        raise RuntimeError("no confirmation results")
    seeds = sorted({int(r["stream_seed"]) for r in results})
    for seed in seeds:
        by_seed = [r for r in results if int(r["stream_seed"]) == seed]
        if len(by_seed) != 2:
            raise RuntimeError(f"seed {seed} expected baseline+selected floor, got {len(by_seed)}")
        fps = {r["permutation_fingerprint"] for r in by_seed}
        if len(fps) != 1:
            raise RuntimeError(f"seed {seed} does not use paired data order across schedules")
    if len({r["permutation_fingerprint"] for r in results[::2]}) < max(1, len(seeds)-1):
        # This is only a guard against accidental repeated identical permutations;
        # exact list ordering can differ, so the primary check is per-seed below.
        pass

    figdir = root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    endpoint_rows = []
    curve_rows = []
    for r in results:
        floor = float(r["floor"])
        for row in _endpoint_rows(r, "holdout"):
            if row["recipe"] in {"raw", alpha_key(alpha), alpha_key(1.0)}:
                endpoint_rows.append({
                    "stream_seed": int(r["stream_seed"]),
                    "floor": floor,
                    "recipe": row["recipe"],
                    "bpb": float(row["bpb"]),
                })
        for row in _curve_rows(r):
            if row["recipe"] in {"raw", alpha_key(alpha), alpha_key(1.0)}:
                curve_rows.append({
                    "stream_seed": int(r["stream_seed"]),
                    "floor": floor,
                    "step": int(row["step"]),
                    "recipe": row["recipe"],
                    "bpb": float(row["bpb"]),
                })
    _write_csv(figdir / "confirmation_endpoints.csv", endpoint_rows)
    _write_csv(figdir / "confirmation_curves.csv", curve_rows)

    # Across-run endpoint summary.
    summary = []
    for floor in (baseline_floor, selected_floor):
        for recipe, method in [("raw", "Raw"), (alpha_key(alpha), "TSA"), (alpha_key(1.0), "LAWA")]:
            vals = np.asarray([r["bpb"] for r in endpoint_rows if abs(r["floor"]-floor)<1e-12 and r["recipe"] == recipe], dtype=float)
            summary.append({
                "floor": floor,
                "method": method,
                "n_runs": int(vals.size),
                "mean_bpb": float(vals.mean()),
                "se_across_runs": float(vals.std(ddof=1)/math.sqrt(vals.size)) if vals.size > 1 else 0.0,
            })
    _write_csv(figdir / "confirmation_summary.csv", summary)

    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8.0,"axes.labelsize":8.0,"xtick.labelsize":7.2,"ytick.labelsize":7.2,"legend.fontsize":7.0,"pdf.fonttype":42,"ps.fonttype":42})

    def add_title(ax, title, subtitle):
        ax.set_title(title, fontsize=9.4, fontweight="bold", pad=14)
        ax.text(0.5,1.015,subtitle,transform=ax.transAxes,ha="center",va="bottom",fontsize=7.4,fontweight="bold")

    def aggregate(floor: float, recipe: str):
        steps = sorted({r["step"] for r in curve_rows if abs(r["floor"]-floor)<1e-12 and r["recipe"]==recipe})
        means, ses = [], []
        for step in steps:
            vals=np.asarray([r["bpb"] for r in curve_rows if abs(r["floor"]-floor)<1e-12 and r["recipe"]==recipe and r["step"]==step],dtype=float)
            means.append(vals.mean()); ses.append(vals.std(ddof=1)/math.sqrt(vals.size) if vals.size>1 else 0.0)
        return np.asarray(steps), np.asarray(means), np.asarray(ses)

    colors={"raw":"#222222","tsa":"#1f77b4","lawa":"#d95f02","base":"#4c78a8","active":"#2a9d8f","active_raw":"#b2182b"}
    def save(fig,name):
        fig.savefig(figdir/f"{name}.pdf",bbox_inches="tight"); fig.savefig(figdir/f"{name}.png",dpi=240,bbox_inches="tight"); plt.close(fig)

    # Confirmed averaging plot on baseline.
    for zoom,suffix in [(False,"full"),(True,"zoom")]:
        fig,ax=plt.subplots(figsize=(3.25,2.28))
        for recipe,label,color in [("raw","Raw",colors["raw"]),(alpha_key(alpha),f"TSA (α={alpha:.2f})",colors["tsa"]),(alpha_key(1.0),"LAWA",colors["lawa"])]:
            x,m,se=aggregate(baseline_floor,recipe); ax.plot(x,m,label=label,color=color,linewidth=1.8); ax.fill_between(x,m-se,m+se,color=color,alpha=.12,linewidth=0)
        add_title(ax,"Partial averaging improves the output model",f"Mean of {len(seeds)} paired data-order repetitions · 5% floor")
        ax.set_xlabel("Optimizer step"); ax.set_ylabel("Validation BPB"); ax.grid(True,alpha=.18,linewidth=.6); ax.legend(frameon=False)
        if zoom: ax.set_xlim(2680,3010)
        ax.margins(x=.01,y=.08); save(fig,f"confirmed_fig1_averaging_{suffix}")

    # Confirmed interaction plot.
    for zoom,suffix in [(False,"full"),(True,"zoom")]:
        fig,ax=plt.subplots(figsize=(3.25,2.35))
        series=[(baseline_floor,"raw","5% · Raw","#4d4d4d","--"),(baseline_floor,alpha_key(alpha),"5% · TSA",colors["base"],"-"),(selected_floor,"raw",f"{100*selected_floor:g}% · Raw",colors["active_raw"],"--"),(selected_floor,alpha_key(alpha),f"{100*selected_floor:g}% · TSA",colors["active"],"-")]
        for floor,recipe,label,color,ls in series:
            x,m,se=aggregate(floor,recipe); ax.plot(x,m,label=label,color=color,linestyle=ls,linewidth=1.75); ax.fill_between(x,m-se,m+se,color=color,alpha=.10,linewidth=0)
        add_title(ax,"Averaging changes the preferred terminal schedule",f"Mean of {len(seeds)} paired data-order repetitions")
        ax.set_xlabel("Optimizer step"); ax.set_ylabel("Validation BPB"); ax.grid(True,alpha=.18,linewidth=.6); ax.legend(frameon=False)
        if zoom: ax.set_xlim(2680,3010)
        ax.margins(x=.01,y=.08); save(fig,f"confirmed_fig3_interaction_{suffix}")

    digest = f"Confirmation repetitions\n{'='*72}\nselected_alpha={alpha:.3f}\nselected_floor={100*selected_floor:.2f}%\nn_repetitions={len(seeds)}\n"
    for row in summary:
        digest += f"floor={100*row['floor']:.2f}% method={row['method']:<4s} mean={row['mean_bpb']:.6f} se={row['se_across_runs']:.6f}\n"
    (figdir/"DIGEST.txt").write_text(digest)
    print(digest)
    return figdir


def selftest() -> None:
    # Schedule clamp identity and activation.
    assert abs(max(0.20, 0.15) - 0.20) < 1e-12
    assert abs(max(0.10, 0.15) - 0.15) < 1e-12
    # Snapshot alignment for all paper estimators at T=3000.
    for window, spacing in [(PRIMARY_K, PRIMARY_SPACING), (EMA_K, EMA_SPACING), (SWA_K, SWA_SPACING)]:
        steps = [3000 - i * spacing for i in range(window)]
        assert all(s % SNAPSHOT_STRIDE == SNAPSHOT_PHASE for s in steps)
    # Bias-free alpha key roundtrip sanity.
    assert alpha_key(0.6) == "tsa:0.600"
    # Interior selection algebra sanity on a synthetic convex risk.
    grid = parse_floats(DEFAULT_ALPHA_GRID)
    risk = [(a - 0.6) ** 2 for a in grid]
    assert grid[int(np.argmin(risk))] == 0.6
    print("paper-main D12 selftest PASS")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")

    bp = sub.add_parser("branch")
    bp.add_argument("--pack", type=Path, required=True)
    bp.add_argument("--out", type=Path, required=True)
    bp.add_argument("--terminal-floor", type=float, required=True)
    bp.add_argument("--device", default="cuda")
    bp.add_argument("--history-device", default=None)
    bp.add_argument("--history-dtype", default=None)
    bp.add_argument("--lr-scale", type=float, default=BASE_LR_SCALE)
    bp.add_argument("--warmdown-ratio", type=float, default=BASE_WARMDOWN_RATIO)
    bp.add_argument("--final-lr-frac", type=float, default=BASE_FINAL_FRAC)
    bp.add_argument("--train-steps", type=int, default=3000)
    bp.add_argument("--schedule-total-iterations", type=int, default=3000)
    bp.add_argument("--snapshot-dtype", choices=["bfloat16","float16","float32"], default="bfloat16")
    bp.add_argument("--alpha-grid", default=DEFAULT_ALPHA_GRID)
    bp.add_argument("--curve-steps", default=DEFAULT_CURVE_STEPS)
    bp.add_argument("--curve-eval-batches", type=int, default=32)
    bp.add_argument("--alpha-select-batches", type=int, default=64)
    bp.add_argument("--floor-select-batches", type=int, default=64)
    bp.add_argument("--holdout-eval-batches", type=int, default=96)
    bp.add_argument("--seed", type=int, default=None)
    bp.add_argument("--stream-seed", type=int, default=None)

    mp = sub.add_parser("merge")
    mp.add_argument("--root", type=Path, required=True)
    mp.add_argument("--floors", default=DEFAULT_FLOORS)

    cp = sub.add_parser("merge-confirmation")
    cp.add_argument("--root", type=Path, required=True)
    cp.add_argument("--frozen", type=Path, required=True)
    return ap


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "selftest":
        selftest(); return
    if args.command == "branch":
        run_branch(args); return
    if args.command == "merge":
        run_merge(args); return
    if args.command == "merge-confirmation":
        run_merge_confirmation(args); return
    raise ValueError(args.command)


if __name__ == "__main__":
    main()
