#!/usr/bin/env python3
"""D12 appendix sensitivity study for Terminal Shrinkage Averaging.

One depth-12 Muon+AdamW trajectory per data-order seed is trained with the frozen
10% terminal-floor recipe. A dense terminal checkpoint bank is retained in host
memory and all estimator comparisons are made post hoc on the identical trained
trajectory and untouched holdout evaluation batches.

Primary sensitivity grid:
    K in {4, 8, 16}
    spacing s in {16, 32, 64}
    TSA alpha = 0.55 fixed

Additional endpoint controls:
    uniform LAWA for each K/s pair
    finite checkpoint EWA on the matched K=8,s=32 window for beta in
        {0.50, 0.75, 0.90, 0.95}
    checkpoint EMA beta=.95 over K=16,s=16 (historical paper control)
    SWA-style uniform average over K=32,s=16
    tensorwise adaptive TSA on all tensors and Muon-managed tensors

This module intentionally changes no NanoChat optimizer/training implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import paper_main_d12 as pm

K_VALUES = (4, 8, 16)
S_VALUES = (16, 32, 64)
PRIMARY_ALPHA = 0.55
MATCHED_EWA_BETAS = (0.50, 0.75, 0.90, 0.95)
HIST_EMA_BETA = 0.95
HIST_EMA_K = 16
HIST_EMA_SPACING = 16
SWA_K = 32
SWA_SPACING = 16
SNAPSHOT_STRIDE = 16
SNAPSHOT_PHASE = 8


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
                fields.append(key); seen.add(key)
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def _fingerprint(values: Sequence[int]) -> str:
    arr = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _eval_candidate(model, spec, current_gpu, candidate_cpu, eval_batches, token_bytes, device):
    from .filter_suite import evaluate_detailed
    from .skip_core import Timer
    try:
        spec.assign_parameters(candidate_cpu)
        with Timer() as timer:
            ev = evaluate_detailed(model, eval_batches, token_bytes, device)
    finally:
        spec.assign_parameters(current_gpu)
    return {**ev, "eval_seconds": timer.seconds}


def _paired_delta(method_vals, raw_vals):
    a = np.asarray(method_vals, dtype=float)
    b = np.asarray(raw_vals, dtype=float)
    d = a - b
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0


def run(runtime: argparse.Namespace) -> Path:
    from .filter_suite import BudgetLedger, OPTIMIZER_OPS_PER_PARAMETER, _install_canonical_build, _stabilize_nanochat_adamw_kernel, evaluate_detailed
    from .skip_core import Timer, seed_everything
    from .skip_suite import PACK_VERSION, exact_gradient, restore_branch
    from .trajectory_groups import build_slice_infos, make_averaged_candidate, tensor_stats

    _install_canonical_build()
    _stabilize_nanochat_adamw_kernel()

    pack = torch.load(runtime.pack, map_location="cpu", weights_only=False)
    if int(pack.get("pack_version", -1)) != PACK_VERSION:
        raise RuntimeError("pack version mismatch")
    if int(pack.get("prefix_steps", 0)) != 0:
        raise RuntimeError("appendix suite expects prefix_steps=0 D12 pack")

    args, model, optimizer, token_bytes, device, spec, _history, _bank = restore_branch(pack, runtime)
    seed_everything(int(runtime.seed))
    infos = build_slice_infos(spec, optimizer)
    n_layers = int(getattr(model.config, "n_layer", getattr(model.config, "depth", args.depth)))

    all_eval = pack["eval_batches"]
    holdout_start = int(runtime.curve_eval_batches + runtime.alpha_select_batches + runtime.floor_select_batches)
    holdout_end = holdout_start + int(runtime.holdout_eval_batches)
    if holdout_end > len(all_eval):
        raise RuntimeError(f"need holdout through {holdout_end}, pack has {len(all_eval)} eval batches")
    holdout = all_eval[holdout_start:holdout_end]

    if int(runtime.train_steps) % SNAPSHOT_STRIDE != SNAPSHOT_PHASE:
        raise ValueError(f"train_steps must be congruent to {SNAPSHOT_PHASE} mod {SNAPSHOT_STRIDE}")

    max_span = max(
        max((k - 1) * s for k in K_VALUES for s in S_VALUES),
        (HIST_EMA_K - 1) * HIST_EMA_SPACING,
        (SWA_K - 1) * SWA_SPACING,
    )
    bank = pm.AlignedSnapshotBank(
        stride=SNAPSHOT_STRIDE,
        phase=SNAPSHOT_PHASE,
        max_span=max_span,
        dtype=pm._dtype(runtime.snapshot_dtype),
    )

    stream = pack["branch_batches"]
    grad_accum = int(args.grad_accum)
    stream_steps = len(stream) // grad_accum
    if stream_steps < int(runtime.train_steps):
        raise RuntimeError(f"pack has {stream_steps} optimizer-step blocks, need {runtime.train_steps}")

    rng = np.random.default_rng(int(runtime.stream_seed))
    permutation = rng.permutation(int(runtime.train_steps)).astype(np.int64)
    perm_fp = _fingerprint(permutation.tolist())

    micro_tokens = int(args.device_batch_size) * int(args.max_seq_len)
    full_tokens = micro_tokens * grad_accum
    ledger = BudgetLedger(spec.numel, full_tokens)
    training_rows: list[dict[str, Any]] = []

    print(
        f"[setup] appendix averaging seed={runtime.stream_seed} floor={runtime.terminal_floor} "
        f"T={runtime.train_steps} H={runtime.schedule_total_iterations} max_span={max_span}", flush=True
    )

    for step in range(1, int(runtime.train_steps) + 1):
        block_index = int(permutation[step - 1])
        lo = block_index * grad_accum
        batches = stream[lo:lo + grad_accum]
        lr_mult = pm._apply_floor_lr(optimizer, step - 1, args, runtime)
        with Timer() as train_timer:
            _g, train_loss, _features, _seconds = exact_gradient(
                model, optimizer, spec, batches, device,
                need_forward_features=False,
                token_sample_max=args.token_sample_max,
            )
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        ledger.add_exact(full_tokens, grad_accum, train_timer.seconds, charged=True)
        ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
        ledger.optimizer_steps += 1
        if bank.should_store(step):
            bank.append(step, spec.flatten_parameters(dtype=torch.float32))
        if step % 250 == 0 or step == int(runtime.train_steps):
            training_rows.append({"step": step, "train_loss": float(train_loss), "lr_multiplier": float(lr_mult)})
            print(f"[train] step={step} loss={float(train_loss):.6f} lr={float(lr_mult):.5f} bank_gib={bank.memory_bytes()/2**30:.2f}", flush=True)

    current_gpu = spec.flatten_parameters(dtype=torch.float32)
    current_cpu = current_gpu.detach().to(device="cpu", dtype=torch.float32)
    with Timer() as timer:
        raw = evaluate_detailed(model, holdout, token_bytes, device)
    rows: list[dict[str, Any]] = [{
        "recipe": "raw", "method": "Raw iterate", "k": 0, "spacing": 0,
        "alpha": 0.0, "beta": float("nan"), "bpb": float(raw["bpb"]),
        "eval_seconds": timer.seconds, "stream_seed": int(runtime.stream_seed),
    }]

    primary_selected = None
    primary_mean = None
    for k in K_VALUES:
        for spacing in S_VALUES:
            selected = bank.select(k, spacing, end_step=int(runtime.train_steps))
            mean = pm.uniform_average(selected)
            if k == 8 and spacing == 32:
                primary_selected = selected
                primary_mean = mean
            delta = mean - current_cpu
            for alpha, label in ((float(runtime.alpha), "TSA"), (1.0, "LAWA")):
                cand = current_cpu + alpha * delta
                ev = _eval_candidate(model, spec, current_gpu, cand, holdout, token_bytes, device)
                d, se = _paired_delta(ev["bpb_values"], raw["bpb_values"])
                rows.append({
                    "recipe": f"{'tsa' if alpha < 1 else 'lawa'}:k{k}:s{spacing}:a{alpha:.3f}",
                    "method": label, "k": k, "spacing": spacing, "alpha": alpha,
                    "beta": float("nan"), "bpb": float(ev["bpb"]),
                    "paired_delta_method_minus_raw": d, "paired_se": se,
                    "eval_seconds": ev["eval_seconds"], "stream_seed": int(runtime.stream_seed),
                })
                del cand
            del mean

    if primary_selected is None or primary_mean is None:
        raise RuntimeError("primary K=8,s=32 window missing")

    # Matched-window finite exponential weighting controls.
    for beta in MATCHED_EWA_BETAS:
        vec = pm.exponential_average(primary_selected, beta)
        ev = _eval_candidate(model, spec, current_gpu, vec, holdout, token_bytes, device)
        d, se = _paired_delta(ev["bpb_values"], raw["bpb_values"])
        rows.append({
            "recipe": f"ewa:beta{beta:.2f}:k8:s32", "method": "Matched-window EWA",
            "k": 8, "spacing": 32, "alpha": float("nan"), "beta": beta,
            "bpb": float(ev["bpb"]), "paired_delta_method_minus_raw": d, "paired_se": se,
            "eval_seconds": ev["eval_seconds"], "stream_seed": int(runtime.stream_seed),
        })

    # Historical checkpoint EMA and SWA-style controls used in earlier runs.
    hist_selected = bank.select(HIST_EMA_K, HIST_EMA_SPACING, end_step=int(runtime.train_steps))
    hist_ema = pm.exponential_average(hist_selected, HIST_EMA_BETA)
    ev = _eval_candidate(model, spec, current_gpu, hist_ema, holdout, token_bytes, device)
    d, se = _paired_delta(ev["bpb_values"], raw["bpb_values"])
    rows.append({
        "recipe": "ema:0.95:16:16", "method": "Checkpoint EMA",
        "k": HIST_EMA_K, "spacing": HIST_EMA_SPACING, "alpha": float("nan"), "beta": HIST_EMA_BETA,
        "bpb": float(ev["bpb"]), "paired_delta_method_minus_raw": d, "paired_se": se,
        "eval_seconds": ev["eval_seconds"], "stream_seed": int(runtime.stream_seed),
    })

    swa_selected = bank.select(SWA_K, SWA_SPACING, end_step=int(runtime.train_steps))
    swa = pm.uniform_average(swa_selected)
    ev = _eval_candidate(model, spec, current_gpu, swa, holdout, token_bytes, device)
    d, se = _paired_delta(ev["bpb_values"], raw["bpb_values"])
    rows.append({
        "recipe": "swastyle:32:16", "method": "SWA-style late average",
        "k": SWA_K, "spacing": SWA_SPACING, "alpha": 1.0, "beta": float("nan"),
        "bpb": float(ev["bpb"]), "paired_delta_method_minus_raw": d, "paired_se": se,
        "eval_seconds": ev["eval_seconds"], "stream_seed": int(runtime.stream_seed),
    })

    # Exploratory tensorwise controls on the same primary window.
    stats = tensor_stats(primary_selected, infos)
    for adaptive_selector, method in (("all", "Adaptive TSA (all tensors)"), ("muon", "Adaptive TSA (Muon tensors)")):
        cand, alpha_rows = make_averaged_candidate(
            current_gpu, primary_mean, infos,
            selector="all", n_layers=n_layers,
            adaptive_rule="hybrid", adaptive_strength=1.0,
            stats_rows=stats, adaptive_selector=adaptive_selector,
        )
        ev = _eval_candidate(model, spec, current_gpu, cand, holdout, token_bytes, device)
        d, se = _paired_delta(ev["bpb_values"], raw["bpb_values"])
        anum = sum(float(x["alpha"]) * int(x["numel"]) for x in alpha_rows)
        aden = max(sum(int(x["numel"]) for x in alpha_rows), 1)
        rows.append({
            "recipe": f"adaptive:hybrid:8:32:1.0:{adaptive_selector}", "method": method,
            "k": 8, "spacing": 32, "alpha": anum/aden, "beta": float("nan"),
            "bpb": float(ev["bpb"]), "paired_delta_method_minus_raw": d, "paired_se": se,
            "eval_seconds": ev["eval_seconds"], "stream_seed": int(runtime.stream_seed),
            "mean_alpha_by_numel": anum/aden,
        })
        del cand

    out = runtime.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    result = {
        "suite": "d12_appendix_averaging_suite_v1",
        "stream_seed": int(runtime.stream_seed),
        "permutation_fingerprint": perm_fp,
        "pack": str(runtime.pack.resolve()),
        "optimizer": pack.get("build_args", {}).get("optimizer", "unknown"),
        "train_steps": int(runtime.train_steps),
        "schedule_total_iterations": int(runtime.schedule_total_iterations),
        "terminal_floor": float(runtime.terminal_floor),
        "alpha": float(runtime.alpha),
        "k_values": list(K_VALUES), "spacing_values": list(S_VALUES),
        "matched_ewa_betas": list(MATCHED_EWA_BETAS),
        "eval_split": {"holdout_start": holdout_start, "holdout_batches": len(holdout)},
        "snapshot_memory_bytes": bank.memory_bytes(), "snapshot_steps": bank.steps(),
        "rows": rows, "training_trace": training_rows, "ledger": ledger.as_dict(),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    _write_csv(out / "endpoint_results.csv", rows)
    _write_csv(out / "training_trace.csv", training_rows)
    print(f"[result] wrote {out/'result.json'}", flush=True)
    return out / "result.json"


def selftest() -> None:
    assert 3000 % SNAPSHOT_STRIDE == SNAPSHOT_PHASE
    assert max((k-1)*s for k in K_VALUES for s in S_VALUES) == 960
    # Algebraic check for EWA normalization.
    fake = [(i, torch.tensor([float(i)])) for i in range(8)]
    for b in MATCHED_EWA_BETAS:
        x = pm.exponential_average(fake, b)
        assert torch.isfinite(x).all()
    print("appendix averaging selftest PASS")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    rp = sub.add_parser("run")
    rp.add_argument("--pack", type=Path, required=True)
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--terminal-floor", type=float, default=0.10)
    rp.add_argument("--alpha", type=float, default=PRIMARY_ALPHA)
    rp.add_argument("--device", default="cuda")
    rp.add_argument("--history-device", default=None)
    rp.add_argument("--history-dtype", default=None)
    rp.add_argument("--lr-scale", type=float, default=0.40)
    rp.add_argument("--warmdown-ratio", type=float, default=pm.BASE_WARMDOWN_RATIO)
    rp.add_argument("--final-lr-frac", type=float, default=pm.BASE_FINAL_FRAC)
    rp.add_argument("--train-steps", type=int, default=3000)
    rp.add_argument("--schedule-total-iterations", type=int, default=3000)
    rp.add_argument("--snapshot-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    rp.add_argument("--curve-eval-batches", type=int, default=32)
    rp.add_argument("--alpha-select-batches", type=int, default=64)
    rp.add_argument("--floor-select-batches", type=int, default=64)
    rp.add_argument("--holdout-eval-batches", type=int, default=96)
    rp.add_argument("--seed", type=int, default=1337)
    rp.add_argument("--stream-seed", type=int, required=True)
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.command == "selftest": selftest(); return
    if args.command == "run": run(args); return
    raise SystemExit(2)

if __name__ == "__main__": main()
