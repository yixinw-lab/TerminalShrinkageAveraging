#!/usr/bin/env python3
"""Depth-12 structured TSA group-search suite.

The suite searches for a better parameter/tensor partition for Terminal Shrinkage
Averaging (TSA) than the existing hidden-matrix-vs-complement split. Development
runs train fresh D12 trajectories at a 10% terminal floor and evaluate a broad,
predeclared candidate library on a calibration block. A freeze stage chooses one
winner per structural family plus an overall winner. Fresh native and pure-AdamW
confirmation trajectories then evaluate only those frozen rules on the untouched
96-batch holdout.

This module intentionally reuses the installed structured_tsa_d12 restoration and
evaluation helpers so it stays aligned with the existing D12 experiment harness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import structured_tsa_d12 as st

SUITE = "d12_tsa_group_search_v1"
TRAIN_STEPS = 3000
K = 8
SPACING = 32
SNAPSHOT_STEPS = tuple(TRAIN_STEPS - (K - 1 - i) * SPACING for i in range(K))
CURVE_RESERVED_BATCHES = 32
DEV_SEARCH_BATCHES = 32
CONFIRM_HOLDOUT_BATCHES = 96
DEFAULT_REST_ALPHA = 0.25


@dataclass(frozen=True)
class ParamRecord:
    name: str
    shape: tuple[int, ...]
    ndim: int
    numel: int
    start: int
    end: int
    layer: int | None
    role: str
    module: str
    hidden: bool
    squareish: bool


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (list, tuple, dict, np.ndarray, torch.Tensor)):
                continue
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _canonical_hash(obj: Mapping[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _slug_floor(value: float) -> str:
    return f"{int(round(100 * float(value))):02d}pct"


def _layer_from_name(name: str) -> int | None:
    # Current NanoChat: transformer.h.7.attn.c_q.weight
    m = re.search(r"(?:^|\.)h\.(\d+)(?:\.|$)", name)
    if m:
        return int(m.group(1))
    # Tolerate variants such as blocks.7.* or layers.7.*
    m = re.search(r"(?:blocks|layers)\.(\d+)(?:\.|$)", name)
    return int(m.group(1)) if m else None


def _classify_role(name: str, shape: tuple[int, ...], layer: int | None) -> tuple[str, str, bool, bool]:
    low = name.lower()
    ndim = len(shape)
    hidden = layer is not None
    squareish = False
    if ndim >= 2 and min(shape[-2:]) > 0:
        squareish = max(shape[-2:]) / min(shape[-2:]) <= 1.5

    if hidden and (".attn." in low or ".attention." in low):
        if any(tok in low for tok in ("c_q", "c_k", "c_v", "q_proj", "k_proj", "v_proj", "qkv")):
            return "attn_qkv", "attn", True, squareish
        if any(tok in low for tok in ("c_proj", "o_proj", "out_proj")):
            return "attn_out", "attn", True, squareish
        return "attn_aux", "attn", True, squareish

    if hidden and (".mlp." in low or ".ffn." in low or ".feed_forward." in low):
        if any(tok in low for tok in ("c_fc", "fc1", "up_proj", "gate_proj", "w1", "w3")):
            return "mlp_up", "mlp", True, squareish
        if any(tok in low for tok in ("c_proj", "fc2", "down_proj", "w2")):
            return "mlp_down", "mlp", True, squareish
        return "mlp_aux", "mlp", True, squareish

    if hidden:
        return "hidden_other", "other_hidden", True, squareish

    if "lm_head" in low or "unembed" in low or "output" in low and ndim >= 2:
        return "unembed", "unembed", False, squareish
    if "value_embed" in low or "value_emb" in low:
        return "value_embed", "embedding", False, squareish
    if "wte" in low or "token_embed" in low or "embedding" in low and ndim >= 2:
        return "token_embed", "embedding", False, squareish
    if ndim <= 1:
        return "scalar", "scalar", False, squareish
    return "other", "other", False, squareish


def _parameter_records(model, spec, flat_gpu: torch.Tensor | None = None) -> tuple[list[ParamRecord], dict[str, Any]]:
    named = list(model.named_parameters())
    total = sum(int(p.numel()) for _, p in named)
    if total != int(spec.numel):
        raise RuntimeError(
            f"named-parameter total {total} != flattened spec numel {int(spec.numel)}; "
            "cannot safely construct groupwise candidates"
        )

    records: list[ParamRecord] = []
    offset = 0
    for name, p in named:
        n = int(p.numel())
        shape = tuple(int(x) for x in p.shape)
        layer = _layer_from_name(name)
        role, module, hidden, squareish = _classify_role(name, shape, layer)
        records.append(
            ParamRecord(
                name=name,
                shape=shape,
                ndim=int(p.ndim),
                numel=n,
                start=offset,
                end=offset + n,
                layer=layer,
                role=role,
                module=module,
                hidden=hidden,
                squareish=squareish,
            )
        )
        offset += n

    # Sample-check that model.named_parameters() order agrees with FlattenSpec.
    if flat_gpu is not None:
        if int(flat_gpu.numel()) != total:
            raise RuntimeError("flat_gpu size mismatch")
        mismatches: list[str] = []
        for rec, (_name, p) in zip(records, named):
            if rec.numel == 0:
                continue
            idxs = sorted(set((0, rec.numel // 2, rec.numel - 1)))
            local = p.detach().reshape(-1)
            for idx in idxs:
                a = flat_gpu[rec.start + idx].float()
                b = local[idx].float()
                if not bool(torch.isclose(a, b, rtol=0.0, atol=0.0).item()):
                    mismatches.append(f"{rec.name}@{idx}")
                    break
            if len(mismatches) >= 5:
                break
        if mismatches:
            raise RuntimeError(
                "FlattenSpec order does not match model.named_parameters() at sampled coordinates: "
                + ", ".join(mismatches)
            )

    role_numel: dict[str, int] = {}
    role_tensors: dict[str, int] = {}
    for rec in records:
        role_numel[rec.role] = role_numel.get(rec.role, 0) + rec.numel
        role_tensors[rec.role] = role_tensors.get(rec.role, 0) + 1
    hidden_numel = sum(r.numel for r in records if r.hidden)
    layer_ids = sorted({r.layer for r in records if r.layer is not None})
    meta = {
        "total_numel": total,
        "hidden_numel": hidden_numel,
        "hidden_fraction": hidden_numel / total,
        "n_layers_seen": len(layer_ids),
        "layer_ids": layer_ids,
        "role_numel": role_numel,
        "role_tensors": role_tensors,
    }
    return records, meta


def _paired_gain(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[float, float]:
    c = np.asarray(candidate.get("bpb_values", []), dtype=float)
    b = np.asarray(baseline.get("bpb_values", []), dtype=float)
    if len(c) == len(b) and len(c) > 0:
        d = b - c
        mean = float(d.mean())
        se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
        return mean, se
    return float(baseline["bpb"] - candidate["bpb"]), math.nan


def _trajectory_stats(records: Sequence[ParamRecord], snapshots: Mapping[int, torch.Tensor]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    steps = list(SNAPSHOT_STEPS)
    score_map: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    eps = 1e-30
    for rec in records:
        if not rec.hidden:
            continue
        updates = [
            snapshots[steps[i]][rec.start:rec.end] - snapshots[steps[i - 1]][rec.start:rec.end]
            for i in range(1, len(steps))
        ]
        mean_update = torch.stack(updates, dim=0).mean(dim=0)
        drift = float(torch.dot(mean_update, mean_update).item())
        residual = 0.0
        for u in updates:
            z = u - mean_update
            residual += float(torch.dot(z, z).item())
        residual /= max(len(updates), 1)
        nu = residual / (residual + drift + eps)
        cosines: list[float] = []
        for a, b in zip(updates[:-1], updates[1:]):
            a2 = float(torch.dot(a, a).item())
            b2 = float(torch.dot(b, b).item())
            if a2 > 0.0 and b2 > 0.0:
                cosines.append(float(torch.dot(a, b).item()) / math.sqrt(a2 * b2))
        c = float(np.mean(cosines)) if cosines else 0.0
        directional = min(max(1.0 - max(c, 0.0), 0.0), 1.0)
        score = min(max(0.5 * (nu + directional), 0.0), 1.0)
        score_map[rec.name] = score
        rows.append(
            {
                "name": rec.name,
                "layer": rec.layer,
                "role": rec.role,
                "module": rec.module,
                "numel": rec.numel,
                "shape": "x".join(str(x) for x in rec.shape),
                "squareish": int(rec.squareish),
                "drift_energy": drift,
                "residual_energy": residual,
                "noise_fraction": nu,
                "mean_update_cosine": c,
                "trajectory_score": score,
            }
        )
        del updates, mean_update
    return score_map, rows


def _spec_id(family: str, **params: Any) -> str:
    bits = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, float):
            bits.append(f"{key}={value:.3f}")
        else:
            bits.append(f"{key}={value}")
    return family + ("|" + "|".join(bits) if bits else "")


def _candidate(family: str, **params: Any) -> dict[str, Any]:
    return {"candidate_id": _spec_id(family, **params), "family": family, "params": params}


def _candidate_library(n_layers: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    # Dense scalar baseline, deliberately including the known neighborhood.
    for i in range(21):
        out.append(_candidate("scalar", alpha=0.05 * i))

    hidden_grid = (0.45, 0.55, 0.625, 0.70, 0.80)
    rest_grid = (0.0, 0.125, 0.25, 0.375, 0.50)
    for ah in hidden_grid:
        for ar in rest_grid:
            out.append(_candidate("hidden_rest", hidden=ah, rest=ar))

    pair_grid = (0.45, 0.55, 0.65, 0.75, 0.85)
    for a in pair_grid:
        for b in pair_grid:
            out.append(_candidate("attn_mlp_rest", attn=a, mlp=b, rest=DEFAULT_REST_ALPHA))
            out.append(_candidate("input_write_rest", input=a, write=b, rest=DEFAULT_REST_ALPHA))
            out.append(_candidate("square_rect_rest", square=a, rectangular=b, rest=DEFAULT_REST_ALPHA))
            out.append(_candidate("depth_half_rest", early=a, late=b, rest=DEFAULT_REST_ALPHA))

    depth_patterns = (
        (0.55, 0.55, 0.55), (0.65, 0.65, 0.65), (0.75, 0.75, 0.75),
        (0.45, 0.60, 0.75), (0.40, 0.60, 0.80), (0.50, 0.65, 0.80),
        (0.75, 0.60, 0.45), (0.80, 0.60, 0.40), (0.80, 0.65, 0.50),
        (0.45, 0.75, 0.45), (0.50, 0.80, 0.50),
        (0.75, 0.45, 0.75), (0.80, 0.50, 0.80),
        (0.55, 0.55, 0.80), (0.45, 0.65, 0.85),
    )
    for early, middle, late in depth_patterns:
        out.append(_candidate(
            "depth_thirds_rest", early=early, middle=middle, late=late, rest=DEFAULT_REST_ALPHA
        ))

    # Four functional matrix roles. Rather than a 5^4 cartesian sweep, use a
    # coordinate design around 0.65 plus targeted input/write asymmetries.
    base = 0.65
    roles = ("qkv", "attn_out", "mlp_up", "mlp_down")
    out.append(_candidate("functional4", qkv=base, attn_out=base, mlp_up=base, mlp_down=base, rest=DEFAULT_REST_ALPHA))
    for role in roles:
        for value in (0.45, 0.55, 0.75, 0.85):
            p = {"qkv": base, "attn_out": base, "mlp_up": base, "mlp_down": base, "rest": DEFAULT_REST_ALPHA}
            p[role] = value
            out.append(_candidate("functional4", **p))
    targeted = (
        (0.75, 0.55, 0.75, 0.55), (0.85, 0.45, 0.85, 0.45),
        (0.55, 0.75, 0.55, 0.75), (0.45, 0.85, 0.45, 0.85),
        (0.75, 0.75, 0.55, 0.55), (0.55, 0.55, 0.75, 0.75),
    )
    for qkv, ao, mu, md in targeted:
        out.append(_candidate("functional4", qkv=qkv, attn_out=ao, mlp_up=mu, mlp_down=md, rest=DEFAULT_REST_ALPHA))

    # Complement decomposition, because the previous Muon/rest split lumped
    # token/value embeddings, unembedding, and scalars together.
    for emb in (0.0, 0.15, 0.30, 0.45):
        for unemb in (0.0, 0.15, 0.30, 0.45):
            out.append(_candidate(
                "complement_split", hidden=0.65, embedding=emb, unembed=unemb, scalar=0.25, other=0.25
            ))

    # Snapshot-statistic adaptive rules. "high" means noisier/less persistent,
    # so the hypothesis is high >= low.
    for low in (0.45, 0.55, 0.65):
        for high in (0.65, 0.75, 0.85):
            if high >= low:
                out.append(_candidate("adaptive_binary", low=low, high=high, rest=DEFAULT_REST_ALPHA))
    for intercept in (0.25, 0.35, 0.45):
        for slope in (0.25, 0.50, 0.75):
            out.append(_candidate("adaptive_affine", intercept=intercept, slope=slope, rest=DEFAULT_REST_ALPHA))

    # Layer/module perturbation scans: these are both candidate rules and a
    # diagnostic map of where the scalar coefficient wants to move.
    for layer in range(n_layers):
        for target in (0.45, 0.85):
            out.append(_candidate("layer_scan", layer=layer, base=0.65, target=target, rest=DEFAULT_REST_ALPHA))
            for module in ("attn", "mlp"):
                out.append(_candidate(
                    "module_layer_scan", layer=layer, module=module, base=0.65, target=target,
                    rest=DEFAULT_REST_ALPHA,
                ))

    # Deduplicate by candidate_id in case a targeted pattern overlaps another.
    unique: dict[str, dict[str, Any]] = {}
    for item in out:
        unique[item["candidate_id"]] = item
    return list(unique.values())


def _alpha_for_record(
    rec: ParamRecord,
    candidate: Mapping[str, Any],
    *,
    n_layers: int,
    score_map: Mapping[str, float],
    score_median: float,
) -> float:
    family = str(candidate["family"])
    p = candidate["params"]

    if family == "scalar":
        return float(p["alpha"])
    if family == "hidden_rest":
        return float(p["hidden"] if rec.hidden else p["rest"])
    if family == "attn_mlp_rest":
        if rec.module == "attn": return float(p["attn"])
        if rec.module == "mlp": return float(p["mlp"])
        return float(p["rest"])
    if family == "input_write_rest":
        if rec.role in ("attn_qkv", "attn_aux", "mlp_up", "mlp_aux"):
            return float(p["input"])
        if rec.role in ("attn_out", "mlp_down"):
            return float(p["write"])
        return float(p["rest"])
    if family == "square_rect_rest":
        if rec.hidden:
            return float(p["square"] if rec.squareish else p["rectangular"])
        return float(p["rest"])
    if family == "depth_half_rest":
        if rec.layer is None:
            return float(p["rest"])
        return float(p["early"] if rec.layer < n_layers / 2 else p["late"])
    if family == "depth_thirds_rest":
        if rec.layer is None:
            return float(p["rest"])
        bucket = min(2, (3 * rec.layer) // max(n_layers, 1))
        return float((p["early"], p["middle"], p["late"])[bucket])
    if family == "functional4":
        if rec.role in ("attn_qkv", "attn_aux"):
            return float(p["qkv"])
        if rec.role == "attn_out":
            return float(p["attn_out"])
        if rec.role in ("mlp_up", "mlp_aux"):
            return float(p["mlp_up"])
        if rec.role == "mlp_down":
            return float(p["mlp_down"])
        return float(p["rest"])
    if family == "complement_split":
        if rec.hidden: return float(p["hidden"])
        if rec.role in ("token_embed", "value_embed"): return float(p["embedding"])
        if rec.role == "unembed": return float(p["unembed"])
        if rec.role == "scalar": return float(p["scalar"])
        return float(p["other"])
    if family == "adaptive_binary":
        if not rec.hidden:
            return float(p["rest"])
        score = float(score_map.get(rec.name, score_median))
        return float(p["high"] if score >= score_median else p["low"])
    if family == "adaptive_affine":
        if not rec.hidden:
            return float(p["rest"])
        score = float(score_map.get(rec.name, score_median))
        return min(max(float(p["intercept"]) + float(p["slope"]) * score, 0.0), 1.0)
    if family == "layer_scan":
        if rec.layer is None:
            return float(p["rest"])
        return float(p["target"] if rec.layer == int(p["layer"]) else p["base"])
    if family == "module_layer_scan":
        if rec.layer is None:
            return float(p["rest"])
        if rec.layer == int(p["layer"]) and rec.module == str(p["module"]):
            return float(p["target"])
        return float(p["base"])
    raise KeyError(f"unknown candidate family {family}")


def _materialize_candidate(
    buffer: torch.Tensor,
    origin: torch.Tensor,
    delta: torch.Tensor,
    records: Sequence[ParamRecord],
    candidate: Mapping[str, Any],
    *,
    n_layers: int,
    score_map: Mapping[str, float],
    score_median: float,
) -> torch.Tensor:
    family = str(candidate["family"])
    if family == "scalar":
        buffer.copy_(origin)
        alpha = float(candidate["params"]["alpha"])
        if alpha != 0.0:
            buffer.add_(delta, alpha=alpha)
        return buffer
    buffer.copy_(origin)
    for rec in records:
        alpha = _alpha_for_record(
            rec, candidate, n_layers=n_layers, score_map=score_map, score_median=score_median
        )
        if alpha != 0.0:
            buffer[rec.start:rec.end].add_(delta[rec.start:rec.end], alpha=float(alpha))
    return buffer


def _snapshot_mean(snapshots: Mapping[int, torch.Tensor]) -> torch.Tensor:
    mean = torch.zeros_like(snapshots[SNAPSHOT_STEPS[-1]], dtype=torch.float32)
    for step in SNAPSHOT_STEPS:
        mean.add_(snapshots[step], alpha=1.0 / K)
    return mean


def _load_frozen(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if obj.get("suite") != SUITE:
        raise RuntimeError(f"expected {SUITE}, got {obj.get('suite')}")
    claimed = obj.get("freeze_sha256")
    payload = dict(obj)
    payload.pop("freeze_sha256", None)
    if claimed != _canonical_hash(payload):
        raise RuntimeError("frozen group-rule hash mismatch")
    return obj


def _runtime_defaults(args: argparse.Namespace) -> argparse.Namespace:
    return st._runtime_defaults(args)


def _train_trajectory(runtime: argparse.Namespace):
    from . import paper_main_d12 as pm
    from .filter_suite import BudgetLedger, OPTIMIZER_OPS_PER_PARAMETER
    from .skip_core import Timer, seed_everything
    from .skip_suite import exact_gradient

    runtime = _runtime_defaults(runtime)
    pack, restored = st._restore(runtime.pack.resolve(), runtime)
    args, model, optimizer, token_bytes, device, spec, _history, _bank = restored
    actual_optimizer = st._pack_optimizer(pack)
    aliases = {"native": ("native", "muon"), "pure_adamw": ("pure_adamw", "adamw", "adam")}
    if not any(alias in actual_optimizer.lower() for alias in aliases[runtime.optimizer_label]):
        raise RuntimeError(
            f"manifest optimizer {runtime.optimizer_label} does not match pack optimizer {actual_optimizer}"
        )
    if int(runtime.train_steps) != TRAIN_STEPS:
        raise ValueError(f"suite requires train_steps={TRAIN_STEPS}")

    seed_everything(int(runtime.seed))
    stream = pack["branch_batches"]
    grad_accum = int(args.grad_accum)
    if len(stream) // grad_accum < TRAIN_STEPS:
        raise RuntimeError("pack does not contain 3000 optimizer-step blocks")
    permutation = np.random.default_rng(int(runtime.stream_seed)).permutation(TRAIN_STEPS).astype(np.int64)
    permutation_fp = hashlib.sha256(permutation.tobytes()).hexdigest()

    micro_tokens = int(args.device_batch_size) * int(args.max_seq_len)
    full_tokens = micro_tokens * grad_accum
    ledger = BudgetLedger(spec.numel, full_tokens)
    snapshots: dict[int, torch.Tensor] = {}
    training_rows: list[dict[str, Any]] = []
    elapsed = 0.0

    print(
        f"[setup] optimizer={runtime.optimizer_label} floor={float(runtime.terminal_floor):.2f} "
        f"stream_seed={runtime.stream_seed} snapshots={SNAPSHOT_STEPS}",
        flush=True,
    )
    for step in range(1, TRAIN_STEPS + 1):
        block = int(permutation[step - 1])
        batches = stream[block * grad_accum : (block + 1) * grad_accum]
        lr_mult = pm._apply_floor_lr(optimizer, step - 1, args, runtime)
        with Timer() as timer:
            _g, train_loss, _features, _seconds = exact_gradient(
                model, optimizer, spec, batches, device,
                need_forward_features=False, token_sample_max=args.token_sample_max,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        elapsed += float(timer.seconds)
        ledger.add_exact(full_tokens, grad_accum, timer.seconds, charged=True)
        ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
        ledger.optimizer_steps += 1
        if step in SNAPSHOT_STEPS:
            snapshots[step] = spec.flatten_parameters(dtype=torch.float32).detach().to("cpu", dtype=torch.float32)
            print(f"[snapshot] step={step} count={len(snapshots)}/{K}", flush=True)
        if step % 250 == 0:
            training_rows.append({
                "step": step,
                "train_loss": float(train_loss),
                "lr_multiplier": float(lr_mult),
                "elapsed_training_seconds": elapsed,
            })
            print(f"[train] step={step} loss={float(train_loss):.6f} lr={float(lr_mult):.6f}", flush=True)

    missing = sorted(set(SNAPSHOT_STEPS) - set(snapshots))
    if missing:
        raise RuntimeError(f"missing snapshots: {missing}")
    restore_gpu = spec.flatten_parameters(dtype=torch.float32)
    records, param_meta = _parameter_records(model, spec, restore_gpu)
    return {
        "runtime": runtime,
        "pack": pack,
        "args": args,
        "model": model,
        "optimizer": optimizer,
        "token_bytes": token_bytes,
        "device": device,
        "spec": spec,
        "snapshots": snapshots,
        "restore_gpu": restore_gpu,
        "records": records,
        "param_meta": param_meta,
        "actual_optimizer": actual_optimizer,
        "permutation_fp": permutation_fp,
        "training_rows": training_rows,
        "ledger": ledger,
    }


def smoke(runtime: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("smoke must run on a GPU allocation")
    runtime = _runtime_defaults(runtime)
    pack, restored = st._restore(runtime.native_pack.resolve(), runtime)
    _args, model, _optimizer, token_bytes, device, spec, _history, _bank = restored
    flat = spec.flatten_parameters(dtype=torch.float32)
    records, meta = _parameter_records(model, spec, flat)
    required = {"attn_qkv", "attn_out", "mlp_up", "mlp_down"}
    present = {r.role for r in records}
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"required functional groups missing: {missing}")
    if len(pack.get("eval_batches", [])) < CURVE_RESERVED_BATCHES + CONFIRM_HOLDOUT_BATCHES:
        raise RuntimeError("pack does not contain enough eval batches")
    ev = st._eval_candidate(model, spec, flat, flat.detach().to("cpu"), pack["eval_batches"][:1], token_bytes, device)
    out = runtime.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "name": r.name, "shape": "x".join(str(x) for x in r.shape), "numel": r.numel,
            "layer": r.layer, "role": r.role, "module": r.module, "hidden": int(r.hidden),
            "squareish": int(r.squareish),
        }
        for r in records
    ]
    _write_csv(out / "SMOKE_PARAMETER_MAP.csv", rows)
    lines = [
        f"suite={SUITE}",
        f"python={os.sys.executable}",
        f"torch={torch.__version__}",
        f"native_pack={runtime.native_pack.resolve()}",
        f"total_numel={meta['total_numel']}",
        f"hidden_numel={meta['hidden_numel']}",
        f"hidden_fraction={meta['hidden_fraction']:.6f}",
        f"n_layers_seen={meta['n_layers_seen']}",
        f"candidate_count={len(_candidate_library(int(meta['n_layers_seen'])))}",
        f"one_batch_bpb={float(ev['bpb'])}",
        "TSA GROUP SEARCH SMOKE PASS",
    ]
    (out / "SMOKE.txt").write_text("\n".join(lines) + "\n")
    print((out / "SMOKE.txt").read_text(), flush=True)
    return out / "SMOKE.txt"


def develop(runtime: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("development must run on a GPU allocation")
    state = _train_trajectory(runtime)
    runtime = state["runtime"]
    pack = state["pack"]
    model, spec = state["model"], state["spec"]
    token_bytes, device = state["token_bytes"], state["device"]
    snapshots, restore_gpu = state["snapshots"], state["restore_gpu"]
    records, meta = state["records"], state["param_meta"]

    score_map, stat_rows = _trajectory_stats(records, snapshots)
    hidden_scores = [score_map[r.name] for r in records if r.hidden and r.name in score_map]
    score_median = float(np.median(hidden_scores)) if hidden_scores else 0.5
    n_layers = int(meta["n_layers_seen"])
    library = _candidate_library(n_layers)
    search_batches = pack["eval_batches"][CURVE_RESERVED_BATCHES:CURVE_RESERVED_BATCHES + int(runtime.search_eval_batches)]
    if len(search_batches) != int(runtime.search_eval_batches):
        raise RuntimeError("insufficient development eval batches")

    origin = snapshots[TRAIN_STEPS]
    mean = _snapshot_mean(snapshots)
    delta = mean - origin
    buffer = torch.empty_like(origin)
    raw_ev = st._eval_candidate(model, spec, restore_gpu, origin, search_batches, token_bytes, device)
    rows: list[dict[str, Any]] = []
    print(f"[develop] evaluating {len(library)} candidates on {len(search_batches)} batches", flush=True)
    for idx, candidate in enumerate(library):
        vector = _materialize_candidate(
            buffer, origin, delta, records, candidate,
            n_layers=n_layers, score_map=score_map, score_median=score_median,
        )
        ev = st._eval_candidate(model, spec, restore_gpu, vector, search_batches, token_bytes, device)
        gain, se = _paired_gain(ev, raw_ev)
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "family": candidate["family"],
            "bpb": float(ev["bpb"]),
            "gain_vs_raw": gain,
            "paired_se": se,
            "spec_json": json.dumps(candidate, sort_keys=True, separators=(",", ":")),
        })
        if (idx + 1) % 25 == 0 or idx + 1 == len(library):
            print(f"[develop] {idx + 1}/{len(library)}", flush=True)

    out = runtime.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "candidate_results.csv", rows)
    _write_csv(out / "trajectory_tensor_stats.csv", stat_rows)
    _write_csv(out / "training_trace.csv", state["training_rows"])
    result = {
        "suite": SUITE,
        "stage": "develop",
        "optimizer_label": runtime.optimizer_label,
        "actual_pack_optimizer": state["actual_optimizer"],
        "terminal_floor": float(runtime.terminal_floor),
        "stream_seed": int(runtime.stream_seed),
        "seed": int(runtime.seed),
        "permutation_fingerprint": state["permutation_fp"],
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "search_eval_batches": int(runtime.search_eval_batches),
        "candidate_count": len(library),
        "trajectory_score_median": score_median,
        "parameter_meta": meta,
        "pack": st._pack_summary(runtime.pack.resolve(), pack),
        "ledger": state["ledger"].as_dict(),
    }
    _json_dump(out / "result.json", result)
    print(f"[result] wrote {out / 'result.json'}", flush=True)
    return out / "result.json"


def confirm(runtime: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("confirmation must run on a GPU allocation")
    frozen = _load_frozen(runtime.frozen_rules.resolve())
    state = _train_trajectory(runtime)
    runtime = state["runtime"]
    pack = state["pack"]
    model, spec = state["model"], state["spec"]
    token_bytes, device = state["token_bytes"], state["device"]
    snapshots, restore_gpu = state["snapshots"], state["restore_gpu"]
    records, meta = state["records"], state["param_meta"]

    score_map, stat_rows = _trajectory_stats(records, snapshots)
    hidden_scores = [score_map[r.name] for r in records if r.hidden and r.name in score_map]
    score_median = float(np.median(hidden_scores)) if hidden_scores else 0.5
    n_layers = int(meta["n_layers_seen"])

    # Always include two external scalar anchors for transfer diagnostics.
    candidates: dict[str, dict[str, Any]] = {}
    for item in frozen["selected_rules"]:
        candidates[item["candidate_id"]] = item["spec"]
    for alpha in (0.50, 0.55):
        item = _candidate("scalar", alpha=alpha)
        candidates[item["candidate_id"]] = item

    holdout = pack["eval_batches"][-int(runtime.holdout_eval_batches):]
    if len(holdout) != int(runtime.holdout_eval_batches):
        raise RuntimeError("insufficient confirmation holdout batches")
    origin = snapshots[TRAIN_STEPS]
    mean = _snapshot_mean(snapshots)
    delta = mean - origin
    buffer = torch.empty_like(origin)
    raw_ev = st._eval_candidate(model, spec, restore_gpu, origin, holdout, token_bytes, device)

    rows: list[dict[str, Any]] = [{
        "candidate_id": "raw", "family": "raw", "bpb": float(raw_ev["bpb"]),
        "gain_vs_raw": 0.0, "paired_se": 0.0,
    }]
    print(f"[confirm] evaluating {len(candidates)} frozen/anchor candidates on {len(holdout)} batches", flush=True)
    for candidate_id, candidate in candidates.items():
        vector = _materialize_candidate(
            buffer, origin, delta, records, candidate,
            n_layers=n_layers, score_map=score_map, score_median=score_median,
        )
        ev = st._eval_candidate(model, spec, restore_gpu, vector, holdout, token_bytes, device)
        gain, se = _paired_gain(ev, raw_ev)
        rows.append({
            "candidate_id": candidate_id,
            "family": candidate["family"],
            "bpb": float(ev["bpb"]),
            "gain_vs_raw": gain,
            "paired_se": se,
            "spec_json": json.dumps(candidate, sort_keys=True, separators=(",", ":")),
        })

    out = runtime.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "confirmation_results.csv", rows)
    _write_csv(out / "trajectory_tensor_stats.csv", stat_rows)
    _write_csv(out / "training_trace.csv", state["training_rows"])
    result = {
        "suite": SUITE,
        "stage": "confirm",
        "optimizer_label": runtime.optimizer_label,
        "actual_pack_optimizer": state["actual_optimizer"],
        "terminal_floor": float(runtime.terminal_floor),
        "stream_seed": int(runtime.stream_seed),
        "seed": int(runtime.seed),
        "permutation_fingerprint": state["permutation_fp"],
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "holdout_eval_batches": int(runtime.holdout_eval_batches),
        "frozen_rules": str(runtime.frozen_rules.resolve()),
        "frozen_rules_sha256": frozen["freeze_sha256"],
        "trajectory_score_median": score_median,
        "parameter_meta": meta,
        "pack": st._pack_summary(runtime.pack.resolve(), pack),
        "ledger": state["ledger"].as_dict(),
    }
    _json_dump(out / "result.json", result)
    print(f"[result] wrote {out / 'result.json'}", flush=True)
    return out / "result.json"


def selftest() -> None:
    # Classification and candidate logic tests do not require a model.
    role, module, hidden, sq = _classify_role("transformer.h.3.attn.c_q.weight", (768, 768), 3)
    assert (role, module, hidden, sq) == ("attn_qkv", "attn", True, True)
    role, module, hidden, sq = _classify_role("transformer.h.3.mlp.c_fc.weight", (3072, 768), 3)
    assert role == "mlp_up" and module == "mlp" and hidden and not sq
    role, module, hidden, _ = _classify_role("lm_head.weight", (32768, 768), None)
    assert role == "unembed" and not hidden
    assert _layer_from_name("transformer.h.11.mlp.c_proj.weight") == 11
    lib = _candidate_library(12)
    ids = [x["candidate_id"] for x in lib]
    assert len(ids) == len(set(ids))
    assert any(x["family"] == "adaptive_affine" for x in lib)
    assert any(x["family"] == "module_layer_scan" for x in lib)
    fake = {"suite": SUITE, "selected_rules": [], "x": 1}
    fake["freeze_sha256"] = _canonical_hash(fake)
    payload = dict(fake); claimed = payload.pop("freeze_sha256")
    assert claimed == _canonical_hash(payload)
    print(f"tsa_group_search_d12 selftest PASS candidate_count={len(lib)}")


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history-device", default=None)
    parser.add_argument("--history-dtype", default=None)
    parser.add_argument("--lr-scale", type=float, default=0.40)
    parser.add_argument("--warmdown-ratio", type=float, default=None)
    parser.add_argument("--final-lr-frac", type=float, default=None)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--schedule-total-iterations", type=int, default=TRAIN_STEPS)
    parser.add_argument("--snapshot-dtype", choices=["bfloat16", "float16", "float32"], default="float32")
    parser.add_argument("--curve-eval-batches", type=int, default=CURVE_RESERVED_BATCHES)
    parser.add_argument("--alpha-select-batches", type=int, default=0)
    parser.add_argument("--floor-select-batches", type=int, default=0)
    parser.add_argument("--search-eval-batches", type=int, default=DEV_SEARCH_BATCHES)
    parser.add_argument("--holdout-eval-batches", type=int, default=CONFIRM_HOLDOUT_BATCHES)
    parser.add_argument("--token-sample-max", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")
    sp = sub.add_parser("smoke")
    sp.add_argument("--native-pack", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    _add_runtime_args(sp)
    for name in ("develop", "confirm"):
        p = sub.add_parser(name)
        p.add_argument("--pack", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--optimizer-label", choices=["native", "pure_adamw"], required=True)
        p.add_argument("--terminal-floor", type=float, required=True)
        p.add_argument("--stream-seed", type=int, required=True)
        if name == "confirm":
            p.add_argument("--frozen-rules", type=Path, required=True)
        _add_runtime_args(p)
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "develop":
        develop(args)
    elif args.command == "confirm":
        confirm(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
