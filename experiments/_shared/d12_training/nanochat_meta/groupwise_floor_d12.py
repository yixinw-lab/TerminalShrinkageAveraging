#!/usr/bin/env python3
"""Depth-12 predeclared groupwise terminal-floor experiment.

This suite tests whether the parameter-role heterogeneity already confirmed for
Terminal Shrinkage Averaging (TSA) also predicts heterogeneous terminal learning-
rate activity.  It deliberately performs NO hyperparameter selection on the new
runs.  A frozen structured-TSA rule is mapped monotonically onto the already
studied 5%-15% terminal-floor range, then compared with uniform-floor controls on
fresh paired training streams.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import structured_tsa_d12 as st

SUITE = "d12_tsa_groupwise_floor_v1"
TRAIN_STEPS = 3000
K = 8
SPACING = 32
SNAPSHOT_STEPS = tuple(TRAIN_STEPS - (K - 1 - i) * SPACING for i in range(K))
HOLDOUT_BATCHES = 96

# These are frozen from the preceding fresh-confirmation structured-TSA result.
EXPECTED_ALPHA = {
    "embedding": 0.00,
    "hidden": 0.65,
    "unembed": 0.45,
    "rest": 0.25,  # scalar and other had the same selected coefficient
}
EXPECTED_FLOOR_MIN = 0.05
EXPECTED_FLOOR_MAX = 0.15
EXPECTED_ALPHA_MAX = 0.65
EXPECTED_ARMS = ("uniform05", "uniform10", "matched_uniform", "mapped")


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


def _load_protocol(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    claimed = obj.get("protocol_sha256")
    payload = dict(obj)
    payload.pop("protocol_sha256", None)
    actual = _canonical_hash(payload)
    if claimed != actual:
        raise RuntimeError(f"protocol hash mismatch: claimed={claimed} actual={actual}")
    if obj.get("suite") != SUITE:
        raise RuntimeError(f"wrong protocol suite: {obj.get('suite')}")
    got_alpha = {k: float(v) for k, v in obj["frozen_tsa_alpha"].items()}
    if got_alpha != EXPECTED_ALPHA:
        raise RuntimeError(f"frozen alpha mismatch: {got_alpha} != {EXPECTED_ALPHA}")
    fmap = obj["floor_map"]
    checks = (
        (float(fmap["floor_min"]), EXPECTED_FLOOR_MIN, "floor_min"),
        (float(fmap["floor_max"]), EXPECTED_FLOOR_MAX, "floor_max"),
        (float(fmap["alpha_max"]), EXPECTED_ALPHA_MAX, "alpha_max"),
    )
    for got, want, name in checks:
        if not math.isclose(got, want, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"{name} mismatch: {got} != {want}")
    arms = tuple(obj["arms"])
    if arms != EXPECTED_ARMS:
        raise RuntimeError(f"arm mismatch: {arms} != {EXPECTED_ARMS}")
    return obj


def _layer_from_name(name: str) -> int | None:
    m = re.search(r"(?:^|\.)h\.(\d+)(?:\.|$)", name)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:blocks|layers)\.(\d+)(?:\.|$)", name)
    return int(m.group(1)) if m else None


def _classify_role(name: str, shape: tuple[int, ...], layer: int | None) -> tuple[str, str, bool]:
    # Keep the same broad classification used by the preceding group-search suite
    # so the frozen TSA coefficients have exactly the same meaning here.
    low = name.lower()
    ndim = len(shape)
    hidden = layer is not None

    if hidden and (".attn." in low or ".attention." in low):
        if any(tok in low for tok in ("c_q", "c_k", "c_v", "q_proj", "k_proj", "v_proj", "qkv")):
            return "attn_qkv", "attn", True
        if any(tok in low for tok in ("c_proj", "o_proj", "out_proj")):
            return "attn_out", "attn", True
        return "attn_aux", "attn", True

    if hidden and (".mlp." in low or ".ffn." in low or ".feed_forward." in low):
        if any(tok in low for tok in ("c_fc", "fc1", "up_proj", "gate_proj", "w1", "w3")):
            return "mlp_up", "mlp", True
        if any(tok in low for tok in ("c_proj", "fc2", "down_proj", "w2")):
            return "mlp_down", "mlp", True
        return "mlp_aux", "mlp", True

    if hidden:
        return "hidden_other", "other_hidden", True
    if "lm_head" in low or "unembed" in low or ("output" in low and ndim >= 2):
        return "unembed", "unembed", False
    if "value_embed" in low or "value_emb" in low:
        return "value_embed", "embedding", False
    if "wte" in low or "token_embed" in low or ("embedding" in low and ndim >= 2):
        return "token_embed", "embedding", False
    if ndim <= 1:
        return "scalar", "scalar", False
    return "other", "other", False


def _parameter_records(model, spec, flat_gpu: torch.Tensor | None = None) -> tuple[list[ParamRecord], dict[str, Any]]:
    named = list(model.named_parameters())
    total = sum(int(p.numel()) for _, p in named)
    if total != int(spec.numel):
        raise RuntimeError(f"named parameter total {total} != FlattenSpec numel {int(spec.numel)}")

    records: list[ParamRecord] = []
    offset = 0
    for name, p in named:
        n = int(p.numel())
        shape = tuple(int(x) for x in p.shape)
        layer = _layer_from_name(name)
        role, module, hidden = _classify_role(name, shape, layer)
        records.append(ParamRecord(
            name=name, shape=shape, ndim=int(p.ndim), numel=n,
            start=offset, end=offset + n, layer=layer,
            role=role, module=module, hidden=hidden,
        ))
        offset += n

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
            raise RuntimeError("FlattenSpec/model parameter order mismatch: " + ", ".join(mismatches))

    bucket_numel = {k: 0 for k in EXPECTED_ALPHA}
    bucket_tensors = {k: 0 for k in EXPECTED_ALPHA}
    tsa_bucket_numel = {k: 0 for k in EXPECTED_ALPHA}
    for rec in records:
        b = _schedule_bucket(rec)
        bucket_numel[b] += rec.numel
        bucket_tensors[b] += 1
        tsa_bucket_numel[_tsa_bucket(rec)] += rec.numel
    return records, {
        "total_numel": total,
        "bucket_numel": bucket_numel,
        "bucket_tensors": bucket_tensors,
        "tsa_bucket_numel": tsa_bucket_numel,
        "n_layers_seen": len(sorted({r.layer for r in records if r.layer is not None})),
    }


def _tsa_bucket(rec: ParamRecord) -> str:
    # Exact bucket semantics used by the preceding structured-TSA search.
    # In that implementation any parameter carrying a transformer-layer index
    # was in the hidden bucket, including the very small set of 1D layer scales.
    if rec.hidden:
        return "hidden"
    if rec.role in ("token_embed", "value_embed"):
        return "embedding"
    if rec.role == "unembed":
        return "unembed"
    return "rest"


def _schedule_bucket(rec: ParamRecord) -> str:
    # For learning-rate routing, "hidden" means the matrix-valued hidden block
    # that receives the matrix learning rate / Muon updates in the native recipe.
    # 1D layer scales are routed with the rest/scalar group.  This makes the
    # training intervention correspond to actual optimizer tensor groups while
    # leaving the returned structured-TSA estimator exactly frozen.
    if rec.hidden and rec.ndim >= 2:
        return "hidden"
    if rec.role in ("token_embed", "value_embed"):
        return "embedding"
    if rec.role == "unembed":
        return "unembed"
    return "rest"


def _mapped_floors(protocol: Mapping[str, Any]) -> dict[str, float]:
    fmap = protocol["floor_map"]
    lo = float(fmap["floor_min"])
    hi = float(fmap["floor_max"])
    amax = float(fmap["alpha_max"])
    return {
        b: lo + (hi - lo) * float(protocol["frozen_tsa_alpha"][b]) / amax
        for b in EXPECTED_ALPHA
    }


def _matched_uniform_floor(meta: Mapping[str, Any], mapped: Mapping[str, float]) -> float:
    total = float(meta["total_numel"])
    return sum(float(meta["bucket_numel"][b]) * float(mapped[b]) for b in EXPECTED_ALPHA) / total


def _floors_for_arm(arm: str, matched_floor: float, mapped: Mapping[str, float]) -> dict[str, float]:
    if arm == "uniform05":
        return {b: 0.05 for b in EXPECTED_ALPHA}
    if arm == "uniform10":
        return {b: 0.10 for b in EXPECTED_ALPHA}
    if arm == "matched_uniform":
        return {b: float(matched_floor) for b in EXPECTED_ALPHA}
    if arm == "mapped":
        return {b: float(mapped[b]) for b in EXPECTED_ALPHA}
    raise KeyError(arm)


def _snapshot_mean(snapshots: Mapping[int, torch.Tensor]) -> torch.Tensor:
    mean = torch.zeros_like(snapshots[SNAPSHOT_STEPS[-1]], dtype=torch.float32)
    for step in SNAPSHOT_STEPS:
        mean.add_(snapshots[step], alpha=1.0 / K)
    return mean


def _structured_vector(origin: torch.Tensor, mean: torch.Tensor, records: Sequence[ParamRecord], alpha: Mapping[str, float]) -> torch.Tensor:
    delta = mean - origin
    out = origin.clone()
    for rec in records:
        a = float(alpha[_tsa_bucket(rec)])
        if a:
            out[rec.start:rec.end].add_(delta[rec.start:rec.end], alpha=a)
    return out


def _runtime_defaults(args: argparse.Namespace) -> argparse.Namespace:
    return st._runtime_defaults(args)


def _with_floor(runtime: argparse.Namespace, floor: float) -> argparse.Namespace:
    d = dict(vars(runtime))
    d["terminal_floor"] = float(floor)
    return argparse.Namespace(**d)


def _optimizer_group_routing(model, optimizer, records: Sequence[ParamRecord]) -> tuple[bool, list[str | None], list[dict[str, Any]]]:
    id_to_bucket: dict[int, str] = {}
    id_to_name: dict[int, str] = {}
    for (name, p), rec in zip(model.named_parameters(), records):
        id_to_bucket[id(p)] = _schedule_bucket(rec)
        id_to_name[id(p)] = name

    group_bucket: list[str | None] = []
    rows: list[dict[str, Any]] = []
    all_direct = True
    for gi, pg in enumerate(optimizer.param_groups):
        buckets: set[str] = set()
        names: list[str] = []
        numel = 0
        for p in pg.get("params", []):
            if id(p) not in id_to_bucket:
                raise RuntimeError(f"optimizer group {gi} contains a parameter not found in model.named_parameters()")
            buckets.add(id_to_bucket[id(p)])
            names.append(id_to_name[id(p)])
            numel += int(p.numel())
        b = next(iter(buckets)) if len(buckets) == 1 else None
        if b is None:
            all_direct = False
        group_bucket.append(b)
        rows.append({
            "optimizer_group": gi,
            "parameter_count": len(names),
            "numel": numel,
            "buckets": "+".join(sorted(buckets)),
            "direct_bucket": b or "MIXED",
            "example_names": ";".join(names[:8]),
            "lr_at_restore": float(pg.get("lr", math.nan)),
        })
    return all_direct, group_bucket, rows


def _schedule_call(optimizer, step0: int, args, runtime: argparse.Namespace, floor: float) -> float:
    from . import paper_main_d12 as pm
    return float(pm._apply_floor_lr(optimizer, int(step0), args, _with_floor(runtime, float(floor))))


def _set_direct_groupwise_lr(
    optimizer, step0: int, args, runtime: argparse.Namespace,
    floors: Mapping[str, float], group_bucket: Sequence[str | None],
) -> dict[str, float]:
    unique = sorted(set(float(v) for v in floors.values()))
    tables: dict[float, list[float]] = {}
    mults: dict[float, float] = {}
    for floor in unique:
        mults[floor] = _schedule_call(optimizer, step0, args, runtime, floor)
        tables[floor] = [float(pg["lr"]) for pg in optimizer.param_groups]
    for gi, b in enumerate(group_bucket):
        if b is None:
            raise RuntimeError("direct LR routing requested with a mixed optimizer parameter group")
        optimizer.param_groups[gi]["lr"] = tables[float(floors[b])][gi]
    return {b: mults[float(floors[b])] for b in EXPECTED_ALPHA}


def _prepare_poststep_groupwise(
    optimizer, step0: int, args, runtime: argparse.Namespace, floors: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, float]]:
    # Run the real repository scheduler for every requested floor, then leave the
    # optimizer at the minimum floor.  After optimizer.step() we multiply each
    # tensor's realized update by the ratio of the requested scheduler multiplier
    # to the minimum-floor multiplier.  AdamW and Muon keep their momentum/state
    # independently of the scalar LR, so this is the natural per-tensor LR route
    # when an optimizer param_group mixes two tensor roles.
    unique = sorted(set(float(v) for v in floors.values()))
    mults: dict[float, float] = {}
    for floor in unique:
        mults[floor] = _schedule_call(optimizer, step0, args, runtime, floor)
    base_floor = min(unique)
    base_mult = _schedule_call(optimizer, step0, args, runtime, base_floor)
    if abs(base_mult) < 1e-30:
        if max(abs(v) for v in mults.values()) < 1e-30:
            ratios = {b: 1.0 for b in EXPECTED_ALPHA}
        else:
            raise RuntimeError(f"zero base multiplier but nonzero requested multiplier at step {step0+1}: {mults}")
    else:
        ratios = {b: float(mults[float(floors[b])] / base_mult) for b in EXPECTED_ALPHA}
    return {b: mults[float(floors[b])] for b in EXPECTED_ALPHA}, ratios


def _clone_for_postscale(model, records: Sequence[ParamRecord], ratios: Mapping[str, float]):
    saved: list[tuple[torch.nn.Parameter, torch.Tensor, float]] = []
    for (_name, p), rec in zip(model.named_parameters(), records):
        r = float(ratios[_schedule_bucket(rec)])
        if not math.isclose(r, 1.0, rel_tol=0.0, abs_tol=1e-12):
            saved.append((p, p.detach().clone(), r))
    return saved


def _apply_postscale(saved) -> None:
    if not saved:
        return
    with torch.no_grad():
        for p, old, ratio in saved:
            # p <- old + ratio * (p - old), in place with no extra full tensor.
            p.data.sub_(old)
            p.data.mul_(ratio)
            p.data.add_(old)


def _train_trajectory(runtime: argparse.Namespace):
    from .filter_suite import BudgetLedger, OPTIMIZER_OPS_PER_PARAMETER
    from .skip_core import Timer, seed_everything
    from .skip_suite import exact_gradient

    runtime = _runtime_defaults(runtime)
    protocol = _load_protocol(runtime.protocol.resolve())
    pack, restored = st._restore(runtime.pack.resolve(), runtime)
    args, model, optimizer, token_bytes, device, spec, _history, _bank = restored
    actual_optimizer = st._pack_optimizer(pack)
    if not any(alias in actual_optimizer.lower() for alias in ("native", "muon")):
        raise RuntimeError(f"groupwise-floor v1 is native-only, pack optimizer={actual_optimizer}")
    if int(runtime.train_steps) != TRAIN_STEPS:
        raise ValueError(f"suite requires train_steps={TRAIN_STEPS}")

    flat0 = spec.flatten_parameters(dtype=torch.float32)
    records, meta = _parameter_records(model, spec, flat0)
    del flat0
    mapped = _mapped_floors(protocol)
    matched_floor = _matched_uniform_floor(meta, mapped)
    floors = _floors_for_arm(runtime.arm, matched_floor, mapped)
    direct_possible, group_bucket, optimizer_group_rows = _optimizer_group_routing(model, optimizer, records)
    routing_mode = "uniform"
    if runtime.arm == "mapped":
        routing_mode = "direct_param_group" if direct_possible else "poststep_tensor_update"
        if not direct_possible and not bool(runtime.allow_poststep_routing):
            raise RuntimeError(
                "mapped floor requires mixed-group post-step routing, but the predeclared paper run "
                "requires direct optimizer param-group routing. Inspect the smoke optimizer-group map "
                "before deciding whether to rerun with --allow-poststep-routing."
            )

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
    routed_steps = 0

    print(
        f"[setup] arm={runtime.arm} stream_seed={runtime.stream_seed} routing={routing_mode} "
        f"floors={json.dumps(floors, sort_keys=True)} matched_uniform={matched_floor:.9f}",
        flush=True,
    )

    for step in range(1, TRAIN_STEPS + 1):
        block = int(permutation[step - 1])
        batches = stream[block * grad_accum : (block + 1) * grad_accum]

        post_ratios: dict[str, float] | None = None
        if runtime.arm == "mapped" and routing_mode == "direct_param_group":
            mult_by_bucket = _set_direct_groupwise_lr(
                optimizer, step - 1, args, runtime, floors, group_bucket
            )
        elif runtime.arm == "mapped":
            mult_by_bucket, post_ratios = _prepare_poststep_groupwise(
                optimizer, step - 1, args, runtime, floors
            )
        else:
            f = float(next(iter(floors.values())))
            m = _schedule_call(optimizer, step - 1, args, runtime, f)
            mult_by_bucket = {b: m for b in EXPECTED_ALPHA}

        with Timer() as timer:
            _g, train_loss, _features, _seconds = exact_gradient(
                model, optimizer, spec, batches, device,
                need_forward_features=False, token_sample_max=args.token_sample_max,
            )
            saved = []
            if post_ratios is not None:
                saved = _clone_for_postscale(model, records, post_ratios)
                if saved:
                    routed_steps += 1
            optimizer.step()
            if saved:
                _apply_postscale(saved)
            optimizer.zero_grad(set_to_none=True)

        elapsed += float(timer.seconds)
        ledger.add_exact(full_tokens, grad_accum, timer.seconds, charged=True)
        ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
        ledger.optimizer_steps += 1

        if step in SNAPSHOT_STEPS:
            snapshots[step] = spec.flatten_parameters(dtype=torch.float32).detach().to("cpu", dtype=torch.float32)
            print(f"[snapshot] step={step} count={len(snapshots)}/{K}", flush=True)
        if step % 250 == 0:
            row = {
                "step": step,
                "train_loss": float(train_loss),
                "elapsed_training_seconds": elapsed,
                "mult_embedding": float(mult_by_bucket["embedding"]),
                "mult_hidden": float(mult_by_bucket["hidden"]),
                "mult_unembed": float(mult_by_bucket["unembed"]),
                "mult_rest": float(mult_by_bucket["rest"]),
            }
            training_rows.append(row)
            print(
                f"[train] step={step} loss={float(train_loss):.6f} "
                f"mult(embed/hid/unembed/rest)="
                f"{row['mult_embedding']:.6f}/{row['mult_hidden']:.6f}/"
                f"{row['mult_unembed']:.6f}/{row['mult_rest']:.6f}",
                flush=True,
            )

    missing = sorted(set(SNAPSHOT_STEPS) - set(snapshots))
    if missing:
        raise RuntimeError(f"missing snapshots: {missing}")
    restore_gpu = spec.flatten_parameters(dtype=torch.float32)
    return {
        "runtime": runtime,
        "protocol": protocol,
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
        "meta": meta,
        "mapped_floors": mapped,
        "matched_uniform_floor": matched_floor,
        "floors": floors,
        "routing_mode": routing_mode,
        "optimizer_group_rows": optimizer_group_rows,
        "actual_optimizer": actual_optimizer,
        "permutation_fp": permutation_fp,
        "training_rows": training_rows,
        "ledger": ledger,
        "routed_steps": routed_steps,
    }


def run(runtime: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("training must run on a GPU allocation")
    state = _train_trajectory(runtime)
    runtime = state["runtime"]
    pack, model, spec = state["pack"], state["model"], state["spec"]
    token_bytes, device = state["token_bytes"], state["device"]
    snapshots, restore_gpu = state["snapshots"], state["restore_gpu"]
    records = state["records"]

    holdout = pack["eval_batches"][-int(runtime.holdout_eval_batches):]
    if len(holdout) != int(runtime.holdout_eval_batches):
        raise RuntimeError("insufficient holdout batches")
    origin = snapshots[TRAIN_STEPS]
    mean = _snapshot_mean(snapshots)
    scalar055 = origin + 0.55 * (mean - origin)
    structured = _structured_vector(origin, mean, records, EXPECTED_ALPHA)

    candidates = {
        "raw": origin,
        "scalar055": scalar055,
        "structured": structured,
    }
    rows: list[dict[str, Any]] = []
    eval_payload: dict[str, Any] = {}
    for cid, vector in candidates.items():
        ev = st._eval_candidate(model, spec, restore_gpu, vector, holdout, token_bytes, device)
        vals = [float(x) for x in ev.get("bpb_values", [])]
        rows.append({"candidate_id": cid, "bpb": float(ev["bpb"])})
        eval_payload[cid] = {"bpb": float(ev["bpb"]), "bpb_values": vals}
        print(f"[evaluate] {cid} bpb={float(ev['bpb']):.9f}", flush=True)

    out = runtime.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "evaluation.csv", rows)
    _write_csv(out / "training_trace.csv", state["training_rows"])
    _write_csv(out / "optimizer_group_routing.csv", state["optimizer_group_rows"])
    param_rows = [{
        "name": r.name,
        "shape": "x".join(str(x) for x in r.shape),
        "numel": r.numel,
        "layer": r.layer,
        "role": r.role,
        "module": r.module,
        "hidden": int(r.hidden),
        "tsa_bucket": _tsa_bucket(r),
        "schedule_bucket": _schedule_bucket(r),
        "frozen_alpha": EXPECTED_ALPHA[_tsa_bucket(r)],
        "mapped_floor": state["mapped_floors"][_schedule_bucket(r)],
    } for r in records]
    _write_csv(out / "parameter_map.csv", param_rows)

    result = {
        "suite": SUITE,
        "stage": "run",
        "arm": runtime.arm,
        "optimizer_label": "native",
        "actual_pack_optimizer": state["actual_optimizer"],
        "stream_seed": int(runtime.stream_seed),
        "seed": int(runtime.seed),
        "permutation_fingerprint": state["permutation_fp"],
        "protocol_sha256": state["protocol"]["protocol_sha256"],
        "frozen_tsa_alpha": EXPECTED_ALPHA,
        "mapped_floors": state["mapped_floors"],
        "matched_uniform_floor": float(state["matched_uniform_floor"]),
        "arm_floors": state["floors"],
        "routing_mode": state["routing_mode"],
        "routed_steps": int(state["routed_steps"]),
        "snapshot_steps": list(SNAPSHOT_STEPS),
        "holdout_eval_batches": int(runtime.holdout_eval_batches),
        "parameter_meta": state["meta"],
        "evaluations": eval_payload,
        "pack": st._pack_summary(runtime.pack.resolve(), pack),
        "ledger": state["ledger"].as_dict(),
    }
    _json_dump(out / "result.json", result)
    print(f"[result] wrote {out / 'result.json'}", flush=True)
    return out / "result.json"


def _scalar_group_metadata(optimizer) -> list[dict[str, Any]]:
    rows = []
    for gi, pg in enumerate(optimizer.param_groups):
        row = {"group": gi}
        for k, v in pg.items():
            if k in ("params", "lr"):
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                row[k] = v
        rows.append(row)
    return rows


def smoke(runtime: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("smoke must run on a GPU allocation")
    runtime = _runtime_defaults(runtime)
    protocol = _load_protocol(runtime.protocol.resolve())
    pack, restored = st._restore(runtime.native_pack.resolve(), runtime)
    args, model, optimizer, token_bytes, device, spec, _history, _bank = restored
    flat = spec.flatten_parameters(dtype=torch.float32)
    records, meta = _parameter_records(model, spec, flat)
    mapped = _mapped_floors(protocol)
    matched = _matched_uniform_floor(meta, mapped)
    direct, group_bucket, group_rows = _optimizer_group_routing(model, optimizer, records)

    # Repeated scheduler calls are the direct-routing mechanism. Prime the
    # scheduler once (in case it lazily adds bookkeeping fields), then verify
    # subsequent calls are idempotent and do not mutate scalar group metadata
    # other than lr.
    test_step = 2850
    _schedule_call(optimizer, test_step, args, runtime, 0.05)
    metadata_before = _scalar_group_metadata(optimizer)
    m1 = _schedule_call(optimizer, test_step, args, runtime, 0.05)
    lr1 = [float(pg["lr"]) for pg in optimizer.param_groups]
    _schedule_call(optimizer, test_step, args, runtime, 0.15)
    m2 = _schedule_call(optimizer, test_step, args, runtime, 0.05)
    lr2 = [float(pg["lr"]) for pg in optimizer.param_groups]
    metadata_after = _scalar_group_metadata(optimizer)
    if not math.isclose(m1, m2, rel_tol=0.0, abs_tol=1e-15) or any(
        not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-15) for a, b in zip(lr1, lr2)
    ):
        raise RuntimeError("paper_main_d12 floor scheduler is not idempotent under repeated calls")
    if metadata_before != metadata_after:
        raise RuntimeError("paper_main_d12 floor scheduler mutated optimizer metadata other than lr")

    schedule_rows: list[dict[str, Any]] = []
    for step0 in (0, 1000, 2500, 2750, 2850, 2999):
        row: dict[str, Any] = {"optimizer_step": step0 + 1}
        for label, floor in (
            ("floor05", 0.05),
            ("floor10", 0.10),
            ("matched", matched),
            ("embed", mapped["embedding"]),
            ("hidden", mapped["hidden"]),
            ("unembed", mapped["unembed"]),
            ("rest", mapped["rest"]),
        ):
            row[label] = _schedule_call(optimizer, step0, args, runtime, float(floor))
        schedule_rows.append(row)

    ev = st._eval_candidate(model, spec, flat, flat.detach().to("cpu"), pack["eval_batches"][:1], token_bytes, device)
    out = runtime.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    param_rows = [{
        "name": r.name,
        "shape": "x".join(str(x) for x in r.shape),
        "numel": r.numel,
        "layer": r.layer,
        "role": r.role,
        "module": r.module,
        "hidden": int(r.hidden),
        "tsa_bucket": _tsa_bucket(r),
        "schedule_bucket": _schedule_bucket(r),
        "alpha": EXPECTED_ALPHA[_tsa_bucket(r)],
        "mapped_floor": mapped[_schedule_bucket(r)],
    } for r in records]
    _write_csv(out / "SMOKE_PARAMETER_MAP.csv", param_rows)
    _write_csv(out / "SMOKE_OPTIMIZER_GROUPS.csv", group_rows)
    _write_csv(out / "SMOKE_SCHEDULE.csv", schedule_rows)

    lines = [
        f"suite={SUITE}",
        f"python={os.sys.executable}",
        f"torch={torch.__version__}",
        f"native_pack={runtime.native_pack.resolve()}",
        f"protocol_sha256={protocol['protocol_sha256']}",
        f"total_numel={meta['total_numel']}",
        f"bucket_numel={json.dumps(meta['bucket_numel'], sort_keys=True)}",
        f"mapped_floors={json.dumps(mapped, sort_keys=True)}",
        f"matched_uniform_floor={matched:.9f}",
        f"optimizer_param_groups_directly_routable={int(direct)}",
        f"mapped_routing_mode={'direct_param_group' if direct else 'poststep_tensor_update'}",
        f"one_batch_bpb={float(ev['bpb'])}",
        ("GROUPWISE FLOOR D12 SMOKE PASS" if direct or bool(runtime.allow_poststep_routing)
         else "GROUPWISE FLOOR D12 SMOKE BLOCKED: direct routing unavailable"),
    ]
    (out / "SMOKE.txt").write_text("\n".join(lines) + "\n")
    print((out / "SMOKE.txt").read_text(), flush=True)
    if not direct and not bool(runtime.allow_poststep_routing):
        raise RuntimeError(
            "optimizer parameter groups mix schedule buckets; direct LR routing unavailable. "
            "See SMOKE_OPTIMIZER_GROUPS.csv. The training array is intentionally blocked."
        )
    return out / "SMOKE.txt"


def selftest() -> None:
    fake = {
        "suite": SUITE,
        "frozen_tsa_alpha": EXPECTED_ALPHA,
        "floor_map": {
            "floor_min": EXPECTED_FLOOR_MIN,
            "floor_max": EXPECTED_FLOOR_MAX,
            "alpha_max": EXPECTED_ALPHA_MAX,
        },
    }
    mapped = {
        b: EXPECTED_FLOOR_MIN + (EXPECTED_FLOOR_MAX - EXPECTED_FLOOR_MIN) * a / EXPECTED_ALPHA_MAX
        for b, a in EXPECTED_ALPHA.items()
    }
    assert math.isclose(mapped["embedding"], 0.05)
    assert math.isclose(mapped["hidden"], 0.15)
    assert math.isclose(mapped["unembed"], 0.11923076923076924)
    assert math.isclose(mapped["rest"], 0.08846153846153847)
    fake["mapped"] = mapped
    assert len(_canonical_hash(fake)) == 64
    print("groupwise_floor_d12 selftest PASS")


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history-device", default=None)
    parser.add_argument("--history-dtype", default=None)
    parser.add_argument("--lr-scale", type=float, default=0.40)
    parser.add_argument("--warmdown-ratio", type=float, default=None)
    parser.add_argument("--final-lr-frac", type=float, default=None)
    parser.add_argument("--terminal-floor", type=float, default=0.05)
    parser.add_argument("--train-steps", type=int, default=TRAIN_STEPS)
    parser.add_argument("--schedule-total-iterations", type=int, default=TRAIN_STEPS)
    parser.add_argument("--snapshot-dtype", choices=["bfloat16", "float16", "float32"], default="float32")
    parser.add_argument("--curve-eval-batches", type=int, default=32)
    parser.add_argument("--alpha-select-batches", type=int, default=0)
    parser.add_argument("--floor-select-batches", type=int, default=0)
    parser.add_argument("--holdout-eval-batches", type=int, default=HOLDOUT_BATCHES)
    parser.add_argument("--token-sample-max", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--allow-poststep-routing", action="store_true",
                        help="allow the mixed-param-group post-step LR fallback; off by default for the paper protocol")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")

    sp = sub.add_parser("smoke")
    sp.add_argument("--native-pack", type=Path, required=True)
    sp.add_argument("--protocol", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    _add_runtime_args(sp)

    rp = sub.add_parser("run")
    rp.add_argument("--pack", type=Path, required=True)
    rp.add_argument("--protocol", type=Path, required=True)
    rp.add_argument("--out", type=Path, required=True)
    rp.add_argument("--arm", choices=EXPECTED_ARMS, required=True)
    rp.add_argument("--stream-seed", type=int, required=True)
    _add_runtime_args(rp)
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "run":
        run(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
