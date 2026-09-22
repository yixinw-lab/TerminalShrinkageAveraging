#!/usr/bin/env python3
"""Development and confirmation suite for structured Terminal Shrinkage Averaging.

The suite trains fresh depth-12 trajectories under the native Muon+AdamW and
pure-AdamW packs already used by the project.  The terminal estimator uses the
same K=8, spacing=32 checkpoint window as the main paper, but assigns separate
shrinkage strengths to the coordinates that are Muon-managed in the native
optimizer and to their complement.

The experiment is intentionally split into two stages:

* development trajectories evaluate a predeclared two-dimensional alpha grid on
  the calibration validation block;
* confirmation trajectories save their terminal state without evaluating the
  grid.  A separate freeze job selects one structured rule and one scalar rule
  using only development outputs.  Only then are confirmation trajectories
  evaluated on the untouched holdout block.

This module depends on the existing ExtrapProj/NanoChat helper modules and does
not replace the optimizer or model implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import os
import tempfile
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

SUITE = "d12_structured_tsa_suite_v1"
K = 8
SPACING = 32
SNAPSHOT_STRIDE = 32
SNAPSHOT_PHASE = 3000 % SNAPSHOT_STRIDE  # 24
STRUCTURED_GRID = tuple(round(x, 6) for x in np.linspace(0.0, 1.0, 9))
SCALAR_GRID = tuple(round(x, 6) for x in np.linspace(0.0, 1.0, 21))


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (list, tuple, dict)):
                continue
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def _fingerprint_ints(values: Sequence[int]) -> str:
    arr = np.asarray(values, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return _sha256_bytes(payload)


def _slug_floor(value: float) -> str:
    return f"{int(round(100 * value)):02d}pct"


def _runtime_defaults(runtime: argparse.Namespace) -> argparse.Namespace:
    from . import paper_main_d12 as pm

    if getattr(runtime, "warmdown_ratio", None) is None:
        runtime.warmdown_ratio = pm.BASE_WARMDOWN_RATIO
    if getattr(runtime, "final_lr_frac", None) is None:
        runtime.final_lr_frac = pm.BASE_FINAL_FRAC
    return runtime


def _pack_optimizer(pack: Mapping[str, Any]) -> str:
    candidates = [
        pack.get("optimizer"),
        pack.get("build_args", {}).get("optimizer") if isinstance(pack.get("build_args"), Mapping) else None,
        pack.get("args", {}).get("optimizer") if isinstance(pack.get("args"), Mapping) else None,
    ]
    for value in candidates:
        if value is not None:
            return str(value)
    return "unknown"



def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return {"tensor_shape": list(value.shape), "tensor_dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)

def _pack_summary(path: Path, pack: Mapping[str, Any]) -> dict[str, Any]:
    build_args = pack.get("build_args", {})
    if not isinstance(build_args, Mapping):
        build_args = {"repr": repr(build_args)}
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "optimizer": _pack_optimizer(pack),
        "pack_version": pack.get("pack_version"),
        "prefix_steps": pack.get("prefix_steps"),
        "num_branch_batches": len(pack.get("branch_batches", [])),
        "num_eval_batches": len(pack.get("eval_batches", [])),
        "build_args": _json_safe(dict(build_args)),
    }


def _sample_tensor_fingerprint(value: Any, max_values: int = 4096) -> str:
    """Cheap, deterministic fingerprint for comparing pack data provenance."""
    h = hashlib.sha256()

    def update(x: Any) -> None:
        if torch.is_tensor(x):
            t = x.detach().to("cpu").reshape(-1)
            h.update(str(tuple(x.shape)).encode())
            h.update(str(x.dtype).encode())
            if t.numel() > max_values:
                idx = torch.linspace(0, t.numel() - 1, max_values, dtype=torch.long)
                t = t[idx]
            if t.is_floating_point():
                t = t.to(torch.float32)
            elif t.dtype == torch.bool:
                t = t.to(torch.uint8)
            else:
                t = t.to(torch.int64)
            h.update(t.contiguous().numpy().tobytes())
        elif isinstance(x, np.ndarray):
            a = np.asarray(x).reshape(-1)
            h.update(str(x.shape).encode())
            h.update(str(x.dtype).encode())
            if a.size > max_values:
                idx = np.linspace(0, a.size - 1, max_values).astype(np.int64)
                a = a[idx]
            h.update(np.ascontiguousarray(a).tobytes())
        elif isinstance(x, Mapping):
            for k in sorted(x, key=str):
                h.update(str(k).encode())
                update(x[k])
        elif isinstance(x, (list, tuple)):
            for y in x[:4]:
                update(y)
            if len(x) > 4:
                for y in x[-4:]:
                    update(y)
        else:
            h.update(repr(x).encode())

    update(value)
    return h.hexdigest()


def _restore(pack_path: Path, runtime: argparse.Namespace):
    from .filter_suite import _install_canonical_build, _stabilize_nanochat_adamw_kernel
    from .skip_suite import PACK_VERSION, restore_branch

    _install_canonical_build()
    _stabilize_nanochat_adamw_kernel()
    runtime = _runtime_defaults(runtime)
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    if int(pack.get("pack_version", -1)) != PACK_VERSION:
        raise RuntimeError(
            f"pack version mismatch for {pack_path}: got {pack.get('pack_version')}, expected {PACK_VERSION}"
        )
    if int(pack.get("prefix_steps", 0)) != 0:
        raise RuntimeError(f"expected a prefix_steps=0 pack, got {pack.get('prefix_steps')} from {pack_path}")
    restored = restore_branch(pack, runtime)
    return pack, restored


def _as_mapping(info: Any) -> dict[str, Any]:
    if isinstance(info, Mapping):
        return dict(info)
    # Prefer the shallow instance dictionary. dataclasses.asdict recursively copies
    # values and can be expensive if a helper dataclass ever stores tensors.
    if hasattr(info, "__dict__"):
        try:
            return dict(vars(info))
        except Exception:
            pass
    if is_dataclass(info):
        try:
            return {field.name: getattr(info, field.name) for field in info.__dataclass_fields__.values()}
        except Exception:
            pass
    return {"repr": repr(info)}


def _extract_slice(mapping: Mapping[str, Any]) -> tuple[int, int] | None:
    # Direct slice-like fields.
    for key in ("slice", "sl", "flat_slice", "param_slice", "vector_slice"):
        value = mapping.get(key)
        if isinstance(value, slice) and value.start is not None and value.stop is not None:
            return int(value.start), int(value.stop)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            try:
                return int(value[0]), int(value[1])
            except Exception:
                pass

    pairs = (
        ("start", "stop"),
        ("start", "end"),
        ("flat_start", "flat_stop"),
        ("flat_start", "flat_end"),
        ("offset", "stop"),
        ("lo", "hi"),
        ("begin", "end"),
    )
    for left, right in pairs:
        if left in mapping and right in mapping:
            try:
                a, b = int(mapping[left]), int(mapping[right])
                if b > a:
                    return a, b
            except Exception:
                pass

    for offset_key in ("offset", "start", "flat_start", "begin"):
        for count_key in ("numel", "length", "size", "n"):
            if offset_key in mapping and count_key in mapping:
                try:
                    a = int(mapping[offset_key])
                    n = int(mapping[count_key])
                    if n > 0:
                        return a, a + n
                except Exception:
                    pass
    return None


def _mapping_mentions_muon(mapping: Mapping[str, Any]) -> bool:
    positive = False
    negative = False
    for key, value in mapping.items():
        key_l = str(key).lower()
        if isinstance(value, bool):
            if "muon" in key_l and value:
                positive = True
            if "adam" in key_l and value:
                negative = True
        if isinstance(value, str):
            text = value.lower()
            if "muon" in text:
                positive = True
            if "adam" in text and "muon" not in text:
                negative = True
        elif key_l in {"optimizer", "optimizer_kind", "optimizer_name", "group", "kind", "family"}:
            text = str(value).lower()
            if "muon" in text:
                positive = True
            if "adam" in text and "muon" not in text:
                negative = True
    return positive and not negative


def _normalize_ranges(ranges: Iterable[tuple[int, int]], total: int) -> list[list[int]]:
    clean = sorted((int(a), int(b)) for a, b in ranges if int(b) > int(a))
    merged: list[list[int]] = []
    for a, b in clean:
        if a < 0 or b > total:
            raise ValueError(f"group range [{a},{b}) is outside [0,{total})")
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return merged


def _ranges_from_infos(infos: Sequence[Any], total: int) -> tuple[list[list[int]], list[dict[str, Any]]]:
    ranges: list[tuple[int, int]] = []
    records: list[dict[str, Any]] = []
    for idx, info in enumerate(infos):
        mapping = _as_mapping(info)
        sl = _extract_slice(mapping)
        is_muon = _mapping_mentions_muon(mapping)
        records.append(
            {
                "index": idx,
                "is_muon": is_muon,
                "slice": list(sl) if sl else None,
                "mapping_repr": repr(mapping)[:2000],
            }
        )
        if is_muon and sl is not None:
            ranges.append(sl)
    return _normalize_ranges(ranges, total), records


def _call_fixed_group_candidate(
    current_gpu: torch.Tensor,
    mean_cpu: torch.Tensor,
    infos: Sequence[Any],
    n_layers: int,
):
    from .trajectory_groups import make_averaged_candidate

    signature = inspect.signature(make_averaged_candidate)
    proposed = {
        "selector": "muon",
        "n_layers": n_layers,
        "alpha": 1.0,
        "strength": 1.0,
        "adaptive_rule": None,
        "adaptive_strength": 1.0,
        "stats_rows": None,
        "adaptive_selector": "all",
    }
    kwargs: dict[str, Any] = {}
    for name, value in proposed.items():
        if name in signature.parameters:
            parameter = signature.parameters[name]
            # Do not override optional adaptive defaults with None unless required.
            if name == "adaptive_rule" and parameter.default is not inspect._empty:
                continue
            kwargs[name] = value
    out = make_averaged_candidate(current_gpu, mean_cpu, infos, **kwargs)
    candidate = out[0] if isinstance(out, tuple) else out
    if not torch.is_tensor(candidate):
        raise TypeError(f"make_averaged_candidate returned {type(candidate)!r}, expected a tensor")
    return candidate.detach().to("cpu", dtype=torch.float32), str(signature), kwargs


def _ranges_from_candidate(
    current_gpu: torch.Tensor,
    current_cpu: torch.Tensor,
    infos: Sequence[Any],
    n_layers: int,
) -> tuple[list[list[int]], dict[str, Any]]:
    # A constant displacement makes a hard selector easy to recognize exactly.
    displacement = 0.125
    synthetic_mean = current_cpu + displacement
    candidate, signature, kwargs = _call_fixed_group_candidate(
        current_gpu=current_gpu,
        mean_cpu=synthetic_mean,
        infos=infos,
        n_layers=n_layers,
    )
    delta = candidate - current_cpu
    selected = delta.abs() > displacement * 0.5
    selected_values = delta[selected]
    unselected_values = delta[~selected]
    if selected_values.numel() == 0 or unselected_values.numel() == 0:
        raise RuntimeError(
            "the fixed Muon selector selected zero or all coordinates; cannot define a two-group partition"
        )
    max_selected_error = float((selected_values - displacement).abs().max().item())
    max_unselected_error = float(unselected_values.abs().max().item())
    if max_selected_error > 2e-4 or max_unselected_error > 2e-4:
        raise RuntimeError(
            "selector candidate was not a hard 0/1 coordinate partition: "
            f"selected_error={max_selected_error:.3e}, unselected_error={max_unselected_error:.3e}"
        )

    # Find run boundaries without materializing all selected indices.
    mask = selected.to(dtype=torch.int8)
    transitions = torch.nonzero(mask[1:] != mask[:-1], as_tuple=False).flatten().tolist()
    boundaries = [0] + [int(x) + 1 for x in transitions] + [int(mask.numel())]
    ranges: list[tuple[int, int]] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if bool(selected[left].item()):
            ranges.append((left, right))
    meta = {
        "make_averaged_candidate_signature": signature,
        "make_averaged_candidate_kwargs": kwargs,
        "max_selected_error": max_selected_error,
        "max_unselected_error": max_unselected_error,
    }
    return _normalize_ranges(ranges, int(current_cpu.numel())), meta


def _range_numel(ranges: Sequence[Sequence[int]]) -> int:
    return int(sum(int(b) - int(a) for a, b in ranges))


def _apply_group_split(
    current: torch.Tensor,
    mean: torch.Tensor,
    ranges: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if current.shape != mean.shape:
        raise ValueError(f"current/mean shape mismatch: {tuple(current.shape)} vs {tuple(mean.shape)}")
    delta_all = mean.to(dtype=torch.float32) - current.to(dtype=torch.float32)
    delta_group = torch.zeros_like(delta_all)
    for left, right in ranges:
        delta_group[int(left):int(right)].copy_(delta_all[int(left):int(right)])
    delta_other = delta_all - delta_group
    return delta_group, delta_other


def _candidate(current: torch.Tensor, delta_group: torch.Tensor, delta_other: torch.Tensor, a_group: float, a_other: float) -> torch.Tensor:
    return current + float(a_group) * delta_group + float(a_other) * delta_other


def _eval_candidate(model, spec, restore_gpu, candidate_cpu, eval_batches, token_bytes, device):
    from .filter_suite import evaluate_detailed
    from .skip_core import Timer

    try:
        spec.assign_parameters(candidate_cpu)
        with Timer() as timer:
            ev = evaluate_detailed(model, eval_batches, token_bytes, device)
    finally:
        spec.assign_parameters(restore_gpu)
    return {**ev, "eval_seconds": timer.seconds}


def _paired_delta(method_vals: Sequence[float], raw_vals: Sequence[float]) -> tuple[float, float]:
    a = np.asarray(method_vals, dtype=float)
    b = np.asarray(raw_vals, dtype=float)
    d = a - b
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0
    return float(d.mean()), se


def _eval_split(pack: Mapping[str, Any], runtime: argparse.Namespace, kind: str):
    all_eval = pack["eval_batches"]
    curve = int(runtime.curve_eval_batches)
    alpha = int(runtime.alpha_select_batches)
    floor = int(runtime.floor_select_batches)
    holdout = int(runtime.holdout_eval_batches)
    if kind == "calibration":
        start, count = curve, alpha
    elif kind == "holdout":
        start, count = curve + alpha + floor, holdout
    else:
        raise ValueError(kind)
    end = start + count
    if end > len(all_eval):
        raise RuntimeError(f"need eval batches through {end}, pack contains {len(all_eval)}")
    return all_eval[start:end], {"kind": kind, "start": start, "count": count, "end": end}


def smoke(runtime: argparse.Namespace) -> Path:
    from .trajectory_groups import build_slice_infos

    if not torch.cuda.is_available():
        raise RuntimeError("smoke must run on a DeltaAI GPU allocation")

    out = runtime.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    probe_lines: list[str] = [f"suite={SUITE}", f"torch={torch.__version__}"]

    native_pack, native_restored = _restore(runtime.native_pack.resolve(), runtime)
    n_args, n_model, n_optimizer, _n_token_bytes, _n_device, n_spec, _h, _b = native_restored
    current_gpu = n_spec.flatten_parameters(dtype=torch.float32)
    current_cpu = current_gpu.detach().to("cpu", dtype=torch.float32)
    infos = build_slice_infos(n_spec, n_optimizer)
    n_layers = int(getattr(n_model.config, "n_layer", getattr(n_model.config, "depth", 0)))

    info_ranges, info_records = _ranges_from_infos(infos, int(n_spec.numel))
    candidate_error = None
    candidate_meta: dict[str, Any] = {}
    candidate_ranges: list[list[int]] = []
    try:
        candidate_ranges, candidate_meta = _ranges_from_candidate(
            current_gpu=current_gpu,
            current_cpu=current_cpu,
            infos=infos,
            n_layers=n_layers,
        )
    except Exception as exc:  # preserve the diagnostic and use info parsing if possible
        candidate_error = f"{type(exc).__name__}: {exc}"

    if candidate_ranges:
        ranges = candidate_ranges
        source = "trajectory_groups.make_averaged_candidate(selector=muon)"
        if info_ranges and info_ranges != candidate_ranges:
            probe_lines.append(
                "WARNING: info-derived and candidate-derived ranges differ; candidate-derived ranges are authoritative"
            )
    elif info_ranges:
        ranges = info_ranges
        source = "trajectory_groups.build_slice_infos metadata"
    else:
        _json_dump(out / "GROUP_PROBE_infos.json", info_records)
        raise RuntimeError(
            "could not recover native Muon-managed coordinate ranges. "
            f"candidate probe error: {candidate_error}. Inspect {out/'GROUP_PROBE_infos.json'}"
        )

    group_numel = _range_numel(ranges)
    total_numel = int(n_spec.numel)
    if not (0 < group_numel < total_numel):
        raise RuntimeError(f"invalid group partition: selected {group_numel} of {total_numel} coordinates")

    native_parameter_fp = _sample_tensor_fingerprint(current_cpu)
    native_branch_fp = _sample_tensor_fingerprint(
        native_pack.get("branch_batches", [])[:2] + native_pack.get("branch_batches", [])[-2:]
    )
    native_eval_fp = _sample_tensor_fingerprint(
        native_pack.get("eval_batches", [])[:2] + native_pack.get("eval_batches", [])[-2:]
    )

    # Release the first restored model before placing the pure-AdamW model on the GPU.
    del current_gpu, current_cpu, infos, n_model, n_optimizer, n_spec, native_restored
    torch.cuda.empty_cache()

    adam_pack, adam_restored = _restore(runtime.adamw_pack.resolve(), runtime)
    a_args, a_model, _a_optimizer, _a_token_bytes, _a_device, a_spec, _ah, _ab = adam_restored
    if int(a_spec.numel) != total_numel:
        raise RuntimeError(
            f"native and pure-AdamW packs have different flattened sizes: {total_numel} vs {a_spec.numel}"
        )
    a_layers = int(getattr(a_model.config, "n_layer", getattr(a_model.config, "depth", 0)))
    if a_layers != n_layers:
        raise RuntimeError(f"native and pure-AdamW packs have different depth: {n_layers} vs {a_layers}")
    adam_current_gpu = a_spec.flatten_parameters(dtype=torch.float32)
    adam_current_cpu = adam_current_gpu.detach().to("cpu", dtype=torch.float32)
    adam_parameter_fp = _sample_tensor_fingerprint(adam_current_cpu)
    adam_branch_fp = _sample_tensor_fingerprint(
        adam_pack.get("branch_batches", [])[:2] + adam_pack.get("branch_batches", [])[-2:]
    )
    adam_eval_fp = _sample_tensor_fingerprint(
        adam_pack.get("eval_batches", [])[:2] + adam_pack.get("eval_batches", [])[-2:]
    )
    parameter_match = native_parameter_fp == adam_parameter_fp
    branch_match = native_branch_fp == adam_branch_fp
    eval_match = native_eval_fp == adam_eval_fp
    if not parameter_match:
        raise RuntimeError(
            "native and pure-AdamW packs do not share the same sampled initial-parameter fingerprint; "
            "rebuild or locate matched packs before running the optimizer comparison"
        )
    if not branch_match or not eval_match:
        raise RuntimeError(
            "native and pure-AdamW packs do not share matching sampled train/evaluation data fingerprints; "
            "rebuild or locate matched packs before running the paired optimizer comparison"
        )

    schema_without_hash = {
        "suite": SUITE,
        "total_numel": total_numel,
        "n_layers": n_layers,
        "group_definition": (
            "coordinates assigned to Muon by the native NanoChat optimizer; the identical flattened-coordinate "
            "partition is reused for pure-AdamW runs so that the optimizer comparison preserves tensor roles"
        ),
        "group_a": {
            "id": "native_muon_managed",
            "display_name_native": "Muon-managed",
            "display_name_pure_adamw": "Muon-eligible matrix group (trained with AdamW)",
            "numel": group_numel,
            "fraction": group_numel / total_numel,
            "ranges": ranges,
        },
        "group_b": {
            "id": "native_adamw_managed_complement",
            "display_name_native": "AdamW-managed",
            "display_name_pure_adamw": "remaining parameter group (trained with AdamW)",
            "numel": total_numel - group_numel,
            "fraction": (total_numel - group_numel) / total_numel,
        },
        "range_source": source,
        "candidate_probe": candidate_meta,
        "candidate_probe_error": candidate_error,
        "native_pack": _pack_summary(runtime.native_pack.resolve(), native_pack),
        "pure_adamw_pack": _pack_summary(runtime.adamw_pack.resolve(), adam_pack),
        "sampled_initial_parameter_fingerprints": {
            "native": native_parameter_fp,
            "pure_adamw": adam_parameter_fp,
            "match": parameter_match,
        },
        "sampled_data_fingerprints": {
            "native_branch": native_branch_fp,
            "pure_adamw_branch": adam_branch_fp,
            "branch_match": branch_match,
            "native_eval": native_eval_fp,
            "pure_adamw_eval": adam_eval_fp,
            "eval_match": eval_match,
        },
    }
    schema = dict(schema_without_hash)
    schema["schema_sha256"] = _canonical_hash(schema_without_hash)
    _json_dump(out / "GROUP_SCHEMA.json", schema)
    _json_dump(out / "GROUP_PROBE_infos.json", info_records)

    probe_lines.extend(
        [
            f"native_optimizer={_pack_optimizer(native_pack)}",
            f"pure_adamw_optimizer={_pack_optimizer(adam_pack)}",
            f"total_numel={total_numel}",
            f"group_a_numel={group_numel}",
            f"group_a_fraction={group_numel/total_numel:.8f}",
            f"range_count={len(ranges)}",
            f"range_source={source}",
            f"initial_parameter_sample_fingerprint_match={parameter_match}",
            f"branch_sample_fingerprint_match={branch_match}",
            f"eval_sample_fingerprint_match={eval_match}",
            f"schema_sha256={schema['schema_sha256']}",
            "SMOKE PASS",
        ]
    )
    (out / "SMOKE.txt").write_text("\n".join(probe_lines) + "\n")
    print((out / "SMOKE.txt").read_text(), flush=True)
    return out / "GROUP_SCHEMA.json"


def _load_schema(path: Path) -> dict[str, Any]:
    schema = json.loads(path.read_text())
    expected = schema.pop("schema_sha256", None)
    observed = _canonical_hash(schema)
    schema["schema_sha256"] = expected
    if expected != observed:
        raise RuntimeError(f"group schema hash mismatch: expected {expected}, observed {observed}")
    return schema


def _save_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.save(dict(payload), tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def train(runtime: argparse.Namespace) -> Path:
    from . import paper_main_d12 as pm
    from .filter_suite import BudgetLedger, OPTIMIZER_OPS_PER_PARAMETER
    from .skip_core import Timer, seed_everything
    from .skip_suite import exact_gradient

    if not torch.cuda.is_available():
        raise RuntimeError("training must run on a GPU allocation")
    runtime = _runtime_defaults(runtime)
    schema = _load_schema(runtime.group_schema.resolve())
    ranges = schema["group_a"]["ranges"]

    pack, restored = _restore(runtime.pack.resolve(), runtime)
    args, model, optimizer, token_bytes, device, spec, _history, _bank = restored
    actual_optimizer = _pack_optimizer(pack)
    if runtime.optimizer_label not in actual_optimizer and actual_optimizer not in runtime.optimizer_label:
        aliases = {"native": {"native", "muon", "muon_adamw"}, "pure_adamw": {"pure_adamw", "adamw", "adam"}}
        if not any(alias in actual_optimizer.lower() for alias in aliases.get(runtime.optimizer_label, set())):
            raise RuntimeError(
                f"manifest optimizer {runtime.optimizer_label!r} does not match pack optimizer {actual_optimizer!r}"
            )
    if int(spec.numel) != int(schema["total_numel"]):
        raise RuntimeError(f"pack numel {spec.numel} does not match group schema {schema['total_numel']}")

    seed_everything(int(runtime.seed))
    calibration, calibration_meta = _eval_split(pack, runtime, "calibration")

    if int(runtime.train_steps) % SNAPSHOT_STRIDE != SNAPSHOT_PHASE:
        raise ValueError(
            f"train_steps={runtime.train_steps} must be congruent to {SNAPSHOT_PHASE} modulo {SNAPSHOT_STRIDE}"
        )
    bank = pm.AlignedSnapshotBank(
        stride=SNAPSHOT_STRIDE,
        phase=SNAPSHOT_PHASE,
        max_span=(K - 1) * SPACING,
        dtype=pm._dtype(runtime.snapshot_dtype),
    )

    stream = pack["branch_batches"]
    grad_accum = int(args.grad_accum)
    stream_steps = len(stream) // grad_accum
    if stream_steps < int(runtime.train_steps):
        raise RuntimeError(f"pack has {stream_steps} optimizer-step blocks, need {runtime.train_steps}")

    rng = np.random.default_rng(int(runtime.stream_seed))
    permutation = rng.permutation(int(runtime.train_steps)).astype(np.int64)
    permutation_fingerprint = _fingerprint_ints(permutation.tolist())

    micro_tokens = int(args.device_batch_size) * int(args.max_seq_len)
    full_tokens = micro_tokens * grad_accum
    ledger = BudgetLedger(spec.numel, full_tokens)
    training_rows: list[dict[str, Any]] = []
    elapsed_training_seconds = 0.0

    print(
        f"[setup] role={runtime.role} optimizer={runtime.optimizer_label} floor={runtime.terminal_floor:.3f} "
        f"stream_seed={runtime.stream_seed} T={runtime.train_steps}",
        flush=True,
    )

    for step in range(1, int(runtime.train_steps) + 1):
        block_index = int(permutation[step - 1])
        lo = block_index * grad_accum
        batches = stream[lo : lo + grad_accum]
        lr_mult = pm._apply_floor_lr(optimizer, step - 1, args, runtime)
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
        elapsed_training_seconds += float(train_timer.seconds)
        ledger.add_exact(full_tokens, grad_accum, train_timer.seconds, charged=True)
        ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
        ledger.optimizer_steps += 1
        if bank.should_store(step):
            bank.append(step, spec.flatten_parameters(dtype=torch.float32))
        if step % 250 == 0 or step == int(runtime.train_steps):
            row = {
                "step": step,
                "train_loss": float(train_loss),
                "lr_multiplier": float(lr_mult),
                "elapsed_training_seconds": elapsed_training_seconds,
            }
            training_rows.append(row)
            print(
                f"[train] step={step} loss={float(train_loss):.6f} lr={float(lr_mult):.5f} "
                f"bank_gib={bank.memory_bytes()/2**30:.2f}",
                flush=True,
            )

    current_gpu = spec.flatten_parameters(dtype=torch.float32)
    current_cpu = current_gpu.detach().to("cpu", dtype=torch.float32)
    selected = bank.select(K, SPACING, end_step=int(runtime.train_steps))
    mean_cpu = pm.uniform_average(selected).detach().to("cpu", dtype=torch.float32)
    selected_steps = [int(step) for step, _ in selected]

    out = runtime.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / "terminal_artifact.pt"
    _save_artifact(
        artifact_path,
        {
            "suite": SUITE,
            "role": runtime.role,
            "optimizer_label": runtime.optimizer_label,
            "actual_pack_optimizer": actual_optimizer,
            "terminal_floor": float(runtime.terminal_floor),
            "stream_seed": int(runtime.stream_seed),
            "train_steps": int(runtime.train_steps),
            "schedule_total_iterations": int(runtime.schedule_total_iterations),
            "group_schema_sha256": schema["schema_sha256"],
            "current": current_cpu,
            "mean": mean_cpu,
            "selected_steps": selected_steps,
        },
    )

    calibration_rows: list[dict[str, Any]] = []
    calibration_details: dict[str, Any] = {}
    if runtime.role == "development":
        delta_group, delta_other = _apply_group_split(current_cpu, mean_cpu, ranges)
        pair_cache: dict[tuple[float, float], dict[str, Any]] = {}

        def evaluate_pair(a_group: float, a_other: float) -> dict[str, Any]:
            key = (round(float(a_group), 6), round(float(a_other), 6))
            if key not in pair_cache:
                cand = _candidate(current_cpu, delta_group, delta_other, *key)
                pair_cache[key] = _eval_candidate(
                    model, spec, current_gpu, cand, calibration, token_bytes, device
                )
                del cand
            return pair_cache[key]

        raw = evaluate_pair(0.0, 0.0)
        for a_group in STRUCTURED_GRID:
            for a_other in STRUCTURED_GRID:
                ev = evaluate_pair(a_group, a_other)
                d, se = _paired_delta(ev["bpb_values"], raw["bpb_values"])
                calibration_rows.append(
                    {
                        "family": "structured_grid",
                        "recipe": f"structured:g{a_group:.3f}:o{a_other:.3f}",
                        "alpha_group": a_group,
                        "alpha_other": a_other,
                        "bpb": float(ev["bpb"]),
                        "method_minus_raw": d,
                        "paired_eval_se": se,
                        "eval_seconds": float(ev["eval_seconds"]),
                        "optimizer": runtime.optimizer_label,
                        "floor": float(runtime.terminal_floor),
                        "stream_seed": int(runtime.stream_seed),
                    }
                )
        for alpha in SCALAR_GRID:
            ev = evaluate_pair(alpha, alpha)
            d, se = _paired_delta(ev["bpb_values"], raw["bpb_values"])
            calibration_rows.append(
                {
                    "family": "scalar_grid",
                    "recipe": f"scalar:a{alpha:.3f}",
                    "alpha_group": alpha,
                    "alpha_other": alpha,
                    "bpb": float(ev["bpb"]),
                    "method_minus_raw": d,
                    "paired_eval_se": se,
                    "eval_seconds": float(ev["eval_seconds"]),
                    "optimizer": runtime.optimizer_label,
                    "floor": float(runtime.terminal_floor),
                    "stream_seed": int(runtime.stream_seed),
                }
            )
        # Keep per-batch values only once per unique pair in JSON, not in CSV.
        calibration_details = {
            f"g{g:.3f}_o{o:.3f}": {
                "bpb": float(ev["bpb"]),
                "bpb_values": [float(x) for x in ev.get("bpb_values", [])],
            }
            for (g, o), ev in pair_cache.items()
        }
        _write_csv(out / "calibration_results.csv", calibration_rows)
        _json_dump(out / "calibration_details.json", calibration_details)

    result = {
        "suite": SUITE,
        "role": runtime.role,
        "optimizer_label": runtime.optimizer_label,
        "actual_pack_optimizer": actual_optimizer,
        "pack": _pack_summary(runtime.pack.resolve(), pack),
        "group_schema": str(runtime.group_schema.resolve()),
        "group_schema_sha256": schema["schema_sha256"],
        "terminal_floor": float(runtime.terminal_floor),
        "stream_seed": int(runtime.stream_seed),
        "seed": int(runtime.seed),
        "permutation_fingerprint": permutation_fingerprint,
        "train_steps": int(runtime.train_steps),
        "schedule_total_iterations": int(runtime.schedule_total_iterations),
        "checkpoint_k": K,
        "checkpoint_spacing": SPACING,
        "selected_steps": selected_steps,
        "snapshot_dtype": runtime.snapshot_dtype,
        "artifact": str(artifact_path),
        "artifact_size_bytes": artifact_path.stat().st_size,
        "calibration_split": calibration_meta,
        "structured_grid": list(STRUCTURED_GRID),
        "scalar_grid": list(SCALAR_GRID),
        "group_delta_l2": float(math.sqrt(sum(
            float(torch.sum((mean_cpu[int(a):int(b)] - current_cpu[int(a):int(b)]) ** 2).item())
            for a, b in ranges
        ))),
        "all_delta_l2": float(torch.linalg.vector_norm(mean_cpu - current_cpu).item()),
        "training_trace": training_rows,
        "ledger": ledger.as_dict(),
        "development_rows": len(calibration_rows),
    }
    _json_dump(out / "result.json", result)
    _write_csv(out / "training_trace.csv", training_rows)
    print(f"[result] wrote {out/'result.json'}", flush=True)
    return out / "result.json"


def _load_artifact(path: Path, schema: Mapping[str, Any]) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("suite") != SUITE:
        raise RuntimeError(f"artifact suite mismatch: {artifact.get('suite')}")
    if artifact.get("group_schema_sha256") != schema.get("schema_sha256"):
        raise RuntimeError("artifact group schema does not match frozen group schema")
    current = artifact.get("current")
    mean = artifact.get("mean")
    if not torch.is_tensor(current) or not torch.is_tensor(mean):
        raise RuntimeError("artifact is missing current/mean tensors")
    if current.numel() != int(schema["total_numel"]):
        raise RuntimeError("artifact tensor size does not match schema")
    return artifact


def evaluate(runtime: argparse.Namespace) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation must run on a GPU allocation")
    runtime = _runtime_defaults(runtime)
    schema = _load_schema(runtime.group_schema.resolve())
    frozen = json.loads(runtime.frozen_rules.read_text())
    if frozen.get("suite") != SUITE:
        raise RuntimeError(f"frozen rule suite mismatch: {frozen.get('suite')}")
    claimed_freeze_hash = frozen.get("freeze_sha256")
    frozen_without_hash = dict(frozen)
    frozen_without_hash.pop("freeze_sha256", None)
    observed_freeze_hash = _canonical_hash(frozen_without_hash)
    if claimed_freeze_hash != observed_freeze_hash:
        raise RuntimeError(
            f"frozen rule hash mismatch: claimed {claimed_freeze_hash}, observed {observed_freeze_hash}"
        )

    pack, restored = _restore(runtime.pack.resolve(), runtime)
    args, model, optimizer, token_bytes, device, spec, _history, _bank = restored
    holdout, holdout_meta = _eval_split(pack, runtime, "holdout")

    artifact = _load_artifact(runtime.artifact.resolve(), schema)
    if artifact.get("optimizer_label") != runtime.optimizer_label:
        raise RuntimeError("artifact optimizer label does not match evaluation manifest")
    if abs(float(artifact.get("terminal_floor")) - float(runtime.terminal_floor)) > 1e-12:
        raise RuntimeError("artifact floor does not match evaluation manifest")
    if int(artifact.get("stream_seed")) != int(runtime.stream_seed):
        raise RuntimeError("artifact stream seed does not match evaluation manifest")

    current_cpu = artifact["current"].to(dtype=torch.float32)
    mean_cpu = artifact["mean"].to(dtype=torch.float32)
    spec.assign_parameters(current_cpu)
    current_gpu = spec.flatten_parameters(dtype=torch.float32)
    delta_group, delta_other = _apply_group_split(current_cpu, mean_cpu, schema["group_a"]["ranges"])

    own_rule = frozen["rules"][runtime.optimizer_label]
    native_rule = frozen["rules"]["native"]
    primary_specs: list[tuple[str, str, float, float]] = [
        ("raw", "Raw endpoint", 0.0, 0.0),
        ("scalar_fixed_050", "Scalar TSA (alpha=0.50)", 0.5, 0.5),
        ("scalar_fixed_055", "Scalar TSA (alpha=0.55)", 0.55, 0.55),
        (
            "scalar_dev_selected",
            "Development-selected scalar TSA",
            float(own_rule["scalar_alpha"]),
            float(own_rule["scalar_alpha"]),
        ),
        (
            "structured_dev_selected",
            "Development-selected structured TSA",
            float(own_rule["alpha_group"]),
            float(own_rule["alpha_other"]),
        ),
        (
            "group_only_dev_selected",
            "Group-A-only shrinkage",
            float(own_rule["alpha_group"]),
            0.0,
        ),
        (
            "other_only_dev_selected",
            "Group-B-only shrinkage",
            0.0,
            float(own_rule["alpha_other"]),
        ),
        ("lawa", "Uniform LAWA", 1.0, 1.0),
    ]
    if runtime.optimizer_label != "native":
        primary_specs.append(
            (
                "native_structured_rule_transfer",
                "Native-developed structured rule transferred to AdamW",
                float(native_rule["alpha_group"]),
                float(native_rule["alpha_other"]),
            )
        )

    pair_cache: dict[tuple[float, float], dict[str, Any]] = {}

    def evaluate_pair(a_group: float, a_other: float) -> dict[str, Any]:
        key = (round(float(a_group), 6), round(float(a_other), 6))
        if key not in pair_cache:
            cand = _candidate(current_cpu, delta_group, delta_other, *key)
            pair_cache[key] = _eval_candidate(model, spec, current_gpu, cand, holdout, token_bytes, device)
            del cand
        return pair_cache[key]

    raw = evaluate_pair(0.0, 0.0)
    rows: list[dict[str, Any]] = []
    for recipe, method, a_group, a_other in primary_specs:
        ev = evaluate_pair(a_group, a_other)
        d, se = _paired_delta(ev["bpb_values"], raw["bpb_values"])
        rows.append(
            {
                "eval_kind": "primary",
                "recipe": recipe,
                "method": method,
                "alpha_group": a_group,
                "alpha_other": a_other,
                "bpb": float(ev["bpb"]),
                "gain_vs_raw": float(raw["bpb"] - ev["bpb"]),
                "method_minus_raw": d,
                "paired_eval_se": se,
                "eval_seconds": float(ev["eval_seconds"]),
                "optimizer": runtime.optimizer_label,
                "floor": float(runtime.terminal_floor),
                "stream_seed": int(runtime.stream_seed),
            }
        )

    # These grids are evaluated only after the rule is frozen.  They are diagnostic
    # surfaces and are never used by the primary confirmation selection.
    for a_group in STRUCTURED_GRID:
        for a_other in STRUCTURED_GRID:
            ev = evaluate_pair(a_group, a_other)
            rows.append(
                {
                    "eval_kind": "structured_grid_diagnostic",
                    "recipe": f"structured:g{a_group:.3f}:o{a_other:.3f}",
                    "method": "Structured grid diagnostic",
                    "alpha_group": a_group,
                    "alpha_other": a_other,
                    "bpb": float(ev["bpb"]),
                    "gain_vs_raw": float(raw["bpb"] - ev["bpb"]),
                    "eval_seconds": float(ev["eval_seconds"]),
                    "optimizer": runtime.optimizer_label,
                    "floor": float(runtime.terminal_floor),
                    "stream_seed": int(runtime.stream_seed),
                }
            )
    for alpha in SCALAR_GRID:
        ev = evaluate_pair(alpha, alpha)
        rows.append(
            {
                "eval_kind": "scalar_grid_diagnostic",
                "recipe": f"scalar:a{alpha:.3f}",
                "method": "Scalar grid diagnostic",
                "alpha_group": alpha,
                "alpha_other": alpha,
                "bpb": float(ev["bpb"]),
                "gain_vs_raw": float(raw["bpb"] - ev["bpb"]),
                "eval_seconds": float(ev["eval_seconds"]),
                "optimizer": runtime.optimizer_label,
                "floor": float(runtime.terminal_floor),
                "stream_seed": int(runtime.stream_seed),
            }
        )

    out = runtime.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "holdout_results.csv", rows)
    details = {
        f"g{g:.3f}_o{o:.3f}": {
            "bpb": float(ev["bpb"]),
            "bpb_values": [float(x) for x in ev.get("bpb_values", [])],
            "eval_seconds": float(ev["eval_seconds"]),
        }
        for (g, o), ev in pair_cache.items()
    }
    _json_dump(out / "holdout_details.json", details)
    result = {
        "suite": SUITE,
        "optimizer_label": runtime.optimizer_label,
        "terminal_floor": float(runtime.terminal_floor),
        "stream_seed": int(runtime.stream_seed),
        "pack": _pack_summary(runtime.pack.resolve(), pack),
        "artifact": str(runtime.artifact.resolve()),
        "frozen_rules": str(runtime.frozen_rules.resolve()),
        "frozen_rules_sha256": _sha256_bytes(runtime.frozen_rules.read_bytes()),
        "group_schema_sha256": schema["schema_sha256"],
        "holdout_split": holdout_meta,
        "primary_rows": [r for r in rows if r["eval_kind"] == "primary"],
        "num_unique_models_evaluated": len(pair_cache),
    }
    _json_dump(out / "evaluation_result.json", result)
    print(f"[result] wrote {out/'holdout_results.csv'}", flush=True)
    return out / "holdout_results.csv"


def selftest() -> None:
    current = torch.tensor([0.0, 1.0, 2.0, 3.0])
    mean = torch.tensor([1.0, 3.0, 5.0, 7.0])
    dg, do = _apply_group_split(current, mean, [[0, 2]])
    assert torch.allclose(dg, torch.tensor([1.0, 2.0, 0.0, 0.0]))
    assert torch.allclose(do, torch.tensor([0.0, 0.0, 3.0, 4.0]))
    got = _candidate(current, dg, do, 0.5, 0.25)
    assert torch.allclose(got, torch.tensor([0.5, 2.0, 2.75, 4.0]))
    ranges = _normalize_ranges([(0, 2), (2, 4), (7, 9)], 10)
    assert ranges == [[0, 4], [7, 9]]
    schema = {"a": 1, "b": [2, 3]}
    assert _canonical_hash(schema) == _canonical_hash({"b": [2, 3], "a": 1})
    assert len(STRUCTURED_GRID) == 9 and len(SCALAR_GRID) == 21
    assert 3000 % SNAPSHOT_STRIDE == SNAPSHOT_PHASE
    print("structured TSA D12 selftest PASS")


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history-device", default=None)
    parser.add_argument("--history-dtype", default=None)
    parser.add_argument("--lr-scale", type=float, default=0.40)
    parser.add_argument("--warmdown-ratio", type=float, default=None)
    parser.add_argument("--final-lr-frac", type=float, default=None)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--schedule-total-iterations", type=int, default=3000)
    parser.add_argument("--snapshot-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--curve-eval-batches", type=int, default=32)
    parser.add_argument("--alpha-select-batches", type=int, default=64)
    parser.add_argument("--floor-select-batches", type=int, default=64)
    parser.add_argument("--holdout-eval-batches", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1337)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest")

    sp = sub.add_parser("smoke")
    sp.add_argument("--native-pack", type=Path, required=True)
    sp.add_argument("--adamw-pack", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    _add_runtime_args(sp)

    tp = sub.add_parser("train")
    tp.add_argument("--pack", type=Path, required=True)
    tp.add_argument("--out", type=Path, required=True)
    tp.add_argument("--group-schema", type=Path, required=True)
    tp.add_argument("--role", choices=["development", "confirmation"], required=True)
    tp.add_argument("--optimizer-label", choices=["native", "pure_adamw"], required=True)
    tp.add_argument("--terminal-floor", type=float, required=True)
    tp.add_argument("--stream-seed", type=int, required=True)
    _add_runtime_args(tp)

    ep = sub.add_parser("evaluate")
    ep.add_argument("--pack", type=Path, required=True)
    ep.add_argument("--artifact", type=Path, required=True)
    ep.add_argument("--out", type=Path, required=True)
    ep.add_argument("--group-schema", type=Path, required=True)
    ep.add_argument("--frozen-rules", type=Path, required=True)
    ep.add_argument("--optimizer-label", choices=["native", "pure_adamw"], required=True)
    ep.add_argument("--terminal-floor", type=float, required=True)
    ep.add_argument("--stream-seed", type=int, required=True)
    _add_runtime_args(ep)
    return ap


def main() -> None:
    args = parser().parse_args()
    if args.command == "selftest":
        selftest()
    elif args.command == "smoke":
        smoke(args)
    elif args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
