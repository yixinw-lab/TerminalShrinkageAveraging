#!/usr/bin/env python3
"""Parameter grouping, sparse snapshot, and adaptive trajectory-filter utilities.

This module is intentionally independent of NanoChat internals beyond named
parameters and optimizer param groups. It supports the mixed Muon/AdamW stack by
reading each optimizer group's ``kind`` field when present.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class SliceInfo:
    index: int
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]
    optimizer_kind: str
    role: str
    layer: Optional[int]

    @property
    def numel(self) -> int:
        return self.stop - self.start


def _role_for_name(name: str) -> tuple[str, Optional[int]]:
    layer = None
    m = re.search(r"(?:^|\.)h\.(\d+)(?:\.|$)", name)
    if m:
        layer = int(m.group(1))
    low = name.lower()
    if "lm_head" in low:
        return "lm_head", layer
    if "value_embed" in low:
        return "value_embed", layer
    if "wte" in low or "token_embedding" in low or "embedding" in low and layer is None:
        return "embedding", layer
    if layer is not None:
        if ".attn." in low or ".attention." in low:
            return "attention", layer
        if ".mlp." in low or ".ffn." in low or "feed_forward" in low:
            return "mlp", layer
        return "transformer_other", layer
    if any(x in low for x in ("resid_lambda", "x0_lambda", "smear", "backout", "scale", "gain")):
        return "scalar", layer
    if low.endswith("bias") or len(low) and low.split(".")[-1] in {"bias", "weight"}:
        return "other", layer
    return "other", layer


def build_slice_infos(spec: Any, optimizer: torch.optim.Optimizer) -> list[SliceInfo]:
    kind_by_param: dict[int, str] = {}
    for group in optimizer.param_groups:
        kind = str(group.get("kind", "adamw")).lower()
        for p in group["params"]:
            kind_by_param[id(p)] = kind
    infos: list[SliceInfo] = []
    for i, (sl, p) in enumerate(zip(spec.slices, spec.params)):
        role, layer = _role_for_name(sl.name)
        infos.append(SliceInfo(
            index=i,
            name=sl.name,
            start=int(sl.start),
            stop=int(sl.stop),
            shape=tuple(sl.shape),
            optimizer_kind=kind_by_param.get(id(p), "unknown"),
            role=role,
            layer=layer,
        ))
    return infos


def selector_matches(info: SliceInfo, selector: str, n_layers: Optional[int] = None) -> bool:
    selector = selector.lower()
    if selector in {"all", "*"}:
        return True
    if selector in {"muon", "adamw", "unknown"}:
        return info.optimizer_kind == selector
    if selector in {
        "attention", "mlp", "embedding", "value_embed", "lm_head", "scalar",
        "transformer_other", "other",
    }:
        return info.role == selector
    if selector == "matrices":
        return len(info.shape) >= 2
    if selector == "vectors":
        return len(info.shape) < 2
    if selector in {"early", "middle", "late"}:
        if info.layer is None or not n_layers:
            return False
        frac = (info.layer + 0.5) / n_layers
        if selector == "early":
            return frac <= 1 / 3
        if selector == "middle":
            return 1 / 3 < frac <= 2 / 3
        return frac > 2 / 3
    if selector.startswith("layer"):
        try:
            return info.layer == int(selector[5:])
        except ValueError:
            return False
    if selector == "muon_attention":
        return info.optimizer_kind == "muon" and info.role == "attention"
    if selector == "muon_mlp":
        return info.optimizer_kind == "muon" and info.role == "mlp"
    if selector == "adam_embeddings":
        return info.optimizer_kind == "adamw" and info.role in {"embedding", "value_embed"}
    return False


class SparseSnapshotBank:
    """Bounded CPU snapshot bank indexed by exact optimizer step.

    Snapshots are stored only every ``stride`` steps. This is much more scalable
    than retaining every iterate for d12/d24 experiments.
    """

    def __init__(self, max_snapshots: int, stride: int, dtype: torch.dtype = torch.bfloat16):
        if max_snapshots < 2:
            raise ValueError("max_snapshots must be >=2")
        if stride < 1:
            raise ValueError("stride must be >=1")
        self.max_snapshots = int(max_snapshots)
        self.stride = int(stride)
        self.dtype = dtype
        self._rows: deque[tuple[int, torch.Tensor]] = deque(maxlen=max_snapshots)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def memory_bytes(self) -> int:
        return sum(v.numel() * v.element_size() for _, v in self._rows)

    def append(self, step: int, vec: torch.Tensor, *, force: bool = False) -> bool:
        if not force and step % self.stride != 0:
            return False
        cpu = vec.detach().reshape(-1).to(device="cpu", dtype=self.dtype).clone()
        if self._rows and self._rows[-1][0] == int(step):
            self._rows[-1] = (int(step), cpu)
        else:
            self._rows.append((int(step), cpu))
        return True

    def steps(self) -> list[int]:
        return [s for s, _ in self._rows]

    def latest(self) -> tuple[int, torch.Tensor]:
        if not self._rows:
            raise RuntimeError("empty snapshot bank")
        return self._rows[-1]

    def select(self, window: int, spacing_steps: int, *, end_step: Optional[int] = None
               ) -> list[tuple[int, torch.Tensor]]:
        if window < 1 or spacing_steps < 1:
            raise ValueError("window/spacing must be positive")
        rows = list(self._rows)
        if not rows:
            raise RuntimeError("empty snapshot bank")
        end = rows[-1][0] if end_step is None else int(end_step)
        by_step = {s: v for s, v in rows}
        available = np.asarray([s for s, _ in rows], dtype=int)
        target_steps = [end - j * int(spacing_steps) for j in range(window)]
        chosen: list[tuple[int, torch.Tensor]] = []
        used: set[int] = set()
        for target in reversed(target_steps):
            if target in by_step and target not in used:
                step = target
            else:
                order = np.argsort(np.abs(available - target))
                step = None
                for idx in order:
                    candidate = int(available[idx])
                    if candidate not in used and abs(candidate - target) <= self.stride:
                        step = candidate; break
                if step is None:
                    raise RuntimeError(
                        f"snapshot bank has no step near {target} for window={window}, "
                        f"spacing={spacing_steps}, stride={self.stride}; available [{rows[0][0]}, {rows[-1][0]}]"
                    )
            used.add(step); chosen.append((step, by_step[step]))
        return chosen


def average_selected(rows: Sequence[tuple[int, torch.Tensor]], *, dtype: torch.dtype = torch.float32
                     ) -> torch.Tensor:
    if not rows:
        raise ValueError("no snapshots")
    out = torch.zeros_like(rows[0][1], dtype=dtype, device="cpu")
    inv = 1.0 / len(rows)
    for _, v in rows:
        out.add_(v.to(dtype=dtype), alpha=inv)
    return out


def _slice_stats(rows: Sequence[tuple[int, torch.Tensor]], info: SliceInfo) -> dict[str, float]:
    vals = [v[info.start:info.stop].float() for _, v in rows]
    if len(vals) < 2:
        return {
            "autocorr": float("nan"),
            "noise_fraction": 0.0,
            "drift_rms": 0.0,
            "update_rms": 0.0,
            "average_displacement_rms": 0.0,
        }
    updates = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    drift = torch.stack(updates).mean(dim=0)
    drift_e = float(torch.dot(drift, drift) / max(drift.numel(), 1))
    update_e = float(sum(torch.dot(u, u) for u in updates) / (len(updates) * max(drift.numel(), 1)))
    noise_e = max(update_e - drift_e, 0.0)
    noise_fraction = noise_e / max(noise_e + drift_e, 1e-30)
    corrs: list[float] = []
    for a, b in zip(updates[:-1], updates[1:]):
        den = float(a.norm() * b.norm())
        if den > 0:
            corrs.append(float(torch.dot(a, b) / den))
    autocorr = float(np.mean(corrs)) if corrs else float("nan")
    avg = torch.stack(vals).mean(dim=0)
    disp = avg - vals[-1]
    return {
        "autocorr": autocorr,
        "noise_fraction": float(noise_fraction),
        "drift_rms": math.sqrt(max(drift_e, 0.0)),
        "update_rms": math.sqrt(max(update_e, 0.0)),
        "average_displacement_rms": float(disp.norm() / math.sqrt(max(disp.numel(), 1))),
    }


def adaptive_alpha(rule: str, stats: dict[str, float], strength: float = 1.0) -> float:
    noise = float(np.clip(stats.get("noise_fraction", 0.0), 0.0, 1.0))
    corr = stats.get("autocorr", float("nan"))
    corr = 1.0 if not math.isfinite(corr) else float(np.clip(corr, -1.0, 1.0))
    oscillation = float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    rule = rule.lower()
    if rule == "snr":
        raw = noise
    elif rule == "autocorr":
        raw = oscillation
    elif rule == "hybrid":
        raw = math.sqrt(max(noise * oscillation, 0.0))
    elif rule == "max":
        raw = max(noise, oscillation)
    elif rule == "product":
        raw = noise * oscillation
    else:
        raise ValueError(f"unknown adaptive rule {rule!r}")
    return float(np.clip(strength * raw, 0.0, 1.0))


def tensor_stats(rows: Sequence[tuple[int, torch.Tensor]], infos: Sequence[SliceInfo]
                 ) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for info in infos:
        s = _slice_stats(rows, info)
        out.append({
            "name": info.name,
            "numel": info.numel,
            "optimizer_kind": info.optimizer_kind,
            "role": info.role,
            "layer": info.layer,
            **s,
        })
    return out


def make_averaged_candidate(
    current: torch.Tensor,
    average_cpu: torch.Tensor,
    infos: Sequence[SliceInfo],
    *,
    selector: str = "all",
    n_layers: Optional[int] = None,
    adaptive_rule: Optional[str] = None,
    adaptive_strength: float = 1.0,
    stats_rows: Optional[Sequence[dict[str, Any]]] = None,
    adaptive_selector: str = "all",
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    candidate = current.clone()
    alpha_rows: list[dict[str, Any]] = []
    by_name = {r["name"]: r for r in (stats_rows or [])}
    for info in infos:
        selected = selector_matches(info, selector, n_layers)
        alpha = 1.0 if selected else 0.0
        if adaptive_rule is not None:
            if selector_matches(info, adaptive_selector, n_layers):
                alpha = adaptive_alpha(adaptive_rule, by_name.get(info.name, {}), adaptive_strength)
            else:
                alpha = 0.0
        if alpha:
            avg = average_cpu[info.start:info.stop].to(device=current.device, dtype=current.dtype)
            dst = candidate[info.start:info.stop]
            dst.lerp_(avg, float(alpha))
        alpha_rows.append({
            "name": info.name,
            "selector": selector,
            "alpha": float(alpha),
            "optimizer_kind": info.optimizer_kind,
            "role": info.role,
            "layer": info.layer,
            "numel": info.numel,
        })
    return candidate, alpha_rows


def selected_params(spec: Any, infos: Sequence[SliceInfo], selector: str,
                    n_layers: Optional[int] = None) -> list[torch.nn.Parameter]:
    return [p for p, info in zip(spec.params, infos) if selector_matches(info, selector, n_layers)]


@torch.no_grad()
def transport_selected_state(optimizer: torch.optim.Optimizer,
                             params: Iterable[torch.nn.Parameter], policy: str) -> int:
    """Apply a conservative state policy to selected parameters only.

    ``reset_m`` clears first-moment / momentum-like buffers while keeping second
    moments. ``damp_mX`` multiplies them by X. ``reset_all`` clears every tensor
    state associated with the selected parameters.
    """
    policy = policy.lower()
    if policy == "preserve":
        return 0
    damp = None
    if policy.startswith("damp_m"):
        damp = float(policy[len("damp_m"):])
    touched = 0
    first_keys = {"exp_avg", "momentum", "momentum_buffer", "m", "muon_momentum"}
    for p in params:
        state = optimizer.state.get(p, {})
        for key, value in list(state.items()):
            if not torch.is_tensor(value):
                continue
            lk = str(key).lower()
            is_first = lk in first_keys or ("moment" in lk and "sq" not in lk and "second" not in lk)
            if policy == "reset_all":
                value.zero_(); touched += value.numel()
            elif policy == "reset_m" and is_first:
                value.zero_(); touched += value.numel()
            elif damp is not None and is_first:
                value.mul_(damp); touched += value.numel()
    if policy not in {"reset_all", "reset_m"} and damp is None:
        raise ValueError(f"unknown state policy {policy!r}")
    return touched


def weighted_group_aggregate(rows: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault((str(r.get("optimizer_kind")), str(r.get("role"))), []).append(r)
    out = []
    for (kind, role), vals in groups.items():
        w = np.array([max(float(v.get("numel", 0)), 0.0) for v in vals])
        x = np.array([float(v.get(key, float("nan"))) for v in vals])
        mask = np.isfinite(x) & (w > 0)
        mean = float(np.average(x[mask], weights=w[mask])) if mask.any() else float("nan")
        out.append({"optimizer_kind": kind, "role": role, "numel": int(w.sum()), key: mean})
    return sorted(out, key=lambda r: (-r["numel"], r["optimizer_kind"], r["role"]))


def selftest() -> None:
    class Sl:
        def __init__(self, name: str, start: int, stop: int, shape: tuple[int, ...]):
            self.name, self.start, self.stop, self.shape = name, start, stop, shape
    class Spec:
        pass
    p1 = torch.nn.Parameter(torch.zeros(4, 4))
    p2 = torch.nn.Parameter(torch.zeros(4))
    spec = Spec()
    spec.params = [p1, p2]
    spec.slices = [Sl("transformer.h.0.attn.q.weight", 0, 16, (4, 4)), Sl("x0_lambdas", 16, 20, (4,))]
    opt = torch.optim.AdamW([{"params": [p1], "kind": "muon"}, {"params": [p2], "kind": "adamw"}])
    infos = build_slice_infos(spec, opt)
    assert infos[0].role == "attention" and infos[0].optimizer_kind == "muon"
    bank = SparseSnapshotBank(16, 1)
    for t in range(8):
        v = torch.arange(20).float() + t + 0.1 * (-1) ** t
        bank.append(t, v, force=True)
    selected = bank.select(4, 2, end_step=7)
    avg = average_selected(selected)
    stats = tensor_stats(selected, infos)
    cand, alphas = make_averaged_candidate(torch.arange(20).float() + 7.1, avg, infos, selector="muon")
    assert alphas[0]["alpha"] == 1 and alphas[1]["alpha"] == 0
    assert torch.isfinite(cand).all() and len(stats) == 2
    print("[selftest] trajectory groups PASS")


if __name__ == "__main__":
    selftest()
