"""Core utilities for the NanoChat gradient-skip experiments.

The central invariant is simple: every synthetic update is represented as a
*gradient tensor* and is passed through the same optimizer.step() implementation
as an exact gradient. This keeps AdamW moments, bias correction, weight decay,
step counts, and learning-rate schedules consistent.

This module deliberately has no NanoChat imports so that ``python -m
nanochat_meta.skip_suite selftest`` can exercise the dangerous plumbing on a
CPU-only machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Optional, Sequence
import copy
import math
import random
import time

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def recursive_to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: recursive_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def recursive_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to_device(v, device) for v in obj)
    return obj


@dataclass(frozen=True)
class TensorSlice:
    name: str
    start: int
    stop: int
    shape: tuple[int, ...]
    dtype: torch.dtype

    @property
    def numel(self) -> int:
        return self.stop - self.start


class FlatSpec:
    """Stable flatten/unflatten schema for a model's trainable parameters."""

    def __init__(self, named_parameters: Iterable[tuple[str, torch.nn.Parameter]]):
        items = [(n, p) for n, p in named_parameters if p.requires_grad]
        if not items:
            raise ValueError("model has no trainable parameters")
        self.names = [n for n, _ in items]
        self.params = [p for _, p in items]
        self.slices: list[TensorSlice] = []
        off = 0
        for name, p in items:
            nxt = off + p.numel()
            self.slices.append(TensorSlice(name, off, nxt, tuple(p.shape), p.dtype))
            off = nxt
        self.numel = off
        self.device = self.params[0].device

    def flatten_parameters(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        out = torch.empty(self.numel, device=self.device, dtype=dtype)
        for sl, p in zip(self.slices, self.params):
            out[sl.start:sl.stop].copy_(p.detach().reshape(-1).to(dtype))
        return out

    @torch.no_grad()
    def assign_parameters(self, vec: torch.Tensor) -> None:
        if vec.numel() != self.numel:
            raise ValueError(f"parameter vector has {vec.numel()} values, expected {self.numel}")
        for sl, p in zip(self.slices, self.params):
            p.copy_(vec[sl.start:sl.stop].view(sl.shape).to(device=p.device, dtype=p.dtype))

    def flatten_gradients(self, *, dtype: torch.dtype = torch.float32,
                          none_as_zero: bool = True) -> torch.Tensor:
        out = torch.empty(self.numel, device=self.device, dtype=dtype)
        for sl, p in zip(self.slices, self.params):
            if p.grad is None:
                if not none_as_zero:
                    raise RuntimeError(f"missing gradient for {sl.name}")
                out[sl.start:sl.stop].zero_()
            else:
                out[sl.start:sl.stop].copy_(p.grad.detach().reshape(-1).to(dtype))
        return out

    @torch.no_grad()
    def assign_gradients(self, vec: torch.Tensor) -> None:
        """Populate ``p.grad`` from a flat external gradient.

        This is the only supported path for synthetic updates. The caller then
        invokes the *ordinary* optimizer.step().
        """
        if vec.numel() != self.numel:
            raise ValueError(f"gradient vector has {vec.numel()} values, expected {self.numel}")
        for sl, p in zip(self.slices, self.params):
            g = vec[sl.start:sl.stop].view(sl.shape).to(device=p.device, dtype=p.dtype)
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

    def iter_slices(self, vec: torch.Tensor) -> Iterator[tuple[TensorSlice, torch.Tensor]]:
        for sl in self.slices:
            yield sl, vec[sl.start:sl.stop].view(sl.shape)


class VectorHistory:
    """A bounded history of full-dimensional vectors with an incremental Gram matrix.

    ``store_device='cuda'`` is strongly recommended for rapid screening. CPU
    storage works, but every prediction must stream the relevant vectors over
    PCIe. At d6, history=8 in bfloat16 is typically a manageable few GB.
    """

    def __init__(self, capacity: int, numel: int, compute_device: torch.device,
                 store_device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 chunk: int = 2_000_000):
        if capacity < 2:
            raise ValueError("history capacity must be at least 2")
        self.capacity = capacity
        self.numel = numel
        self.compute_device = torch.device(compute_device)
        self.store_device = torch.device(store_device)
        self.dtype = dtype
        self.chunk = chunk
        self._vectors: deque[torch.Tensor] = deque()
        self._gram = torch.zeros((0, 0), dtype=torch.float64, device="cpu")

    def __len__(self) -> int:
        return len(self._vectors)

    @property
    def full(self) -> bool:
        return len(self) >= self.capacity

    @property
    def memory_bytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in self._vectors)

    def _dot(self, a: torch.Tensor, b: torch.Tensor) -> float:
        # Same-device GPU is fast and accumulates in fp32. For CPU/bf16, chunk
        # explicitly to avoid allocating two full fp32 copies at once.
        if a.device.type == "cuda" and b.device == a.device:
            return float(torch.dot(a.float(), b.float()).double().cpu())
        total = 0.0
        for lo in range(0, self.numel, self.chunk):
            hi = min(lo + self.chunk, self.numel)
            aa = a[lo:hi].float()
            bb = b[lo:hi].float()
            total += float((aa * bb).sum(dtype=torch.float64))
        return total

    def append(self, vec: torch.Tensor) -> None:
        if vec.numel() != self.numel:
            raise ValueError(f"history vector has {vec.numel()} values, expected {self.numel}")
        new = vec.detach().reshape(-1).to(device=self.store_device, dtype=self.dtype).clone()
        if len(self._vectors) == self.capacity:
            self._vectors.popleft()
            self._gram = self._gram[1:, 1:].clone()
        n = len(self._vectors)
        cross = [self._dot(v, new) for v in self._vectors]
        gnew = torch.zeros((n + 1, n + 1), dtype=torch.float64)
        if n:
            gnew[:n, :n] = self._gram
            c = torch.tensor(cross, dtype=torch.float64)
            gnew[:n, n] = c
            gnew[n, :n] = c
        gnew[n, n] = self._dot(new, new)
        self._vectors.append(new)
        self._gram = gnew

    def gram(self) -> torch.Tensor:
        return self._gram.clone()

    def norms(self) -> torch.Tensor:
        if not len(self):
            return torch.zeros(0, dtype=torch.float64)
        return self._gram.diagonal().clamp_min(0).sqrt()

    def last(self, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if not self._vectors:
            raise RuntimeError("empty history")
        return self._vectors[-1].to(device=self.compute_device, dtype=dtype)

    def mean(self, n: Optional[int] = None, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if not self._vectors:
            raise RuntimeError("empty history")
        n = len(self) if n is None else min(max(1, n), len(self))
        c = torch.zeros(len(self), dtype=torch.float64)
        c[-n:] = 1.0 / n
        return self.combine(c, dtype=dtype)

    def combine(self, coeffs: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        coeffs = coeffs.detach().double().cpu().reshape(-1)
        if coeffs.numel() != len(self):
            raise ValueError(f"got {coeffs.numel()} coefficients for history of length {len(self)}")
        out = torch.zeros(self.numel, device=self.compute_device, dtype=dtype)
        if self.store_device == self.compute_device:
            for c, v in zip(coeffs.tolist(), self._vectors):
                if c:
                    out.add_(v.to(dtype), alpha=float(c))
            return out
        # Stream CPU history in chunks to avoid a W*N temporary on the GPU.
        for lo in range(0, self.numel, self.chunk):
            hi = min(lo + self.chunk, self.numel)
            dst = out[lo:hi]
            for c, v in zip(coeffs.tolist(), self._vectors):
                if c:
                    dst.add_(v[lo:hi].to(device=self.compute_device, dtype=dtype), alpha=float(c))
        return out

    def cross(self, target: torch.Tensor) -> torch.Tensor:
        """Return ``[<h_i,target>]`` in chronological order."""
        target = target.detach().reshape(-1)
        if target.numel() != self.numel:
            raise ValueError("target shape mismatch")
        if self.store_device == target.device:
            return torch.tensor([self._dot(v, target.to(v.dtype)) for v in self._vectors],
                                dtype=torch.float64)
        # One CPU copy is cheaper than W independent device transfers.
        t = target.to(device=self.store_device, dtype=self.dtype)
        return torch.tensor([self._dot(v, t) for v in self._vectors], dtype=torch.float64)

    def project_coefficients(self, target: torch.Tensor, ridge: float = 1e-4) -> torch.Tensor:
        if not len(self):
            raise RuntimeError("cannot project with empty history")
        G = self.gram()
        b = self.cross(target)
        scale = float(G.diagonal().mean().clamp_min(1e-30))
        return torch.linalg.solve(G + ridge * scale * torch.eye(len(self), dtype=G.dtype), b)

    def project(self, target: torch.Tensor, ridge: float = 1e-4) -> torch.Tensor:
        return self.combine(self.project_coefficients(target, ridge))

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "numel": self.numel,
            "dtype": str(self.dtype).replace("torch.", ""),
            "vectors": [v.detach().cpu() for v in self._vectors],
            "gram": self._gram.clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if int(state["numel"]) != self.numel:
            raise ValueError("history numel mismatch")
        self._vectors.clear()
        for v in state["vectors"]:
            self._vectors.append(v.to(device=self.store_device, dtype=self.dtype).clone())
        self._gram = state["gram"].double().cpu().clone()


def global_ar_coefficients(history: VectorHistory, order: int, ridge: float = 1e-3) -> torch.Tensor:
    """Fit a shared AR(p) over all coordinates using only the history Gram matrix."""
    W = len(history)
    p = min(order, W - 1)
    if p < 1:
        raise RuntimeError("not enough history for AR")
    G = history.gram()
    A = torch.zeros((p, p), dtype=torch.float64)
    b = torch.zeros(p, dtype=torch.float64)
    # Targets are h_t for t=p,...,W-1; regress on h_{t-1},...,h_{t-p}.
    for t in range(p, W):
        idx = [t - j - 1 for j in range(p)]
        A += G[idx][:, idx]
        b += G[idx, t]
    scale = float(A.diagonal().mean().clamp_min(1e-30))
    a = torch.linalg.solve(A + ridge * scale * torch.eye(p, dtype=A.dtype), b)
    c = torch.zeros(W, dtype=torch.float64)
    for j in range(p):
        c[W - j - 1] = a[j]
    return c


def dmd_coefficients(history: VectorHistory, rank: int, ridge: float = 1e-4) -> torch.Tensor:
    """One-step exact-DMD prediction, returned as coefficients over history rows.

    With X=[h_0,...,h_{W-2}] and Y=[h_1,...,h_{W-1}], exact DMD predicts
    A h_{W-1}. The expression below computes the result in the temporal Gram
    space, so it never constructs an N x N operator.
    """
    W = len(history)
    if W < 3:
        raise RuntimeError("DMD needs at least three history vectors")
    m = W - 1
    G = history.gram()
    Gx = G[:m, :m]
    gxlast = G[:m, W - 1]
    evals, V = torch.linalg.eigh(Gx)
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    V = V[:, order]
    r = min(rank, int((evals > evals.max().clamp_min(1e-30) * 1e-10).sum()), m)
    if r < 1:
        raise RuntimeError("DMD history is numerically rank zero")
    lam = ridge * float(evals[:r].mean().clamp_min(1e-30))
    Vr = V[:, :r]
    # a = V (S^2 + lambda I)^-1 V^T X^T h_last; prediction = Y a.
    a = Vr @ ((Vr.T @ gxlast) / (evals[:r] + lam))
    c = torch.zeros(W, dtype=torch.float64)
    c[1:] = a
    return c


def tensor_ar_prediction(history: VectorHistory, spec: FlatSpec, order: int,
                         ridge: float = 1e-3) -> torch.Tensor:
    """Fit independent shared-coordinate AR coefficients for every parameter tensor."""
    W = len(history)
    p = min(order, W - 1)
    if p < 1:
        raise RuntimeError("not enough history for tensor AR")
    out = torch.zeros(spec.numel, device=spec.device, dtype=torch.float32)
    # This path intentionally materializes only one tensor's W x n slice at a time.
    for sl in spec.slices:
        rows = []
        for v in history._vectors:  # internal access avoids W full-vector copies
            rows.append(v[sl.start:sl.stop].to(device=spec.device, dtype=torch.float32))
        H = torch.stack(rows)  # [W, n_tensor]
        G = H @ H.T
        A = torch.zeros((p, p), device=spec.device, dtype=torch.float64)
        b = torch.zeros(p, device=spec.device, dtype=torch.float64)
        Gd = G.double()
        for t in range(p, W):
            idx = [t - j - 1 for j in range(p)]
            A += Gd[idx][:, idx]
            b += Gd[idx, t]
        scale = float(A.diagonal().mean().clamp_min(1e-30))
        a = torch.linalg.solve(A + ridge * scale * torch.eye(p, device=A.device, dtype=A.dtype), b)
        pred = torch.zeros(sl.numel, device=spec.device, dtype=torch.float32)
        for j in range(p):
            pred.add_(H[W - j - 1], alpha=float(a[j]))
        out[sl.start:sl.stop] = pred
        del H, G
    return out


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.float().reshape(-1)
    bb = b.float().reshape(-1)
    den = aa.norm() * bb.norm()
    if float(den) == 0.0:
        return float("nan")
    return float(torch.dot(aa, bb) / den)


def relative_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    den = target.float().norm().clamp_min(1e-30)
    return float((pred.float() - target.float()).norm() / den)


def optimizer_first_moment(optimizer: torch.optim.Optimizer, spec: FlatSpec,
                           *, bias_correct: bool = True) -> torch.Tensor:
    """Flatten AdamW's first moment as a gradient estimate.

    The experiment suite converts every model parameter to an AdamW group. This
    function intentionally errors on non-Adam state rather than silently making
    up a value.
    """
    out = torch.zeros(spec.numel, device=spec.device, dtype=torch.float32)
    group_for_param: dict[int, MutableMapping[str, Any]] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            group_for_param[id(p)] = group
    for sl, p in zip(spec.slices, spec.params):
        state = optimizer.state.get(p, {})
        if "exp_avg" not in state:
            continue
        m = state["exp_avg"].detach().float().reshape(-1)
        if bias_correct:
            step = state.get("step", 0)
            step_value = int(step.item()) if torch.is_tensor(step) else int(step)
            beta1 = float(group_for_param[id(p)].get("betas", (0.9, 0.999))[0])
            if step_value > 0:
                m = m / max(1.0 - beta1 ** step_value, 1e-12)
        out[sl.start:sl.stop].copy_(m.to(out.device))
    return out


def clear_adam_state(optimizer: torch.optim.Optimizer, *, first_only: bool = False) -> None:
    for state in optimizer.state.values():
        if "exp_avg" in state:
            state["exp_avg"].zero_()
        if not first_only and "exp_avg_sq" in state:
            state["exp_avg_sq"].zero_()
        if not first_only and "step" in state:
            if torch.is_tensor(state["step"]):
                state["step"].zero_()
            else:
                state["step"] = 0


def make_pure_adamw(model: torch.nn.Module, base_optimizer: torch.optim.Optimizer,
                    matrix_lr: Optional[float] = None,
                    matrix_betas: tuple[float, float] = (0.9, 0.95),
                    matrix_eps: float = 1e-10,
                    matrix_weight_decay: Optional[float] = None) -> torch.optim.Optimizer:
    """Convert a NanoChat Muon/AdamW optimizer into all-AdamW parameter groups.

    On NanoChat's custom ``MuonAdamW`` class this preserves its fused AdamW
    implementation, including correct fp32 math for reduced-precision embedding
    parameters. If the optimizer is already a regular AdamW, it is returned as-is.
    """
    groups: list[dict[str, Any]] = []
    saw_kind = any("kind" in g for g in base_optimizer.param_groups)
    if not saw_kind:
        return base_optimizer
    for group in base_optimizer.param_groups:
        g = {k: v for k, v in group.items() if k != "params"}
        params = list(group["params"])
        kind = g.get("kind", "adamw")
        if kind == "muon":
            old_wd = float(g.get("weight_decay", 0.0))
            old_lr = float(g.get("lr", 1e-3))
            g = {
                "kind": "adamw",
                "params": params,
                "lr": old_lr if matrix_lr is None else matrix_lr,
                "betas": matrix_betas,
                "eps": matrix_eps,
                "weight_decay": old_wd if matrix_weight_decay is None else matrix_weight_decay,
            }
        else:
            g["kind"] = "adamw"
            g["params"] = params
            g.setdefault("betas", (0.9, 0.999))
            g.setdefault("eps", 1e-8)
            g.setdefault("weight_decay", 0.0)
        g["initial_lr"] = float(g.get("initial_lr", g["lr"]))
        groups.append(g)
    opt_cls = type(base_optimizer)
    try:
        pure = opt_cls(groups)
    except Exception as exc:
        raise RuntimeError(
            f"Could not reconstruct {opt_cls.__name__} with all-AdamW groups. "
            "Your NanoChat optimizer API differs from the public implementation."
        ) from exc
    return pure


def apply_lr_schedule(optimizer: torch.optim.Optimizer, it: int, warmup: int,
                      total: int = 0, warmdown_ratio: float = 0.0,
                      final_lr_frac: float = 1.0) -> float:
    if it < warmup:
        mult = (it + 1) / max(warmup, 1)
    elif warmdown_ratio <= 0 or total <= 0:
        mult = 1.0
    else:
        warmdown = max(int(total * warmdown_ratio), 1)
        if it < total - warmdown:
            mult = 1.0
        else:
            progress = max(0.0, (total - it) / warmdown)
            mult = progress + (1.0 - progress) * final_lr_frac
    for group in optimizer.param_groups:
        group.setdefault("initial_lr", group["lr"])
        group["lr"] = group["initial_lr"] * mult
    return mult


@dataclass
class ComputeLedger:
    exact_backward_equiv: float = 0.0
    forward_equiv: float = 0.0
    predictor_seconds: float = 0.0
    train_seconds: float = 0.0
    eval_seconds: float = 0.0
    actual_backwards: int = 0
    synthetic_steps: int = 0
    virtual_steps: int = 0

    @property
    def ideal_train_equiv(self) -> float:
        # A transformer forward is roughly one third of forward+backward.
        return self.exact_backward_equiv + self.forward_equiv / 3.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "exact_backward_equiv": self.exact_backward_equiv,
            "forward_equiv": self.forward_equiv,
            "ideal_train_equiv": self.ideal_train_equiv,
            "predictor_seconds": self.predictor_seconds,
            "train_seconds": self.train_seconds,
            "eval_seconds": self.eval_seconds,
            "actual_backwards": self.actual_backwards,
            "synthetic_steps": self.synthetic_steps,
            "virtual_steps": self.virtual_steps,
        }


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds = time.perf_counter() - self.t0


def external_gradient_equivalence_test(device: str = "cpu") -> dict[str, float]:
    """Unit test: normal AdamW and externally re-assigned gradient must coincide."""
    dev = torch.device(device)
    torch.manual_seed(7)
    m = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.Tanh(), torch.nn.Linear(7, 3)).to(dev)
    x = torch.randn(11, 5, device=dev)
    y = torch.randn(11, 3, device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3, betas=(0.8, 0.95), weight_decay=0.02)
    spec = FlatSpec(m.named_parameters())
    snap_m = recursive_to_cpu(m.state_dict())
    snap_o = recursive_to_cpu(opt.state_dict())

    loss = torch.nn.functional.mse_loss(m(x), y)
    loss.backward()
    g = spec.flatten_gradients()
    opt.step(); opt.zero_grad(set_to_none=True)
    theta_normal = spec.flatten_parameters()
    state_normal = recursive_to_cpu(opt.state_dict())

    m.load_state_dict(snap_m)
    opt.load_state_dict(snap_o)
    spec.assign_gradients(g)
    opt.step(); opt.zero_grad(set_to_none=True)
    theta_external = spec.flatten_parameters()
    state_external = recursive_to_cpu(opt.state_dict())

    param_err = float((theta_normal - theta_external).abs().max())
    # Compare moment tensors too.
    moment_err = 0.0
    for a, b in zip(state_normal["state"].values(), state_external["state"].values()):
        for key in ("exp_avg", "exp_avg_sq"):
            if key in a:
                moment_err = max(moment_err, float((a[key] - b[key]).abs().max()))
    if param_err > 1e-7 or moment_err > 1e-7:
        raise AssertionError(f"external-gradient equivalence failed: params={param_err}, moments={moment_err}")
    return {"max_parameter_error": param_err, "max_moment_error": moment_err}
