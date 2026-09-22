#!/usr/bin/env python3
"""Focused equal-compute trajectory-filter experiments for NanoChat.

This suite follows the strongest result from the broad gradient-imputation screen:
cheap in-place transformations of recent parameter iterates can beat tuned exact
training at the same charged FLOPs.  The purpose here is to determine whether the
effect is genuine optimization progress, ordinary checkpoint averaging, optimizer
state reset, or a more general trajectory-filter phenomenon.

Key controls:
  exact_full / exact_lr:*                   tuned exact baselines
  eval_lawa:<window>:<spacing>              standard evaluation-only LAWA
  filter_lawa:<window>:<policy>:<alpha>:<history>:<spacing>
                                             in-place rolling average
  oneshot_lawa:<window>:<policy>:<alpha>:<history>:<spacing>
                                             one recenter, then exact training
  reset_only:<zero_m|zero_all|preserve>      state-reset control, no parameter move
  noop_filter                               zero-cost/direct-action bookkeeping control
  lookahead:<k>:<alpha>:<policy>             Lookahead-style slow/fast interpolation
  pswa:<k>:<alpha>:<policy>                  periodic SWA over the last k exact states
  arblock_filter:<order>:<horizon>:<policy>:<alpha>:<history>
                                             dynamics-derived block filter

The branch can turn all filters off at --filter-stop-equiv and continue with exact
training, which directly tests whether a gain persists after the filter is removed.
All reported comparisons use the same cached data stream and paired validation batches.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
import math
import inspect
from pathlib import Path
import re
import time
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

from .skip_core import (
    ComputeLedger,
    FlatSpec,
    Timer,
    VectorHistory,
    apply_lr_schedule,
    clear_adam_state,
    cosine,
    dmd_coefficients,
    global_ar_coefficients,
    optimizer_first_moment,
    relative_error,
    seed_everything,
    tensor_ar_prediction,
)
from .skip_predictors import PredictorBank, PredictorContext
from .skip_suite import (
    PACK_VERSION,
    active_predictors_for_arm,
    add_build_args,
    arm_requires_forward,
    arm_requires_momentum,
    arm_requires_raw,
    arm_slug,
    candidate_from_history,
    clone_batch_cpu,
    exact_gradient,
    forward_only_context,
    namespace_from_pack,
    parse_arm,
    prepare_pack,
    predictor_bank_from_args,
    restore_branch,
    token_sample_from_batches,
    _dtype_from_name,
)


# -----------------------------------------------------------------------------
# NanoChat compile guard
# -----------------------------------------------------------------------------


def _stabilize_nanochat_adamw_kernel() -> None:
    """Run NanoChat's shape-specialized AdamW kernel eagerly when possible.

    Pure-AdamW conversion invokes the same kernel on many heterogeneous tensor
    shapes. Some NanoChat/PyTorch combinations compile it with ``fullgraph=True``
    and hit the default recompilation limit. The original callable implements
    identical AdamW math and is preferable for scientific screening.
    """
    try:
        import nanochat.optim as nc_optim

        fn = nc_optim.adamw_step_fused
        original = getattr(fn, "_torchdynamo_orig_callable", None)
        if original is not None:
            nc_optim.adamw_step_fused = original
            print("[build] NanoChat AdamW kernel: eager mode")
            return
        try:
            import torch._dynamo

            current = int(torch._dynamo.config.recompile_limit)
            torch._dynamo.config.recompile_limit = max(current, 128)
            print(f"[build] NanoChat AdamW recompile_limit={torch._dynamo.config.recompile_limit}")
        except Exception:
            pass
    except Exception as exc:
        print(f"[build] compile guard not applied: {type(exc).__name__}: {exc}")



# -----------------------------------------------------------------------------
# Canonical NanoChat model construction
# -----------------------------------------------------------------------------


def canonical_build(args: argparse.Namespace):
    """Construct NanoChat with the repository's depth-scaling head geometry.

    Earlier exploratory suites set ``n_embd = depth*64`` but accidentally left
    ``n_head`` at GPTConfig's default.  That made d6 internally fair but
    non-canonical and made d4/d8 invalid.  NanoChat's standard scaling keeps a
    128-dimensional attention head, so d4/d6/d8/d12 use 2/3/4/6 heads.

    The function is deliberately introspective so it works with both the user's
    current NanoChat checkout and recent upstream GPTConfig field names.
    """
    from nanochat.common import autodetect_device_type
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

    device_type = autodetect_device_type()
    device = torch.device(args.device if getattr(args, "device", None) else device_type)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else len(tokenizer)

    base_dim = int(args.depth) * int(args.aspect_ratio)
    head_dim = int(getattr(args, "head_dim", 128))
    # Mirror current NanoChat base_train: nudge width up to a clean head multiple.
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = model_dim // head_dim

    cfg_fields = set(GPTConfig.__dataclass_fields__)
    want = {
        "sequence_len": args.max_seq_len,
        "max_seq_len": args.max_seq_len,
        "vocab_size": vocab_size,
        "n_layer": args.depth,
        "depth": args.depth,
        "model_dim": model_dim,
        "n_embd": model_dim,
        "n_head": n_head,
        "num_heads": n_head,
        "n_kv_head": n_head,
        "num_kv_heads": n_head,
        "head_dim": head_dim,
        "window_pattern": getattr(args, "window_pattern", "SSSL"),
    }
    used = {k: v for k, v in want.items() if k in cfg_fields}
    cfg = GPTConfig(**used)
    print(f"[build] canonical GPTConfig fields used: {used}")
    model = GPT(cfg).to(device)
    n = sum(q.numel() for q in model.parameters())
    print(f"[build] canonical d{args.depth}: {n/1e6:.1f}M params, dim={model_dim}, heads={n_head}, head_dim={head_dim}")

    sig = inspect.signature(model.setup_optimizer)
    cand = {
        "embedding_lr": args.embedding_lr,
        "unembedding_lr": args.unembedding_lr,
        "matrix_lr": args.matrix_lr,
        "scalar_lr": args.scalar_lr,
        "weight_decay": args.weight_decay,
        "init_lr_frac": 1.0,
    }
    kwargs = {k: v for k, v in cand.items() if k in sig.parameters}
    print(f"[build] setup_optimizer{sig}")
    print(f"[build] optimizer kwargs: {kwargs}")
    optimizer = model.setup_optimizer(**kwargs)

    def mk(split: str):
        return tokenizing_distributed_data_loader_bos_bestfit(
            tokenizer, args.device_batch_size, args.max_seq_len, split=split, device=device
        )

    token_bytes = get_token_bytes()
    if hasattr(token_bytes, "to"):
        token_bytes = token_bytes.to(device)
    return model, optimizer, mk, token_bytes, device


def _install_canonical_build() -> None:
    """Make the existing pack/restore machinery use ``canonical_build``."""
    from . import skip_suite as _skip_suite
    _skip_suite._import_existing_build = lambda: canonical_build


# -----------------------------------------------------------------------------
# Detailed evaluation and paired statistics
# -----------------------------------------------------------------------------


@torch.no_grad()
def evaluate_detailed(model: torch.nn.Module,
                      eval_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
                      token_bytes: Any, device: torch.device) -> dict[str, Any]:
    from nanochat.loss_eval import evaluate_bpb

    was_training = model.training
    model.eval()
    bpbs: list[float] = []
    nlls: list[float] = []
    for x_cpu, y_cpu in eval_batches:
        x = x_cpu.to(device, non_blocking=True)
        y = y_cpu.to(device, non_blocking=True)
        loss = model(x, y)
        nlls.append(float(loss.mean() if loss.ndim else loss))
        bpbs.append(float(evaluate_bpb(model, iter([(x, y)]), 1, token_bytes)))
        del x, y, loss
    if was_training:
        model.train()

    def mean_se(values: list[float]) -> tuple[float, float]:
        a = np.asarray(values, dtype=float)
        se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
        return float(a.mean()), se

    bpb, bpb_se = mean_se(bpbs)
    nll, nll_se = mean_se(nlls)
    return {
        "bpb": bpb,
        "bpb_se": bpb_se,
        "nll": nll,
        "nll_se": nll_se,
        "bpb_values": bpbs,
        "nll_values": nlls,
        "n_eval_batches": len(bpbs),
    }


# -----------------------------------------------------------------------------
# FLOP accounting
# -----------------------------------------------------------------------------


@dataclass
class BudgetLedger:
    parameter_count: int
    full_step_tokens: int
    forward_tokens_charged: int = 0
    backward_tokens_charged: int = 0
    forward_tokens_actual: int = 0
    backward_tokens_actual: int = 0
    parameter_ops_charged: float = 0.0
    parameter_ops_actual: float = 0.0
    optimizer_steps: int = 0
    exact_backward_calls: int = 0
    synthetic_steps: int = 0
    data_microbatches: int = 0
    data_tokens: int = 0
    model_seconds: float = 0.0
    predictor_seconds: float = 0.0
    eval_seconds: float = 0.0

    @property
    def baseline_step_flops(self) -> float:
        return 6.0 * self.parameter_count * self.full_step_tokens

    @property
    def charged_flops(self) -> float:
        return (
            2.0 * self.parameter_count * self.forward_tokens_charged
            + 4.0 * self.parameter_count * self.backward_tokens_charged
            + self.parameter_ops_charged
        )

    @property
    def actual_flops(self) -> float:
        return (
            2.0 * self.parameter_count * self.forward_tokens_actual
            + 4.0 * self.parameter_count * self.backward_tokens_actual
            + self.parameter_ops_actual
        )

    @property
    def charged_flop_equiv(self) -> float:
        return self.charged_flops / max(self.baseline_step_flops, 1.0)

    @property
    def actual_flop_equiv(self) -> float:
        return self.actual_flops / max(self.baseline_step_flops, 1.0)

    @property
    def charged_seconds(self) -> float:
        return self.model_seconds + self.predictor_seconds

    def add_exact(self, tokens: int, microbatches: int, seconds: float,
                  *, charged: bool = True) -> None:
        self.forward_tokens_actual += tokens
        self.backward_tokens_actual += tokens
        if charged:
            self.forward_tokens_charged += tokens
            self.backward_tokens_charged += tokens
        self.data_microbatches += microbatches
        self.data_tokens += tokens
        self.model_seconds += seconds
        self.exact_backward_calls += 1

    def add_forward(self, tokens: int, microbatches: int, seconds: float,
                    *, charged: bool = True) -> None:
        self.forward_tokens_actual += tokens
        if charged:
            self.forward_tokens_charged += tokens
        self.data_microbatches += microbatches
        self.data_tokens += tokens
        self.model_seconds += seconds

    def add_parameter_ops(self, operations: float, *, charged: bool = True) -> None:
        self.parameter_ops_actual += float(operations)
        if charged:
            self.parameter_ops_charged += float(operations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "charged_flops": self.charged_flops,
            "actual_flops": self.actual_flops,
            "charged_flop_equiv": self.charged_flop_equiv,
            "actual_flop_equiv": self.actual_flop_equiv,
            "forward_tokens_charged": self.forward_tokens_charged,
            "backward_tokens_charged": self.backward_tokens_charged,
            "forward_tokens_actual": self.forward_tokens_actual,
            "backward_tokens_actual": self.backward_tokens_actual,
            "parameter_ops_charged": self.parameter_ops_charged,
            "parameter_ops_actual": self.parameter_ops_actual,
            "optimizer_steps": self.optimizer_steps,
            "exact_backward_calls": self.exact_backward_calls,
            "synthetic_steps": self.synthetic_steps,
            "data_microbatches": self.data_microbatches,
            "data_tokens": self.data_tokens,
            "model_seconds": self.model_seconds,
            "predictor_seconds": self.predictor_seconds,
            "eval_seconds": self.eval_seconds,
            "charged_seconds": self.charged_seconds,
        }


# Approximate non-model operation counts. These are intentionally conservative;
# measured wall time is reported separately and remains the final arbiter.
OPTIMIZER_OPS_PER_PARAMETER = 20.0
DIRECT_ASSIGN_OPS_PER_PARAMETER = 2.0


def history_combine_ops(history_len: int, numel: int) -> float:
    return 2.0 * max(history_len, 1) * numel


def coordinate_inference_ops(bank: PredictorBank, numel: int, kind: str) -> float:
    module = bank.modules[kind]
    params = sum(p.numel() for p in module.parameters())
    return 2.0 * params * numel


def predictor_inference_ops(base_arm: str, bank: PredictorBank,
                            history_len: int, numel: int) -> float:
    name, _ = parse_arm(base_arm)
    ops = 0.0
    if name in {"last", "momentum", "zero"}:
        return numel
    if name in {"mean", "ema", "two_ema", "ar", "dmd", "tensor_ar",
                "coeff", "coeff_resid", "coeff_raw", "coeff_forward",
                "random_span"}:
        ops += history_combine_ops(history_len, numel)
    if name.startswith("coord"):
        ops += coordinate_inference_ops(bank, numel, name)
    return ops


def predictor_training_ops(active: set[str], bank: PredictorBank,
                           coord_sample: int) -> float:
    total = 0.0
    for name in active:
        module = bank.modules[name]
        p = sum(q.numel() for q in module.parameters())
        multiplier = coord_sample if name.startswith("coord") else 1
        # Rough forward+backward+optimizer multiplier.
        total += 8.0 * p * multiplier
    return total


# -----------------------------------------------------------------------------
# Arm parsing and base aliases
# -----------------------------------------------------------------------------


BASE_ALIASES: dict[str, str] = {
    "zero": "zero",
    "last": "last",
    "mom": "momentum",
    "momentum": "momentum",
    "mean4": "mean:4",
    "mean8": "mean:8",
    "mean16": "mean:16",
    "ema08": "ema:0.8",
    "ema09": "ema:0.9",
    "ema095": "ema:0.95",
    "ema099": "ema:0.99",
    "twoema": "two_ema:0.8:0.98:1.0",
    "ar1": "ar:1",
    "ar2": "ar:2",
    "ar4": "ar:4",
    "ar8": "ar:8",
    "dmd2": "dmd:2",
    "dmd4": "dmd:4",
    "dmd8": "dmd:8",
    "tensorar2": "tensor_ar:2",
    "tensorar4": "tensor_ar:4",
    "coeff": "coeff",
    "coeffresid": "coeff_resid",
    "coeffraw": "coeff_raw",
    "coefffwd": "coeff_forward",
    "coord": "coord",
    "coordresid": "coord_resid",
    "coordraw": "coord_raw",
    "coordfwd": "coord_forward",
}

WRAPPERS = {"scale", "cal", "blend", "bias", "bias_span", "gate", "triage", "tgate", "clip", "ensemble"}
DIRECT_ARMS = {"lawa", "update_last", "update_mean", "update_ema", "update_ar", "update_dmd",
               "update_tensor_ar", "update_ar_block",
               "update_cal_ar", "update_cal_dmd", "update_gate_ar", "update_gate_dmd",
               "update_cal_gate_ar", "update_cal_gate_dmd",
               "filter_lawa", "oneshot_lawa", "reset_only", "noop_filter",
               "lookahead", "pswa", "arblock_filter"}
# eval_lawa trains the exact optimizer path and only changes the model used for instrumentation.
EXACT_ARMS = {"exact_full", "exact_lr", "exact_micro", "gd", "eval_lawa"}
ORACLE_ARMS = {"oracle_batch", "projected_oracle", "oracle_scale", "oracle_noise",
               "oracle_backward_free", "oracle_partial", "tensor_projected_oracle"}


def base_arm(alias: str) -> str:
    if alias not in BASE_ALIASES:
        raise ValueError(f"unknown base alias {alias!r}; known: {sorted(BASE_ALIASES)}")
    return BASE_ALIASES[alias]


def arm_bases(arm: str) -> list[str]:
    name, extra = parse_arm(arm)
    if name in WRAPPERS:
        if not extra:
            raise ValueError(f"{name} arm requires a base alias")
        aliases = extra[0].split("+") if name == "ensemble" else [extra[0]]
        return [base_arm(a) for a in aliases]
    if name == "micro_pred":
        if len(extra) < 2:
            raise ValueError("micro_pred grammar is micro_pred:<k>:<base_alias>:<rho>")
        return [base_arm(extra[1])]
    if name in EXACT_ARMS | DIRECT_ARMS | ORACLE_ARMS | {"micro", "micro_blend"}:
        return []
    return [arm]


def arm_active_predictors(arm: str) -> set[str]:
    active: set[str] = set()
    for base in arm_bases(arm):
        active |= active_predictors_for_arm(base)
    return active


def any_base_requires_forward(arm: str) -> bool:
    return any(arm_requires_forward(b) for b in arm_bases(arm))


def any_base_requires_raw(arm: str) -> bool:
    return any(arm_requires_raw(b) for b in arm_bases(arm))


def any_base_requires_momentum(arm: str) -> bool:
    return any(arm_requires_momentum(b) for b in arm_bases(arm))


def is_deployable_arm(arm: str) -> bool:
    return parse_arm(arm)[0] not in ORACLE_ARMS


def is_exact_arm(arm: str) -> bool:
    return parse_arm(arm)[0] in EXACT_ARMS


def is_direct_arm(arm: str) -> bool:
    return parse_arm(arm)[0] in DIRECT_ARMS


def is_history_ready(base: str, history: VectorHistory, bank: PredictorBank,
                     min_predictor_updates: int) -> bool:
    name, extra = parse_arm(base)
    W = len(history)
    if name in {"zero", "momentum"}:
        return W >= 1
    if name in {"last", "mean", "ema", "two_ema", "random_span", "random_full"}:
        return W >= 2
    if name == "ar":
        p = int(extra[0]) if extra else 2
        return W >= p + 1
    if name == "dmd":
        return W >= 3
    if name == "tensor_ar":
        p = int(extra[0]) if extra else 2
        return W >= p + 1
    if name.startswith("coeff"):
        return W >= 4 and bank.train_steps >= min_predictor_updates
    if name.startswith("coord"):
        return W >= bank.coord_order and bank.train_steps >= min_predictor_updates
    return W >= 2


def arm_ready(arm: str, history: VectorHistory, update_history: VectorHistory,
              bank: PredictorBank, weight_history: deque[torch.Tensor],
              min_predictor_updates: int,
              exact_update_history: Optional[VectorHistory] = None,
              exact_weight_history: Optional[deque[torch.Tensor]] = None,
              filter_state: Optional["FilterState"] = None) -> bool:
    name, extra = parse_arm(arm)
    if name in EXACT_ARMS | ORACLE_ARMS | {"micro", "micro_blend"}:
        return True
    if name == "micro_pred":
        return all(is_history_ready(b, history, bank, min_predictor_updates) for b in arm_bases(arm))
    if name == "lawa":
        window = int(extra[0]) if extra else 4
        return len(weight_history) >= window
    if name in {"filter_lawa", "oneshot_lawa"}:
        window = int(extra[0]) if extra else 4
        hist_kind = extra[3] if len(extra) > 3 else "all"
        spacing = int(extra[4]) if len(extra) > 4 else 1
        wh = exact_weight_history if hist_kind == "exact" and exact_weight_history is not None else weight_history
        return len(wh) >= 1 + (window - 1) * max(spacing, 1)
    if name in {"reset_only", "noop_filter"}:
        return True
    if name == "lookahead":
        k = int(extra[0]) if extra else 5
        return filter_state is not None and filter_state.exact_since_action >= k
    if name == "pswa":
        k = int(extra[0]) if extra else 8
        return (filter_state is not None and filter_state.exact_since_action >= k
                and exact_weight_history is not None and len(exact_weight_history) >= k)
    if name == "arblock_filter":
        order = int(extra[0]) if extra else 2
        hist_kind = extra[4] if len(extra) > 4 else "exact"
        uh = exact_update_history if hist_kind == "exact" and exact_update_history is not None else update_history
        return len(uh) >= order + 1
    if name.startswith("update_"):
        if name == "update_last":
            return len(update_history) >= 1
        if name in {"update_mean", "update_ema"}:
            return len(update_history) >= 2
        if name == "update_ar":
            order = int(extra[0]) if extra else 2
            return len(update_history) >= order + 1
        if name == "update_dmd":
            return len(update_history) >= 3
        if name == "update_tensor_ar":
            order = int(extra[0]) if extra else 2
            return len(update_history) >= order + 1
        if name == "update_ar_block":
            order = int(extra[0]) if extra else 2
            return len(update_history) >= order + 1
        if name in {"update_cal_ar", "update_gate_ar", "update_cal_gate_ar"}:
            order = int(extra[0]) if extra else 2
            return len(update_history) >= order + 1
        if name in {"update_cal_dmd", "update_gate_dmd", "update_cal_gate_dmd"}:
            return len(update_history) >= 3
    return all(is_history_ready(b, history, bank, min_predictor_updates) for b in arm_bases(arm))


# -----------------------------------------------------------------------------
# Online calibration, confidence, and tensor gating
# -----------------------------------------------------------------------------


@dataclass
class QualityTracker:
    spec: FlatSpec
    beta: float = 0.9
    count: int = 0
    alpha: float = 1.0
    cosine_ema: float = float("nan")
    relerr_ema: float = float("nan")
    bias: Optional[torch.Tensor] = None
    bias_span: Optional[torch.Tensor] = None
    tensor_cos: np.ndarray = field(init=False)
    tensor_alpha: np.ndarray = field(init=False)
    tensor_norm: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.spec.slices)
        self.tensor_cos = np.full(n, np.nan, dtype=np.float64)
        self.tensor_alpha = np.ones(n, dtype=np.float64)
        self.tensor_norm = np.full(n, np.nan, dtype=np.float64)

    @staticmethod
    def _ema(old: float, new: float, beta: float) -> float:
        return new if not math.isfinite(old) else beta * old + (1.0 - beta) * new

    @torch.no_grad()
    def update(self, pred: torch.Tensor, true: torch.Tensor, history: VectorHistory,
               *, need_bias: bool = False, need_bias_span: bool = False,
               beta: Optional[float] = None) -> dict[str, float]:
        b = self.beta if beta is None else beta
        p = pred.float()
        t = true.float()
        pp = float(torch.dot(p, p))
        alpha_now = float(torch.dot(p, t) / max(pp, 1e-30))
        alpha_now = float(np.clip(alpha_now, -1.0, 3.0))
        c = cosine(p, t)
        r = relative_error(p, t)
        self.alpha = self._ema(self.alpha, alpha_now, b)
        self.cosine_ema = self._ema(self.cosine_ema, c, b)
        self.relerr_ema = self._ema(self.relerr_ema, r, b)

        calibrated = self.alpha * p
        residual = t - calibrated
        if need_bias:
            if self.bias is None:
                self.bias = residual.clone()
            else:
                self.bias.mul_(b).add_(residual, alpha=1.0 - b)
        if need_bias_span:
            projected = history.project(residual)
            if self.bias_span is None:
                self.bias_span = projected.clone()
            else:
                self.bias_span.mul_(b).add_(projected, alpha=1.0 - b)

        for i, sl in enumerate(self.spec.slices):
            ps = p[sl.start:sl.stop]
            ts = t[sl.start:sl.stop]
            pn = float(ps.norm())
            tn = float(ts.norm())
            if pn > 0 and tn > 0:
                ci = float(torch.dot(ps, ts) / (ps.norm() * ts.norm()))
                ai = float(torch.dot(ps, ts) / max(float(torch.dot(ps, ps)), 1e-30))
                ai = float(np.clip(ai, -1.0, 3.0))
                self.tensor_cos[i] = ci if not math.isfinite(self.tensor_cos[i]) else (
                    b * self.tensor_cos[i] + (1.0 - b) * ci
                )
                self.tensor_alpha[i] = b * self.tensor_alpha[i] + (1.0 - b) * ai
            self.tensor_norm[i] = tn if not math.isfinite(self.tensor_norm[i]) else (
                b * self.tensor_norm[i] + (1.0 - b) * tn
            )
        self.count += 1
        return {"alpha": alpha_now, "cosine": c, "relerr": r}

    @torch.no_grad()
    def calibrated(self, pred: torch.Tensor) -> torch.Tensor:
        return pred * float(self.alpha)

    @torch.no_grad()
    def tensor_gate(self, pred: torch.Tensor, threshold: float,
                    fallback: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = pred.clone()
        for i, sl in enumerate(self.spec.slices):
            good = math.isfinite(self.tensor_cos[i]) and self.tensor_cos[i] >= threshold
            if not good:
                if fallback is None:
                    out[sl.start:sl.stop].zero_()
                else:
                    out[sl.start:sl.stop].copy_(fallback[sl.start:sl.stop])
            else:
                out[sl.start:sl.stop].mul_(float(self.tensor_alpha[i]))
        return out

    @torch.no_grad()
    def tensor_clip(self, pred: torch.Tensor, ratio: float) -> torch.Tensor:
        out = pred.clone()
        for i, sl in enumerate(self.spec.slices):
            target_norm = self.tensor_norm[i]
            if not math.isfinite(target_norm) or target_norm <= 0:
                continue
            part = out[sl.start:sl.stop]
            n = float(part.norm())
            cap = ratio * target_norm
            if n > cap > 0:
                part.mul_(cap / n)
        return out


# -----------------------------------------------------------------------------
# Candidate construction
# -----------------------------------------------------------------------------


def _safe_context(bank: PredictorBank, history: VectorHistory,
                  token_ids: Optional[torch.Tensor], ff: Optional[torch.Tensor],
                  step_frac: float, lr_mult: float) -> PredictorContext:
    if ff is not None:
        ff = torch.nan_to_num(ff.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-50, 50)
    return bank.make_context(history, token_ids, ff, step_frac, lr_mult)


def _base_candidate(base: str, history: VectorHistory, spec: FlatSpec,
                    bank: PredictorBank, momentum: torch.Tensor,
                    context: PredictorContext, generator: torch.Generator) -> torch.Tensor:
    pred = candidate_from_history(base, history, spec, bank, momentum, context, generator)
    if not torch.isfinite(pred).all():
        raise FloatingPointError(f"non-finite candidate from {base}")
    return pred


def wrapped_candidate(arm: str, history: VectorHistory, spec: FlatSpec,
                      bank: PredictorBank, momentum: torch.Tensor,
                      context: PredictorContext, generator: torch.Generator,
                      trackers: dict[str, QualityTracker]) -> tuple[torch.Tensor, float]:
    """Return candidate and rough non-model operation count."""
    name, extra = parse_arm(arm)
    if name not in WRAPPERS:
        pred = _base_candidate(arm, history, spec, bank, momentum, context, generator)
        return pred, predictor_inference_ops(arm, bank, len(history), spec.numel)

    if not extra:
        raise ValueError(f"wrapper {name} needs a base")
    aliases = extra[0].split("+") if name == "ensemble" else [extra[0]]
    bases = [base_arm(a) for a in aliases]
    preds = [_base_candidate(b, history, spec, bank, momentum, context, generator) for b in bases]
    ops = sum(predictor_inference_ops(b, bank, len(history), spec.numel) for b in bases)

    if name == "scale":
        alpha = float(extra[1])
        return preds[0] * alpha, ops + spec.numel
    if name == "cal":
        return trackers[bases[0]].calibrated(preds[0]), ops + spec.numel
    if name == "blend":
        rho = float(extra[1])
        p = trackers[bases[0]].calibrated(preds[0])
        return rho * p + (1.0 - rho) * momentum, ops + 3.0 * spec.numel
    if name == "bias":
        tr = trackers[bases[0]]
        p = tr.calibrated(preds[0])
        return p if tr.bias is None else p + tr.bias, ops + 2.0 * spec.numel
    if name == "bias_span":
        tr = trackers[bases[0]]
        p = tr.calibrated(preds[0])
        return p if tr.bias_span is None else p + tr.bias_span, ops + 2.0 * spec.numel
    if name in {"gate", "triage"}:
        # The exact/partial-backward decision is made before this function.
        return trackers[bases[0]].calibrated(preds[0]), ops + spec.numel
    if name == "tgate":
        threshold = float(extra[1])
        return trackers[bases[0]].tensor_gate(preds[0], threshold, fallback=momentum), ops + 2.0 * spec.numel
    if name == "clip":
        ratio = float(extra[1])
        p = trackers[bases[0]].calibrated(preds[0])
        return trackers[bases[0]].tensor_clip(p, ratio), ops + 2.0 * spec.numel
    if name == "ensemble":
        calibrated = [trackers[b].calibrated(p) for b, p in zip(bases, preds)]
        # Normalize each member to the median norm to stop one bad scale dominating.
        norms = torch.tensor([float(p.norm()) for p in calibrated], device=spec.device)
        target = float(norms.median().clamp_min(1e-30))
        out = torch.zeros_like(calibrated[0])
        for p, n in zip(calibrated, norms.tolist()):
            out.add_(p, alpha=1.0 / max(n, 1e-30) / len(calibrated) * target)
        return out, ops + 3.0 * len(calibrated) * spec.numel
    raise AssertionError(name)


def tracker_requirements(arm: str) -> tuple[bool, bool, float]:
    name, extra = parse_arm(arm)
    beta = float(extra[1]) if name in {"bias", "bias_span"} and len(extra) > 1 else 0.9
    return name == "bias", name == "bias_span", beta


def gate_allows_synthetic(arm: str, trackers: dict[str, QualityTracker],
                          min_quality_points: int) -> bool:
    name, extra = parse_arm(arm)
    if name != "gate":
        return True
    base = base_arm(extra[0])
    threshold = float(extra[1])
    tr = trackers[base]
    return tr.count >= min_quality_points and math.isfinite(tr.cosine_ema) and tr.cosine_ema >= threshold


def triage_decision(arm: str, trackers: dict[str, QualityTracker],
                    min_quality_points: int) -> tuple[str, int, float]:
    """Return (action, microbatches, micro_weight).

    ``action`` is one of ``exact``, ``micro``, or ``synthetic``.  The medium
    confidence action spends a small backward and blends that fresh estimate
    with the calibrated history predictor. ``micro_weight`` is the weight on
    the fresh microgradient.
    """
    name, extra = parse_arm(arm)
    if name != "triage":
        return "none", 0, 0.0
    if len(extra) < 5:
        raise ValueError("triage grammar is triage:<base>:<lo>:<hi>:<k>:<rho>")
    base = base_arm(extra[0])
    lo, hi = float(extra[1]), float(extra[2])
    k, rho = int(extra[3]), float(extra[4])
    if hi < lo:
        raise ValueError("triage high threshold must be >= low threshold")
    if k < 1:
        raise ValueError("triage microbatch count must be positive")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("triage rho must be in [0,1]")
    tr = trackers[base]
    if tr.count < min_quality_points or not math.isfinite(tr.cosine_ema):
        return "exact", k, rho
    if tr.cosine_ema >= hi:
        return "synthetic", k, rho
    if tr.cosine_ema >= lo:
        return "micro", k, rho
    return "exact", k, rho



# -----------------------------------------------------------------------------
# Next-wave representation/oracle helpers
# -----------------------------------------------------------------------------

@torch.no_grad()
def tensorwise_project(history: VectorHistory, target: torch.Tensor, spec: FlatSpec,
                       *, window: int = 8, ridge: float = 1e-4,
                       chunk: int = 500_000) -> tuple[torch.Tensor, float]:
    """Project ``target`` independently into each parameter tensor's history span.

    A global history span forces one coefficient vector to explain embeddings,
    attention, MLPs, norms, etc. This oracle instead fits independent coefficients
    per tensor. ``window`` caps the temporal rank so this diagnostic remains
    tractable on a GH200. Returns (projection, rough arithmetic operation count).
    """
    if len(history) < 1:
        raise RuntimeError("tensorwise projection needs nonempty history")
    rows = list(history._vectors)[-min(window, len(history)):]
    W = len(rows)
    out = torch.zeros(spec.numel, device=spec.device, dtype=torch.float32)
    ops = 0.0
    eye = torch.eye(W, device=spec.device, dtype=torch.float64)
    for sl in spec.slices:
        G = torch.zeros((W, W), device=spec.device, dtype=torch.float64)
        b = torch.zeros(W, device=spec.device, dtype=torch.float64)
        # First streaming pass: tensor-local Gram and target cross-products.
        for lo in range(sl.start, sl.stop, chunk):
            hi = min(lo + chunk, sl.stop)
            H = torch.stack([
                v[lo:hi].to(device=spec.device, dtype=torch.float32) for v in rows
            ], dim=0)
            tt = target[lo:hi].to(device=spec.device, dtype=torch.float32)
            G.add_((H @ H.T).double())
            b.add_((H @ tt).double())
            ops += 2.0 * W * W * (hi - lo) + 2.0 * W * (hi - lo)
            del H, tt
        scale = float(G.diagonal().mean().clamp_min(1e-30))
        c = torch.linalg.solve(G + ridge * scale * eye, b).float()
        # Second streaming pass: materialize only this tensor's projection.
        for lo in range(sl.start, sl.stop, chunk):
            hi = min(lo + chunk, sl.stop)
            H = torch.stack([
                v[lo:hi].to(device=spec.device, dtype=torch.float32) for v in rows
            ], dim=0)
            out[lo:hi].copy_(c @ H)
            ops += 2.0 * W * (hi - lo)
            del H
    return out, ops


def ar_block_coefficients(history: VectorHistory, order: int, horizon: int,
                          ridge: float = 1e-3) -> torch.Tensor:
    """Coefficients over the *current* history for the sum of h future AR updates.

    The AR coefficients are fit once at the anchor. Future updates are then rolled
    in coefficient space, so a K-step block nowcast only materializes one N-vector.
    """
    if horizon < 1:
        raise ValueError("block horizon must be positive")
    W = len(history)
    c1 = global_ar_coefficients(history, order, ridge)
    p = min(order, W - 1)
    # global_ar_coefficients stores a_j on history[W-j-1].
    a = torch.tensor([float(c1[W - j - 1]) for j in range(p)], dtype=torch.float64)
    recent: list[torch.Tensor] = []
    for j in range(p):
        e = torch.zeros(W, dtype=torch.float64)
        e[W - p + j] = 1.0
        recent.append(e)
    total = torch.zeros(W, dtype=torch.float64)
    for _ in range(horizon):
        nxt = torch.zeros(W, dtype=torch.float64)
        for j in range(p):
            nxt.add_(recent[-j - 1], alpha=float(a[j]))
        total.add_(nxt)
        recent.append(nxt)
    return total


@dataclass
class DirectQualityTracker:
    beta: float = 0.9
    count: int = 0
    alpha: float = 1.0
    cosine_ema: float = float("nan")
    relerr_ema: float = float("nan")

    @staticmethod
    def _ema(old: float, new: float, beta: float) -> float:
        return new if not math.isfinite(old) else beta * old + (1.0 - beta) * new

    @torch.no_grad()
    def update(self, pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
        p, t = pred.float(), true.float()
        pp = float(torch.dot(p, p))
        alpha_now = float(torch.dot(p, t) / max(pp, 1e-30))
        alpha_now = float(np.clip(alpha_now, -1.0, 3.0))
        c = cosine(p, t)
        r = relative_error(p, t)
        self.alpha = self._ema(self.alpha, alpha_now, self.beta)
        self.cosine_ema = self._ema(self.cosine_ema, c, self.beta)
        self.relerr_ema = self._ema(self.relerr_ema, r, self.beta)
        self.count += 1
        return {"alpha": alpha_now, "cosine": c, "relerr": r}


def direct_quality_spec(arm: str) -> tuple[bool, bool, float]:
    """Return (track_quality, calibrate, gate_threshold). NaN threshold means no gate."""
    name, extra = parse_arm(arm)
    if name in {"update_cal_ar", "update_cal_dmd"}:
        return True, True, float("nan")
    if name in {"update_gate_ar", "update_gate_dmd"}:
        return True, False, float(extra[1])
    if name in {"update_cal_gate_ar", "update_cal_gate_dmd"}:
        return True, True, float(extra[1])
    return False, False, float("nan")


def direct_quality_base_arm(arm: str) -> str:
    name, extra = parse_arm(arm)
    if name in {"update_cal_ar", "update_gate_ar", "update_cal_gate_ar"}:
        policy_idx = 2 if "gate" in name else 1
        policy = extra[policy_idx] if len(extra) > policy_idx else "preserve"
        return f"update_ar:{int(extra[0])}:{policy}"
    if name in {"update_cal_dmd", "update_gate_dmd", "update_cal_gate_dmd"}:
        policy_idx = 2 if "gate" in name else 1
        policy = extra[policy_idx] if len(extra) > policy_idx else "preserve"
        return f"update_dmd:{int(extra[0])}:{policy}"
    return arm

# -----------------------------------------------------------------------------
# Direct trajectory filters
# -----------------------------------------------------------------------------


@dataclass
class FilterState:
    exact_steps: int = 0
    exact_since_action: int = 0
    actions: int = 0
    oneshot_done: bool = False
    slow: Optional[torch.Tensor] = None


def _spaced_weights(weights: deque[torch.Tensor], window: int, spacing: int) -> list[torch.Tensor]:
    spacing = max(int(spacing), 1)
    rows = list(weights)
    idx = [len(rows) - 1 - j * spacing for j in range(window)]
    if not idx or min(idx) < 0:
        raise RuntimeError(
            f"need {1 + (window-1)*spacing} weight states for window={window}, spacing={spacing}; "
            f"have {len(rows)}"
        )
    return [rows[i] for i in reversed(idx)]


def _mean_weight_delta(weights: deque[torch.Tensor], current: torch.Tensor,
                       window: int, spacing: int = 1) -> torch.Tensor:
    selected = _spaced_weights(weights, window, spacing)
    mean = torch.zeros_like(current)
    inv = 1.0 / len(selected)
    for w in selected:
        mean.add_(w.to(device=current.device, dtype=current.dtype), alpha=inv)
    return mean - current


def _direct_policy(optimizer: torch.optim.Optimizer, spec: FlatSpec,
                   policy: str, *, before_delta: bool) -> float:
    """Apply optimizer-state policy; return rough parameter operations."""
    if policy == "coast" and before_delta:
        spec.assign_gradients(torch.zeros(spec.numel, device=spec.device))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return OPTIMIZER_OPS_PER_PARAMETER * spec.numel
    if before_delta:
        return 0.0
    if policy == "zero_m":
        clear_adam_state(optimizer, first_only=True)
    elif policy == "zero_all":
        clear_adam_state(optimizer, first_only=False)
    elif policy not in {"preserve", "coast"}:
        raise ValueError(f"unknown direct-update policy {policy}")
    return spec.numel if policy.startswith("zero") else 0.0


def direct_delta(arm: str, update_history: VectorHistory,
                 weight_history: deque[torch.Tensor], spec: FlatSpec,
                 optimizer: torch.optim.Optimizer,
                 *, exact_update_history: Optional[VectorHistory] = None,
                 exact_weight_history: Optional[deque[torch.Tensor]] = None,
                 filter_state: Optional[FilterState] = None) -> tuple[torch.Tensor, str, float]:
    """Return (parameter displacement, optimizer-state policy, rough ops)."""
    arm = direct_quality_base_arm(arm)
    name, extra = parse_arm(arm)
    current = spec.flatten_parameters(dtype=torch.float32)

    # Historical alias: reproduces the winning exploratory branch exactly.
    if name == "lawa":
        window = int(extra[0]) if extra else 4
        policy = extra[1] if len(extra) > 1 else "preserve"
        delta = _mean_weight_delta(weight_history, current, window, 1)
        return delta, policy, 2.0 * window * spec.numel

    if name == "filter_lawa":
        window = int(extra[0]) if extra else 4
        policy = extra[1] if len(extra) > 1 else "preserve"
        alpha = float(extra[2]) if len(extra) > 2 else 1.0
        hist_kind = extra[3] if len(extra) > 3 else "all"
        spacing = int(extra[4]) if len(extra) > 4 else 1
        wh = exact_weight_history if hist_kind == "exact" and exact_weight_history is not None else weight_history
        return alpha * _mean_weight_delta(wh, current, window, spacing), policy, 2.0 * window * spec.numel

    if name == "oneshot_lawa":
        window = int(extra[0]) if extra else 4
        policy = extra[1] if len(extra) > 1 else "preserve"
        alpha = float(extra[2]) if len(extra) > 2 else 1.0
        hist_kind = extra[3] if len(extra) > 3 else "exact"
        spacing = int(extra[4]) if len(extra) > 4 else 1
        wh = exact_weight_history if hist_kind == "exact" and exact_weight_history is not None else weight_history
        return alpha * _mean_weight_delta(wh, current, window, spacing), policy, 2.0 * window * spec.numel

    if name == "reset_only":
        policy = extra[0] if extra else "zero_m"
        return torch.zeros_like(current), policy, spec.numel

    if name == "noop_filter":
        return torch.zeros_like(current), "preserve", 0.0

    if name == "lookahead":
        # Standard Lookahead geometry: slow <- slow + alpha*(fast-slow); fast <- slow.
        # The `k` cadence is enforced by filter_planned().
        alpha = float(extra[1]) if len(extra) > 1 else 0.5
        policy = extra[2] if len(extra) > 2 else "preserve"
        if filter_state is None:
            raise RuntimeError("lookahead requires FilterState")
        if filter_state.slow is None:
            filter_state.slow = current.detach().clone()
        new_slow = filter_state.slow + alpha * (current - filter_state.slow)
        delta = new_slow - current
        filter_state.slow = new_slow.detach().clone()
        return delta, policy, 5.0 * spec.numel

    if name == "pswa":
        k = int(extra[0]) if extra else 8
        alpha = float(extra[1]) if len(extra) > 1 else 1.0
        policy = extra[2] if len(extra) > 2 else "preserve"
        if exact_weight_history is None:
            raise RuntimeError("pswa requires exact-weight history")
        selected = list(exact_weight_history)[-k:]
        if len(selected) < k:
            raise RuntimeError(f"pswa:{k} has only {len(selected)} exact states")
        mean = torch.zeros_like(current)
        for w in selected:
            mean.add_(w.to(device=current.device, dtype=current.dtype), alpha=1.0 / k)
        return alpha * (mean - current), policy, 2.0 * k * spec.numel

    if name == "arblock_filter":
        order = int(extra[0]) if extra else 2
        horizon = int(extra[1]) if len(extra) > 1 else 4
        policy = extra[2] if len(extra) > 2 else "zero_m"
        alpha = float(extra[3]) if len(extra) > 3 else 1.0
        hist_kind = extra[4] if len(extra) > 4 else "exact"
        uh = exact_update_history if hist_kind == "exact" and exact_update_history is not None else update_history
        c = ar_block_coefficients(uh, order, horizon)
        return alpha * uh.combine(c), policy, history_combine_ops(len(uh), spec.numel)

    # Existing direct-update controls retained as baselines.
    if name == "update_last":
        policy = extra[0] if extra else "preserve"
        return update_history.last(), policy, spec.numel
    if name == "update_mean":
        n = int(extra[0]) if extra else len(update_history)
        policy = extra[1] if len(extra) > 1 else "preserve"
        return update_history.mean(n), policy, history_combine_ops(min(n, len(update_history)), spec.numel)
    if name == "update_ema":
        beta = float(extra[0]) if extra else 0.9
        policy = extra[1] if len(extra) > 1 else "preserve"
        W = len(update_history)
        ages = torch.arange(W - 1, -1, -1, dtype=torch.float64)
        c = (1.0 - beta) * beta ** ages
        c /= c.sum().clamp_min(1e-30)
        return update_history.combine(c), policy, history_combine_ops(W, spec.numel)
    if name == "update_ar":
        order = int(extra[0]) if extra else 2
        policy = extra[1] if len(extra) > 1 else "preserve"
        c = global_ar_coefficients(update_history, order)
        return update_history.combine(c), policy, history_combine_ops(len(update_history), spec.numel)
    if name == "update_dmd":
        rank = int(extra[0]) if extra else 4
        policy = extra[1] if len(extra) > 1 else "preserve"
        c = dmd_coefficients(update_history, rank)
        return update_history.combine(c), policy, history_combine_ops(len(update_history), spec.numel)
    if name == "update_tensor_ar":
        order = int(extra[0]) if extra else 2
        policy = extra[1] if len(extra) > 1 else "preserve"
        pred = tensor_ar_prediction(update_history, spec, order)
        return pred, policy, history_combine_ops(len(update_history), spec.numel)
    if name == "update_ar_block":
        order = int(extra[0]) if extra else 2
        horizon = int(extra[1]) if len(extra) > 1 else 4
        policy = extra[2] if len(extra) > 2 else "preserve"
        c = ar_block_coefficients(update_history, order, horizon)
        return update_history.combine(c), policy, history_combine_ops(len(update_history), spec.numel)
    raise ValueError(arm)


def filter_planned(arm: str, ledger: BudgetLedger, runtime: argparse.Namespace,
                   state: FilterState) -> bool:
    """Whether this loop iteration should attempt a cheap trajectory action."""
    start = float(runtime.filter_start_equiv)
    stop = float(runtime.filter_stop_equiv)
    if not (ledger.charged_flop_equiv >= start and ledger.charged_flop_equiv < stop):
        return False
    name, extra = parse_arm(arm)
    if name in EXACT_ARMS:
        return False
    if name == "oneshot_lawa" and state.oneshot_done:
        return False
    if name in {"lookahead", "pswa"}:
        k = int(extra[0]) if extra else (5 if name == "lookahead" else 8)
        return state.exact_since_action >= k
    return _planned_synthetic(arm, ledger.optimizer_steps, runtime.skip_period, runtime.skip_count)


def _proposal_diagnostics(update_history: VectorHistory, weight_history: deque[torch.Tensor],
                          exact_update_history: VectorHistory,
                          exact_weight_history: deque[torch.Tensor],
                          spec: FlatSpec) -> dict[str, float]:
    """Compare the two leading cheap directions without changing the model."""
    out: dict[str, float] = {}
    current = spec.flatten_parameters(dtype=torch.float32)
    try:
        lawa = _mean_weight_delta(weight_history, current, 4, 1)
        out["lawa_norm"] = float(lawa.norm())
    except Exception:
        lawa = None
    try:
        c = ar_block_coefficients(exact_update_history, 2, 4)
        ar = exact_update_history.combine(c)
        out["arblock_norm"] = float(ar.norm())
    except Exception:
        ar = None
    if lawa is not None and ar is not None:
        out["arblock_lawa_cosine"] = cosine(ar, lawa)
        out["arblock_to_lawa_norm"] = float(ar.norm() / lawa.norm().clamp_min(1e-30))
    return out


# -----------------------------------------------------------------------------
# Branch execution
# -----------------------------------------------------------------------------


def _micro_tokens(args: argparse.Namespace) -> int:
    return int(args.device_batch_size) * int(args.max_seq_len)


def _schedule_index(clock: str, pack_prefix: int, ledger: BudgetLedger,
                    full_step_tokens: int) -> int:
    if clock == "updates":
        return pack_prefix + ledger.optimizer_steps
    if clock == "flops":
        return pack_prefix + int(math.floor(ledger.charged_flop_equiv))
    if clock == "tokens":
        return pack_prefix + int(ledger.data_tokens // max(full_step_tokens, 1))
    raise ValueError(clock)


def _apply_lr(optimizer: torch.optim.Optimizer, schedule_step: int,
              args: argparse.Namespace, lr_scale: float) -> float:
    mult = apply_lr_schedule(optimizer, schedule_step, args.warmup_steps,
                             args.total_iterations, args.warmdown_ratio,
                             args.final_lr_frac)
    for group in optimizer.param_groups:
        group["lr"] *= lr_scale
    return mult * lr_scale


def _arm_lr_scale(arm: str, default: float) -> float:
    name, extra = parse_arm(arm)
    if name == "exact_lr":
        return float(extra[0])
    if name == "exact_micro" and len(extra) > 1:
        return float(extra[1])
    return default


def _exact_micro_count(arm: str, full: int) -> int:
    name, extra = parse_arm(arm)
    if name == "exact_micro":
        return int(extra[0])
    return full


def _planned_synthetic(arm: str, update_index: int, period: int, count: int) -> bool:
    if is_exact_arm(arm):
        return False
    if count <= 0:
        return False
    return update_index % period >= period - count


def _candidate_context(arm: str, bank: PredictorBank, history: VectorHistory,
                       batches: Optional[Sequence[tuple[torch.Tensor, torch.Tensor]]],
                       model: torch.nn.Module, device: torch.device,
                       step_frac: float, lr_mult: float, ledger: BudgetLedger,
                       token_sample_max: int, micro_tokens: int) -> PredictorContext:
    tok: Optional[torch.Tensor] = None
    ff: Optional[torch.Tensor] = None
    if any_base_requires_raw(arm):
        if batches is None:
            raise RuntimeError("raw-conditioned predictor requires batches")
        tok = token_sample_from_batches(batches, token_sample_max).to(device)
    if any_base_requires_forward(arm):
        if batches is None:
            raise RuntimeError("forward-conditioned predictor requires batches")
        with Timer() as ft:
            _, tok_ff, ff = forward_only_context(model, batches, device,
                                                  token_sample_max=token_sample_max)
        tokens = len(batches) * micro_tokens
        ledger.add_forward(tokens, len(batches), ft.seconds, charged=True)
        tok = tok if tok is not None else tok_ff
    with Timer() as pt:
        ctx = _safe_context(bank, history, tok, ff, step_frac, lr_mult)
    ledger.predictor_seconds += pt.seconds
    return ctx


def _init_update_history(pack: dict[str, Any], args: argparse.Namespace,
                         spec: FlatSpec, device: torch.device) -> tuple[VectorHistory, deque[torch.Tensor]]:
    hdev = torch.device(args.history_device if args.history_device != "cuda" else device)
    dtype = _dtype_from_name(args.history_dtype)
    updates = VectorHistory(args.history, spec.numel, device, hdev, dtype)
    weights = deque(pack.get("weight_history", []), maxlen=max(args.history + 2, int(getattr(args, "lawa_window", 34)), 128))
    if len(weights) >= 2:
        ws = list(weights)
        for a, b in zip(ws[:-1], ws[1:]):
            updates.append(b.to(device=device, dtype=torch.float32) - a.to(device=device, dtype=torch.float32))
    return updates, weights


def run_budget_branch(runtime: argparse.Namespace) -> Path:
    _stabilize_nanochat_adamw_kernel()
    pack = torch.load(runtime.pack, map_location="cpu", weights_only=False)
    if int(pack.get("pack_version", -1)) != PACK_VERSION:
        raise RuntimeError(f"pack version mismatch: {pack.get('pack_version')} != {PACK_VERSION}")

    args, model, optimizer, token_bytes, device, spec, history, bank = restore_branch(pack, runtime)
    arm = runtime.arm
    name, extra = parse_arm(arm)
    seed = int(pack.get("seed", 1337)) if runtime.seed is None else runtime.seed
    seed_everything(seed)
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed(seed + 99173)

    micro_tokens = _micro_tokens(args)
    full_step_tokens = micro_tokens * int(args.grad_accum)
    ledger = BudgetLedger(spec.numel, full_step_tokens)
    eval_batches = pack["eval_batches"]
    stream: list[tuple[torch.Tensor, torch.Tensor]] = pack["branch_batches"]
    cursor = 0
    trace: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    update_history, weight_history = _init_update_history(pack, args, spec, device)
    exact_update_history, exact_weight_history = _init_update_history(pack, args, spec, device)
    if not weight_history:
        w0 = spec.flatten_parameters(dtype=torch.float32).cpu().to(_dtype_from_name(args.history_dtype))
        weight_history.append(w0)
        exact_weight_history.append(w0.clone())
    filter_state = FilterState()
    filter_state.slow = spec.flatten_parameters(dtype=torch.float32).detach().clone()

    active_predictors = arm_active_predictors(arm)
    bases = arm_bases(arm)
    wrapper_name = parse_arm(arm)[0]
    tracking_bases = bases if wrapper_name in WRAPPERS else []
    trackers = {b: QualityTracker(spec, beta=runtime.quality_beta) for b in tracking_bases}
    need_bias, need_bias_span, bias_beta = tracker_requirements(arm)
    dq_track, dq_calibrate, dq_threshold = direct_quality_spec(arm)
    direct_tracker = DirectQualityTracker(beta=runtime.quality_beta) if dq_track else None

    def take_batches(k: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        nonlocal cursor
        if k <= 0:
            return []
        if cursor + k > len(stream):
            raise EOFError(f"branch microbatch stream exhausted at {cursor}/{len(stream)}")
        out = stream[cursor:cursor + k]
        cursor += k
        return out

    def record() -> None:
        # Evaluation-only LAWA is the standard checkpoint-averaging control: the
        # live training trajectory stays exact, we only swap averaged parameters
        # in for instrumentation and immediately restore the fast weights.
        if name == "eval_lawa":
            window = int(extra[0]) if extra else 4
            spacing = int(extra[1]) if len(extra) > 1 else 1
            current = spec.flatten_parameters(dtype=torch.float32)
            with Timer() as et:
                fast_ev = evaluate_detailed(model, eval_batches, token_bytes, device)
                try:
                    delta = _mean_weight_delta(exact_weight_history, current, window, spacing)
                    spec.assign_parameters(current + delta)
                    ev = evaluate_detailed(model, eval_batches, token_bytes, device)
                finally:
                    spec.assign_parameters(current)
            ev["fast_bpb"] = fast_ev["bpb"]
            ev["fast_bpb_se"] = fast_ev["bpb_se"]
            ev["fast_bpb_values"] = fast_ev["bpb_values"]
        else:
            with Timer() as et:
                ev = evaluate_detailed(model, eval_batches, token_bytes, device)
        ledger.eval_seconds += et.seconds
        row = {
            "optimizer_step": ledger.optimizer_steps,
            "schedule_step": _schedule_index(runtime.schedule_clock, int(pack["prefix_steps"]),
                                              ledger, full_step_tokens),
            "filter_active": bool(runtime.filter_start_equiv <= ledger.charged_flop_equiv < runtime.filter_stop_equiv),
            "filter_actions": filter_state.actions,
            **ev,
            **ledger.as_dict(),
        }
        trace.append(row)
        print(
            f"[{arm}] upd={ledger.optimizer_steps:5d} "
            f"F={ledger.charged_flop_equiv:8.2f} "
            f"actualF={ledger.actual_flop_equiv:8.2f} "
            f"bw={ledger.exact_backward_calls:5d} "
            f"tok={ledger.data_tokens/1e6:8.2f}M "
            f"bpb={ev['bpb']:.6f}+/-{ev['bpb_se']:.6f}"
        )

    record()
    next_eval = runtime.eval_every_equiv
    termination = "budget"

    while ledger.charged_flop_equiv < runtime.compute_budget:
        if ledger.optimizer_steps >= runtime.max_updates:
            termination = "max_updates"
            break

        synthetic_window_open = (
            ledger.charged_flop_equiv >= runtime.filter_start_equiv
            and ledger.charged_flop_equiv < runtime.filter_stop_equiv
        )
        planned = filter_planned(arm, ledger, runtime, filter_state)
        ready = arm_ready(
            arm, history, update_history, bank, weight_history,
            runtime.min_predictor_updates,
            exact_update_history=exact_update_history,
            exact_weight_history=exact_weight_history,
            filter_state=filter_state,
        )
        triage_action, triage_k, triage_rho = triage_decision(
            arm, trackers, runtime.min_quality_points
        )
        triage_micro = planned and ready and triage_action == "micro"
        synthetic = planned and ready and triage_action != "exact"
        if triage_action == "micro":
            synthetic = False
        if triage_action == "synthetic":
            synthetic = planned and ready
        if triage_action == "none":
            synthetic = planned and ready
        if synthetic and not gate_allows_synthetic(arm, trackers, runtime.min_quality_points):
            synthetic = False
        if synthetic and direct_tracker is not None and math.isfinite(dq_threshold):
            if (direct_tracker.count < runtime.min_quality_points
                    or not math.isfinite(direct_tracker.cosine_ema)
                    or direct_tracker.cosine_ema < dq_threshold):
                synthetic = False

        # Stop cleanly when the deterministic cached stream cannot fund the next
        # data-bearing action. History-only synthetic updates require no data.
        required_micro = 0
        if triage_micro:
            required_micro = triage_k
        elif not synthetic:
            required_micro = _exact_micro_count(arm, int(args.grad_accum))
        elif name in ORACLE_ARMS:
            required_micro = int(args.grad_accum)
        elif name in {"micro", "micro_blend"}:
            required_micro = int(extra[0]) if extra else 1
        elif name == "micro_pred":
            required_micro = int(extra[0]) if extra else 1
        elif not is_direct_arm(arm) and (any_base_requires_raw(arm) or any_base_requires_forward(arm)):
            required_micro = int(args.grad_accum)
        if cursor + required_micro > len(stream):
            termination = "stream_exhausted"
            break

        schedule_step = _schedule_index(runtime.schedule_clock, int(pack["prefix_steps"]),
                                        ledger, full_step_tokens)
        lr_scale = _arm_lr_scale(arm, runtime.lr_scale)
        lr_mult = _apply_lr(optimizer, schedule_step, args, lr_scale)
        step_frac = schedule_step / max(int(args.total_iterations), 1)
        event: dict[str, Any] = {
            "optimizer_step": ledger.optimizer_steps + 1,
            "planned_synthetic": planned,
            "synthetic_window_open": synthetic_window_open,
            "synthetic": synthetic,
            "triage_micro": triage_micro,
            "triage_action": triage_action,
            "ready": ready,
            "schedule_step": schedule_step,
            "lr_mult": lr_mult,
        }

        # Snapshot only when update-history/direct controls need the parameter delta.
        need_update_delta = runtime.track_update_history or is_direct_arm(arm) or name == "eval_lawa"
        theta_before = spec.flatten_parameters(dtype=torch.float32) if need_update_delta else None
        direct_pred_before = None
        if direct_tracker is not None and not synthetic and ready:
            try:
                with Timer() as dqt:
                    direct_pred_before, _, dq_ops = direct_delta(
                        arm, update_history, weight_history, spec, optimizer,
                        exact_update_history=exact_update_history,
                        exact_weight_history=exact_weight_history,
                        filter_state=filter_state,
                    )
                ledger.predictor_seconds += dqt.seconds
                ledger.add_parameter_ops(dq_ops, charged=True)
            except Exception as exc:
                event["direct_quality_pred_error"] = f"{type(exc).__name__}: {exc}"
                direct_pred_before = None

        try:
            if triage_micro:
                if triage_k > int(args.grad_accum):
                    raise ValueError(
                        f"triage microbatch count {triage_k} exceeds grad_accum={args.grad_accum}"
                    )
                batches = take_batches(triage_k)
                momentum = optimizer_first_moment(optimizer, spec)
                need_ff = any_base_requires_forward(arm)
                with Timer() as tt:
                    mg, train_loss, tok, ff = exact_gradient(
                        model, optimizer, spec, batches, device,
                        need_forward_features=need_ff,
                        token_sample_max=args.token_sample_max,
                    )
                ledger.add_exact(triage_k * micro_tokens, triage_k, tt.seconds, charged=True)
                ctx = _safe_context(bank, history, tok, ff, step_frac, lr_mult)
                with Timer() as pt:
                    pred, ops = wrapped_candidate(
                        arm, history, spec, bank, momentum, ctx, gen, trackers
                    )
                    cand = triage_rho * mg + (1.0 - triage_rho) * pred
                ledger.predictor_seconds += pt.seconds
                ledger.add_parameter_ops(ops + 3.0 * spec.numel, charged=True)
                if not torch.isfinite(cand).all():
                    raise FloatingPointError("triage candidate became non-finite")
                spec.assign_gradients(cand)
                with Timer() as ot:
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                ledger.model_seconds += ot.seconds
                ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
                if runtime.synthetic_history == "predicted":
                    history.append(cand)
                elif runtime.synthetic_history == "zero":
                    history.append(torch.zeros_like(cand))
                event.update({
                    "train_loss": train_loss,
                    "microbatches": triage_k,
                    "micro_weight": triage_rho,
                    "candidate_norm": float(cand.norm()),
                })

            elif not synthetic:
                k = _exact_micro_count(arm, int(args.grad_accum))
                if not 1 <= k <= int(args.grad_accum):
                    raise ValueError(f"exact microbatch count {k} outside [1,{args.grad_accum}]")
                batches = take_batches(k)
                need_ff = any_base_requires_forward(arm)
                momentum = optimizer_first_moment(optimizer, spec) if (
                    any_base_requires_momentum(arm) or bool(active_predictors) or bool(bases)
                ) else torch.zeros(spec.numel, device=device)
                with Timer() as tt:
                    g, train_loss, tok, ff = exact_gradient(
                        model, optimizer, spec, batches, device,
                        need_forward_features=need_ff,
                        token_sample_max=args.token_sample_max,
                    )
                ledger.add_exact(k * micro_tokens, k, tt.seconds, charged=True)

                # Evaluate candidate quality BEFORE training the predictor on this target.
                if tracking_bases and len(history) >= 2:
                    ctx = _safe_context(bank, history, tok, ff, step_frac, lr_mult)
                    for b in tracking_bases:
                        if not is_history_ready(b, history, bank, 0):
                            continue
                        try:
                            this_bias = need_bias and b == tracking_bases[0]
                            this_bias_span = need_bias_span and b == tracking_bases[0]
                            with Timer() as pt:
                                pred = _base_candidate(b, history, spec, bank, momentum, ctx, gen)
                                metrics = trackers[b].update(
                                    pred, g, history,
                                    need_bias=this_bias,
                                    need_bias_span=this_bias_span,
                                    beta=bias_beta,
                                )
                            ledger.predictor_seconds += pt.seconds
                            qops = predictor_inference_ops(b, bank, len(history), spec.numel)
                            qops += 12.0 * spec.numel  # cosine, scale, residual, tensor statistics
                            if this_bias_span:
                                qops += history_combine_ops(len(history), spec.numel)
                            ledger.add_parameter_ops(qops, charged=True)
                            event[f"quality_{b}"] = metrics
                        except Exception as exc:
                            event[f"quality_error_{b}"] = f"{type(exc).__name__}: {exc}"

                if active_predictors and len(history) >= max(3, bank.coord_order):
                    ff_train = None if ff is None else torch.nan_to_num(
                        ff.float(), nan=0.0, posinf=20.0, neginf=-20.0
                    ).clamp(-50, 50)
                    with Timer() as pt:
                        train_metrics = bank.observe_true_gradient(
                            history, g, momentum, tok, ff_train, step_frac, lr_mult,
                            active=active_predictors,
                            inner_steps=runtime.predictor_inner_steps,
                        )
                    ledger.predictor_seconds += pt.seconds
                    ledger.add_parameter_ops(
                        predictor_training_ops(active_predictors, bank, args.coord_sample), charged=True
                    )
                    event["predictor_train"] = train_metrics

                with Timer() as ot:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                ledger.model_seconds += ot.seconds
                ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
                history.append(g)
                event.update({"train_loss": train_loss, "gradient_norm": float(g.norm()),
                              "microbatches": k})

            elif name in ORACLE_ARMS:
                # Oracle frontiers: hidden exact computation is always recorded in actualF.
                # The charged budget reflects the hypothetical information available to the
                # deployable method represented by each oracle.
                batches = take_batches(int(args.grad_accum))
                momentum = optimizer_first_moment(optimizer, spec)
                with Timer() as tt:
                    true_g, train_loss, tok, ff = exact_gradient(
                        model, optimizer, spec, batches, device,
                        need_forward_features=False,
                        token_sample_max=args.token_sample_max,
                    )
                ledger.add_exact(full_step_tokens, int(args.grad_accum), tt.seconds, charged=False)
                projection_ops = 0.0
                if name == "oracle_batch":
                    cand = true_g
                elif name == "oracle_backward_free":
                    # Perfect backward imputation after paying only for the base-model forward.
                    ledger.forward_tokens_charged += full_step_tokens
                    cand = true_g
                elif name == "oracle_partial":
                    k = int(extra[0]) if extra else 1
                    if not 1 <= k <= int(args.grad_accum):
                        raise ValueError(f"oracle_partial k={k} outside [1,{args.grad_accum}]")
                    ledger.forward_tokens_charged += k * micro_tokens
                    ledger.backward_tokens_charged += k * micro_tokens
                    cand = true_g
                    event["oracle_charged_microbatches"] = k
                elif name == "projected_oracle":
                    cand = history.project(true_g, args.projection_ridge)
                    projection_ops = history_combine_ops(len(history), spec.numel)
                elif name == "tensor_projected_oracle":
                    window = int(extra[0]) if extra else min(8, len(history))
                    cand, projection_ops = tensorwise_project(
                        history, true_g, spec, window=window, ridge=args.projection_ridge
                    )
                    event["tensor_projection_window"] = window
                elif name == "oracle_scale":
                    cand = float(extra[0]) * true_g
                else:
                    rel = float(extra[0])
                    mode = extra[1] if len(extra) > 1 else "iid"
                    if mode == "span":
                        coeff = torch.randn(len(history), generator=gen, device=device).double().cpu()
                        noise = history.combine(coeff)
                    else:
                        noise = torch.randn(spec.numel, generator=gen, device=device)
                    noise = noise / noise.norm().clamp_min(1e-30) * (rel * true_g.norm())
                    cand = true_g + noise
                if projection_ops:
                    ledger.add_parameter_ops(projection_ops, charged=True)
                spec.assign_gradients(cand)
                with Timer() as ot:
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                ledger.model_seconds += ot.seconds
                ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
                history.append(cand if runtime.synthetic_history == "predicted" else true_g)
                event.update({"train_loss": train_loss, "candidate_norm": float(cand.norm()),
                              "cos_to_oracle": cosine(cand, true_g),
                              "relerr_to_oracle": relative_error(cand, true_g)})

            elif name == "micro_pred":
                if len(extra) < 3:
                    raise ValueError("micro_pred grammar is micro_pred:<k>:<base_alias>:<rho>")
                k = int(extra[0])
                base = base_arm(extra[1])
                rho = float(extra[2])
                if not 1 <= k <= int(args.grad_accum):
                    raise ValueError(f"micro_pred k={k} outside [1,{args.grad_accum}]")
                if not 0.0 <= rho <= 1.0:
                    raise ValueError("micro_pred rho must be in [0,1]")
                batches = take_batches(k)
                momentum = optimizer_first_moment(optimizer, spec)
                need_ff = arm_requires_forward(base)
                with Timer() as tt:
                    mg, train_loss, tok, ff = exact_gradient(
                        model, optimizer, spec, batches, device,
                        need_forward_features=need_ff,
                        token_sample_max=args.token_sample_max,
                    )
                ledger.add_exact(k * micro_tokens, k, tt.seconds, charged=True)
                ctx = _safe_context(bank, history, tok, ff, step_frac, lr_mult)
                with Timer() as pt:
                    pred = _base_candidate(base, history, spec, bank, momentum, ctx, gen)
                    cand = rho * mg + (1.0 - rho) * pred
                ledger.predictor_seconds += pt.seconds
                ledger.add_parameter_ops(
                    predictor_inference_ops(base, bank, len(history), spec.numel)
                    + 3.0 * spec.numel, charged=True
                )
                if not torch.isfinite(cand).all():
                    raise FloatingPointError("micro_pred candidate became non-finite")
                spec.assign_gradients(cand)
                with Timer() as ot:
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                ledger.model_seconds += ot.seconds
                ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
                history.append(cand if runtime.synthetic_history == "predicted" else mg)
                event.update({"train_loss": train_loss, "microbatches": k,
                              "micro_weight": rho, "base": base,
                              "candidate_norm": float(cand.norm())})

            elif name in {"micro", "micro_blend"}:
                k = int(extra[0]) if extra else 1
                batches = take_batches(k)
                momentum = optimizer_first_moment(optimizer, spec)
                with Timer() as tt:
                    mg, train_loss, tok, ff = exact_gradient(
                        model, optimizer, spec, batches, device,
                        need_forward_features=False,
                        token_sample_max=args.token_sample_max,
                    )
                ledger.add_exact(k * micro_tokens, k, tt.seconds, charged=True)
                if name == "micro_blend":
                    rho = float(extra[1]) if len(extra) > 1 else 0.5
                    cand = rho * mg + (1.0 - rho) * momentum
                else:
                    cand = mg
                spec.assign_gradients(cand)
                with Timer() as ot:
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                ledger.model_seconds += ot.seconds
                ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
                history.append(cand if runtime.synthetic_history == "predicted" else mg)
                event.update({"train_loss": train_loss, "microbatches": k,
                              "candidate_norm": float(cand.norm())})

            elif is_direct_arm(arm):
                with Timer() as pt:
                    delta, policy, ops = direct_delta(
                        arm, update_history, weight_history, spec, optimizer,
                        exact_update_history=exact_update_history,
                        exact_weight_history=exact_weight_history,
                        filter_state=filter_state,
                    )
                    event["proposal_comparison"] = _proposal_diagnostics(
                        update_history, weight_history, exact_update_history, exact_weight_history, spec
                    )
                    if direct_tracker is not None and dq_calibrate and direct_tracker.count > 0:
                        delta = delta * float(direct_tracker.alpha)
                    ops += _direct_policy(optimizer, spec, policy, before_delta=True)
                    current = spec.flatten_parameters(dtype=torch.float32)
                    spec.assign_parameters(current + delta)
                    ops += DIRECT_ASSIGN_OPS_PER_PARAMETER * spec.numel
                    ops += _direct_policy(optimizer, spec, policy, before_delta=False)
                ledger.predictor_seconds += pt.seconds
                ledger.add_parameter_ops(ops, charged=True)
                event.update({"direct_delta_norm": float(delta.norm()), "state_policy": policy})

            else:
                # Fixed or wrapped synthetic gradient.
                needs_data = any_base_requires_raw(arm) or any_base_requires_forward(arm)
                batches = take_batches(int(args.grad_accum)) if needs_data else None
                momentum = optimizer_first_moment(optimizer, spec) if (
                    any_base_requires_momentum(arm) or parse_arm(arm)[0] in WRAPPERS
                ) else torch.zeros(spec.numel, device=device)
                ctx = _candidate_context(
                    arm, bank, history, batches, model, device, step_frac, lr_mult,
                    ledger, args.token_sample_max, micro_tokens,
                )
                # Raw-only methods consume data even though the base model does not.
                if any_base_requires_raw(arm) and not any_base_requires_forward(arm):
                    ledger.data_microbatches += int(args.grad_accum)
                    ledger.data_tokens += full_step_tokens
                with Timer() as pt:
                    cand, ops = wrapped_candidate(
                        arm, history, spec, bank, momentum, ctx, gen, trackers
                    )
                ledger.predictor_seconds += pt.seconds
                ledger.add_parameter_ops(ops, charged=True)
                if not torch.isfinite(cand).all():
                    raise FloatingPointError("synthetic candidate became non-finite")
                spec.assign_gradients(cand)
                with Timer() as ot:
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                ledger.model_seconds += ot.seconds
                ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
                if runtime.synthetic_history == "predicted":
                    history.append(cand)
                elif runtime.synthetic_history == "zero":
                    history.append(torch.zeros_like(cand))
                elif runtime.synthetic_history != "hold":
                    raise ValueError(runtime.synthetic_history)
                event.update({
                    "candidate_norm": float(cand.norm()),
                    "candidate_cos_last": cosine(cand, history.last()) if len(history) else float("nan"),
                })

        except (FloatingPointError, RuntimeError) as exc:
            if not runtime.fallback_exact:
                raise
            # Safe failure mode: spend the backward rather than poison the run.
            event["fallback_exact"] = f"{type(exc).__name__}: {exc}"
            k = int(args.grad_accum)
            batches = take_batches(k)
            with Timer() as tt:
                g, train_loss, tok, ff = exact_gradient(
                    model, optimizer, spec, batches, device,
                    need_forward_features=any_base_requires_forward(arm),
                    token_sample_max=args.token_sample_max,
                )
            ledger.add_exact(full_step_tokens, k, tt.seconds, charged=True)
            with Timer() as ot:
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            ledger.model_seconds += ot.seconds
            ledger.add_parameter_ops(OPTIMIZER_OPS_PER_PARAMETER * spec.numel, charged=True)
            history.append(g)
            synthetic = False

        ledger.optimizer_steps += 1
        if synthetic:
            ledger.synthetic_steps += 1
            filter_state.actions += 1
            if name == "oneshot_lawa":
                filter_state.oneshot_done = True
            if name in {"lookahead", "pswa"}:
                filter_state.exact_since_action = 0
        else:
            filter_state.exact_steps += 1
            filter_state.exact_since_action += 1

        if need_update_delta:
            assert theta_before is not None
            theta_after = spec.flatten_parameters(dtype=torch.float32)
            true_delta = theta_after - theta_before
            if direct_tracker is not None and not synthetic and direct_pred_before is not None:
                event["direct_quality"] = direct_tracker.update(direct_pred_before, true_delta)
                ledger.add_parameter_ops(8.0 * spec.numel, charged=True)
            # Keep the historical block-nowcast baseline's teacher history exact-only.
            if not (synthetic and name == "update_ar_block"):
                update_history.append(true_delta)
            else:
                event["update_history_policy"] = "hold_after_block_nowcast"
            w_after = theta_after.cpu().to(_dtype_from_name(args.history_dtype))
            weight_history.append(w_after)
            if not synthetic:
                exact_update_history.append(true_delta)
                exact_weight_history.append(w_after.clone())
        elif runtime.track_weight_history:
            w_after = spec.flatten_parameters(dtype=torch.float32).cpu().to(
                _dtype_from_name(args.history_dtype)
            )
            weight_history.append(w_after)
            if not synthetic:
                exact_weight_history.append(w_after.clone())

        event.update({
            "charged_flop_equiv": ledger.charged_flop_equiv,
            "actual_flop_equiv": ledger.actual_flop_equiv,
            "data_tokens": ledger.data_tokens,
        })
        for b, tr in trackers.items():
            event[f"tracker_{b}"] = {
                "count": tr.count,
                "alpha": tr.alpha,
                "cosine_ema": tr.cosine_ema,
                "relerr_ema": tr.relerr_ema,
            }
        if direct_tracker is not None:
            event["direct_tracker"] = {
                "count": direct_tracker.count,
                "alpha": direct_tracker.alpha,
                "cosine_ema": direct_tracker.cosine_ema,
                "relerr_ema": direct_tracker.relerr_ema,
            }
        events.append(event)

        if ledger.charged_flop_equiv >= next_eval:
            record()
            while next_eval <= ledger.charged_flop_equiv:
                next_eval += runtime.eval_every_equiv

    if not trace or trace[-1]["optimizer_step"] != ledger.optimizer_steps:
        record()

    runtime.out.mkdir(parents=True, exist_ok=True)
    result = {
        "arm": arm,
        "arm_slug": arm_slug(arm),
        "deployable": is_deployable_arm(arm),
        "pack": str(runtime.pack),
        "pack_version": PACK_VERSION,
        "optimizer": pack["build_args"].get("optimizer", "unknown"),
        "prefix_steps": int(pack["prefix_steps"]),
        "anchor_eval": pack["anchor_eval"],
        "runtime_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(runtime).items()},
        "build_args": pack["build_args"],
        "trace": trace,
        "events": events,
        "termination": termination,
        "microbatch_cursor": cursor,
        "final_ledger": ledger.as_dict(),
        "filter_state": {
            "exact_steps": filter_state.exact_steps,
            "actions": filter_state.actions,
            "oneshot_done": filter_state.oneshot_done,
        },
    }
    path = runtime.out / "result.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=True))
    print(f"[result] wrote {path}")
    return path


# -----------------------------------------------------------------------------
# Self-test and CLI
# -----------------------------------------------------------------------------


def run_selftest(device: str = "cpu") -> None:
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.GELU(), torch.nn.Linear(8, 3)).to(dev)
    spec = FlatSpec(model.named_parameters())
    hist = VectorHistory(8, spec.numel, dev, dev, torch.float32)
    x = torch.randn(spec.numel, device=dev)
    A = 0.92
    for _ in range(8):
        x = A * x + 0.01 * torch.randn_like(x)
        hist.append(x)
    truth = A * x
    tracker = QualityTracker(spec)
    tracker.update(hist.last(), truth, hist, need_bias=True, need_bias_span=True)
    assert tracker.count == 1
    assert math.isfinite(tracker.alpha)
    assert torch.isfinite(tracker.calibrated(hist.last())).all()
    for base in ("ar:2", "dmd:4"):
        if base.startswith("ar"):
            c = global_ar_coefficients(hist, 2)
        else:
            c = dmd_coefficients(hist, 4)
        pred = hist.combine(c)
        print(f"[selftest] {base} cos={cosine(pred, truth):+.5f} rel={relative_error(pred, truth):.5f}")
    tproj, _ = tensorwise_project(hist, truth, spec, window=4, chunk=17)
    assert torch.isfinite(tproj).all()
    print(f"[selftest] tensor projection cos={cosine(tproj, truth):+.5f}")
    bc = ar_block_coefficients(hist, 1, 4)
    bpred = hist.combine(bc)
    assert torch.isfinite(bpred).all()
    print(f"[selftest] AR block norm={float(bpred.norm()):.5f}")
    ledger = BudgetLedger(spec.numel, 1024)
    ledger.add_exact(1024, 1, 0.1)
    assert abs(ledger.charged_flop_equiv - 1.0) < 1e-9
    ledger.add_forward(1024, 1, 0.03)
    assert abs(ledger.charged_flop_equiv - (1.0 + 1.0 / 3.0)) < 1e-9

    # Trajectory-filter primitives: exact rolling mean and AR block must be finite.
    base = spec.flatten_parameters(dtype=torch.float32)
    wh = deque(maxlen=16)
    for j in range(8):
        wh.append((base + 0.01 * j).detach().cpu())
    delta = _mean_weight_delta(wh, base, window=4, spacing=1)
    expected = torch.full_like(base, 0.055)  # mean(0.04,0.05,0.06,0.07) - 0
    assert torch.allclose(delta.cpu(), expected.cpu(), atol=1e-6), float((delta-expected).abs().max())
    spaced = _spaced_weights(wh, 3, 2)
    assert len(spaced) == 3
    fs = FilterState(exact_since_action=4)
    dummy = argparse.Namespace(filter_start_equiv=0.0, filter_stop_equiv=10.0,
                               skip_period=2, skip_count=1)
    assert filter_planned("lookahead:4:0.5:preserve", BudgetLedger(spec.numel, 1024), dummy, fs)
    print("[selftest] filter primitives PASS")
    print("[selftest] PASS")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("selftest")
    sp.add_argument("--device", default="cpu")

    pp = sub.add_parser("prepare")
    add_build_args(pp)
    pp.add_argument("--head-dim", type=int, default=128,
                    help="canonical NanoChat attention head dimension")
    pp.add_argument("--window-pattern", type=str, default="SSSL",
                    help="NanoChat attention window pattern when supported by GPTConfig")
    pp.add_argument("--pack", type=Path, required=True)
    pp.add_argument("--prefix-steps", type=int, default=0)
    pp.add_argument("--max-virtual-steps", type=int, default=1200,
                    help="number of full-grad-accum groups cached in the linear microbatch stream")
    pp.add_argument("--eval-batches", type=int, default=64)
    pp.add_argument("--prefix-eval-every", type=int, default=100)
    pp.add_argument("--predictor-inner-steps", type=int, default=1)
    pp.add_argument("--lawa-window", type=int, default=34)
    pp.add_argument("--oracle-reference-accum", type=int, default=0)
    pp.add_argument("--seed", type=int, default=1337)
    pp.add_argument("--skip-prefix-predictor-training", action=argparse.BooleanOptionalAction, default=False,
                    help="for oracle/dynamics screens, keep the shared prefix predictor bank untrained")

    bp = sub.add_parser("branch")
    bp.add_argument("--pack", type=Path, required=True)
    bp.add_argument("--arm", required=True)
    bp.add_argument("--out", type=Path, required=True)
    bp.add_argument("--device", default="cuda")
    bp.add_argument("--history-device", choices=["cuda", "cpu"], default=None)
    bp.add_argument("--history-dtype", choices=["float32", "bfloat16", "float16"], default=None)
    bp.add_argument("--compute-budget", type=float, default=600.0,
                    help="charged estimated full-training-step FLOP equivalents")
    bp.add_argument("--max-updates", type=int, default=20000)
    bp.add_argument("--schedule-clock", choices=["flops", "updates", "tokens"], default="flops")
    bp.add_argument("--skip-period", type=int, default=2)
    bp.add_argument("--skip-count", type=int, default=1)
    bp.add_argument("--filter-start-equiv", type=float, default=0.0,
                    help="charged branch FLOP equivalents before trajectory filters may act")
    bp.add_argument("--filter-stop-equiv", type=float, default=1e30,
                    help="charged branch FLOP equivalent at which filters turn off; subsequent training is exact")
    bp.add_argument("--eval-every-equiv", type=float, default=50.0)
    bp.add_argument("--synthetic-history", choices=["predicted", "hold", "zero"], default="predicted")
    bp.add_argument("--predictor-inner-steps", type=int, default=1)
    bp.add_argument("--min-predictor-updates", type=int, default=8)
    bp.add_argument("--min-quality-points", type=int, default=4)
    bp.add_argument("--quality-beta", type=float, default=0.9)
    bp.add_argument("--lr-scale", type=float, default=1.0)
    bp.add_argument("--fallback-exact", action=argparse.BooleanOptionalAction, default=True)
    bp.add_argument("--track-update-history", action=argparse.BooleanOptionalAction, default=False)
    bp.add_argument("--track-weight-history", action=argparse.BooleanOptionalAction, default=False)
    bp.add_argument("--seed", type=int, default=None)

    return ap


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command == "selftest":
        run_selftest(args.device)
    elif args.command == "prepare":
        _install_canonical_build()
        _stabilize_nanochat_adamw_kernel()
        if args.skip_prefix_predictor_training:
            # Predictor training is orthogonal to the oracle/dynamics screen and can make
            # a long shared prefix unnecessarily expensive. The bank remains initialized
            # so branch restore stays format-compatible.
            original_observe = PredictorBank.observe_true_gradient
            try:
                PredictorBank.observe_true_gradient = lambda self, *a, **k: {}
                prepare_pack(args)
            finally:
                PredictorBank.observe_true_gradient = original_observe
        else:
            prepare_pack(args)
    elif args.command == "branch":
        _install_canonical_build()
        run_budget_branch(args)
    else:
        parser.error(args.command)


if __name__ == "__main__":
    main()
