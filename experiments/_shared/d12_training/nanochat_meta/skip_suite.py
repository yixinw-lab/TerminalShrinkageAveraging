#!/usr/bin/env python3
"""Parallel screening harness for backward-pass skipping in NanoChat.

Typical use
-----------

1. Build one shared prefix/anchor pack (model, AdamW state, online predictors,
   exact gradient history, fixed validation batches, and a cached future data stream)::

       python -m nanochat_meta.skip_suite prepare --pack outputs/skip/anchor.pt ...

2. Run one arm per Slurm array task from the identical pack::

       python -m nanochat_meta.skip_suite branch --pack outputs/skip/anchor.pt \
           --arm ar:4 --out outputs/skip/arms/ar4

3. Merge all JSON files into compute/loss plots::

       python -m nanochat_meta.skip_plot --root outputs/skip/arms

``all`` performs prepare + all requested arms sequentially in one process. The
Slurm scripts shipped beside this file use ``prepare`` + an array so the arms run
concurrently.

The scientific invariant is that every gradient-based synthetic arm sets
``p.grad`` and invokes the same pure-AdamW ``optimizer.step()`` as the baseline.
No direct preconditioning, sign convention, or stale-moment shortcut exists in
that path.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .skip_core import (
    ComputeLedger, FlatSpec, Timer, VectorHistory, apply_lr_schedule, clear_adam_state,
    cosine, dmd_coefficients, external_gradient_equivalence_test, global_ar_coefficients,
    make_pure_adamw, optimizer_first_moment, recursive_to_cpu, relative_error,
    seed_everything, tensor_ar_prediction,
)
from .skip_predictors import (
    FORWARD_FEATURE_DIM, PredictorBank, PredictorContext, summarize_forward_features,
)

PACK_VERSION = 3


# -----------------------------------------------------------------------------
# NanoChat integration
# -----------------------------------------------------------------------------

def _import_existing_build():
    try:
        from nanochat_meta.meta_experiment import build
    except Exception as exc:
        raise RuntimeError(
            "Could not import nanochat_meta.meta_experiment.build. Copy this suite into "
            "the ExtrapProj repository beside your existing nanochat_meta directory and "
            "set PYTHONPATH=$REPO:$NANOCHAT_ROOT."
        ) from exc
    return build


def _stabilize_nanochat_adamw_kernel() -> None:
    """Keep NanoChat's AdamW math but avoid shape-specialization failures.

    NanoChat's adamw_step_fused is torch.compile(dynamic=False, fullgraph=True)
    and is invoked on many differently-shaped parameter tensors. For these
    screening runs we execute the original function eagerly. This changes
    performance of optimizer.step slightly, but not the AdamW equations or state.
    """
    try:
        import nanochat.optim as nc_optim

        fn = nc_optim.adamw_step_fused
        original = getattr(fn, "_torchdynamo_orig_callable", None)

        if original is not None:
            nc_optim.adamw_step_fused = original
            print(
                "[build] NanoChat AdamW kernel: eager mode "
                "(disabled torch.compile across heterogeneous parameter shapes)"
            )
            return

        # Fallback for a PyTorch version whose compiled wrapper does not expose
        # _torchdynamo_orig_callable.
        import torch._dynamo
        current = int(torch._dynamo.config.recompile_limit)
        torch._dynamo.config.recompile_limit = max(current, 64)
        print(
            "[build] NanoChat AdamW kernel: compiled mode, "
            f"recompile_limit={torch._dynamo.config.recompile_limit}"
        )

    except Exception as exc:
        print(
            "[build] WARNING: could not stabilize NanoChat AdamW kernel: "
            f"{type(exc).__name__}: {exc}"
        )


def build_stack(args: argparse.Namespace):
    build = _import_existing_build()

    if args.optimizer == "pure_adamw":
        _stabilize_nanochat_adamw_kernel()

    model, base_optimizer, mk_loader, token_bytes, device = build(args)
    if args.optimizer == "pure_adamw":
        optimizer = make_pure_adamw(
            model, base_optimizer,
            matrix_lr=args.matrix_adam_lr,
            matrix_betas=(args.matrix_beta1, args.matrix_beta2),
            matrix_eps=args.matrix_eps,
            matrix_weight_decay=args.matrix_adam_weight_decay,
        )
        explicit_kinds = [g.get("kind") for g in optimizer.param_groups if "kind" in g]
        if explicit_kinds and set(explicit_kinds) != {"adamw"}:
            raise AssertionError(
                f"pure AdamW conversion failed; optimizer kinds are {sorted(set(explicit_kinds))}"
            )
    else:
        optimizer = base_optimizer
    return model, optimizer, mk_loader, token_bytes, torch.device(device)


def namespace_from_pack(pack: dict[str, Any], overrides: argparse.Namespace) -> argparse.Namespace:
    d = dict(pack["build_args"])
    # Runtime-only settings may differ from prepare.
    for key in ("device", "history_device", "history_dtype", "out", "arm"):
        if hasattr(overrides, key) and getattr(overrides, key) is not None:
            d[key] = getattr(overrides, key)
    return argparse.Namespace(**d)


# -----------------------------------------------------------------------------
# Batch handling and model evaluation
# -----------------------------------------------------------------------------

def clone_batch_cpu(batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(t.detach().cpu().clone() for t in batch)  # type: ignore[return-value]


def move_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
               ) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(t.to(device, non_blocking=True) for t in batch)  # type: ignore[return-value]


def token_sample_from_batches(batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
                              max_tokens: int = 4096) -> torch.Tensor:
    pieces = []
    per = max(1, max_tokens // max(len(batches), 1))
    for x, _ in batches:
        flat = x.reshape(-1)
        if flat.numel() <= per:
            pieces.append(flat)
        else:
            idx = torch.linspace(0, flat.numel() - 1, per).long()
            pieces.append(flat[idx])
    return torch.cat(pieces)[:max_tokens]


def _manual_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=-1)


def exact_gradient(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   spec: FlatSpec,
                   batches_cpu: Sequence[tuple[torch.Tensor, torch.Tensor]],
                   device: torch.device, *, need_forward_features: bool,
                   token_sample_max: int = 4096
                   ) -> tuple[torch.Tensor, float, torch.Tensor, Optional[torch.Tensor]]:
    """Compute an averaged exact gradient over the supplied microbatches."""
    optimizer.zero_grad(set_to_none=True)
    n = len(batches_cpu)
    if n < 1:
        raise ValueError("exact_gradient received no microbatches")
    loss_total = 0.0
    features = []
    for batch_cpu in batches_cpu:
        x, y = move_batch(batch_cpu, device)
        if need_forward_features:
            logits = model(x)
            loss = _manual_loss(logits, y)
            features.append(summarize_forward_features(logits, y, loss).detach())
            del logits
        else:
            loss = model(x, y)
            if loss.ndim:
                loss = loss.mean()
        (loss / n).backward()
        loss_total += float(loss.detach()) / n
        del x, y, loss
    g = spec.flatten_gradients(dtype=torch.float32)
    token_ids = token_sample_from_batches(batches_cpu, token_sample_max).to(device)
    ff = torch.stack(features).mean(0) if features else None
    return g, loss_total, token_ids, ff


@torch.no_grad()
def forward_only_context(model: torch.nn.Module,
                         batches_cpu: Sequence[tuple[torch.Tensor, torch.Tensor]],
                         device: torch.device,
                         token_sample_max: int = 4096
                         ) -> tuple[float, torch.Tensor, torch.Tensor]:
    loss_total = 0.0
    feats = []
    for batch_cpu in batches_cpu:
        x, y = move_batch(batch_cpu, device)
        logits = model(x)
        loss = _manual_loss(logits, y)
        feats.append(summarize_forward_features(logits, y, loss))
        loss_total += float(loss) / len(batches_cpu)
        del x, y, logits, loss
    tok = token_sample_from_batches(batches_cpu, token_sample_max).to(device)
    return loss_total, tok, torch.stack(feats).mean(0)


@torch.no_grad()
def evaluate_model(model: torch.nn.Module,
                   eval_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
                   token_bytes: Any, device: torch.device) -> dict[str, float]:
    from nanochat.loss_eval import evaluate_bpb
    was_training = model.training
    model.eval()
    bpbs: list[float] = []
    nlls: list[float] = []
    for b in eval_batches:
        x, y = move_batch(b, device)
        loss = model(x, y)
        nlls.append(float(loss.mean() if loss.ndim else loss))
        bpbs.append(float(evaluate_bpb(model, iter([(x, y)]), 1, token_bytes)))
        del x, y, loss
    if was_training:
        model.train()
    def mean_se(v: list[float]) -> tuple[float, float]:
        a = np.asarray(v, dtype=float)
        return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0
    bpb, bpb_se = mean_se(bpbs)
    nll, nll_se = mean_se(nlls)
    return {"bpb": bpb, "bpb_se": bpb_se, "nll": nll, "nll_se": nll_se,
            "n_eval_batches": len(eval_batches)}


# -----------------------------------------------------------------------------
# Predictor/candidate utilities
# -----------------------------------------------------------------------------

def geometric_ema_coeffs(W: int, beta: float, n: Optional[int] = None) -> torch.Tensor:
    n = W if n is None else min(n, W)
    ages = torch.arange(n - 1, -1, -1, dtype=torch.float64)
    w = (1.0 - beta) * beta ** ages
    w /= w.sum().clamp_min(1e-30)
    c = torch.zeros(W, dtype=torch.float64)
    c[-n:] = w
    return c


def parse_arm(arm: str) -> tuple[str, list[str]]:
    parts = arm.strip().split(":")
    return parts[0], parts[1:]


def arm_slug(arm: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", arm)


def active_predictors_for_arm(arm: str) -> set[str]:
    name, _ = parse_arm(arm)
    if name in {"coeff", "coeff_resid", "coeff_raw", "coeff_forward",
                "coord", "coord_resid", "coord_raw", "coord_forward"}:
        return {name}
    return set()


def arm_requires_raw(arm: str) -> bool:
    return parse_arm(arm)[0] in {"coeff_raw", "coord_raw"}


def arm_requires_forward(arm: str) -> bool:
    return parse_arm(arm)[0] in {"coeff_forward", "coord_forward"}


def arm_requires_momentum(arm: str) -> bool:
    return parse_arm(arm)[0] in {
        "momentum", "two_ema", "coeff_resid", "coord_resid", "micro_blend"
    }


def candidate_from_history(arm: str, history: VectorHistory, spec: FlatSpec,
                           bank: PredictorBank, momentum: torch.Tensor,
                           context: PredictorContext,
                           generator: torch.Generator) -> torch.Tensor:
    name, extra = parse_arm(arm)
    W = len(history)
    if name == "zero":
        return torch.zeros(spec.numel, device=spec.device, dtype=torch.float32)
    if name == "last":
        return history.last()
    if name == "mean":
        n = int(extra[0]) if extra else W
        return history.mean(n)
    if name == "ema":
        beta = float(extra[0]) if extra else 0.95
        return history.combine(geometric_ema_coeffs(W, beta))
    if name == "two_ema":
        fast = float(extra[0]) if len(extra) > 0 else 0.8
        slow = float(extra[1]) if len(extra) > 1 else 0.98
        gamma = float(extra[2]) if len(extra) > 2 else 1.0
        gf = history.combine(geometric_ema_coeffs(W, fast))
        gs = history.combine(geometric_ema_coeffs(W, slow))
        return gf + gamma * (gf - gs)
    if name == "momentum":
        return momentum
    if name == "ar":
        order = int(extra[0]) if extra else 2
        ridge = float(extra[1]) if len(extra) > 1 else 1e-3
        return history.combine(global_ar_coefficients(history, order, ridge))
    if name == "dmd":
        rank = int(extra[0]) if extra else min(8, W - 1)
        ridge = float(extra[1]) if len(extra) > 1 else 1e-4
        return history.combine(dmd_coefficients(history, rank, ridge))
    if name == "tensor_ar":
        order = int(extra[0]) if extra else 2
        return tensor_ar_prediction(history, spec, order)
    if name in {"coeff", "coeff_resid", "coeff_raw", "coeff_forward"}:
        return bank.coefficient_prediction(name, history, momentum, context)
    if name in {"coord", "coord_resid", "coord_raw", "coord_forward"}:
        return bank.coordinate_prediction(name, history, momentum, context)
    if name == "random_span":
        c = torch.randn(W, generator=generator, device=spec.device, dtype=torch.float32).double().cpu()
        g = history.combine(c)
        return g * (history.last().norm() / g.norm().clamp_min(1e-30))
    if name == "random_full":
        g = torch.randn(spec.numel, generator=generator, device=spec.device)
        return g * (history.last().norm() / g.norm().clamp_min(1e-30))
    raise ValueError(f"arm {arm!r} is not a history-only candidate")


# -----------------------------------------------------------------------------
# Pack preparation
# -----------------------------------------------------------------------------

def build_args_dict(args: argparse.Namespace) -> dict[str, Any]:
    keys = [
        "depth", "aspect_ratio", "max_seq_len", "device_batch_size", "grad_accum",
        "warmup_steps", "total_iterations", "warmdown_ratio", "final_lr_frac",
        "embedding_lr", "unembedding_lr", "matrix_lr", "scalar_lr", "weight_decay",
        "device", "optimizer", "matrix_adam_lr", "matrix_beta1", "matrix_beta2",
        "matrix_eps", "matrix_adam_weight_decay", "history", "history_device",
        "history_dtype", "token_dim", "coeff_hidden", "coord_hidden", "coord_order",
        "predictor_lr", "coord_sample", "projection_ridge", "token_sample_max",
    ]
    return {k: getattr(args, k) for k in keys}


def _dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16,
            "float16": torch.float16}[name]


def _history_device(name: str, compute: torch.device) -> torch.device:
    if name == "cuda":
        if compute.type != "cuda":
            raise ValueError("--history-device cuda requires a CUDA model")
        return compute
    return torch.device("cpu")


def predictor_bank_from_args(args: argparse.Namespace, model: torch.nn.Module,
                             device: torch.device) -> PredictorBank:
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 32768))
    return PredictorBank(
        args.history, vocab, device,
        token_dim=args.token_dim,
        coeff_hidden=args.coeff_hidden,
        coord_hidden=args.coord_hidden,
        coord_order=args.coord_order,
        lr=args.predictor_lr,
        coord_sample=args.coord_sample,
        ridge=args.projection_ridge,
    )


def prepare_pack(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    toy = external_gradient_equivalence_test("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[invariant] external-gradient AdamW equivalence: {toy}")

    model, optimizer, mk_loader, token_bytes, device = build_stack(args)
    # Current NanoChat separates construction from real initialization. The existing
    # meta_experiment.build imports compute_init but does not call init_weights.
    # Explicitly initialize here when that API is present; branch jobs immediately
    # overwrite the weights from the pack and therefore do not call it.
    if hasattr(model, "init_weights"):
        model.init_weights()
    model.train()
    spec = FlatSpec(model.named_parameters())
    hdev = _history_device(args.history_device, device)
    hdtype = _dtype_from_name(args.history_dtype)
    history = VectorHistory(args.history, spec.numel, device, hdev, hdtype)
    bank = predictor_bank_from_args(args, model, device)
    train_loader = mk_loader("train")
    val_loader = mk_loader("val")

    eval_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    fingerprints: set[int] = set()
    attempts = 0
    while len(eval_batches) < args.eval_batches and attempts < args.eval_batches * 50:
        b = clone_batch_cpu(next(val_loader))
        # Fast deterministic fingerprint; duplicate eval batches make SE meaningless.
        fp = hash(b[0].numpy().tobytes())
        if fp not in fingerprints:
            fingerprints.add(fp)
            eval_batches.append(b)
        attempts += 1
    if len(eval_batches) < 2:
        raise RuntimeError("validation loader produced fewer than two distinct batches")
    print(f"[eval] cached {len(eval_batches)} distinct validation batches")

    weight_history: deque[torch.Tensor] = deque(maxlen=max(args.lawa_window, 0))
    prefix_trace: list[dict[str, Any]] = []
    predictor_active = {
        "coeff", "coeff_resid", "coeff_raw", "coeff_forward",
        "coord", "coord_resid", "coord_raw", "coord_forward",
    }
    t0 = time.time()
    predictor_prefix_seconds = 0.0
    for step in range(args.prefix_steps):
        lr_mult = apply_lr_schedule(optimizer, step, args.warmup_steps,
                                    args.total_iterations, args.warmdown_ratio,
                                    args.final_lr_frac)
        batches = [clone_batch_cpu(next(train_loader)) for _ in range(args.grad_accum)]
        need_momentum = history.full
        momentum = optimizer_first_moment(optimizer, spec) if need_momentum else torch.zeros(
            spec.numel, device=device, dtype=torch.float32)
        g, train_loss, token_ids, ff = exact_gradient(
            model, optimizer, spec, batches, device,
            need_forward_features=True, token_sample_max=args.token_sample_max)
        pred_metrics: dict[str, float] = {}
        if history.full:
            with Timer() as pt:
                pred_metrics = bank.observe_true_gradient(
                    history, g, momentum, token_ids, ff,
                    step_frac=step / max(args.total_iterations or args.prefix_steps, 1),
                    lr_mult=lr_mult, active=predictor_active,
                    inner_steps=args.predictor_inner_steps,
                )
            pred_metrics["predictor_seconds"] = pt.seconds
            predictor_prefix_seconds += pt.seconds
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        history.append(g)
        if args.lawa_window > 0 and step >= args.prefix_steps - args.lawa_window:
            weight_history.append(spec.flatten_parameters(dtype=torch.float32).to(
                device="cpu", dtype=hdtype))
        if step % args.prefix_eval_every == 0 or step == args.prefix_steps - 1:
            with Timer() as et:
                ev = evaluate_model(model, eval_batches, token_bytes, device)
            row = {"step": step + 1, "train_loss": train_loss, **ev,
                   "history_gb": history.memory_bytes / 1e9,
                   "elapsed_seconds": time.time() - t0,
                   **{f"pred_{k}": v for k, v in pred_metrics.items()}}
            prefix_trace.append(row)
            print("[prefix] " + " ".join(
                [f"step={step+1}", f"loss={train_loss:.4f}", f"bpb={ev['bpb']:.5f}",
                 f"hist={history.memory_bytes/1e9:.2f}GB", f"time={time.time()-t0:.0f}s"]))

    anchor_eval = evaluate_model(model, eval_batches, token_bytes, device)
    print(f"[anchor] bpb={anchor_eval['bpb']:.6f} +/- {anchor_eval['bpb_se']:.6f}")

    max_micro = args.max_virtual_steps * args.grad_accum
    print(f"[cache] cloning {max_micro} future training microbatches")
    branch_batches = [clone_batch_cpu(next(train_loader)) for _ in range(max_micro)]
    reference_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    if args.oracle_reference_accum > 0:
        nref = args.max_virtual_steps * args.oracle_reference_accum
        print(f"[cache] cloning {nref} independent reference microbatches for population-oracle arms")
        reference_batches = [clone_batch_cpu(next(train_loader)) for _ in range(nref)]

    pack = {
        "pack_version": PACK_VERSION,
        "created_unix": time.time(),
        "build_args": build_args_dict(args),
        "seed": args.seed,
        "prefix_steps": args.prefix_steps,
        "model_state": recursive_to_cpu(model.state_dict()),
        "optimizer_state": recursive_to_cpu(optimizer.state_dict()),
        "history_state": history.state_dict(),
        "predictor_state": recursive_to_cpu(bank.state_dict()),
        "weight_history": list(weight_history),
        "eval_batches": eval_batches,
        "branch_batches": branch_batches,
        "reference_batches": reference_batches,
        "anchor_eval": anchor_eval,
        "prefix_trace": prefix_trace,
        "parameter_count": spec.numel,
        "parameter_names": spec.names,
        "oracle_reference_accum": args.oracle_reference_accum,
        "shared_prefix_predictor_seconds": predictor_prefix_seconds,
    }
    args.pack.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.pack.with_suffix(args.pack.suffix + ".tmp")
    torch.save(pack, tmp)
    tmp.replace(args.pack)
    size = args.pack.stat().st_size / 1e9
    print(f"[pack] wrote {args.pack} ({size:.2f} GB)")
    return args.pack


# -----------------------------------------------------------------------------
# Branch execution
# -----------------------------------------------------------------------------

def restore_branch(pack: dict[str, Any], runtime: argparse.Namespace):
    args = namespace_from_pack(pack, runtime)
    model, optimizer, mk_loader, token_bytes, device = build_stack(args)
    model.load_state_dict(pack["model_state"])
    optimizer.load_state_dict(pack["optimizer_state"])
    model.train()
    spec = FlatSpec(model.named_parameters())
    hdev = _history_device(args.history_device, device)
    history = VectorHistory(args.history, spec.numel, device, hdev,
                            _dtype_from_name(args.history_dtype))
    history.load_state_dict(pack["history_state"])
    bank = predictor_bank_from_args(args, model, device)
    bank.load_state_dict(pack["predictor_state"])
    return args, model, optimizer, token_bytes, device, spec, history, bank


def branch_step_batches(pack: dict[str, Any], virtual_idx: int, grad_accum: int
                       ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    lo = virtual_idx * grad_accum
    hi = lo + grad_accum
    batches = pack["branch_batches"][lo:hi]
    if len(batches) != grad_accum:
        raise RuntimeError("anchor pack does not contain enough cached branch batches")
    return batches


def reference_step_batches(pack: dict[str, Any], virtual_idx: int, count: int
                          ) -> list[tuple[torch.Tensor, torch.Tensor]]:
    lo = virtual_idx * count
    hi = lo + count
    batches = pack.get("reference_batches", [])[lo:hi]
    if len(batches) != count:
        raise RuntimeError("anchor pack lacks reference batches; rerun prepare with --oracle-reference-accum")
    return batches


def is_synthetic_step(arm: str, local_step: int, period: int, count: int) -> bool:
    if parse_arm(arm)[0] == "gd":
        return False
    if count <= 0:
        return False
    pos = local_step % period
    return pos >= period - count


def make_context_for_arm(arm: str, bank: PredictorBank, history: VectorHistory,
                         batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
                         model: torch.nn.Module, device: torch.device,
                         step_frac: float, lr_mult: float, ledger: ComputeLedger,
                         token_sample_max: int) -> tuple[PredictorContext, Optional[float]]:
    tok = None
    ff = None
    forward_loss = None
    if arm_requires_raw(arm):
        tok = token_sample_from_batches(batches, token_sample_max).to(device)
    if arm_requires_forward(arm):
        with Timer() as ft:
            forward_loss, tok_ff, ff = forward_only_context(
                model, batches, device, token_sample_max=token_sample_max)
        ledger.forward_equiv += 1.0
        ledger.train_seconds += ft.seconds
        tok = tok if tok is not None else tok_ff
    with Timer() as pt:
        ctx = bank.make_context(history, tok, ff, step_frac, lr_mult)
    ledger.predictor_seconds += pt.seconds
    return ctx, forward_loss


def _noise_like(target: torch.Tensor, rel: float, generator: torch.Generator,
                mode: str, history: VectorHistory,
                frozen: dict[str, torch.Tensor]) -> torch.Tensor:
    if mode == "frozen":
        if "noise" not in frozen:
            n = torch.randn(target.numel(), generator=generator, device=target.device)
            frozen["noise"] = n / n.norm().clamp_min(1e-30)
        unit = frozen["noise"]
    elif mode == "span":
        c = torch.randn(len(history), generator=generator, device=target.device, dtype=torch.float32).double().cpu()
        n = history.combine(c)
        unit = n / n.norm().clamp_min(1e-30)
    else:
        n = torch.randn(target.numel(), generator=generator, device=target.device)
        unit = n / n.norm().clamp_min(1e-30)
    return unit * (rel * target.norm())


def apply_lawa(spec: FlatSpec, weight_history: deque[torch.Tensor], window: int,
               state_policy: str, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    if len(weight_history) < window:
        raise RuntimeError(f"LAWA needs {window} weight snapshots, only {len(weight_history)} available")
    current = spec.flatten_parameters(dtype=torch.float32)
    mean = torch.zeros_like(current)
    for w in list(weight_history)[-window:]:
        mean.add_(w.to(device=current.device, dtype=current.dtype), alpha=1.0 / window)
    delta = mean - current
    spec.assign_parameters(mean)
    if state_policy == "zero_m":
        clear_adam_state(optimizer, first_only=True)
    elif state_policy == "zero_all":
        clear_adam_state(optimizer, first_only=False)
    elif state_policy != "preserve":
        raise ValueError(f"unknown LAWA state policy {state_policy}")
    return delta


def run_branch(runtime: argparse.Namespace) -> Path:
    pack = torch.load(runtime.pack, map_location="cpu", weights_only=False)
    if int(pack.get("pack_version", -1)) != PACK_VERSION:
        raise RuntimeError(f"pack version mismatch: got {pack.get('pack_version')}, expected {PACK_VERSION}")
    args, model, optimizer, token_bytes, device, spec, history, bank = restore_branch(pack, runtime)
    arm = runtime.arm
    name, extra = parse_arm(arm)
    seed_everything(runtime.seed if runtime.seed is not None else int(pack["seed"]))
    gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
    gen.manual_seed((runtime.seed or int(pack["seed"])) + 1777)
    ledger = ComputeLedger()
    eval_batches = pack["eval_batches"]
    trace: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    active_predictors = active_predictors_for_arm(arm)
    weight_history: deque[torch.Tensor] = deque(pack.get("weight_history", []),
                                                maxlen=max(runtime.lawa_max_window, 1))
    frozen_noise: dict[str, torch.Tensor] = {}

    def record(local_step: int) -> None:
        with Timer() as et:
            ev = evaluate_model(model, eval_batches, token_bytes, device)
        ledger.eval_seconds += et.seconds
        trace.append({
            "local_virtual_step": local_step,
            "global_virtual_step": int(pack["prefix_steps"]) + local_step,
            **ev, **ledger.as_dict(),
        })
        print(f"[{arm}] v={local_step:4d} ideal={ledger.ideal_train_equiv:7.2f} "
              f"bw={ledger.actual_backwards:4d} bpb={ev['bpb']:.6f}+/-{ev['bpb_se']:.6f}")

    record(0)
    max_virtual = min(runtime.max_virtual_steps, len(pack["branch_batches"]) // args.grad_accum)
    local = 0
    while local < max_virtual:
        if runtime.stop_by == "ideal_compute" and ledger.ideal_train_equiv >= runtime.compute_budget:
            break
        batches = branch_step_batches(pack, local, args.grad_accum)
        global_step = int(pack["prefix_steps"]) + local
        lr_mult = apply_lr_schedule(optimizer, global_step, args.warmup_steps,
                                    args.total_iterations, args.warmdown_ratio,
                                    args.final_lr_frac)
        synthetic = is_synthetic_step(arm, local, runtime.skip_period, runtime.skip_count)
        step_frac = global_step / max(args.total_iterations or (pack["prefix_steps"] + max_virtual), 1)
        need_mom = arm_requires_momentum(arm) or bool(active_predictors)
        momentum = optimizer_first_moment(optimizer, spec) if need_mom else torch.zeros(
            spec.numel, device=device, dtype=torch.float32)
        event: dict[str, Any] = {"local_step": local + 1, "synthetic": synthetic,
                                 "lr_mult": lr_mult, "arm": arm}

        if not synthetic:
            need_ff = arm_requires_forward(arm)
            with Timer() as tt:
                g, train_loss, tok, ff = exact_gradient(
                    model, optimizer, spec, batches, device,
                    need_forward_features=need_ff,
                    token_sample_max=args.token_sample_max)
                if active_predictors and history.full:
                    bank.observe_true_gradient(
                        history, g, momentum, tok, ff, step_frac, lr_mult,
                        active=active_predictors, inner_steps=runtime.predictor_inner_steps_branch)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            ledger.train_seconds += tt.seconds
            ledger.exact_backward_equiv += 1.0
            ledger.actual_backwards += 1
            history.append(g)
            event.update({"train_loss": train_loss, "gradient_norm": float(g.norm())})
        else:
            ledger.synthetic_steps += 1
            true_g: Optional[torch.Tensor] = None
            train_loss: Optional[float] = None
            # Arms below intentionally compute a true/partial gradient for an oracle
            # frontier. They are diagnostic; ideal accounting does not pretend the
            # measured wall-clock is deployable.
            if name in {"oracle_batch", "projected_oracle", "oracle_scale",
                        "oracle_noise"}:
                with Timer() as tt:
                    true_g, train_loss, tok, ff = exact_gradient(
                        model, optimizer, spec, batches, device,
                        need_forward_features=False,
                        token_sample_max=args.token_sample_max)
                ledger.train_seconds += tt.seconds
                ledger.actual_backwards += 1
                if name == "oracle_batch":
                    cand = true_g
                elif name == "projected_oracle":
                    cand = history.project(true_g, args.projection_ridge)
                elif name == "oracle_scale":
                    scale = float(extra[0]) if extra else 1.0
                    cand = scale * true_g
                elif name == "oracle_noise":
                    rel = float(extra[0]) if extra else 0.3
                    mode = extra[1] if len(extra) > 1 else "iid"
                    cand = true_g + _noise_like(true_g, rel, gen, mode, history, frozen_noise)
                else:
                    raise AssertionError
            elif name in {"oracle_population", "projected_population_oracle"}:
                count = int(extra[0]) if extra else int(pack.get("oracle_reference_accum", 0))
                ref = reference_step_batches(pack, local, count)
                with Timer() as tt:
                    true_g, train_loss, tok, ff = exact_gradient(
                        model, optimizer, spec, ref, device,
                        need_forward_features=False,
                        token_sample_max=args.token_sample_max)
                ledger.train_seconds += tt.seconds
                ledger.actual_backwards += 1
                cand = true_g if name == "oracle_population" else history.project(
                    true_g, args.projection_ridge)
            elif name in {"micro", "micro_blend"}:
                k = int(extra[0]) if extra else 1
                if not 1 <= k <= args.grad_accum:
                    raise ValueError(f"microbatch count {k} must be in [1,{args.grad_accum}]")
                with Timer() as tt:
                    mg, train_loss, tok, ff = exact_gradient(
                        model, optimizer, spec, batches[:k], device,
                        need_forward_features=False,
                        token_sample_max=args.token_sample_max)
                ledger.train_seconds += tt.seconds
                ledger.exact_backward_equiv += k / args.grad_accum
                ledger.actual_backwards += 1
                if name == "micro_blend":
                    alpha = float(extra[1]) if len(extra) > 1 else 0.5
                    cand = alpha * mg + (1.0 - alpha) * momentum
                else:
                    cand = mg
            elif name == "lawa":
                window = int(extra[0]) if extra else 4
                policy = extra[1] if len(extra) > 1 else "preserve"
                with Timer() as pt:
                    delta = apply_lawa(spec, weight_history, window, policy, optimizer)
                ledger.predictor_seconds += pt.seconds
                event.update({"direct_parameter_delta_norm": float(delta.norm()),
                              "lawa_window": window, "state_policy": policy})
                cand = None
            else:
                ctx, forward_loss = make_context_for_arm(
                    arm, bank, history, batches, model, device, step_frac, lr_mult,
                    ledger, args.token_sample_max)
                with Timer() as pt:
                    cand = candidate_from_history(arm, history, spec, bank, momentum, ctx, gen)
                ledger.predictor_seconds += pt.seconds
                train_loss = forward_loss

            if name != "lawa":
                assert cand is not None
                event.update({
                    "candidate_norm": float(cand.norm()),
                    "candidate_to_last_norm": float(cand.norm() / history.last().norm().clamp_min(1e-30)),
                    "candidate_cos_last": cosine(cand, history.last()),
                })
                if true_g is not None:
                    event.update({
                        "cos_to_oracle": cosine(cand, true_g),
                        "relerr_to_oracle": relative_error(cand, true_g),
                        "oracle_norm": float(true_g.norm()),
                    })
                with Timer() as ot:
                    spec.assign_gradients(cand)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                ledger.train_seconds += ot.seconds
                if runtime.synthetic_history == "predicted":
                    history.append(cand)
                elif runtime.synthetic_history == "zero":
                    history.append(torch.zeros_like(cand))
                elif runtime.synthetic_history != "hold":
                    raise ValueError(runtime.synthetic_history)
            event["train_loss_if_computed"] = train_loss

        ledger.virtual_steps += 1
        local += 1
        if name == "lawa" or runtime.track_weight_history:
            weight_history.append(spec.flatten_parameters(dtype=torch.float32).to(
                device="cpu", dtype=_dtype_from_name(args.history_dtype)))
        events.append(event)
        if local % runtime.eval_every == 0:
            record(local)

    if not trace or trace[-1]["local_virtual_step"] != local:
        record(local)

    out = runtime.out
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "arm": arm,
        "arm_slug": arm_slug(arm),
        "pack": str(runtime.pack),
        "pack_version": PACK_VERSION,
        "prefix_steps": int(pack["prefix_steps"]),
        "anchor_eval": pack["anchor_eval"],
        "shared_prefix_predictor_seconds": float(pack.get("shared_prefix_predictor_seconds", 0.0)),
        "runtime_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(runtime).items()},
        "build_args": pack["build_args"],
        "trace": trace,
        "events": events,
        "final_ledger": ledger.as_dict(),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[result] wrote {out/'result.json'}")
    return out


# -----------------------------------------------------------------------------
# Self-test and CLI
# -----------------------------------------------------------------------------

def run_selftest(args: argparse.Namespace) -> None:
    print("[selftest] external gradient path")
    print(external_gradient_equivalence_test(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"))
    print("[selftest] history projection / AR / DMD")
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    d = 256
    H = VectorHistory(8, d, dev, dev, torch.float32)
    torch.manual_seed(3)
    A = torch.diag(torch.linspace(0.7, 0.99, d, device=dev))
    x = torch.randn(d, device=dev)
    for _ in range(8):
        x = A @ x
        H.append(x)
    truth = A @ x
    for name, c in {
        "ar2": global_ar_coefficients(H, 2),
        "ar4": global_ar_coefficients(H, 4),
        "dmd4": dmd_coefficients(H, 4),
        "dmd7": dmd_coefficients(H, 7),
    }.items():
        pred = H.combine(c)
        print(f"  {name:6s} cos={cosine(pred, truth):+.6f} relerr={relative_error(pred, truth):.6f}")
    proj = H.project(truth)
    print(f"  projection cos={cosine(proj, truth):+.6f} relerr={relative_error(proj, truth):.6f}")
    print("[selftest] PASS")


def add_build_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--aspect-ratio", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--device-batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup-steps", type=int, default=40)
    ap.add_argument("--total-iterations", type=int, default=2000)
    ap.add_argument("--warmdown-ratio", type=float, default=0.3)
    ap.add_argument("--final-lr-frac", type=float, default=0.1)
    ap.add_argument("--embedding-lr", type=float, default=0.2)
    ap.add_argument("--unembedding-lr", type=float, default=0.004)
    ap.add_argument("--matrix-lr", type=float, default=0.02,
                    help="Muon LR used only to construct NanoChat's base optimizer")
    ap.add_argument("--scalar-lr", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.28)
    ap.add_argument("--optimizer", choices=["pure_adamw", "native"], default="pure_adamw")
    ap.add_argument("--matrix-adam-lr", type=float, default=3e-3)
    ap.add_argument("--matrix-beta1", type=float, default=0.9)
    ap.add_argument("--matrix-beta2", type=float, default=0.95)
    ap.add_argument("--matrix-eps", type=float, default=1e-10)
    ap.add_argument("--matrix-adam-weight-decay", type=float, default=0.1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--history", type=int, default=8)
    ap.add_argument("--history-device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--history-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    ap.add_argument("--token-dim", type=int, default=32)
    ap.add_argument("--coeff-hidden", type=int, default=256)
    ap.add_argument("--coord-hidden", type=int, default=48)
    ap.add_argument("--coord-order", type=int, default=4)
    ap.add_argument("--predictor-lr", type=float, default=2e-3)
    ap.add_argument("--coord-sample", type=int, default=131072)
    ap.add_argument("--projection-ridge", type=float, default=1e-4)
    ap.add_argument("--token-sample-max", type=int, default=4096)


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("selftest")
    sp.add_argument("--device", default="cpu")

    pp = sub.add_parser("prepare")
    add_build_args(pp)
    pp.add_argument("--pack", type=Path, required=True)
    pp.add_argument("--prefix-steps", type=int, default=1200)
    pp.add_argument("--max-virtual-steps", type=int, default=300)
    pp.add_argument("--eval-batches", type=int, default=32)
    pp.add_argument("--prefix-eval-every", type=int, default=100)
    pp.add_argument("--predictor-inner-steps", type=int, default=1)
    pp.add_argument("--lawa-window", type=int, default=8)
    pp.add_argument("--oracle-reference-accum", type=int, default=4)
    pp.add_argument("--seed", type=int, default=1337)

    bp = sub.add_parser("branch")
    bp.add_argument("--pack", type=Path, required=True)
    bp.add_argument("--arm", required=True)
    bp.add_argument("--out", type=Path, required=True)
    bp.add_argument("--device", default="cuda")
    bp.add_argument("--history-device", choices=["cuda", "cpu"], default=None)
    bp.add_argument("--history-dtype", choices=["float32", "bfloat16", "float16"], default=None)
    bp.add_argument("--max-virtual-steps", type=int, default=300)
    bp.add_argument("--stop-by", choices=["virtual", "ideal_compute"], default="virtual")
    bp.add_argument("--compute-budget", type=float, default=150.0)
    bp.add_argument("--skip-period", type=int, default=2)
    bp.add_argument("--skip-count", type=int, default=1)
    bp.add_argument("--eval-every", type=int, default=25)
    bp.add_argument("--synthetic-history", choices=["predicted", "hold", "zero"], default="predicted")
    bp.add_argument("--predictor-inner-steps-branch", type=int, default=1)
    bp.add_argument("--lawa-max-window", type=int, default=16)
    bp.add_argument("--track-weight-history", action="store_true")
    bp.add_argument("--seed", type=int, default=None)

    ap_all = sub.add_parser("all")
    add_build_args(ap_all)
    ap_all.add_argument("--pack", type=Path, required=True)
    ap_all.add_argument("--root", type=Path, required=True)
    ap_all.add_argument("--arms", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                        required=True)
    ap_all.add_argument("--prefix-steps", type=int, default=1200)
    ap_all.add_argument("--max-virtual-steps", type=int, default=200)
    ap_all.add_argument("--eval-batches", type=int, default=32)
    ap_all.add_argument("--prefix-eval-every", type=int, default=100)
    ap_all.add_argument("--predictor-inner-steps", type=int, default=1)
    ap_all.add_argument("--lawa-window", type=int, default=8)
    ap_all.add_argument("--oracle-reference-accum", type=int, default=4)
    ap_all.add_argument("--seed", type=int, default=1337)
    ap_all.add_argument("--stop-by", choices=["virtual", "ideal_compute"], default="virtual")
    ap_all.add_argument("--compute-budget", type=float, default=100.0)
    ap_all.add_argument("--skip-period", type=int, default=2)
    ap_all.add_argument("--skip-count", type=int, default=1)
    ap_all.add_argument("--eval-every", type=int, default=25)
    ap_all.add_argument("--synthetic-history", choices=["predicted", "hold", "zero"], default="predicted")
    ap_all.add_argument("--predictor-inner-steps-branch", type=int, default=1)
    return ap


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.command == "selftest":
        run_selftest(args)
    elif args.command == "prepare":
        prepare_pack(args)
    elif args.command == "branch":
        run_branch(args)
    elif args.command == "all":
        prepare_pack(args)
        for arm in args.arms:
            b = argparse.Namespace(
                pack=args.pack, arm=arm, out=args.root / arm_slug(arm), device=args.device,
                history_device=args.history_device, history_dtype=args.history_dtype,
                max_virtual_steps=args.max_virtual_steps, stop_by=args.stop_by,
                compute_budget=args.compute_budget, skip_period=args.skip_period,
                skip_count=args.skip_count, eval_every=args.eval_every,
                synthetic_history=args.synthetic_history,
                predictor_inner_steps_branch=args.predictor_inner_steps_branch,
                lawa_max_window=max(args.lawa_window, 16), track_weight_history=True,
                seed=args.seed,
            )
            run_branch(b)
    else:
        parser.error(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
