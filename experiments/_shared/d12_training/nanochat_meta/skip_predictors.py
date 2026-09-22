"""Online predictors used by :mod:`nanochat_meta.skip_suite`.

There are two deliberately different learned families:

1. ``CoefficientNet`` predicts coefficients over the recent gradient history.
   It is cheap and tests *temporal predictability inside a known subspace*.
2. ``CoordinateNet`` is shared across coordinates and predicts every gradient
   coordinate from its recent values. It is full-rank and tests whether the
   history-span restriction is the real bottleneck.

Both can be history-only, raw-token-conditioned, or forward-feature-conditioned.
The raw-token arm is a control. The forward-feature arm runs the base model's
forward pass but skips its backward pass, exposing how the current model responds
on the upcoming batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
import math
import time

import torch
import torch.nn as nn

from .skip_core import VectorHistory, optimizer_first_moment


FORWARD_FEATURE_DIM = 12


def history_feature_vector(history: VectorHistory, capacity: int,
                           step_frac: float, lr_mult: float) -> torch.Tensor:
    """Scale-free feature vector from the chronological history Gram matrix."""
    G = history.gram()
    W = len(history)
    pad = torch.zeros((capacity, capacity), dtype=torch.float32)
    norms = torch.zeros(capacity, dtype=torch.float32)
    if W:
        d = G.diagonal().clamp_min(1e-30).sqrt()
        corr = G / (d[:, None] * d[None, :]).clamp_min(1e-30)
        pad[-W:, -W:] = corr.float().clamp(-4, 4)
        logn = d.log()
        logn = logn - logn.mean()
        norms[-W:] = logn.float().clamp(-20, 20)
    scalars = torch.tensor([
        float(W) / capacity,
        float(step_frac),
        float(lr_mult),
        math.log1p(max(step_frac, 0.0)),
    ], dtype=torch.float32)
    return torch.cat([pad.reshape(-1), norms, scalars])


class TokenEncoder(nn.Module):
    """Small token-set encoder; deliberately much cheaper than a base forward."""

    def __init__(self, vocab_size: int, dim: int = 32, max_tokens: int = 1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.proj = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.dim = dim
        self.max_tokens = max_tokens

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.reshape(-1)
        if ids.numel() > self.max_tokens:
            # Deterministic subsample so branch comparisons see the same feature.
            idx = torch.linspace(0, ids.numel() - 1, self.max_tokens,
                                 device=ids.device).long()
            ids = ids[idx]
        z = self.embedding(ids)
        feat = torch.cat([z.mean(0), z.float().var(0, unbiased=False).sqrt().to(z.dtype)], 0)
        return self.proj(feat)


class CoefficientNet(nn.Module):
    def __init__(self, history: int, context_dim: int = 0, hidden: int = 256):
        super().__init__()
        base = history * history + history + 4
        self.history = history
        self.context_dim = context_dim
        self.net = nn.Sequential(
            nn.Linear(base + context_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, history),
        )
        # Start at zero. Residual variants then exactly recover the momentum baseline.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hist_features: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if hist_features.ndim == 1:
            hist_features = hist_features.unsqueeze(0)
        if self.context_dim:
            if context is None:
                context = torch.zeros(hist_features.shape[0], self.context_dim,
                                      device=hist_features.device,
                                      dtype=hist_features.dtype)
            elif context.ndim == 1:
                context = context.unsqueeze(0)
            x = torch.cat([hist_features, context.to(hist_features.dtype)], -1)
        else:
            x = hist_features
        return self.net(x)


class CoordinateNet(nn.Module):
    """Shared map from p recent coordinate values to the next value."""

    def __init__(self, order: int = 4, context_dim: int = 0, hidden: int = 48):
        super().__init__()
        self.order = order
        self.context_dim = context_dim
        d = order + context_dim
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.skip = nn.Linear(d, 1, bias=False)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        nn.init.zeros_(self.skip.weight)
        # A weak last-gradient initialization is safer than a random full-rank output.
        with torch.no_grad():
            self.skip.weight[0, order - 1] = 0.1

    def forward(self, recent: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.context_dim:
            if context is None:
                c = torch.zeros(recent.shape[0], self.context_dim,
                                device=recent.device, dtype=recent.dtype)
            else:
                c = context.reshape(1, -1).expand(recent.shape[0], -1).to(recent.dtype)
            recent = torch.cat([recent, c], -1)
        return (self.net(recent) + self.skip(recent)).squeeze(-1)


@dataclass
class PredictorContext:
    hist_features: torch.Tensor
    token_ids: Optional[torch.Tensor]
    raw_embedding: Optional[torch.Tensor]
    forward_features: Optional[torch.Tensor]
    step_frac: float
    lr_mult: float


class PredictorBank:
    """Owns and trains all neural predictors during a shared exact prefix."""

    def __init__(self, history_capacity: int, vocab_size: int,
                 device: torch.device, token_dim: int = 32,
                 coeff_hidden: int = 256, coord_hidden: int = 48,
                 coord_order: int = 4, lr: float = 2e-3,
                 coord_sample: int = 131_072, ridge: float = 1e-4):
        self.device = torch.device(device)
        self.history_capacity = history_capacity
        self.coord_order = coord_order
        self.coord_sample = coord_sample
        self.ridge = ridge
        self.token_encoder = TokenEncoder(vocab_size, token_dim).to(self.device)
        self.coeff = CoefficientNet(history_capacity, 0, coeff_hidden).to(self.device)
        self.coeff_resid = CoefficientNet(history_capacity, 0, coeff_hidden).to(self.device)
        self.coeff_raw = CoefficientNet(history_capacity, token_dim, coeff_hidden).to(self.device)
        self.coeff_forward = CoefficientNet(history_capacity, FORWARD_FEATURE_DIM, coeff_hidden).to(self.device)
        self.coord = CoordinateNet(coord_order, 0, coord_hidden).to(self.device)
        self.coord_resid = CoordinateNet(coord_order, 0, coord_hidden).to(self.device)
        self.coord_raw = CoordinateNet(coord_order, token_dim, coord_hidden).to(self.device)
        self.coord_forward = CoordinateNet(coord_order, FORWARD_FEATURE_DIM, coord_hidden).to(self.device)
        modules = [self.token_encoder, self.coeff, self.coeff_resid, self.coeff_raw,
                   self.coeff_forward, self.coord, self.coord_resid,
                   self.coord_raw, self.coord_forward]
        params = [p for m in modules for p in m.parameters()]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        self.train_steps = 0
        self.metrics: list[dict[str, float]] = []

    @property
    def modules(self) -> dict[str, nn.Module]:
        return {
            "token_encoder": self.token_encoder,
            "coeff": self.coeff,
            "coeff_resid": self.coeff_resid,
            "coeff_raw": self.coeff_raw,
            "coeff_forward": self.coeff_forward,
            "coord": self.coord,
            "coord_resid": self.coord_resid,
            "coord_raw": self.coord_raw,
            "coord_forward": self.coord_forward,
        }

    def train(self, mode: bool = True) -> None:
        for m in self.modules.values():
            m.train(mode)

    def eval(self) -> None:
        self.train(False)

    def make_context(self, history: VectorHistory, token_ids: Optional[torch.Tensor],
                     forward_features: Optional[torch.Tensor], step_frac: float,
                     lr_mult: float, *, require_raw_grad: bool = False) -> PredictorContext:
        hf = history_feature_vector(history, self.history_capacity, step_frac, lr_mult).to(self.device)
        raw = None
        if token_ids is not None:
            raw = self.token_encoder(token_ids.to(self.device))
            if not require_raw_grad:
                raw = raw.detach()
        ff = None if forward_features is None else forward_features.to(self.device).float()
        return PredictorContext(hf, token_ids, raw, ff, step_frac, lr_mult)

    def _pad_coeffs(self, c: torch.Tensor, W: int) -> torch.Tensor:
        # Networks always emit capacity coefficients. History occupies the last W slots.
        return c[..., -W:]

    def _coefficient_loss(self, pred: torch.Tensor, target: torch.Tensor,
                          G: torch.Tensor, target_norm_sq: float) -> torch.Tensor:
        d = pred.double() - target.double()
        q = d @ G.to(d.device) @ d
        return q.float() / max(target_norm_sq, 1e-30)

    def observe_true_gradient(self, history: VectorHistory, true_grad: torch.Tensor,
                              momentum: torch.Tensor, token_ids: Optional[torch.Tensor],
                              forward_features: Optional[torch.Tensor], step_frac: float,
                              lr_mult: float, active: Optional[set[str]] = None,
                              inner_steps: int = 1) -> dict[str, float]:
        """Online supervised update using information available before this gradient.

        ``history`` must *not* yet contain ``true_grad``. This is the causal/no-leakage
        invariant that the earlier experiments did not consistently enforce.
        """
        if len(history) < max(3, self.coord_order):
            return {}
        active = active or {
            "coeff", "coeff_resid", "coeff_raw", "coeff_forward",
            "coord", "coord_resid", "coord_raw", "coord_forward",
        }
        W = len(history)
        G = history.gram().to(self.device)
        true_norm_sq = float(true_grad.float().pow(2).sum())
        resid = true_grad - momentum
        resid_norm_sq = float(resid.float().pow(2).sum())
        c_true = history.project_coefficients(true_grad, self.ridge).to(self.device).float()
        c_resid = history.project_coefficients(resid, self.ridge).to(self.device).float()
        # Raw embedding needs gradients during training.
        ctx = self.make_context(history, token_ids, forward_features, step_frac, lr_mult,
                                require_raw_grad=True)

        # Coordinate sample and normalization. The same indices train every coordinate arm.
        k = min(self.coord_sample, history.numel)
        idx = torch.randint(0, history.numel, (k,), device=history.compute_device)
        recent_rows = list(history._vectors)[-self.coord_order:]
        recent = torch.stack([
            v[idx.to(v.device)].to(device=self.device, dtype=torch.float32)
            for v in recent_rows
        ], dim=1)
        scale = recent.square().mean(1).sqrt().clamp_min(1e-8)
        recent_n = recent / scale[:, None]
        tgt = true_grad[idx].to(self.device).float() / scale
        tgt_resid = resid[idx].to(self.device).float() / scale

        losses: dict[str, torch.Tensor] = {}
        for _ in range(max(1, inner_steps)):
            self.optimizer.zero_grad(set_to_none=True)
            if "coeff" in active:
                cp = self._pad_coeffs(self.coeff(ctx.hist_features).squeeze(0), W)
                losses["coeff"] = self._coefficient_loss(cp, c_true, G, true_norm_sq)
            if "coeff_resid" in active:
                cp = self._pad_coeffs(self.coeff_resid(ctx.hist_features).squeeze(0), W)
                losses["coeff_resid"] = self._coefficient_loss(cp, c_resid, G, resid_norm_sq)
            if "coeff_raw" in active and ctx.raw_embedding is not None:
                cp = self._pad_coeffs(self.coeff_raw(ctx.hist_features, ctx.raw_embedding).squeeze(0), W)
                losses["coeff_raw"] = self._coefficient_loss(cp, c_true, G, true_norm_sq)
            if "coeff_forward" in active and ctx.forward_features is not None:
                cp = self._pad_coeffs(self.coeff_forward(ctx.hist_features, ctx.forward_features).squeeze(0), W)
                losses["coeff_forward"] = self._coefficient_loss(cp, c_true, G, true_norm_sq)
            if "coord" in active:
                losses["coord"] = (self.coord(recent_n) - tgt).square().mean() / tgt.square().mean().clamp_min(1e-8)
            if "coord_resid" in active:
                losses["coord_resid"] = (self.coord_resid(recent_n) - tgt_resid).square().mean() / tgt_resid.square().mean().clamp_min(1e-8)
            if "coord_raw" in active and ctx.raw_embedding is not None:
                losses["coord_raw"] = (self.coord_raw(recent_n, ctx.raw_embedding) - tgt).square().mean() / tgt.square().mean().clamp_min(1e-8)
            if "coord_forward" in active and ctx.forward_features is not None:
                losses["coord_forward"] = (self.coord_forward(recent_n, ctx.forward_features) - tgt).square().mean() / tgt.square().mean().clamp_min(1e-8)
            if not losses:
                return {}
            total = torch.stack(list(losses.values())).mean()
            total.backward()
            nn.utils.clip_grad_norm_([p for m in self.modules.values() for p in m.parameters()], 1.0)
            self.optimizer.step()
        self.train_steps += 1
        out = {k: float(v.detach()) for k, v in losses.items()}
        self.metrics.append(out)
        return out

    @torch.no_grad()
    def coefficient_prediction(self, kind: str, history: VectorHistory,
                               momentum: torch.Tensor, context: PredictorContext) -> torch.Tensor:
        W = len(history)
        if kind == "coeff":
            c = self.coeff(context.hist_features).squeeze(0)
            base = None
        elif kind == "coeff_resid":
            c = self.coeff_resid(context.hist_features).squeeze(0)
            base = momentum
        elif kind == "coeff_raw":
            if context.raw_embedding is None:
                raise RuntimeError("raw-token predictor requested without token context")
            c = self.coeff_raw(context.hist_features, context.raw_embedding).squeeze(0)
            base = None
        elif kind == "coeff_forward":
            if context.forward_features is None:
                raise RuntimeError("forward predictor requested without forward features")
            c = self.coeff_forward(context.hist_features, context.forward_features).squeeze(0)
            base = None
        else:
            raise ValueError(kind)
        pred = history.combine(c[-W:].double())
        return pred if base is None else base + pred

    @torch.no_grad()
    def coordinate_prediction(self, kind: str, history: VectorHistory,
                              momentum: torch.Tensor, context: PredictorContext,
                              chunk: int = 1_000_000) -> torch.Tensor:
        if len(history) < self.coord_order:
            raise RuntimeError("insufficient history for coordinate predictor")
        if kind == "coord":
            net, ctx, base = self.coord, None, None
        elif kind == "coord_resid":
            net, ctx, base = self.coord_resid, None, momentum
        elif kind == "coord_raw":
            if context.raw_embedding is None:
                raise RuntimeError("raw-token predictor requested without token context")
            net, ctx, base = self.coord_raw, context.raw_embedding, None
        elif kind == "coord_forward":
            if context.forward_features is None:
                raise RuntimeError("forward predictor requested without forward context")
            net, ctx, base = self.coord_forward, context.forward_features, None
        else:
            raise ValueError(kind)
        rows = list(history._vectors)[-self.coord_order:]
        out = torch.empty(history.numel, device=history.compute_device, dtype=torch.float32)
        for lo in range(0, history.numel, chunk):
            hi = min(lo + chunk, history.numel)
            recent = torch.stack([
                v[lo:hi].to(device=self.device, dtype=torch.float32) for v in rows
            ], 1)
            scale = recent.square().mean(1).sqrt().clamp_min(1e-8)
            y = net(recent / scale[:, None], ctx) * scale
            out[lo:hi].copy_(y.to(out.device))
        return out if base is None else base + out

    def state_dict(self) -> dict[str, Any]:
        return {
            "modules": {k: v.state_dict() for k, v in self.modules.items()},
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "metrics": self.metrics[-500:],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        for k, sd in state["modules"].items():
            self.modules[k].load_state_dict(sd)
        self.optimizer.load_state_dict(state["optimizer"])
        self.train_steps = int(state.get("train_steps", 0))
        self.metrics = list(state.get("metrics", []))


def summarize_forward_features(logits: torch.Tensor, targets: torch.Tensor,
                               loss: torch.Tensor, max_tokens: int = 2048) -> torch.Tensor:
    """Cheap detached summaries of a base-model forward pass.

    The vector intentionally contains no activations requiring architectural hooks,
    so it works across NanoChat commits. It can later be replaced with richer layer
    sketches without changing the rest of the experiment harness.
    """
    with torch.no_grad():
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        valid = flat_targets >= 0
        flat_logits = flat_logits[valid]
        flat_targets = flat_targets[valid]
        if flat_logits.shape[0] > max_tokens:
            idx = torch.linspace(0, flat_logits.shape[0] - 1, max_tokens,
                                 device=flat_logits.device).long()
            flat_logits = flat_logits[idx]
            flat_targets = flat_targets[idx]
        probs = flat_logits.softmax(-1)
        logp = probs.clamp_min(1e-20).log()
        entropy = -(probs * logp).sum(-1)
        top2 = probs.topk(2, dim=-1).values
        target_p = probs.gather(1, flat_targets[:, None]).squeeze(1)
        maxp = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        vals = torch.stack([
            loss.detach().float(),
            flat_logits.float().mean(),
            flat_logits.float().std(unbiased=False),
            entropy.mean(), entropy.std(unbiased=False),
            target_p.mean(), target_p.std(unbiased=False),
            maxp.mean(), maxp.std(unbiased=False),
            margin.mean(), margin.std(unbiased=False),
            (target_p < 0.01).float().mean(),
        ])
        return vals.float()
