"""Online meta-optimizer: learn to jump forward, with constant memory.

The whole point is that nothing is archived. At any moment we hold:

    * the model's current parameters                    (1 vector, N floats)
    * a sliding window of the last B checkpoint deltas  (B vectors)
    * a tiny neural net operating on r-dimensional coefficients

and nothing else. Storage is O(B*N), constant in the length of the run, versus
O(steps * N) for a checkpoint archive. Each delta is absorbed into the window
the moment it is produced and the source checkpoints are dropped.

Representation
--------------
Let D be the [B, N] matrix of stored deltas. Every candidate update is a linear
combination c^T D, so the model only ever emits B numbers regardless of N. We
never materialise an explicit orthonormal basis: with G = D D^T (a B x B Gram
matrix, cheap), the Cholesky factor G = L L^T gives an orthonormal frame
U = L^{-1} D implicitly, and all the geometry can be done in B dimensions.

Prediction task
---------------
Given the p deltas ending at time t, predict the sum of the next m deltas, in
the orthonormal frame of those p deltas. Both input and target come from the
window itself, so a training pair is available at every step with no extra
compute and no stored history:

    input   deltas [a, b)  with b - a = p, expressed as the Cholesky factor L
    target  sum of deltas [b, b + m), expressed in the same frame

Building the frame from only the input deltas (not the whole window) is what
keeps the target out of the model's input.

Everything here is exact in B dimensions; the only N-dimensional operations are
the Gram matrix, the window update, and materialising the final update.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# compute ledger
# ---------------------------------------------------------------------------

@dataclass
class Ledger:
    """Every FLOP the method spends, itemised, so nothing hides."""
    gd_steps: int = 0
    jump_attempts: int = 0
    jumps_accepted: int = 0
    probe_forwards: int = 0
    flops: Dict[str, float] = field(default_factory=lambda: {
        "gd_step": 0.0, "probe_eval": 0.0, "gram": 0.0,
        "window_update": 0.0, "materialize": 0.0, "meta_net": 0.0})
    seconds: Dict[str, float] = field(default_factory=lambda: {
        "gd_step": 0.0, "probe_eval": 0.0, "meta_overhead": 0.0})

    def total_flops(self) -> float:
        return sum(self.flops.values())

    def total_seconds(self) -> float:
        return sum(self.seconds.values())

    def overhead_fraction(self) -> float:
        t = self.total_flops()
        return 0.0 if t <= 0 else (t - self.flops["gd_step"]) / t

    def report(self) -> str:
        t = max(self.total_flops(), 1e-30)
        lines = [f"  {'component':<16}{'FLOPs':>12}{'share':>9}"]
        for k, v in sorted(self.flops.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {k:<16}{v:>12.3e}{100*v/t:>8.3f}%")
        lines.append(f"  {'TOTAL':<16}{t:>12.3e}")
        lines.append(f"  gd steps {self.gd_steps}, jumps {self.jumps_accepted}"
                     f"/{self.jump_attempts} accepted, probe forwards {self.probe_forwards}")
        lines.append(f"  wall clock: " + ", ".join(
            f"{k} {v:.1f}s" for k, v in self.seconds.items()))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# constant-memory delta window
# ---------------------------------------------------------------------------

class DeltaWindow:
    """Sliding window of the last `size` strided parameter deltas.

    Two design points that determine whether large windows (= large achievable
    rank) are actually affordable:

    MEMORY: this stores B raw N-dimensional vectors densely, so it costs
    B * N * dtype_bytes. At N=73.5M and fp32 that is ~294MB PER STORED DELTA --
    window=256 alone is ~75GB, competing with the model, activations, and
    optimizer state for the same GPU. `store_device='cpu'` moves the window to
    host RAM (typically far more plentiful than GPU memory in a Slurm
    allocation), trading a small per-event transfer for much larger achievable
    B. `store_dtype=torch.bfloat16` halves it again, at the cost of compounding
    with any rounding noise already in the checkpoints.

    COMPUTE: naively recomputing the B x B Gram matrix from scratch costs
    O(B^2 * N) and would dominate training itself once B is more than a few
    hundred. Instead the Gram is maintained INCREMENTALLY: a new delta needs
    only its inner products against the B-1 deltas already in the window --
    O(B*N), linear in B rather than quadratic -- and the Gram matrix rolls the
    same way the window does. This is the difference between rank being
    genuinely cheap at any size (matching the O(rN) cost of LoRA-style
    application) and rank being quadratically expensive; without it, "just
    raise the rank" silently stops being affordable.

    `stride` matters independently of both: consecutive-step deltas are nearly
    parallel and dominated by gradient noise, and when checkpoints are bf16 they
    can be dominated by rounding. Recording every `stride` steps makes each
    delta larger relative to both noise floors and the Gram better conditioned.
    """

    def __init__(self, size: int, stride: int, numel: int, compute_device,
                 store_device=None, dtype=torch.float32):
        self.size, self.stride, self.numel = size, stride, numel
        self.compute_device = compute_device
        self.device = store_device or compute_device   # where D physically lives
        self.dtype = dtype
        self.D = torch.zeros(size, numel, device=self.device, dtype=dtype)
        self.G = torch.zeros(size, size, dtype=torch.float64)   # maintained incrementally, on CPU
        self.count = 0
        self.anchor: Optional[torch.Tensor] = None
        self.since = 0

    @property
    def full(self) -> bool:
        return self.count >= self.size

    def bytes(self) -> int:
        return self.D.numel() * self.D.element_size() + (
            self.anchor.numel() * self.anchor.element_size() if self.anchor is not None else 0)

    @staticmethod
    def memory_report(size: int, numel: int, dtype=torch.float32) -> str:
        bpe = torch.zeros(1, dtype=dtype).element_size()
        gb = size * numel * bpe / 1e9
        return f"window(size={size}) needs {gb:.2f} GB ({bpe} bytes/elt, N={numel:,})"

    def observe(self, theta: torch.Tensor) -> bool:
        """Feed the current parameters. Returns True if a delta was recorded."""
        theta_c = theta.detach().to(self.compute_device)
        if self.anchor is None:
            self.anchor = theta_c.to(self.dtype).to(self.device).clone()
            return False
        self.since += 1
        if self.since < self.stride:
            return False
        self.since = 0
        d = (theta_c.to(self.dtype).to(self.device) - self.anchor)

        if self.count < self.size:
            # window not yet full: append, no roll needed
            i = self.count
            cross = (self.D[:i] @ d.to(self.D.dtype)).double().cpu() if i > 0 else                 torch.zeros(0, dtype=torch.float64)
            self.D[i] = d
            self.G[:i, i] = cross
            self.G[i, :i] = cross
            self.G[i, i] = float(d.double() @ d.double())
        else:
            # full: inner products of the new delta against the CURRENT window
            # (before the roll), then roll both D and G the same way and drop
            # the row/col belonging to the delta that just fell off
            cross = (self.D @ d.to(self.D.dtype)).double().cpu()      # [B]
            self.D = torch.roll(self.D, -1, dims=0)
            self.D[-1] = d
            self.G = torch.roll(torch.roll(self.G, -1, dims=0), -1, dims=1)
            self.G[:-1, -1] = cross[1:]
            self.G[-1, :-1] = cross[1:]
            self.G[-1, -1] = float(d.double() @ d.double())

        self.anchor.copy_(theta_c.to(self.dtype).to(self.device))
        self.count += 1
        return True

    def reset(self, theta: torch.Tensor) -> None:
        """Drop the window. Used after a jump: the stored deltas describe a
        trajectory the parameters have just left discontinuously."""
        self.D.zero_()
        self.G.zero_()
        self.count = 0
        self.since = 0
        self.anchor = theta.detach().to(self.dtype).to(self.device).clone()

    def gram(self) -> torch.Tensor:
        """B x B Gram matrix, maintained incrementally (see class docstring).
        Returned on CPU, where the meta net and all B-dimensional algebra live."""
        return self.G.clone()

    def combine(self, coeffs: torch.Tensor) -> torch.Tensor:
        """c^T D -- the only O(rN) operation at prediction time, cheap at any r."""
        c = coeffs.to(self.D.dtype).to(self.device)
        out = c @ self.D
        return out.to(self.compute_device)


# ---------------------------------------------------------------------------
# frame algebra, all in B dimensions
# ---------------------------------------------------------------------------

def _chol(G: torch.Tensor, jitter: float = 1e-10) -> Optional[torch.Tensor]:
    s = float(torch.diagonal(G).mean().clamp(min=1e-300))
    for k in range(6):
        try:
            return torch.linalg.cholesky(G + (jitter * (10 ** k)) * s * torch.eye(
                G.shape[0], dtype=G.dtype, device=G.device))
        except Exception:
            continue
    return None


def frame_and_target(G: torch.Tensor, a: int, b: int, m: int
                     ) -> Optional[Tuple[torch.Tensor, torch.Tensor, float]]:
    """Latent input and target for 'deltas [a,b) predict sum of [b, b+m)'.

    Returns (L, target_coeffs, captured) where L is the p x p Cholesky factor of
    the input deltas' Gram matrix (their coordinates in their own orthonormal
    frame), target_coeffs are the target's coordinates in that frame, and
    `captured` is the fraction of the target's squared norm that lies inside the
    span at all -- a hard ceiling on any rank-p predictor.
    """
    Gin = G[a:b, a:b]
    L = _chol(Gin)
    if L is None:
        return None
    v = G[a:b, b:b + m].sum(dim=1)                       # D_in @ target
    coeffs = torch.cholesky_solve(v.unsqueeze(1), L).squeeze(1)   # Gin^{-1} D_in t
    latent = L.T @ coeffs                                # coordinates in U frame
    tt = float(G[b:b + m, b:b + m].sum())                # ||target||^2
    captured = float((latent @ latent) / tt) if tt > 0 else float("nan")
    return L, latent, captured


def latent_to_delta_coeffs(L: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """Coordinates in the orthonormal frame -> coefficients over the raw deltas.

    With U = L^{-1} D orthonormal, a vector with frame coordinates z is
    v = U^T z = D^T L^{-T} z, so the delta coefficients are c = L^{-T} z.
    (Solving with L instead of L^T is a silent, catastrophic error: the
    coefficients come out finite and plausible but point nowhere useful.)
    """
    return torch.linalg.solve_triangular(
        L.transpose(-1, -2), latent.unsqueeze(1), upper=True).squeeze(1)


# ---------------------------------------------------------------------------
# per-coordinate map: f(theta_j history, data) -> update_j
# ---------------------------------------------------------------------------

class DataEncoder(nn.Module):
    """Tiny transformer over the upcoming batch's token ids -> one embedding.

    Deliberately small (2 layers, d_model 64) so its cost is negligible against
    the model being optimised. It sees WHICH tokens are coming, not how the big
    model responds to them -- anything requiring the latter would cost a forward
    pass of the big model, which is the thing we are trying to avoid.
    """

    def __init__(self, vocab: int, d_model: int = 64, layers: int = 2,
                 heads: int = 4, out_dim: int = 32, max_tokens: int = 512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, d_model * 2,
                                         batch_first=True, dropout=0.0)
        self.tr = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(d_model, out_dim)
        self.max_tokens = max_tokens
        self.out_dim = out_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        flat = token_ids.reshape(-1)
        if flat.numel() > self.max_tokens:                # subsample for cost
            idx = torch.linspace(0, flat.numel() - 1, self.max_tokens,
                                 device=flat.device).long()
            flat = flat[idx]
        h = self.tr(self.emb(flat).unsqueeze(0))
        return self.head(h.mean(dim=1)).squeeze(0)


class CoordNet(nn.Module):
    """update_j = g(recent deltas at coordinate j [, data embedding]).

    The key difference from the latent proposers: this is applied to EVERY
    coordinate independently and its output is therefore FULL RANK. It can
    express directions outside span(recent deltas), which is the constraint the
    ray profile shows to be binding -- inside that span the loss-optimal move is
    backward, so no re-weighting of past directions can produce forward
    progress.

    The network itself has O(1) parameters regardless of model size: it is
    applied 73.5M times, not sized 73.5M. At hidden 32 that is ~500 FLOPs per
    parameter, about 0.02% of one training step.
    """

    def __init__(self, p: int, data_dim: int = 0, hidden: int = 32):
        super().__init__()
        self.p, self.data_dim = p, data_dim
        d_in = p + data_dim
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))
        self.skip = nn.Linear(d_in, 1)
        for m in list(self.net) + [self.skip]:
            if isinstance(m, nn.Linear):
                m.weight.data *= 0.1
                m.bias.data.zero_()

    def forward(self, feats: torch.Tensor, data_emb: Optional[torch.Tensor] = None,
                drop_data: bool = False) -> torch.Tensor:
        """data_emb=None with data_dim>0 means 'no data', encoded as a zero
        embedding rather than a shorter input -- otherwise the plain `coord`
        branch calls a net built for `coord_data` and the widths disagree.

        Training applies the same zero embedding at random (see learn_coord), so
        both the with-data and without-data settings are in distribution and the
        pair is a fair ablation using one network."""
        if self.data_dim:
            if data_emb is None or drop_data:
                pad = torch.zeros(feats.shape[0], self.data_dim,
                                  device=feats.device, dtype=feats.dtype)
            else:
                pad = data_emb.reshape(1, -1).expand(feats.shape[0], -1).to(feats.dtype)
            feats = torch.cat([feats, pad], dim=-1)
        return (self.net(feats) + self.skip(feats)).squeeze(-1)

    def n_params(self) -> int:
        return sum(q.numel() for q in self.parameters())


def target_weights(m: int, mode: str, device, dtype) -> torch.Tensor:
    """Weights over the next m deltas defining what the model is trained toward.

    'next' : plain sum -> predicts theta_{t+m}, i.e. exactly where gradient
             descent lands. On a trajectory whose steps overshoot (the ray
             profile finds the loss-optimal point half a step BACK from where
             the optimizer put it) a perfect predictor faithfully reproduces
             that overshoot. Predicting GD well and beating GD are different
             objectives.
    'avg'  : weights (m-i)/m -> predicts the MEAN of theta over the next m
             steps, which is where the loss is actually low. Same supervision
             cost, different target.
    """
    if mode == "avg":
        w = torch.tensor([(m - i) / m for i in range(m)], device=device, dtype=dtype)
    else:
        w = torch.ones(m, device=device, dtype=dtype)
    return w


# ---------------------------------------------------------------------------
# the meta net
# ---------------------------------------------------------------------------

class MetaNet(nn.Module):
    """(latent history, horizon) -> latent update.

    Inputs and outputs are p-dimensional, so this net's size is independent of
    the model being optimised: the same architecture works for a 70M model and a
    70B one. A linear skip runs alongside the MLP so the net can represent the
    linear extrapolator exactly and only has to learn the departure from it.
    """

    def __init__(self, p: int, hidden: int = 128):
        super().__init__()
        d_in = p * p + 2
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, p))
        self.skip = nn.Linear(d_in, p)
        for mod in list(self.net) + [self.skip]:
            if isinstance(mod, nn.Linear):
                mod.weight.data *= 0.1
                mod.bias.data.zero_()
        self.p = p

    @staticmethod
    def featurise(L: torch.Tensor, m: int):
        """Returns (features, scale). The input frame is normalised by its own
        scale, so the net sees a scale-free problem; the TARGET must be divided
        by the same scale or the net is asked to predict absolute magnitudes
        from scale-free inputs, which it cannot do. Working in these units also
        makes training pairs comparable across time, so they can be replayed."""
        scale = float(L.diagonal().abs().mean().clamp(min=1e-30))
        feats = torch.cat([(L / scale).reshape(-1),
                           torch.tensor([float(m), math.log1p(m)], dtype=L.dtype,
                                        device=L.device)])
        return feats, scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)

    def n_params(self) -> int:
        return sum(q.numel() for q in self.parameters())


# ---------------------------------------------------------------------------
# candidate proposers: the meta net and its controls
# ---------------------------------------------------------------------------

def propose(kind: str, window: DeltaWindow, G: torch.Tensor, p: int, m: int,
            net: Optional[MetaNet], generator: Optional[torch.Generator] = None
            ) -> Optional[torch.Tensor]:
    """Coefficients over the stored deltas for a proposed jump of horizon m.

    kind:
      meta    the live-trained net
      last    m copies of the most recent delta -- linear extrapolation, the
              cheapest possible forecaster and a real baseline. Fails badly at
              the edge of stability: if consecutive deltas alternate sign, this
              extrapolates the OSCILLATION and overshoots quadratically in m.

      mean    m copies of the MEAN of the last p deltas. Averaging cancels the
              alternating component and leaves the drift, so this steps forward
              along net progress rather than along the oscillation. This is the
              proposer the oscillation diagnosis calls for: it keeps lawa's
              cancellation benefit while still moving forward instead of back.

      damped  extrapolate only the persistent part: deltas are weighted by how
              strongly each correlates with the window mean, so oscillatory
              directions are suppressed and consistent ones survive.

      spectral  the per-mode rule that the other proposers are special cases of.
              In the window's own orthonormal frame, each latent mode is treated
              as AR(1): estimate its contraction ratio r_i from consecutive
              coordinates, then continue it for m steps with the finite
              geometric sum d_i * r_i (1 - r_i^m) / (1 - r_i).
                r near 1   -> jump ~m steps forward (the drift modes; this is
                              where the synthetic-quadratic gains came from)
                r near 0   -> barely move
                r negative -> half-step back to the oscillation centre (the
                              LAWA-like benefit, but applied ONLY to the
                              oscillating modes instead of to everything)
              Uniform averaging and uniform extrapolation are the two global
              limits of this estimator; per-mode, it does the right thing at
              the edge of stability AND on a clean quadratic with one formula.
      lawa    move to the mean of the last p recorded parameter vectors. This is
              latest weight averaging: it moves INTO the hull of recent
              iterates, the opposite direction to extrapolation. Both lower the
              loss, and a loss check alone cannot tell them apart, which is why
              it has to be run as a control.
      random  a random direction in the same span with the norm the meta net
              chose. Passed through the identical acceptance test, this is what
              separates "the net found a good direction" from "the acceptance
              test filtered out the bad ones".
    """
    B = window.size
    a, b = B - p, B
    if kind == "spectral":
        L = _chol(G[a:b, a:b])
        if L is None:
            return None
        # Rows of L are the latent coordinates of the p input deltas in their
        # own orthonormal frame. Fit the one-step transition delta_{t+1} = A
        # delta_t by ridge regression on the p-1 consecutive pairs, then jump
        # with the MATRIX geometric sum  sum_{k=1..m} A^k d_last.
        #
        # This is the latent VAR(1) used as a proposer. Its eigenvalues are the
        # per-mode ratios the scalar version tried to estimate, but in the
        # correct (data-determined) eigenbasis: eigenvalues near +1 extrapolate
        # the drift ~m steps, negative or complex eigenvalues damp/rotate the
        # oscillation toward its centre. Uniform averaging (lawa/mean) and
        # uniform extrapolation (last) are the two global limits.
        X0 = L[:-1, :]
        X1 = L[1:, :]
        gram_in = X0.T @ X0
        lam = 0.05 * float(torch.diagonal(gram_in).mean().clamp(min=1e-300))
        A = torch.linalg.solve(gram_in + lam * torch.eye(p, dtype=L.dtype), X0.T @ X1)
        # keep the geometric sum convergent: scale down if spectral radius >= 1
        try:
            rad = float(torch.linalg.eigvals(A).abs().max())
        except Exception:
            rad = float(torch.linalg.matrix_norm(A, 2))
        if rad > 0.999:
            A = A * (0.999 / rad)
        d_last = L[p - 1, :]
        Pk = A.clone()
        y = Pk.T @ d_last
        for _ in range(1, m):
            Pk = Pk @ A
            y = y + Pk.T @ d_last
        c_sub = latent_to_delta_coeffs(L, y)
        c = torch.zeros(B, dtype=torch.float64)
        c[a:b] = c_sub
        return c
    if kind == "mean":
        c = torch.zeros(B, dtype=torch.float64)
        c[a:b] = float(m) / p
        return c
    if kind == "damped":
        Gs = G[a:b, a:b]
        mean_ip = Gs.mean(dim=1)                       # <d_i, mean delta>
        scale = mean_ip.abs().max().clamp(min=1e-300)
        w = (mean_ip / scale).clamp(min=0.0)           # drop anti-correlated
        if float(w.sum()) <= 0:
            return None
        w = w / w.sum()
        c = torch.zeros(B, dtype=torch.float64)
        c[a:b] = w * float(m)
        return c
    if kind == "lawa":
        # mean of last p thetas minus current theta, in delta coordinates
        c = torch.zeros(B, dtype=torch.float64)
        for i in range(1, p):
            c[B - i:] += -1.0 / p
        return c
    if kind == "last":
        c = torch.zeros(B, dtype=torch.float64)
        c[-1] = float(m)
        return c

    L = _chol(G[a:b, a:b])
    if L is None:
        return None
    if kind == "meta":
        if net is None:
            return None
        with torch.no_grad():
            x, scale = MetaNet.featurise(L.float(), m)
            latent = net(x.unsqueeze(0)).squeeze(0).double() * scale
    elif kind == "random":
        if net is None:
            return None
        with torch.no_grad():
            x, scale = MetaNet.featurise(L.float(), m)
            ref = net(x.unsqueeze(0)).squeeze(0).double() * scale
        g = torch.randn(p, dtype=torch.float64, generator=generator)
        latent = g / g.norm().clamp(min=1e-30) * ref.norm()
    else:
        raise ValueError(f"unknown proposer {kind!r}")

    c_sub = latent_to_delta_coeffs(L, latent)
    c = torch.zeros(B, dtype=torch.float64)
    c[a:b] = c_sub
    return c


# ---------------------------------------------------------------------------
# the live learner
# ---------------------------------------------------------------------------

class OnlineMeta:
    """Maintains the window and trains the net from it, live, every step."""

    def __init__(self, numel: int, device, window: int = 16, stride: int = 4,
                 p: int = 8, hidden: int = 128, lr: float = 3e-3,
                 inner_steps: int = 4, replay: int = 4096, batch: int = 64,
                 dtype=torch.float32, store_device=None):
        assert p < window, "need at least one delta beyond the input frame"
        self.win = DeltaWindow(window, stride, numel, compute_device=device,
                               store_device=store_device, dtype=dtype)
        self.p, self.m_max = p, window - p
        self.net = MetaNet(p, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.inner_steps = inner_steps
        self.numel = numel
        # latent pairs are (p*p+2) + p floats, so thousands cost nothing and
        # they are directly comparable across time thanks to the scale-free
        # parameterisation above
        self.replay_x: List[torch.Tensor] = []
        self.replay_y: List[torch.Tensor] = []
        self.replay_cap = replay
        self.batch = batch
        self.train_loss: List[float] = []
        self.captured: List[float] = []
        # captured[(rank, horizon)] -> list of fractions. Drift accumulates like
        # k and gradient noise like sqrt(k), so if the low-rank picture is right
        # this should RISE with horizon. If it is flat, the update is noise all
        # the way down and no rank helps.
        self.captured_grid: Dict[tuple, List[float]] = {}

    def bytes(self) -> int:
        return self.win.bytes()

    def memory_report(self) -> str:
        return DeltaWindow.memory_report(self.win.size, self.win.numel, self.win.dtype) \
               + f"  [stored on {self.win.device}]"

    def observe_and_learn(self, theta: torch.Tensor, ledger: Optional[Ledger] = None) -> bool:
        """One live update. Call after every optimizer step."""
        recorded = self.win.observe(theta)
        if ledger is not None:
            ledger.flops["window_update"] += 2.0 * self.numel
        if not (recorded and self.win.full):
            return False
        G = self.win.gram()
        if ledger is not None:
            ledger.flops["gram"] += 2.0 * (self.win.size ** 2) * self.numel

        pairs = []
        B = self.win.size
        for m in range(1, self.m_max + 1):
            b = B - m
            a = b - self.p
            if a < 0:
                continue
            got = frame_and_target(G, a, b, m)
            if got is None:
                continue
            L, target, cap = got
            for p_alt in (2, 4, 8, 12):
                if p_alt > self.p or b - p_alt < 0:
                    continue
                alt = frame_and_target(G, b - p_alt, b, m)
                if alt is not None:
                    self.captured_grid.setdefault((p_alt, m), []).append(alt[2])
            feats, scale = MetaNet.featurise(L.float(), m)
            pairs.append((feats, target.float() / scale))     # SAME units
            self.captured.append(cap)
        if not pairs:
            return False

        for x, y in pairs:
            self.replay_x.append(x); self.replay_y.append(y)
        if len(self.replay_x) > self.replay_cap:
            self.replay_x = self.replay_x[-self.replay_cap:]
            self.replay_y = self.replay_y[-self.replay_cap:]

        Xall = torch.stack(self.replay_x)
        Yall = torch.stack(self.replay_y)
        n = Xall.shape[0]
        last = 0.0
        for _ in range(self.inner_steps):
            idx = torch.randint(0, n, (min(self.batch, n),))
            pred = self.net(Xall[idx])
            loss = ((pred - Yall[idx]) ** 2).mean() / (float(Yall[idx].pow(2).mean()) + 1e-30)
            self.opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()
            last = float(loss.detach())
        self.train_loss.append(last)
        if ledger is not None:
            ledger.flops["meta_net"] += 6.0 * self.net.n_params() * len(pairs) * self.inner_steps
        return True

    def capture_now(self, m: int) -> Optional[float]:
        """Captured fraction at the CURRENT window, for horizon m.

        Recorded at every probe so learnability can be tracked as a function of
        training step, rather than reported once as an aggregate. This is what
        makes 'the trajectory only becomes learnable much later in training'
        a testable claim instead of a hypothesis.
        """
        if not self.win.full:
            return None
        G = self.win.gram()
        B = self.win.size
        b = B - m
        a = b - self.p
        if a < 0:
            return None
        got = frame_and_target(G, a, b, m)
        return None if got is None else got[2]

    def oscillation(self) -> Optional[float]:
        """Lag-1 correlation between consecutive deltas, from the Gram matrix.

        Negative means consecutive updates partly cancel: the trajectory is
        oscillating rather than advancing, which is the edge-of-stability
        signature. It also explains why extrapolating the last delta overshoots
        while averaging helps -- and it is the reason to prefer the `mean`
        proposer over `last`.
        """
        if not self.win.full:
            return None
        G = self.win.gram()
        B = self.win.size
        vals = []
        for i in range(B - 1):
            ni = float(G[i, i]) ** 0.5
            nj = float(G[i + 1, i + 1]) ** 0.5
            if ni > 0 and nj > 0:
                vals.append(float(G[i, i + 1]) / (ni * nj))
        return float(np.median(vals)) if vals else None

    def capture_table(self) -> str:
        """Median captured fraction by rank and horizon."""
        import numpy as _np
        ranks = sorted({k[0] for k in self.captured_grid})
        hs = sorted({k[1] for k in self.captured_grid})
        if not ranks:
            return "  (no data)"
        out = ["  fraction of the true update lying in the rank-r span",
               "  (rows: rank, cols: horizon in optimizer steps)",
               "  " + "rank".ljust(6) + "".join(f"{h*self.win.stride:>9}" for h in hs)]
        for r in ranks:
            row = f"  {r:<6}"
            for h in hs:
                v = self.captured_grid.get((r, h))
                row += f"{_np.median(v):>9.4f}" if v else f"{'-':>9}"
            out.append(row)
        out.append("  rising along a row means drift is accumulating faster than")
        out.append("  noise, so longer jumps are MORE forecastable, not less.")
        return "\n".join(out)

    def init_coord(self, data_dim: int = 0, hidden: int = 32, lr: float = 1e-3,
                   sample: int = 200_000, target: str = "avg") -> None:
        self.coord = CoordNet(self.p, data_dim, hidden).to(self.win.device)
        self.coord_opt = torch.optim.Adam(self.coord.parameters(), lr=lr)
        self.coord_sample = sample
        self.coord_target = target
        self.coord_data_dropout = 0.5
        self.coord_loss: List[float] = []
        self.coord_cos: List[float] = []

    def _coord_feats(self, idx: torch.Tensor, a: int, b: int) -> torch.Tensor:
        return self.win.D[a:b, idx].T.float()

    def learn_coord(self, data_emb: Optional[torch.Tensor] = None,
                    ledger: Optional[Ledger] = None) -> Optional[float]:
        """One online update of the per-coordinate map, on sampled coordinates."""
        if not self.win.full or not hasattr(self, "coord"):
            return None
        B = self.win.size
        m = max(1, min(self.m_max, B - self.p))
        b = B - m
        a = b - self.p
        if a < 0:
            return None
        idx = torch.randint(0, self.win.numel, (self.coord_sample,),
                            device=self.win.device)
        feats = self._coord_feats(idx, a, b)
        w = target_weights(m, self.coord_target, self.win.device, torch.float32)
        tgt = (self.win.D[b:b + m, :][:, idx].float() * w.unsqueeze(1)).sum(0)
        scale = float(feats.abs().mean().clamp(min=1e-30))
        drop = (data_emb is not None
                and float(torch.rand(1)) < self.coord_data_dropout)
        pred = self.coord(feats / scale, data_emb, drop_data=drop) * scale
        loss = ((pred - tgt) ** 2).mean() / (float(tgt.pow(2).mean()) + 1e-30)
        self.coord_opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.coord.parameters(), 1.0)
        self.coord_opt.step()
        with torch.no_grad():
            c = float(torch.nn.functional.cosine_similarity(
                pred.detach().reshape(1, -1), tgt.reshape(1, -1)))
        self.coord_loss.append(float(loss.detach()))
        self.coord_cos.append(c)
        if ledger is not None:
            ledger.flops["meta_net"] += 6.0 * self.coord.n_params() * self.coord_sample
        return c

    @torch.no_grad()
    def propose_coord(self, m: int, data_emb: Optional[torch.Tensor] = None,
                      chunk: int = 4_000_000, ledger: Optional[Ledger] = None
                      ) -> Optional[torch.Tensor]:
        """Full-rank update: apply the per-coordinate map to every parameter."""
        if not self.win.full or not hasattr(self, "coord"):
            return None
        B = self.win.size
        b, a = B, B - self.p
        out = torch.zeros(self.win.numel, device=self.win.device,
                          dtype=self.win.D.dtype)
        scale = float(self.win.D[a:b].abs().mean().clamp(min=1e-30))
        mm = max(1, min(self.m_max, m))
        for lo in range(0, self.win.numel, chunk):
            hi = min(lo + chunk, self.win.numel)
            feats = self.win.D[a:b, lo:hi].T.float() / scale
            out[lo:hi] = (self.coord(feats, data_emb) * scale).to(out.dtype)
        if ledger is not None:
            ledger.flops["meta_net"] += 2.0 * self.coord.n_params() * self.win.numel
        return out * (float(m) / mm)

    def propose(self, kind: str, m: int, generator=None) -> Optional[torch.Tensor]:
        if not self.win.full:
            return None
        G = self.win.gram()
        return propose(kind, self.win, G, self.p, m, self.net, generator)

    def materialise(self, coeffs: torch.Tensor, ledger: Optional[Ledger] = None
                    ) -> torch.Tensor:
        if ledger is not None:
            ledger.flops["materialize"] += 2.0 * self.win.size * self.numel
        return self.win.combine(coeffs)
