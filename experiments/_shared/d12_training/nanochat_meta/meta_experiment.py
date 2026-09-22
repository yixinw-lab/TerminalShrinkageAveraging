#!/usr/bin/env python3
"""Live meta-optimization on nanochat, with an equal-compute fork.

  phase 1   train normally for --fork-step steps. After every optimizer step,
            absorb the parameter delta into a constant-memory window and take a
            few SGD steps on the meta net. Nothing is written to disk; the only
            state kept is the window (B full-dim vectors) plus a tiny net.

  fork      snapshot parameters + optimizer state in CPU RAM. No checkpoints.

  phase 2   run several branches from that snapshot, each until the SAME
            compute budget is exhausted, and evaluate all of them on the same
            held-out data at the same budget points.

Branches
    gd       plain training. the thing to beat.
    meta     GD interleaved with jumps proposed by the live-trained net
    last     same loop, jump = horizon x most recent delta (linear extrapolation)
    lawa     same loop, jump = move to the mean of recent weights (latest weight
             averaging). Moves INTO the hull of recent iterates rather than
             along the trajectory; it also lowers loss, and a loss check alone
             cannot distinguish it from extrapolation, so it must be run.
    random   same loop, jump = random direction in the same span with the norm
             the meta net chose. Isolates how much of any gain comes from the
             acceptance test filtering rather than from the proposal.

Fairness
    * The meta net is trained ONLY during the shared prefix, live, and the cost
      of doing so is measured and reported. There is no free burn-in.
    * A rejected jump costs exactly what it costs: one meta forward plus the
      probe evaluations. No invented penalty.
    * Every branch stops at the same cumulative budget, measured in training
      FLOPs, with jump overhead charged to the branch that incurs it.
    * Evaluation for the plot is instrumentation: it is performed at identical
      budget points for every branch and is NOT charged to any of them.
    * The acceptance probe uses fresh batches, never the batch the anchor step
      just used.
    * Optimizer state handling after a jump is a flag, applied identically to
      every jumping branch.

A jump skips its horizon in training steps, so a jumping branch also consumes
fewer tokens. This is a compute saving and a data saving; it is not a claim
about sample efficiency at fixed compute.
"""
from __future__ import annotations

import argparse, copy, inspect, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nanochat_meta.online_meta import (Ledger, OnlineMeta, DeltaWindow,  # noqa: E402
                                       DataEncoder)

BRANCHES = ["gd", "meta", "spectral", "mean", "damped", "last", "lawa",
            "random", "coord", "coord_data"]
COLORS = {"gd": "black", "meta": "#d62728", "spectral": "#ff7f0e",
          "mean": "#1f77b4", "damped": "#17becf", "last": "#2ca02c",
          "lawa": "#7f7f7f", "random": "#9467bd", "coord": "#8c564b",
          "coord_data": "#e377c2"}


def flat(params) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in params])


def unflat(vec: torch.Tensor, params) -> None:
    i = 0
    for p in params:
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p).to(p.dtype))
        i += n


def build(args):
    """Model, optimizer, loaders -- constructed the way base_train does."""
    from nanochat.common import compute_init, get_base_dir, autodetect_device_type
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import get_tokenizer, get_token_bytes
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit

    device_type = autodetect_device_type()
    device = torch.device(args.device if args.device else device_type)
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") \
        else len(tokenizer)

    cfg_fields = set(GPTConfig.__dataclass_fields__)
    want = {"sequence_len": args.max_seq_len, "max_seq_len": args.max_seq_len,
            "vocab_size": vocab_size, "n_layer": args.depth, "depth": args.depth,
            "model_dim": args.depth * args.aspect_ratio,
            "n_embd": args.depth * args.aspect_ratio}
    cfg = GPTConfig(**{k: v for k, v in want.items() if k in cfg_fields})
    print(f"[build] GPTConfig fields used: "
          f"{ {k: v for k, v in want.items() if k in cfg_fields} }")
    model = GPT(cfg).to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[build] {n/1e6:.1f}M parameters")

    sig = inspect.signature(model.setup_optimizer)
    print(f"[build] setup_optimizer{sig}")
    cand = {"embedding_lr": args.embedding_lr, "unembedding_lr": args.unembedding_lr,
            "matrix_lr": args.matrix_lr, "scalar_lr": args.scalar_lr,
            "weight_decay": args.weight_decay, "init_lr_frac": 1.0}
    kwargs = {k: v for k, v in cand.items() if k in sig.parameters}
    print(f"[build] optimizer kwargs: {kwargs}")
    optimizer = model.setup_optimizer(**kwargs)

    mk = lambda split: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split=split, device=device)
    tb = get_token_bytes()
    if hasattr(tb, "to"):
        tb = tb.to(device)          # indexed by y, which is on the accelerator
    return model, optimizer, mk, tb, device


def lr_multiplier(it: int, warmup: int, total: int = 0,
                  warmdown_ratio: float = 0.0, final_lr_frac: float = 1.0) -> float:
    """Warmup -> constant -> linear warmdown, mirroring base_train.

    warmdown_ratio=0 gives the constant schedule (what every run so far used).
    A real decay matters here: Meterez et al. (A Defense of the Quadratic Model,
    arXiv 2607.21716) report that local Taylor expansions predict LLM dynamics
    far better late in training than early -- agreement grows from ~0-2% of the
    budget at the 10-50% checkpoints to 7-10% at the 80-100% checkpoints. A run
    at constant LR with no defined horizon is permanently in the early regime,
    so forecastability there is measured where the effect is weakest."""
    if it < warmup:
        return (it + 1) / warmup
    if warmdown_ratio <= 0 or total <= 0:
        return 1.0
    warmdown_iters = max(int(warmdown_ratio * total), 1)
    if it < total - warmdown_iters:
        return 1.0
    progress = max(0.0, (total - it) / warmdown_iters)      # 1 -> 0
    return progress * 1.0 + (1 - progress) * final_lr_frac


def muon_momentum(it: int, warmup: int) -> float:
    """base_train warms Muon momentum from 0.85 to 0.97 over the warmup."""
    frac = min(it / max(warmup, 1), 1.0)
    return 0.85 * (1 - frac) + 0.97 * frac


def apply_schedules(optimizer, it: int, warmup: int, total: int = 0,
                    warmdown_ratio: float = 0.0, final_lr_frac: float = 1.0) -> None:
    lrm = lr_multiplier(it, warmup, total, warmdown_ratio, final_lr_frac)
    mom = muon_momentum(it, warmup)
    for g in optimizer.param_groups:
        if "initial_lr" not in g:
            g["initial_lr"] = g["lr"]
        g["lr"] = g["initial_lr"] * lrm
        if "momentum" in g:
            g["momentum"] = mom


def train_step(model, optimizer, loader, ledger, key="gd_step", flops_per=0.0,
               accum: int = 1, it: int = 0, warmup: int = 40, total: int = 0,
               warmdown_ratio: float = 0.0, final_lr_frac: float = 1.0):
    t0 = time.time()
    apply_schedules(optimizer, it, warmup, total, warmdown_ratio, final_lr_frac)
    loss_val = 0.0
    for _ in range(accum):
        x, y = next(loader)
        loss = model(x, y)
        if loss.ndim > 0:
            loss = loss.mean()
        (loss / accum).backward()
        loss_val += float(loss.detach()) / accum
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ledger.seconds[key] = ledger.seconds.get(key, 0.0) + (time.time() - t0)
    ledger.flops["gd_step"] += flops_per
    ledger.gd_steps += 1
    return loss_val


@torch.no_grad()
def probe_loss(model, batches, ledger, flops_per_fwd):
    t0 = time.time()
    tot = 0.0
    for x, y in batches:
        l = model(x, y)
        tot += float(l.mean() if l.ndim > 0 else l)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ledger.seconds["probe_eval"] += time.time() - t0
    ledger.flops["probe_eval"] += flops_per_fwd * len(batches)
    ledger.probe_forwards += len(batches)
    return tot / max(len(batches), 1)


@torch.no_grad()
def val_bpb_batches(model, batches, token_bytes):
    """Per-batch bpb. SE comes from the spread ACROSS batches, which works even
    when a loader hands back overlapping shards -- the previous shard-level SE
    silently reported 0.0 when the two shards turned out to be identical data,
    which made every significance verdict in the output meaningless."""
    from nanochat.loss_eval import evaluate_bpb
    return [float(evaluate_bpb(model, iter([b]), 1, token_bytes)) for b in batches]


def summarize(vals):
    import numpy as _np
    v = _np.asarray(vals, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0


def run_quality(args, model, optimizer, params, meta, loader, eval_batches,
                token_bytes, F_STEP, tokens_per_step, N):
    """Does a jump buy real progress? Measured against the run's own curve.

    Training proceeds normally and val bpb is logged every --quality-period
    steps, giving the reference: what plain GD achieves at each step. At each
    logging point every proposer also gets to jump from the current parameters;
    the resulting bpb is measured and the parameters are restored.

    A jump is then scored by the only thing that matters: the number of real GD
    steps needed to reach the same bpb. That is a direct answer to 'is this
    worth doing', it needs no acceptance test, and it cannot be gamed by taking
    fewer steps -- which is exactly how the earlier branch race produced a
    spurious 'meta BETTER'.
    """
    print(f"\n{'='*70}\nQUALITY MODE\n{'='*70}")
    kinds = [k for k in args.branches if k != "gd"] or ["meta", "last", "lawa", "random"]
    rays: List[dict] = []
    track: List[dict] = []          # learnability vs training step
    want_coord = any(k.startswith("coord") for k in kinds)
    want_data = "coord_data" in kinds
    encoder = None
    if want_coord:
        meta.init_coord(data_dim=(args.data_dim if want_data else 0),
                        hidden=args.coord_hidden, lr=args.coord_lr,
                        sample=args.coord_sample, target=args.coord_target)
        print(f"[coord] per-coordinate map: {meta.coord.n_params():,} params, "
              f"target='{args.coord_target}', applied to {meta.win.numel:,} "
              f"coordinates (full rank -- can express directions OUTSIDE "
              f"span(recent deltas), which the ray profile shows is the "
              f"binding constraint)")
        if want_data:
            vocab = int(model.config.vocab_size) if hasattr(model, "config") else 32768
            encoder = DataEncoder(vocab, args.data_model, args.data_layers,
                                  out_dim=args.data_dim).to(flat(params).device)
            meta.coord_opt.add_param_group({"params": list(encoder.parameters())})
            print(f"[coord] data encoder: {sum(q.numel() for q in encoder.parameters()):,} "
                  f"params, {args.data_layers}-layer transformer over the UPCOMING "
                  f"batch's token ids")
    pending = None
    curve: List[tuple] = []          # (step, bpb) for plain GD
    trials: List[dict] = []
    compounds: List[dict] = []
    led = Ledger()
    gen = torch.Generator().manual_seed(0)
    total = args.fork_step

    for step in range(total):
        train_step(model, optimizer, loader, led, flops_per=F_STEP,
                   accum=args.grad_accum, it=step, warmup=args.warmup_steps,
                   total=args.total_iterations, warmdown_ratio=args.warmdown_ratio,
                   final_lr_frac=args.final_lr_frac)
        meta.observe_and_learn(flat(params), led)
        if want_coord:
            # the embedding describes the batch that is about to be consumed,
            # so the map is conditioned on upcoming data rather than past data
            emb = None
            if encoder is not None:
                if pending is None:
                    pending = tuple(t.detach().clone() for t in next(loader))
                emb = encoder(pending[0]).unsqueeze(0)
            meta.learn_coord(emb, led)

        if step % args.quality_period != 0 or step == 0:
            continue
        base_m, base_s = summarize(val_bpb_batches(model, eval_batches, token_bytes))
        curve.append((step, base_m))
        track.append({"step": step, "bpb": base_m,
                      "capture_h1": meta.capture_now(1),
                      "capture_hmax": meta.capture_now(max(args.quality_horizons)),
                      "lag1": meta.oscillation(),
                      "meta_loss": (meta.train_loss[-1] if meta.train_loss else None),
                      "coord_cos": (float(np.mean(meta.coord_cos[-50:]))
                                    if getattr(meta, "coord_cos", None) else None)})

        if not meta.win.full:
            continue
        theta0 = flat(params).clone()

        # RAY PROFILE: loss along theta + t * d_last, for t on a grid.
        # This reads the curvature directly, which the norm-weighted lag-1
        # correlation cannot: lag-1 is +0.80 (coherent drift dominates the
        # NORM) while extrapolation still fails (the sharp, low-norm directions
        # dominate the LOSS). The location of the minimum is the answer to "how
        # far can a jump go": t* near 0 means the optimizer step is already at
        # the curvature limit and no extrapolation along this direction can
        # help, whatever proposes it. t* > 1 means there is room.
        if args.ray_profile and step % (args.quality_period * 4) == 0:
            d_last = meta.win.D[-1].to(theta0.dtype)
            prof = []
            for t in args.ray_ts:
                unflat(theta0 + t * d_last, params)
                mt, _ = summarize(val_bpb_batches(model, eval_batches, token_bytes))
                prof.append((t, mt))
            unflat(theta0, params)
            t_star = min(prof, key=lambda z: z[1])[0]
            rays.append({"step": step, "profile": prof, "t_star": t_star})
            print("    [ray] " + "  ".join(f"t={t:+.1f}:{v:.5f}" for t, v in prof)
                  + f"   argmin t*={t_star:+.1f}", flush=True)
        for kind in kinds:
            for h in args.quality_horizons:
                if kind.startswith("coord"):
                    emb = None
                    if kind == "coord_data" and encoder is not None:
                        if pending is None:
                            pending = tuple(t.detach().clone() for t in next(loader))
                        with torch.no_grad():
                            emb = encoder(pending[0]).unsqueeze(0)
                    upd = meta.propose_coord(h, emb, ledger=led)
                    if upd is None:
                        continue
                else:
                    c = meta.propose(kind, h, generator=gen)
                    if c is None:
                        continue
                    upd = meta.materialise(c, led)
                unflat(theta0 + upd.to(theta0.dtype), params)
                m, sd = summarize(val_bpb_batches(model, eval_batches, token_bytes))
                trials.append({"step": step, "kind": kind,
                               "horizon_steps": h * args.stride,
                               "base_bpb": base_m, "jump_bpb": m, "se": sd,
                               "delta": m - base_m})
                unflat(theta0, params)
        ml = meta.train_loss[-1] if meta.train_loss else float("nan")
        if want_coord and getattr(meta, "coord_cos", None):
            print(f"    [coord] prediction cosine with the true update "
                  f"(last 50): {np.mean(meta.coord_cos[-50:]):+.3f}", flush=True)
        osc = meta.oscillation()
        print(f"  step {step:5d}  bpb {base_m:.5f}  meta_loss {ml:.5f}  "
              f"lag1 {osc if osc is not None else float('nan'):+.3f}  "
              f"trials {len(trials)}", flush=True)

        if args.compound_steps > 0 and step % args.compound_period == 0:
            best = min((t for t in trials if t["step"] == step),
                       key=lambda t: t["jump_bpb"], default=None)
            if best is not None and best["jump_bpb"] < base_m:
                opt_state = copy.deepcopy(optimizer.state_dict())
                c = meta.propose(best["kind"], best["horizon_steps"] // max(args.stride, 1),
                                 generator=gen)
                if c is not None:
                    # arm A: continue from the jumped parameters
                    unflat(theta0 + meta.materialise(c).to(theta0.dtype), params)
                    for j in range(args.compound_steps):
                        train_step(model, optimizer, loader, led, flops_per=F_STEP,
                                   accum=args.grad_accum, it=step + j,
                                   warmup=args.warmup_steps, total=args.total_iterations,
                                   warmdown_ratio=args.warmdown_ratio,
                                   final_lr_frac=args.final_lr_frac)
                    a_m, _ = summarize(val_bpb_batches(model, eval_batches, token_bytes))
                    # arm B: continue from the un-jumped parameters, same budget
                    unflat(theta0, params)
                    optimizer.load_state_dict(copy.deepcopy(opt_state))
                    for j in range(args.compound_steps):
                        train_step(model, optimizer, loader, led, flops_per=F_STEP,
                                   accum=args.grad_accum, it=step + j,
                                   warmup=args.warmup_steps, total=args.total_iterations,
                                   warmdown_ratio=args.warmdown_ratio,
                                   final_lr_frac=args.final_lr_frac)
                    b_m, _ = summarize(val_bpb_batches(model, eval_batches, token_bytes))
                    compounds.append({"step": step, "kind": best["kind"],
                                      "immediate_gain": base_m - best["jump_bpb"],
                                      "after_jump": a_m, "after_plain": b_m,
                                      "persisted_gain": b_m - a_m})
                    print(f"    [compound] {best['kind']}: immediate "
                          f"{base_m - best['jump_bpb']:+.5f}, after "
                          f"{args.compound_steps} more steps "
                          f"{b_m - a_m:+.5f} {'(persists)' if b_m - a_m > 0 else '(gone)'}",
                          flush=True)
                    unflat(theta0, params)
                    optimizer.load_state_dict(copy.deepcopy(opt_state))

    # score each jump against the reference curve
    steps_arr = np.array([c[0] for c in curve])
    bpb_arr = np.array([c[1] for c in curve])
    for t in trials:
        later = steps_arr > t["step"]
        reach = steps_arr[later][bpb_arr[later] <= t["jump_bpb"]]
        t["gd_steps_equivalent"] = int(reach[0] - t["step"]) if len(reach) else None
        # is plain GD actually making progress here? if not, the equivalent is
        # meaningless
        win = (steps_arr > t["step"]) & (steps_arr <= t["step"] + 200)
        t["gd_descending"] = bool(win.any() and
                                  (t["base_bpb"] - bpb_arr[win].min()) > 0.002)

    print(f"\n{'='*70}\nJUMP QUALITY: is a jump worth real gradient steps?\n{'='*70}")
    print("  (restricted to trials where plain GD is measurably descending: in "
          "flat\n   regions any improvement converts to an unbounded step-"
          "equivalent, which is\n   how the earlier branch race produced a "
          "spurious winner.)")
    print(f"{'proposer':>10}{'horizon':>9}{'n':>5}{'median dbpb':>13}"
          f"{'% helped':>10}{'median GD-steps worth':>23}")
    print("-" * 70)
    for kind in kinds:
        for h in sorted({t["horizon_steps"] for t in trials}):
            sub = [t for t in trials if t["kind"] == kind and t["horizon_steps"] == h
                   and t.get("gd_descending", True)]
            if not sub:
                continue
            d = np.array([t["delta"] for t in sub])
            eq = [t["gd_steps_equivalent"] for t in sub
                  if t["gd_steps_equivalent"] is not None]
            helped = 100.0 * float((d < 0).mean())
            med_eq = f"{np.median(eq):.0f}" if eq else "0 (never helps)"
            print(f"{kind:>10}{h:>9}{len(sub):>5}{np.median(d):>+13.5f}"
                  f"{helped:>9.0f}%{med_eq:>23}")
    print("\n  dbpb < 0 means the jump lowered validation loss.")
    print("  'GD-steps worth' is how many real steps reach the same bpb: it is the")
    print("  speedup, and it must exceed the jump's probe cost to be worth doing.")
    print(f"  For reference one probe evaluation here costs about "
          f"{2*len(eval_batches)*args.device_batch_size*args.max_seq_len/(3*tokens_per_step):.2f}"
          f" GD steps.")

    args.out.mkdir(parents=True, exist_ok=True)
    if track:
        import numpy as _np
        print(f"\n{'='*70}\nIS IT GETTING MORE LEARNABLE WITH TRAINING?\n{'='*70}")
        n = len(track)
        for label, sl in (("first third", slice(0, n // 3)),
                          ("middle third", slice(n // 3, 2 * n // 3)),
                          ("last third", slice(2 * n // 3, n))):
            seg = track[sl]
            f = lambda k: _np.median([t[k] for t in seg if t.get(k) is not None]) \
                if any(t.get(k) is not None for t in seg) else float("nan")
            print(f"  {label:>13}  steps {seg[0]['step']:>5}-{seg[-1]['step']:<5}"
                  f"  capture(h=1) {f('capture_h1'):.4f}"
                  f"  lag1 {f('lag1'):+.3f}"
                  f"  meta_loss {f('meta_loss'):.5f}"
                  + (f"  coord_cos {f('coord_cos'):+.3f}"
                     if any(t.get('coord_cos') is not None for t in track) else ""))
        print("  Rising capture / falling meta_loss across thirds is the signature")
        print("  of the trajectory becoming more predictable as training proceeds.")

    if rays:
        import numpy as _np
        ts = [r["t_star"] for r in rays]
        print(f"\n{'='*70}\nRAY PROFILE: how far can a jump go?\n{'='*70}")
        print(f"  median argmin t* = {_np.median(ts):+.2f}  over {len(rays)} profiles")
        print(f"  distribution: {sorted(ts)}")
        if _np.median(ts) <= 0.25:
            print("  t* ~ 0: the optimizer step is ALREADY at the curvature limit.")
            print("  No proposer can gain by moving further along this direction --")
            print("  the ceiling is a property of the loss surface, not the model.")
            print("  Any speedup has to come from a DIFFERENT direction (damping the")
            print("  sharp modes while advancing the flat ones), not a longer step.")
        else:
            print(f"  t* > 0: there is room to move about {_np.median(ts):.1f}x the last")
            print("  update along the same direction. That is the headroom a jump can")
            print("  claim, and it bounds the per-jump speedup.")

    if compounds:
        print(f"\n{'='*70}\nDOES THE GAIN COMPOUND?\n{'='*70}")
        print(f"{'step':>7}{'proposer':>10}{'immediate':>12}"
              f"{'after training':>16}{'verdict':>12}")
        for c in compounds:
            v = "persists" if c["persisted_gain"] > 0 else "gone"
            print(f"{c['step']:>7}{c['kind']:>10}{c['immediate_gain']:>+12.5f}"
                  f"{c['persisted_gain']:>+16.5f}{v:>12}")
        print("\n  A gain that vanishes after continued training gives a better")
        print("  FINAL checkpoint but does not make training faster: the run")
        print("  rejoins the trajectory it would have taken anyway.")

    (args.out / "quality.json").write_text(json.dumps(
        {"curve": curve, "trials": trials, "compounds": compounds, "rays": rays,
         "track": track,
         "stride": args.stride, "rank": args.rank,
         "grad_accum": args.grad_accum, "n_eval_batches": len(eval_batches)}, indent=2))
    print(f"\nwrote {args.out}/quality.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--aspect-ratio", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--device-batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step. nanochat's default "
                         "total batch is ~524288 tokens; with device-batch 16 and "
                         "seq 2048 that is 16. Larger means less gradient noise, "
                         "which makes the trajectory easier to forecast, at "
                         "proportionally more time per step.")
    ap.add_argument("--warmup-steps", type=int, default=40)
    ap.add_argument("--total-iterations", type=int, default=0,
                    help="schedule horizon. Set this with --warmdown-ratio to "
                         "run a real decaying schedule and fork LATE, which is "
                         "the regime where local linearization is reported to "
                         "hold best. 0 = constant LR (the old behaviour).")
    ap.add_argument("--warmdown-ratio", type=float, default=0.0,
                    help="fraction of --total-iterations spent decaying")
    ap.add_argument("--final-lr-frac", type=float, default=0.1,
                    help="final LR as a fraction of peak, when warmdown is on")
    ap.add_argument("--fork-step", type=int, default=3000)
    ap.add_argument("--branch-steps", type=int, default=500,
                    help="budget = this many plain GD steps' worth of FLOPs")
    ap.add_argument("--branches", type=lambda s: s.split(","), default=BRANCHES)
    ap.add_argument("--embedding-lr", type=float, default=0.2)
    ap.add_argument("--unembedding-lr", type=float, default=0.004)
    ap.add_argument("--matrix-lr", type=float, default=0.02)
    ap.add_argument("--scalar-lr", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.28)
    # meta
    ap.add_argument("--window", type=int, default=0,
                    help="history length B > rank. 0 = auto (rank + "
                         "--window-margin).")
    ap.add_argument("--window-margin", type=int, default=32,
                    help="deltas beyond rank, needed so every horizon up to "
                         "window-rank has training pairs to learn from")
    ap.add_argument("--stride", type=int, default=8, help="record a delta every N steps")
    ap.add_argument("--rank", type=int, default=32,
                    help="target rank r. Application cost is O(r*N), cheap at "
                         "any r -- same scaling as LoRA. The window (history "
                         "length) must exceed r; see --window-margin.")
    ap.add_argument("--meta-hidden", type=int, default=128)
    ap.add_argument("--meta-lr", type=float, default=3e-3)
    ap.add_argument("--meta-inner", type=int, default=4)
    # jumping
    ap.add_argument("--jump-horizon", type=int, default=4, help="in window units")
    ap.add_argument("--jump-period", type=int, default=32,
                    help="minimum GD steps between jump ATTEMPTS. Last run "
                         "attempted every single step once armed and burned 63%% "
                         "of its whole budget on probe evaluations.")
    ap.add_argument("--jump-period-max", type=int, default=256,
                    help="rejections double the wait up to this cap; an "
                         "acceptance resets it. Rejection is evidence the "
                         "proposal is not good here yet, so back off.")
    ap.add_argument("--alphas", type=lambda s: [float(x) for x in s.split(",")],
                    default=[1.0],
                    help="jump scales to try per attempt; each costs a probe "
                         "evaluation. Default trusts the learned magnitude.")
    ap.add_argument("--probe-batches", type=int, default=4)
    ap.add_argument("--reset-opt-state", action="store_true",
                    help="clear optimizer moments after an accepted jump")
    ap.add_argument("--keep-window-after-jump", action="store_true")
    # eval
    ap.add_argument("--eval-shards", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--eval-points", type=int, default=8)
    ap.add_argument("--prefix-eval-every", type=int, default=25,
                    help="evaluate val bpb every N steps during the SHARED "
                         "prefix, so the plotted curve starts at step 0 and "
                         "shows the early representation-learning drop, not "
                         "just the post-fork window")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--mode", choices=["fork", "quality"], default="fork",
                    help="fork: the equal-budget branch race. quality: measure "
                         "each proposer's jump against the run's own val-loss "
                         "curve, converting every jump into 'how many GD steps "
                         "of progress was that worth'. quality answers whether "
                         "the mechanism works at all, with no acceptance test "
                         "and no budget accounting to confound it.")
    ap.add_argument("--quality-period", type=int, default=25,
                    help="evaluate, and try jumps, every N steps")
    ap.add_argument("--coord-hidden", type=int, default=32,
                    help="width of the per-coordinate map. Its parameter count "
                         "is independent of model size: it is APPLIED 73.5M "
                         "times, not sized 73.5M.")
    ap.add_argument("--coord-lr", type=float, default=1e-3)
    ap.add_argument("--coord-sample", type=int, default=200_000,
                    help="coordinates sampled per online update")
    ap.add_argument("--coord-target", choices=["next", "avg"], default="next",
                    help="next: squared error to the real delta-theta a true GD "
                         "step produced -- the standard supervised objective. "
                         "avg: the mean of the next m iterates. A model that "
                         "predicts 'next' perfectly reproduces GD exactly, "
                         "including any overshoot, so the gain has to come from "
                         "predicting k steps for the price of one.")
    ap.add_argument("--data-dim", type=int, default=32,
                    help="width of the batch embedding fed to the "
                         "per-coordinate map")
    ap.add_argument("--data-layers", type=int, default=2)
    ap.add_argument("--data-model", type=int, default=64)
    ap.add_argument("--ray-profile", action="store_true",
                    help="periodically profile the loss along theta + t*d_last. "
                         "Reads curvature directly; the argmin t* is the maximum "
                         "useful extrapolation along the current direction.")
    ap.add_argument("--ray-ts", type=lambda s: [float(x) for x in s.split(",")],
                    default=[-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--compound-steps", type=int, default=0,
                    help="after a jump, train this many more steps from BOTH the "
                         "jumped and un-jumped parameters and compare. This is "
                         "the difference between 'a better checkpoint at the end' "
                         "and 'training is genuinely faster': a gain that "
                         "disappears after continued training does not compound. "
                         "0 disables (costs 2x compound-steps per test).")
    ap.add_argument("--compound-period", type=int, default=400,
                    help="run the compound test every N steps (it is expensive)")
    ap.add_argument("--quality-horizons", type=lambda s: [int(x) for x in s.split(",")],
                    default=[1, 2, 4, 8],
                    help="jump horizons in WINDOW units (multiply by --stride "
                         "for optimizer steps)")
    ap.add_argument("--diagnose-only", action="store_true",
                    help="run phase 1 and report the capture table, then stop. "
                         "Cheap way to choose stride/rank/grad-accum before "
                         "spending time on branches.")
    ap.add_argument("--store-device", type=str, default="cuda", choices=["cuda", "cpu"],
                    help="where the delta window physically lives. cuda is fast "
                         "but shares GPU memory with the model/activations/"
                         "optimizer; cpu trades a small per-event transfer for "
                         "much larger achievable window (and hence rank).")
    ap.add_argument("--store-dtype", type=str, default="float32",
                    choices=["float32", "bfloat16"],
                    help="bfloat16 halves window memory. Only use it once "
                         "--diagnose-only confirms deltas are well above the "
                         "checkpoint rounding floor at your stride -- bf16 "
                         "storage compounds with bf16 checkpoint rounding.")
    ap.add_argument("--out", type=Path, default=Path("outputs/meta_fork"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, optimizer, mk_loader, token_bytes, device = build(args)
    params = [p for p in model.parameters()]
    N = sum(p.numel() for p in params)
    tokens_per_step = args.device_batch_size * args.max_seq_len * args.grad_accum
    F_STEP = 6.0 * N * tokens_per_step
    F_FWD = 2.0 * N * args.device_batch_size * args.max_seq_len
    print(f"[cost] {tokens_per_step:,} tokens/step, {F_STEP:.3e} FLOPs/step")

    train_loader = mk_loader("train")
    val_loader = mk_loader("val")
    import hashlib
    n_eval = args.eval_shards * args.eval_batches
    seen, eval_batches = set(), []
    for _ in range(n_eval * 20):            # oversample; keep only distinct
        if len(eval_batches) >= n_eval:
            break
        raw = next(val_loader)
        # CLONE: the dataloader reuses its output buffer, so keeping a reference
        # keeps a view of memory that the next call overwrites. Without this
        # every stored batch aliases the same tensor and the whole eval set
        # collapses to one repeated batch (SE identically 0).
        b = tuple(t.detach().clone() for t in raw)
        h = hashlib.md5(b[0].cpu().numpy().tobytes()).hexdigest()
        if h not in seen:
            seen.add(h); eval_batches.append(b)
    if len(eval_batches) < n_eval:
        print(f"[eval] only {len(eval_batches)} DISTINCT val batches available "
              f"(asked for {n_eval}); the val stream cycles. Evaluation uses what "
              f"exists. If this is 1, every comparison below is unmeasurable -- "
              f"run check_val.py and consider a train-stream holdout.")
    sigs = [hashlib.md5(b[0].cpu().numpy().tobytes()).hexdigest()
            for b in eval_batches]
    if len(set(sigs)) < len(sigs):
        print(f"[eval] WARNING: the val loader returned duplicate batches "
              f"({len(sigs)-len(set(sigs))} repeats among {len(sigs)}). The SE is "
              f"computed per batch so it stays valid, but the effective eval set "
              f"is smaller than requested.")
    print(f"[eval] {n_eval} val batches "
          f"({n_eval*args.device_batch_size*args.max_seq_len:,} tokens); "
          f"SE computed across batches")

    # probe batches for the acceptance test: drawn ONCE from the val stream,
    # fixed for the whole experiment. Drawing them from the training loader --
    # as before -- silently advanced the jumping branches' data stream past
    # batches the gd branch trained on, so the branches were not seeing the
    # same data. Fixed probes also make base-vs-candidate an exact paired
    # comparison on identical tokens.
    probe_batches = [tuple(t.detach().clone() for t in next(val_loader))
                     for _ in range(args.probe_batches)]

    # ---------------- phase 1: shared prefix with live meta learning --------
    window = args.window if args.window > 0 else args.rank + args.window_margin
    store_dtype = torch.bfloat16 if args.store_dtype == "bfloat16" else torch.float32
    store_device = torch.device(args.store_device)

    print(f"[meta] {DeltaWindow.memory_report(window, N, store_dtype)}")
    if store_device.type == "cuda" and torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        need_b = window * N * torch.zeros(1, dtype=store_dtype).element_size()
        print(f"[meta] GPU free: {free_b/1e9:.1f} GB / {total_b/1e9:.1f} GB total, "
              f"window needs {need_b/1e9:.1f} GB")
        if need_b > 0.6 * free_b:
            print(f"[meta] WARNING: window is >60% of free GPU memory. Model, "
                  f"activations, and optimizer state share this. Consider "
                  f"--store-device cpu or a smaller --rank/--window-margin.")

    meta = OnlineMeta(N, device, window=window, stride=args.stride, p=args.rank,
                      hidden=args.meta_hidden, lr=args.meta_lr,
                      inner_steps=args.meta_inner, dtype=store_dtype,
                      store_device=store_device)
    print(f"[meta] window {meta.bytes()/1e9:.2f} GB on {store_device}, "
          f"net {meta.net.n_params():,} params (net size is independent of N)")
    prefix = Ledger()
    prefix_curve: List[dict] = []
    t0 = time.time()
    for step in range(args.fork_step):
        l = train_step(model, optimizer, train_loader, prefix, flops_per=F_STEP,
                       accum=args.grad_accum, it=step, warmup=args.warmup_steps,
                       total=args.total_iterations,
                       warmdown_ratio=args.warmdown_ratio,
                       final_lr_frac=args.final_lr_frac)
        meta.observe_and_learn(flat(params), prefix)
        if args.prefix_eval_every > 0 and (step % args.prefix_eval_every == 0
                                           or step == args.fork_step - 1):
            pm, ps = summarize(val_bpb_batches(model, eval_batches, token_bytes))
            prefix_curve.append({"step": step, "bpb": pm, "se": ps,
                                 "flops": prefix.total_flops()})
        if step % 250 == 0 or step == args.fork_step - 1:
            ml = meta.train_loss[-1] if meta.train_loss else float("nan")
            print(f"  step {step:5d}  loss {l:.4f}  meta_loss {ml:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    print("\n[phase 1 ledger]"); print(prefix.report())
    cap = float(np.median(meta.captured)) if meta.captured else float("nan")
    print(f"\n[meta] median captured fraction at rank {args.rank}: {cap:.4f}")
    print(meta.capture_table())
    osc = meta.oscillation()
    if osc is not None:
        print(f"\n  lag-1 correlation between consecutive deltas: {osc:+.4f}")
        if osc < -0.02:
            print("  NEGATIVE: consecutive updates partly cancel. The trajectory is")
            print("  oscillating (edge of stability), so the dominant low-rank")
            print("  direction is the oscillation, not net progress. Extrapolating")
            print("  the last delta will overshoot and get worse with horizon;")
            print("  averaging first (the `mean`/`damped` proposers) is the fix.")
        elif osc > 0.02:
            print("  POSITIVE: consecutive updates reinforce, so the trajectory is")
            print("  advancing coherently and extrapolation should be safe.")
    print("\n  NOTE: this bounds how well a rank-r jump can REPRODUCE the gradient-"
          "\n  descent path. It does not bound how much it can lower the loss: the"
          "\n  part outside the span is largely gradient noise, which cancels over a"
          "\n  jump rather than contributing to progress. A captured fraction f"
          "\n  implies a drift-to-noise ratio of about sqrt(f/(1-f)) per window.")
    if cap == cap and cap > 0:
        print(f"  drift/noise ~= {math.sqrt(cap/max(1-cap,1e-9)):.3f}")


    if args.mode == "quality":
        run_quality(args, model, optimizer, params, meta, train_loader,
                    eval_batches, token_bytes, F_STEP, tokens_per_step, N)
        return

    # ---------------- fork: snapshot in RAM, nothing on disk ----------------
    snap_theta = flat(params).cpu().clone()
    snap_opt = copy.deepcopy(optimizer.state_dict())
    snap_win = meta.win.D.cpu().clone()
    snap_anchor = meta.win.anchor.cpu().clone()
    snap_count, snap_since = meta.win.count, meta.win.since
    snap_net = copy.deepcopy(meta.net.state_dict())
    print(f"[fork] snapshot in RAM: {(snap_theta.numel()*4 + snap_win.numel()*4)/1e9:.2f} GB")

    BUDGET = args.branch_steps * F_STEP
    eval_at = np.linspace(0, BUDGET, args.eval_points + 1)[1:]
    results: Dict[str, dict] = {}

    for branch in args.branches:
        print(f"\n{'='*64}\nbranch: {branch}\n{'='*64}", flush=True)
        unflat(snap_theta.to(device), params)
        optimizer.load_state_dict(copy.deepcopy(snap_opt))
        meta.win.D.copy_(snap_win.to(device)); meta.win.anchor.copy_(snap_anchor.to(device))
        meta.win.count, meta.win.since = snap_count, snap_since
        meta.net.load_state_dict(copy.deepcopy(snap_net))
        loader = mk_loader("train")
        gen = torch.Generator().manual_seed(1234)
        led = Ledger()
        m0, s0 = summarize(val_bpb_batches(model, eval_batches, token_bytes))
        trace = {"flops": [0.0], "steps": [0], "bpb": [m0], "bpb_se": [s0],
                 "tokens": [0], "jumps": []}
        nxt = 0
        since_jump = 0
        since_attempt = 0
        period = args.jump_period
        while led.total_flops() < BUDGET:
            train_step(model, optimizer, loader, led, flops_per=F_STEP,
                       accum=args.grad_accum,
                       it=args.fork_step + led.gd_steps, warmup=args.warmup_steps,
                       total=args.total_iterations,
                       warmdown_ratio=args.warmdown_ratio,
                       final_lr_frac=args.final_lr_frac)
            since_jump += 1
            since_attempt += 1
            if branch != "gd":
                recorded = meta.observe_and_learn(flat(params), led)
                ready = (meta.win.full
                         and recorded                       # fresh delta this step
                         and since_attempt >= period
                         and (args.keep_window_after_jump or
                              since_jump >= window * args.stride))
                if ready:
                    since_attempt = 0
                    t0 = time.time()
                    coeffs = meta.propose(branch, args.jump_horizon, generator=gen)
                    led.seconds["meta_overhead"] += time.time() - t0
                    if coeffs is not None:
                        led.jump_attempts += 1
                        base = probe_loss(model, probe_batches, led, F_FWD)
                        theta0 = flat(params).clone()
                        upd = meta.materialise(coeffs, led)
                        best, best_a = base, None
                        for a in args.alphas:
                            unflat(theta0 + a * upd.to(theta0.dtype), params)
                            lc = probe_loss(model, probe_batches, led, F_FWD)
                            if lc < best:
                                best, best_a = lc, a
                        if best_a is None:
                            unflat(theta0, params)
                            period = min(period * 2, args.jump_period_max)
                        else:
                            unflat(theta0 + best_a * upd.to(theta0.dtype), params)
                            led.jumps_accepted += 1
                            # evaluate right before and right after, so the
                            # jump shows up on the plotted curve as a discrete
                            # move rather than being averaged into the next
                            # eval point. Instrumentation: not charged.
                            unflat(theta0, params)
                            pre_m, _ = summarize(val_bpb_batches(
                                model, eval_batches, token_bytes))
                            unflat(theta0 + best_a * upd.to(theta0.dtype), params)
                            post_m, _ = summarize(val_bpb_batches(
                                model, eval_batches, token_bytes))
                            trace["jumps"].append(
                                {"flops": led.total_flops(), "alpha": best_a,
                                 "gain": base - best,
                                 "step": led.gd_steps,
                                 "bpb_before": pre_m, "bpb_after": post_m,
                                 "skipped": args.jump_horizon * args.stride})
                            trace["effective_steps"] = trace.get(
                                "effective_steps", 0) + args.jump_horizon * args.stride
                            if args.reset_opt_state:
                                optimizer.state.clear()
                            if not args.keep_window_after_jump:
                                meta.win.reset(flat(params))
                            since_jump = 0
                            period = args.jump_period
            if nxt < len(eval_at) and led.total_flops() >= eval_at[nxt]:
                mv, sv = summarize(val_bpb_batches(model, eval_batches, token_bytes))
                trace["flops"].append(led.total_flops()); trace["steps"].append(led.gd_steps)
                trace["bpb"].append(mv); trace["bpb_se"].append(sv)
                trace["tokens"].append(led.gd_steps * tokens_per_step)
                print(f"  {led.total_flops()/BUDGET*100:5.1f}% budget | "
                      f"{led.gd_steps:4d} gd steps | bpb {trace['bpb'][-1]:.5f} | "
                      f"{led.jumps_accepted}/{led.jump_attempts} jumps", flush=True)
                nxt += 1
        if branch == "gd":
            drop = trace["bpb"][0] - trace["bpb"][-1]
            se0 = math.hypot(trace["bpb_se"][0], trace["bpb_se"][-1])
            print(f"\n  [GUARD] plain GD over the branch window: "
                  f"{trace['bpb'][0]:.5f} -> {trace['bpb'][-1]:.5f} "
                  f"({drop:+.5f}, {drop/max(se0,1e-9):+.1f} SE)")
            if drop < 2 * se0:
                print("  [GUARD] GD ITSELF MAKES NO SIGNIFICANT PROGRESS over this "
                      "window.\n  There is nothing to forecast: the run has reached "
                      "its noise floor for\n  this LR/batch, every jumping branch's "
                      "ranking will be dominated by\n  noise and by how many steps it "
                      "happened to take, and the comparison is\n  VOID. Fork earlier "
                      "(--fork-step), reduce gradient noise (--grad-accum),\n  or "
                      "both. The rest of this run is only useful as plumbing.")
        trace["effective_steps"] = (trace.get("effective_steps", 0)
                                    + led.gd_steps)
        trace["ledger"] = {"flops": led.flops, "seconds": led.seconds,
                           "gd_steps": led.gd_steps, "attempts": led.jump_attempts,
                           "accepted": led.jumps_accepted,
                           "overhead_fraction": led.overhead_fraction()}
        results[branch] = trace
        print(led.report())

    # ---------------- report -------------------------------------------------
    print(f"\n{'='*70}\nEQUAL-BUDGET RESULT ({args.branch_steps} GD steps of FLOPs)\n{'='*70}")
    print(f"{'branch':>8}{'final bpb':>12}{'+-2SE':>9}{'best bpb':>11}"
          f"{'gd steps':>10}{'jumps':>8}{'overhead':>10}{'tokens':>13}")
    print("-" * 81)
    for b, t in results.items():
        print(f"{b:>8}{t['bpb'][-1]:>12.5f}{2*t['bpb_se'][-1]:>9.5f}"
              f"{min(t['bpb']):>11.5f}"
              f"{t['ledger']['gd_steps']:>10}"
              f"{t['ledger']['accepted']:>8}"
              f"{100*t['ledger']['overhead_fraction']:>9.2f}%"
              f"{t['ledger']['gd_steps']*tokens_per_step:>13,}")
    print("  ('best bpb' is the minimum along the curve -- reported because the "
          "curves\n   can be non-monotonic, but selecting it post hoc is cherry-"
          "picking;\n   the honest comparison is 'final bpb' at equal budget.)")
    if "gd" in results:
        g = results["gd"]["bpb"][-1]; se = results["gd"]["bpb_se"][-1]
        print(f"\nvs plain GD at equal compute (negative = better):")
        for b, t in results.items():
            if b == "gd": continue
            d = t["bpb"][-1] - g
            s = math.hypot(se, t["bpb_se"][-1])
            if s < 1e-7:
                print(f"  {b:>8}  {d:+.5f} bpb  (SE unavailable -- eval batches "
                      f"degenerate, verdict withheld)")
                continue
            verdict = "BETTER" if d < -2*s else ("worse" if d > 2*s else "within noise")
            print(f"  {b:>8}  {d:+.5f} bpb  ({d/max(s,1e-9):+.1f} SE)  {verdict}")

    (args.out / "results.json").write_text(json.dumps(
        {"args": {k: str(v) for k, v in vars(args).items()},
         "N": N, "flops_per_step": F_STEP, "captured": cap,
         "fork_step": args.fork_step, "prefix_curve": prefix_curve,
         "prefix_ledger": {"flops": prefix.flops, "seconds": prefix.seconds},
         "results": results}, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed (uv sync prunes it; reinstall with "
              "`uv pip install matplotlib`).\nresults.json is already written -- "
              "plot it anywhere with plot_results.py.")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for b, t in results.items():
        f = np.array(t["flops"]) / F_STEP
        y, e = np.array(t["bpb"]), np.array(t["bpb_se"])
        for ax, x in ((a1, f), (a2, np.array(t["steps"]))):
            ax.plot(x, y, marker="o", ms=3.5, color=COLORS.get(b),
                    lw=2.4 if b == "gd" else 1.7, label=b)
            ax.fill_between(x, y-2*e, y+2*e, color=COLORS.get(b), alpha=.12, lw=0)
    for j in results.get("meta", {}).get("jumps", []):
        a1.axvline(j["flops"]/F_STEP, color=COLORS["meta"], alpha=.25, lw=.8)
    a1.set_xlabel("compute (in units of one plain GD step)")
    a1.set_ylabel("validation bits per byte")
    a1.set_title("equal compute budget -- the fair comparison")
    a2.set_xlabel("gradient steps actually taken")
    a2.set_title("same runs, plotted against gradient steps")
    for ax in (a1, a2):
        ax.grid(True, alpha=.25); ax.legend(fontsize=9)
    fig.suptitle(f"Live meta-optimization, fork at step {args.fork_step} "
                 f"(bands $\\pm$2 SE over {args.eval_shards} shards)", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, .94])
    fig.savefig(args.out / "meta_fork.png", dpi=165)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
