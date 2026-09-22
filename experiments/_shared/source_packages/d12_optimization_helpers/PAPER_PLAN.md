# Paper hypothesis map

## Established baseline

Uniform evaluation-only LAWA is useful, but it is prior art and is not the
paper contribution. It remains the strongest baseline and the easiest path to a
NanoChat leaderboard attempt.

## Candidate contribution A: Muon-aware adaptive trajectory averaging

**Hypothesis.** The matrix parameters updated by Muon contain a larger
high-frequency/noise component than the AdamW-managed embeddings, head, and
scalars. Uniform LAWA filters every parameter identically. A causal, group- or
tensor-specific filter should preserve coherent drift while averaging the
oscillatory groups more strongly.

The package measures, for every tensor:

- optimizer family (Muon or AdamW),
- architectural role (attention, MLP, embedding, head, scalar, layer range),
- consecutive-update autocorrelation,
- estimated noise-to-drift fraction,
- displacement induced by averaging,
- marginal validation gain from averaging that group alone.

A publishable result would require all of the following:

1. The group attribution is stable across seeds and model sizes.
2. A causal adaptive recipe beats the best uniform LAWA recipe after
   hyperparameters are frozen.
3. The gain is not explained by simply tuning learning rate, warmdown, EMA, or
   nonuniform scalar checkpoint weights.
4. The rule generalizes across canonical NanoChat depths and ideally at least
   one non-NanoChat architecture/optimizer.

## Candidate contribution B: live state-consistent selective filtering

**Hypothesis.** A small periodic pull of only noisy Muon groups toward their
causal trajectory center can improve optimization, provided the associated
optimizer state is transported consistently.

The decisive experiment has three intervals:

1. exact shared prefix,
2. active filter interval,
3. exact continuation after the filter turns off.

An optimizer claim requires a durable advantage after interval 3. An immediate
loss drop that disappears under continuation is only checkpoint smoothing.

## Candidate contribution C: current-batch verified fast-forward

**Hypothesis.** Open-loop history-only jumps fail because future minibatch
innovation is unknown. A causal trajectory model can instead propose a jump,
and one or a few current-batch forward evaluations can accept, shrink, or
reject it without paying for a backward pass.

`vf` compares the current and proposed losses. `qf` evaluates a few points on
the proposal ray, fits a directional quadratic, and accepts only an observed
improvement. Forward probes are charged in the FLOP ledger.

A fast-forward claim requires:

1. fewer exact backward calls,
2. equal charged FLOPs and lower measured wall time,
3. lower BPB/CORE than tuned LR + uniform LAWA,
4. `advance` outperforming its matched `hold` control,
5. replicated seeds with frozen hyperparameters.

## Closest prior art / novelty guardrails

The final related-work audit must include at least SWA, EMA, LAWA, Lookahead,
Trainable Weight Averaging, layer-wise averaging/connectivity, schedule-free or
primal-averaging methods, and line-search/forward-evaluation optimizers.

The contribution cannot be "average checkpoints" or "learn checkpoint
coefficients." The novelty bar is the optimizer-family-aware causal rule, its
mechanistic relationship to Muon trajectory geometry, or a forward-verified
method that actually replaces backwards.
