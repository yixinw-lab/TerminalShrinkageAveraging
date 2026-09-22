#!/usr/bin/env python3
"""Patch the exact NanoChat PR #830 base_train.py for a D22 DeltaAI replication.

This script refuses to patch files whose expected e09bc164 anchors are absent.
It writes scripts/base_train_d22_delta.py and never edits upstream base_train.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib

BASE_COMMIT = "e09bc164162f35da0b5b8315be791e9e974a4c3a"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"anchor {label!r}: expected exactly 1 occurrence, found {n}")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    args = ap.parse_args()
    src = args.repo / "scripts" / "base_train.py"
    dst = args.repo / "scripts" / "base_train_d22_delta.py"
    text = src.read_text()
    original_sha = hashlib.sha256(text.encode()).hexdigest()

    # Add experiment-only CLI flags immediately after --model-tag.
    anchor = 'parser.add_argument("--model-tag", type=str, default=None, help="override model tag for checkpoint directory name")\n'
    addition = anchor + '''# D22 DeltaAI replication controls (post-hoc instrumentation; no live estimator feedback)\nparser.add_argument("--experiment-seed", type=int, default=42, help="model-initialization RNG seed; paired arms use the same value")\nparser.add_argument("--stop-after-step", type=int, default=-1, help="physical stop step while retaining the original schedule horizon")\nparser.add_argument("--terminal-lr-floor", type=float, default=-1.0, help="if >=0, clamp the baseline LR multiplier from below")\nparser.add_argument("--snapshot-steps", type=str, default="", help="comma-separated model-only snapshot steps")\nparser.add_argument("--snapshot-dir", type=str, default="", help="directory for bf16 model-only snapshots and metrics")\nparser.add_argument("--skip-standard-final-checkpoint", action="store_true", help="avoid the large optimizer-state checkpoint; model-only snapshots are retained")\nparser.add_argument("--posthoc-eval", action="store_true", help="evaluate raw/TSA/LAWA/EWA from the stored terminal window")\nparser.add_argument("--posthoc-eval-tokens", type=int, default=80*524288, help="canonical BPB evaluation token budget for post-hoc candidates")\nparser.add_argument("--posthoc-alphas", type=str, default="0,0.55,0.587,1", help="TSA coefficients to evaluate")\nparser.add_argument("--posthoc-ewa-beta", type=float, default=0.75, help="finite-window EWA beta")\n'''
    text = replace_once(text, anchor, addition, "CLI")

    # Override NanoChat's global initialization seed before model initialization.
    anchor = 'master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.\n'
    addition = anchor + '''# compute_init() seeds model initialization to 42 upstream. Override only that RNG here\n# so paired schedule arms can share initialization while we obtain three independent repetitions.\ntorch.manual_seed(args.experiment_seed)\nif device_type == "cuda":\n    torch.cuda.manual_seed(args.experiment_seed)\n'''
    text = replace_once(text, anchor, addition, "seed override")

    # Preserve the ratio-9.4 horizon for scaling laws/schedulers, but physically stop at ratio 9.0.
    anchor = 'print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")\n'
    addition = anchor + '''stop_step = num_iterations if args.stop_after_step < 0 else args.stop_after_step\nif stop_step > num_iterations:\n    raise ValueError(f"stop-after-step {stop_step} exceeds schedule horizon {num_iterations}")\nprint0(f"Physical stop step: {stop_step:,} (schedule/scaling horizon remains {num_iterations:,})")\nsnapshot_steps = sorted({int(x) for x in args.snapshot_steps.split(",") if x.strip()})\nif snapshot_steps and not args.snapshot_dir:\n    raise ValueError("--snapshot-dir is required when --snapshot-steps is nonempty")\nif snapshot_steps and snapshot_steps[-1] > stop_step:\n    raise ValueError(f"snapshot step {snapshot_steps[-1]} exceeds physical stop {stop_step}")\nif master_process and args.snapshot_dir:\n    os.makedirs(args.snapshot_dir, exist_ok=True)\n'''
    text = replace_once(text, anchor, addition, "physical stop setup")

    # Clamp the *baseline* schedule from below for treatment; control is untouched when floor < 0.
    old = '''def get_lr_multiplier(it):\n    warmup_iters = args.warmup_steps\n    warmdown_iters = round(args.warmdown_ratio * num_iterations)\n    if it < warmup_iters:\n        return (it + 1) / warmup_iters\n    elif it <= num_iterations - warmdown_iters:\n        return 1.0\n    else:\n        progress = (num_iterations - it) / warmdown_iters\n        return progress * 1.0 + (1 - progress) * args.final_lr_frac\n'''
    new = '''def get_lr_multiplier(it):\n    warmup_iters = args.warmup_steps\n    warmdown_iters = round(args.warmdown_ratio * num_iterations)\n    if it < warmup_iters:\n        mult = (it + 1) / warmup_iters\n    elif it <= num_iterations - warmdown_iters:\n        mult = 1.0\n    else:\n        progress = (num_iterations - it) / warmdown_iters\n        mult = progress * 1.0 + (1 - progress) * args.final_lr_frac\n    if args.terminal_lr_floor >= 0:\n        mult = max(mult, args.terminal_lr_floor)\n    return mult\n'''
    text = replace_once(text, old, new, "LR schedule")

    # Stop at requested step, not at the ratio-9.4 scheduler horizon.
    text = replace_once(
        text,
        'last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end\n',
        'last_step = step == stop_step # physical endpoint; schedulers still use num_iterations\n',
        "last step",
    )

    # Save compact, model-only bf16 snapshots at the exact terminal steps.
    anchor = '    flops_so_far = num_flops_per_token * total_batch_size * step\n'
    addition = anchor + '''    if master_process and step in snapshot_steps:\n        snap_path = os.path.join(args.snapshot_dir, f"model_step{step:05d}.pt")\n        if not os.path.exists(snap_path):\n            cpu_state = {k: v.detach().to(device="cpu", dtype=torch.bfloat16) if v.is_floating_point() else v.detach().cpu()\n                         for k, v in orig_model.state_dict().items()}\n            torch.save(cpu_state, snap_path)\n            del cpu_state\n            print0(f"Saved model-only bf16 snapshot: {snap_path}")\n'''
    text = replace_once(text, anchor, addition, "snapshot save")

    # Avoid the full optimizer checkpoint when model-only snapshots are the experimental artifact.
    old = '    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):\n'
    new = '    if (not (last_step and args.skip_standard_final_checkpoint)) and (last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0)):\n'
    text = replace_once(text, old, new, "checkpoint suppression")

    # Make logging/ETA describe the physical endpoint. This does not alter optimization.
    text = replace_once(text, '    pct_done = 100 * step / num_iterations\n', '    pct_done = 100 * step / stop_step\n', "pct")
    text = replace_once(text, '        remaining_steps = num_iterations - step\n', '        remaining_steps = stop_step - step\n', "eta")
    text = replace_once(text, '    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss:', '    print0(f"step {step:05d}/{stop_step:05d} ({pct_done:.2f}%) | loss:', "log horizon")

    # Post-hoc candidate construction and canonical BPB evaluation.
    anchor = '# print a few more stats\n'
    block = r'''# Post-hoc terminal estimators. No candidate influences optimizer updates.\nif args.posthoc_eval:\n    if not snapshot_steps:\n        raise ValueError("--posthoc-eval requires --snapshot-steps")\n    expected = [os.path.join(args.snapshot_dir, f"model_step{s:05d}.pt") for s in snapshot_steps]\n    if master_process:\n        missing = [p for p in expected if not os.path.exists(p)]\n        if missing:\n            raise FileNotFoundError(f"missing snapshots: {missing}")\n        saved_states = [torch.load(p, map_location="cpu", weights_only=True) for p in expected]\n    else:\n        saved_states = None\n\n    def load_candidate(kind, alpha=None, beta=None):\n        # Only rank 0 reads/composes the stored weights; then broadcast every tensor.\n        for name, target in orig_model.state_dict().items():\n            if master_process:\n                newest = saved_states[-1][name]\n                if not newest.is_floating_point() or kind == "raw":\n                    cand = newest\n                elif kind == "tsa":\n                    acc = torch.zeros_like(newest, dtype=torch.float32)\n                    for st in saved_states:\n                        acc.add_(st[name].float())\n                    acc.div_(len(saved_states))\n                    cand = newest.float().lerp(acc, float(alpha))\n                elif kind == "ewa":\n                    b = float(beta)\n                    weights = [b ** (len(saved_states) - 1 - i) for i in range(len(saved_states))]\n                    z = sum(weights)\n                    acc = torch.zeros_like(newest, dtype=torch.float32)\n                    for w, st in zip(weights, saved_states):\n                        acc.add_(st[name].float(), alpha=w / z)\n                    cand = acc\n                else:\n                    raise ValueError(kind)\n                target.copy_(cand.to(device=target.device, dtype=target.dtype))\n            if is_ddp_initialized():\n                dist.broadcast(target, src=0)\n        if is_ddp_initialized():\n            dist.barrier()\n\n    def eval_current():\n        orig_model.eval()\n        val_loader = build_val_loader()\n        eval_steps = args.posthoc_eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)\n        with disable_fp8(orig_model):\n            out = evaluate_bpb(orig_model, val_loader, eval_steps, token_bytes)\n        orig_model.train()\n        return float(out)\n\n    metrics = {}\n    for a_text in [x for x in args.posthoc_alphas.split(",") if x.strip()]:\n        a = float(a_text)\n        if a == 0.0:\n            key, kind = "raw", "raw"\n            load_candidate(kind)\n        else:\n            key, kind = f"tsa_alpha_{a:g}", "tsa"\n            load_candidate(kind, alpha=a)\n        metrics[key] = eval_current()\n        print0(f"POSTHOC {key}: BPB={metrics[key]:.6f}")\n    load_candidate("ewa", beta=args.posthoc_ewa_beta)\n    ewa_key = f"ewa_beta_{args.posthoc_ewa_beta:g}"\n    metrics[ewa_key] = eval_current()\n    print0(f"POSTHOC {ewa_key}: BPB={metrics[ewa_key]:.6f}")\n\n    if master_process:\n        payload = {\n            "experiment_seed": args.experiment_seed,\n            "terminal_lr_floor": args.terminal_lr_floor,\n            "schedule_horizon_steps": num_iterations,\n            "physical_stop_step": stop_step,\n            "target_param_data_ratio_for_schedule": args.target_param_data_ratio,\n            "snapshot_steps": snapshot_steps,\n            "training_iteration_time_seconds": total_training_time,\n            "gpu_name": torch.cuda.get_device_name(0) if device_type == "cuda" else device_type,\n            "world_size": ddp_world_size,\n            "posthoc_eval_tokens": args.posthoc_eval_tokens,\n            "metrics_bpb": metrics,\n        }\n        with open(os.path.join(args.snapshot_dir, "metrics.json"), "w") as f:\n            json.dump(payload, f, indent=2, sort_keys=True)\n        print0(f"Wrote {os.path.join(args.snapshot_dir, 'metrics.json')}")\n    if master_process:\n        del saved_states\n\n'''
    # Convert the raw string's escaped newlines to real newlines without touching backslashes elsewhere.
    block = block.replace('\\n', '\n')
    text = replace_once(text, anchor, block + anchor, "posthoc insertion")

    dst.write_text(text)
    patched_sha = hashlib.sha256(text.encode()).hexdigest()
    print(f"source={src}")
    print(f"source_sha256={original_sha}")
    print(f"patched={dst}")
    print(f"patched_sha256={patched_sha}")

if __name__ == "__main__":
    main()
