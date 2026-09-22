#!/usr/bin/env python3
"""Convert record snapshots into standard nanochat candidate checkpoints.

This is intentionally post-hoc: training math is untouched apart from the
terminal LR clamp. The candidate family is screened once, then one recipe is
frozen before confirmation runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import torch

EPS = 1e-30


def torch_load(path: Path) -> dict[str, torch.Tensor]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"snapshot is not a state dict: {path}")
    return obj


def parse_recipes(path: Path | None, inline: str | None) -> list[str]:
    rows: list[str] = []
    if path is not None:
        rows.extend(
            line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if inline:
        rows.extend(x.strip() for x in inline.split(",") if x.strip())
    if not rows:
        raise ValueError("no recipes supplied")
    return list(dict.fromkeys(rows))


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = a.reshape(-1).float()
    bb = b.reshape(-1).float()
    denom = float(aa.norm()) * float(bb.norm())
    if denom <= EPS:
        return 1.0
    return float(torch.dot(aa, bb) / denom)


def alpha_for_tensor(states: list[torch.Tensor], rule: str) -> tuple[float, dict[str, float]]:
    updates = [states[i].float() - states[i - 1].float() for i in range(1, len(states))]
    drift = torch.zeros_like(updates[0])
    for u in updates:
        drift.add_(u)
    drift.div_(len(updates))
    drift_energy = float(drift.square().mean())
    noise_energy = 0.0
    for u in updates:
        noise_energy += float((u - drift).square().mean())
    noise_energy /= len(updates)
    snr_alpha = noise_energy / (noise_energy + drift_energy + EPS)

    if len(updates) >= 2 and states[0].numel() >= 2:
        corr = sum(_cosine(updates[i], updates[i - 1]) for i in range(1, len(updates))) / (len(updates) - 1)
    else:
        corr = 1.0 if drift_energy > noise_energy else 0.0
    autocorr_alpha = max(0.0, min(1.0, 1.0 - max(corr, 0.0)))

    if rule == "snr":
        alpha = snr_alpha
    elif rule == "autocorr":
        alpha = autocorr_alpha
    elif rule == "hybrid":
        alpha = 0.5 * (snr_alpha + autocorr_alpha)
    elif rule == "max":
        alpha = max(snr_alpha, autocorr_alpha)
    elif rule == "product":
        alpha = snr_alpha * autocorr_alpha
    else:
        raise ValueError(f"unknown adaptive rule: {rule}")
    alpha = max(0.0, min(1.0, float(alpha)))
    return alpha, {
        "alpha": alpha,
        "noise_energy": noise_energy,
        "drift_energy": drift_energy,
        "snr_alpha": snr_alpha,
        "autocorr": corr,
        "autocorr_alpha": autocorr_alpha,
    }


def parameter_names_from_meta(meta: dict[str, Any]) -> set[str] | None:
    try:
        from nanochat.gpt import GPT, GPTConfig
        config = GPTConfig(**meta["model_config"])
        with torch.device("meta"):
            model = GPT(config)
        return {name for name, _ in model.named_parameters()}
    except Exception as exc:
        print(f"WARNING: could not instantiate GPT to identify parameters ({exc}); averaging all floating tensors")
        return None


def mean_tensor(values: list[torch.Tensor]) -> torch.Tensor:
    out = torch.zeros_like(values[-1], dtype=torch.float32, device="cpu")
    for value in values:
        out.add_(value.float())
    out.div_(len(values))
    return out


def merge_state_dict(
    snapshots: list[dict[str, torch.Tensor]],
    *,
    recipe: str,
    parameter_names: set[str] | None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    latest = snapshots[-1]
    keys = list(latest)
    if any(set(s) != set(keys) for s in snapshots):
        raise RuntimeError("snapshot state_dict key sets differ")

    out: dict[str, torch.Tensor] = {}
    alpha_weighted = 0.0
    alpha_numel = 0
    per_tensor: dict[str, dict[str, float]] = {}

    if recipe == "raw":
        return {k: v.clone() for k, v in latest.items()}, {"mean_alpha_by_numel": 0.0, "per_tensor": {}}
    if recipe == "uniform":
        fixed_alpha: float | None = 1.0
        adaptive_rule: str | None = None
    elif recipe.startswith("blend:"):
        fixed_alpha = float(recipe.split(":", 1)[1])
        adaptive_rule = None
        if not (0.0 <= fixed_alpha <= 1.0):
            raise ValueError(f"bad blend alpha: {recipe}")
    elif recipe.startswith("adaptive:"):
        fixed_alpha = None
        adaptive_rule = recipe.split(":", 1)[1]
    else:
        raise ValueError(f"unknown recipe: {recipe}")

    for key in keys:
        values = [s[key] for s in snapshots]
        latest_value = values[-1]
        should_average = (
            latest_value.is_floating_point()
            and (parameter_names is None or key.removeprefix("_orig_mod.") in parameter_names)
        )
        if not should_average:
            out[key] = latest_value.clone()
            continue
        avg = mean_tensor(values)
        if fixed_alpha is not None:
            alpha = fixed_alpha
            diag = {"alpha": alpha}
        else:
            assert adaptive_rule is not None
            alpha, diag = alpha_for_tensor(values, adaptive_rule)
        candidate = latest_value.float().add(avg - latest_value.float(), alpha=float(alpha))
        out[key] = candidate.to(dtype=latest_value.dtype)
        numel = latest_value.numel()
        alpha_weighted += float(alpha) * numel
        alpha_numel += numel
        per_tensor[key] = diag
    return out, {
        "mean_alpha_by_numel": alpha_weighted / max(alpha_numel, 1),
        "per_tensor": per_tensor,
    }


def endpoint_meta(manifest: dict[str, Any], endpoint_step: int) -> dict[str, Any]:
    captured = manifest["captured"].get(str(endpoint_step))
    if captured is None:
        raise KeyError(f"endpoint step {endpoint_step} missing from manifest capture map")
    return captured


def load_source_meta(checkpoint_dir: Path, final_step: int) -> dict[str, Any]:
    path = checkpoint_dir / f"meta_{final_step:06d}.json"
    if not path.exists():
        raise FileNotFoundError(f"source final metadata missing: {path}")
    return json.loads(path.read_text())


def write_candidate(
    *,
    base_checkpoints: Path,
    model_tag: str,
    step: int,
    state: dict[str, torch.Tensor],
    source_meta: dict[str, Any],
    record_payload: dict[str, Any],
    total_training_time: float,
) -> tuple[Path, Path]:
    out_dir = base_checkpoints / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"model_{step:06d}.pt"
    meta_path = out_dir / f"meta_{step:06d}.json"
    torch.save(state, model_path)
    meta = json.loads(json.dumps(source_meta))
    meta["step"] = int(step)
    meta["val_bpb"] = None
    meta.setdefault("loop_state", {})["total_training_time"] = float(total_training_time)
    meta.setdefault("user_config", {}).update(record_payload)
    meta["user_config"]["target_param_data_ratio"] = record_payload["ratio"]
    meta["user_config"]["model_tag"] = model_tag
    meta["record_attempt"] = {**record_payload, "evaluation_only_checkpoint": True, "resume_safe": False}
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return model_path, meta_path


def run(args: argparse.Namespace) -> Path:
    snapshot_dir = args.snapshot_dir.resolve()
    manifest = json.loads((snapshot_dir / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise RuntimeError("snapshot manifest is not complete; run training to completion first")
    recipes = parse_recipes(args.recipes_file, args.recipes)
    source_checkpoint_dir = args.source_checkpoint_dir.resolve()
    source_meta = load_source_meta(source_checkpoint_dir, int(manifest["train_num_iterations"]))
    parameter_names = parameter_names_from_meta(source_meta)
    base_checkpoints = args.base_checkpoints.resolve()
    base_checkpoints.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    endpoint_items = sorted(manifest["endpoints"].items(), key=lambda x: float(x[0]))
    for ratio_key, endpoint in endpoint_items:
        ratio = float(endpoint["ratio"])
        endpoint_step = int(endpoint["endpoint_step"])
        steps = [int(x) for x in endpoint["snapshot_steps"]]
        snapshot_paths = [snapshot_dir / manifest["captured"][str(s)]["path"] for s in steps]
        print(f"Loading ratio={ratio:g} endpoint={endpoint_step} snapshots={steps}", flush=True)
        snapshots = [torch_load(path) for path in snapshot_paths]
        ep_capture = endpoint_meta(manifest, endpoint_step)

        for recipe in recipes:
            state, diagnostics = merge_state_dict(
                snapshots,
                recipe=recipe,
                parameter_names=parameter_names,
            )
            model_tag = safe_slug(f"{args.output_prefix}_r{ratio:g}_{recipe}")
            payload = {
                "record_version": 1,
                "source_model_tag": manifest["model_tag"],
                "ratio": ratio,
                "endpoint_step": endpoint_step,
                "schedule_ratio": manifest["user_config"].get("record_schedule_param_data_ratio"),
                "terminal_lr_clamp_frac": manifest["user_config"].get("record_terminal_lr_clamp_frac"),
                "snapshot_steps": steps,
                "snapshot_k": manifest["snapshot_k"],
                "snapshot_spacing_steps": manifest["snapshot_spacing_steps"],
                "recipe": recipe,
                "mean_alpha_by_numel": diagnostics["mean_alpha_by_numel"],
                "note": "adaptive recipes in this transfer screen are a portable variance/autocorrelation approximation; freeze before confirmation",
            }
            model_path, meta_path = write_candidate(
                base_checkpoints=base_checkpoints,
                model_tag=model_tag,
                step=endpoint_step,
                state=state,
                source_meta=source_meta,
                record_payload=payload,
                total_training_time=float(ep_capture["total_training_time"]),
            )
            diag_path = model_path.parent / f"diagnostics_{endpoint_step:06d}.json"
            diag_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
            rows.append({
                "ratio": ratio,
                "endpoint_step": endpoint_step,
                "recipe": recipe,
                "model_tag": model_tag,
                "model_path": str(model_path),
                "meta_path": str(meta_path),
                "total_training_time": float(ep_capture["total_training_time"]),
                "mean_alpha_by_numel": diagnostics["mean_alpha_by_numel"],
            })
            print(
                f"wrote {model_tag} step={endpoint_step} recipe={recipe} "
                f"mean_alpha={diagnostics['mean_alpha_by_numel']:.4f}",
                flush=True,
            )
            del state
        del snapshots

    out_csv = args.output_manifest.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"candidate manifest: {out_csv}")
    return out_csv


def selftest(tmp: Path) -> None:
    s0 = {"w": torch.tensor([0.0, 2.0]), "count": torch.tensor(1)}
    s1 = {"w": torch.tensor([1.0, 1.0]), "count": torch.tensor(1)}
    s2 = {"w": torch.tensor([0.0, 2.0]), "count": torch.tensor(1)}
    state, diag = merge_state_dict([s0, s1, s2], recipe="blend:0.5", parameter_names={"w"})
    assert torch.allclose(state["w"], torch.tensor([1 / 6, 11 / 6]), atol=1e-6)
    assert int(state["count"]) == 1
    assert math.isclose(diag["mean_alpha_by_numel"], 0.5)
    state2, diag2 = merge_state_dict([s0, s1, s2], recipe="adaptive:hybrid", parameter_names={"w"})
    assert torch.isfinite(state2["w"]).all()
    assert 0 <= diag2["mean_alpha_by_numel"] <= 1
    print("merge_record_snapshots selftest PASS")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot-dir", type=Path)
    ap.add_argument("--source-checkpoint-dir", type=Path)
    ap.add_argument("--base-checkpoints", type=Path)
    ap.add_argument("--output-prefix", default="record")
    ap.add_argument("--recipes-file", type=Path)
    ap.add_argument("--recipes", default=None)
    ap.add_argument("--output-manifest", type=Path, default=Path("record_results/candidates.csv"))
    ap.add_argument("--selftest", action="store_true")
    return ap


def main() -> None:
    ap = make_parser()
    args = ap.parse_args()
    if args.selftest:
        selftest(Path("."))
        return
    for name in ("snapshot_dir", "source_checkpoint_dir", "base_checkpoints"):
        if getattr(args, name) is None:
            ap.error(f"--{name.replace('_', '-')} is required")
    if args.recipes_file is None and not args.recipes:
        ap.error("--recipes-file or --recipes is required")
    run(args)


if __name__ == "__main__":
    main()
