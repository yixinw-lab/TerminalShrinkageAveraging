"""Minimal, rank-zero model snapshot capture for the nanochat record attempt.

The training loop calls :meth:`RecordSnapshotManager.capture` after an optimizer
step and outside nanochat's timed training interval. Only explicitly planned
steps are written. Snapshots contain the exact model state_dict on CPU; no
optimizer state is included.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json
import os
import time

import torch
import torch.distributed as dist


_RECORD_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _parse_ratios(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        parts = [x.strip() for x in value.split(",") if x.strip()]
        ratios = [float(x) for x in parts]
    else:
        ratios = [float(x) for x in value]
    if not ratios:
        return []
    if any(r <= 0 for r in ratios):
        raise ValueError(f"snapshot ratios must be positive: {ratios}")
    return sorted(set(ratios))


def plan_snapshot_steps(
    ratios: str | Iterable[float],
    *,
    num_scaling_params: int,
    total_batch_size: int,
    schedule_num_iterations: int,
    train_num_iterations: int,
    k: int,
    spacing_fraction: float,
) -> tuple[dict[str, dict[str, Any]], list[int], int]:
    """Return endpoint metadata, all unique capture steps, and spacing in steps."""
    if num_scaling_params <= 0 or total_batch_size <= 0:
        raise ValueError("num_scaling_params and total_batch_size must be positive")
    if schedule_num_iterations <= 0 or train_num_iterations <= 0:
        raise ValueError("iteration counts must be positive")
    if k < 2:
        raise ValueError("snapshot k must be at least 2")
    if not (0 < spacing_fraction < 1):
        raise ValueError("spacing_fraction must be in (0, 1)")

    spacing = max(1, int(round(schedule_num_iterations * spacing_fraction)))
    endpoint_map: dict[str, dict[str, Any]] = {}
    all_steps: set[int] = set()
    for ratio in _parse_ratios(ratios):
        endpoint = int(ratio * num_scaling_params) // total_batch_size
        if endpoint <= 0:
            raise ValueError(f"ratio {ratio} maps to nonpositive step {endpoint}")
        if endpoint > train_num_iterations:
            raise ValueError(
                f"ratio {ratio} maps to step {endpoint}, beyond train horizon {train_num_iterations}"
            )
        steps = [endpoint - i * spacing for i in reversed(range(k))]
        if steps[0] <= 0:
            raise ValueError(
                f"ratio {ratio} does not have {k} positive snapshots at spacing {spacing}: {steps}"
            )
        key = f"{ratio:.8g}"
        endpoint_map[key] = {
            "ratio": ratio,
            "endpoint_step": endpoint,
            "snapshot_steps": steps,
        }
        all_steps.update(steps)
    return endpoint_map, sorted(all_steps), spacing


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Make an exact CPU clone while preserving state_dict key names and dtypes."""
    out: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key, value in model.state_dict().items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"state_dict value for {key!r} is not a tensor")
            out[key] = value.detach().to(device="cpu", copy=True)
    return out


@dataclass
class RecordSnapshotManager:
    enabled: bool
    writer: bool
    snapshot_dir: Path
    endpoints: dict[str, dict[str, Any]]
    required_steps: set[int]
    manifest: dict[str, Any]
    wall_start: float

    @classmethod
    def create(
        cls,
        *,
        enabled: bool,
        writer: bool,
        snapshot_dir: str | os.PathLike[str],
        ratios: str,
        num_scaling_params: int,
        total_batch_size: int,
        schedule_num_iterations: int,
        train_num_iterations: int,
        k: int,
        spacing_fraction: float,
        model_tag: str,
        user_config: dict[str, Any],
        model_config: dict[str, Any],
    ) -> "RecordSnapshotManager":
        path = Path(snapshot_dir).expanduser().resolve()
        if not enabled or not ratios.strip():
            return cls(False, False, path, {}, set(), {}, time.time())
        endpoints, steps, spacing = plan_snapshot_steps(
            ratios,
            num_scaling_params=num_scaling_params,
            total_batch_size=total_batch_size,
            schedule_num_iterations=schedule_num_iterations,
            train_num_iterations=train_num_iterations,
            k=k,
            spacing_fraction=spacing_fraction,
        )
        if writer:
            path.mkdir(parents=True, exist_ok=True)
        manifest: dict[str, Any] = {
            "record_version": _RECORD_VERSION,
            "model_tag": model_tag,
            "num_scaling_params": int(num_scaling_params),
            "total_batch_size": int(total_batch_size),
            "schedule_num_iterations": int(schedule_num_iterations),
            "train_num_iterations": int(train_num_iterations),
            "snapshot_k": int(k),
            "snapshot_spacing_fraction": float(spacing_fraction),
            "snapshot_spacing_steps": int(spacing),
            "endpoints": endpoints,
            "required_steps": steps,
            "captured": {},
            "capture_seconds_total": 0.0,
            "user_config": user_config,
            "model_config": model_config,
        }
        if writer:
            _atomic_json(path / "manifest.json", manifest)
            print(
                f"[record] snapshot plan: {len(steps)} unique snapshots, spacing={spacing}, "
                f"endpoints={','.join(endpoints)} dir={path}",
                flush=True,
            )
        return cls(True, writer, path, endpoints, set(steps), manifest, time.time())

    def capture(
        self,
        *,
        step: int,
        model: torch.nn.Module,
        total_training_time: float,
    ) -> None:
        if not self.enabled or step not in self.required_steps:
            return
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if self.writer:
            key = str(int(step))
            if key not in self.manifest["captured"]:
                started = time.time()
                state = _cpu_state_dict(model)
                tmp = self.snapshot_dir / f"snapshot_{step:06d}.pt.tmp"
                final = self.snapshot_dir / f"snapshot_{step:06d}.pt"
                torch.save(state, tmp)
                os.replace(tmp, final)
                elapsed = time.time() - started
                self.manifest["capture_seconds_total"] = float(
                    self.manifest.get("capture_seconds_total", 0.0) + elapsed
                )
                self.manifest["captured"][key] = {
                    "path": final.name,
                    "total_training_time": float(total_training_time),
                    "wall_elapsed_since_manager_init": float(time.time() - self.wall_start),
                    "capture_seconds": float(elapsed),
                    "bytes": int(final.stat().st_size),
                }
                _atomic_json(self.snapshot_dir / "manifest.json", self.manifest)
                del state
                print(
                    f"[record] captured step={step} file={final.name} "
                    f"size={final.stat().st_size / 2**30:.2f}GiB capture={elapsed:.2f}s",
                    flush=True,
                )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def finalize(self) -> None:
        if not self.enabled or not self.writer:
            return
        missing = sorted(self.required_steps - {int(x) for x in self.manifest["captured"]})
        self.manifest["missing_steps"] = missing
        self.manifest["complete"] = not missing
        _atomic_json(self.snapshot_dir / "manifest.json", self.manifest)
        if missing:
            raise RuntimeError(f"record snapshot capture incomplete; missing steps: {missing}")
        print(
            f"[record] snapshot capture complete: {len(self.required_steps)} files; "
            f"excluded capture overhead={self.manifest['capture_seconds_total']:.2f}s",
            flush=True,
        )


def selftest() -> None:
    endpoints, steps, spacing = plan_snapshot_steps(
        "9.0,9.2,9.4",
        num_scaling_params=1_000_000,
        total_batch_size=1_000,
        schedule_num_iterations=9_400,
        train_num_iterations=9_400,
        k=8,
        spacing_fraction=32 / 3000,
    )
    assert spacing == 100
    assert endpoints["9"]["endpoint_step"] == 9000
    assert len(endpoints["9"]["snapshot_steps"]) == 8
    assert steps[-1] == 9400
    print("record_snapshots selftest PASS")


if __name__ == "__main__":
    selftest()
