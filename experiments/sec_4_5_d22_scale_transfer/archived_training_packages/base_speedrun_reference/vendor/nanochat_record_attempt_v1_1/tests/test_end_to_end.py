from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanochat.record_snapshots import plan_snapshot_steps
from record_tools.merge_record_snapshots import run, make_parser


def main() -> None:
    endpoints, steps, spacing = plan_snapshot_steps(
        "1.0", num_scaling_params=1000, total_batch_size=10,
        schedule_num_iterations=100, train_num_iterations=100,
        k=4, spacing_fraction=0.1,
    )
    assert spacing == 10 and steps == [70, 80, 90, 100]
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        snap = root / "snap"; snap.mkdir()
        captured = {}
        for i, step in enumerate(steps):
            path = snap / f"snapshot_{step:06d}.pt"
            torch.save({"w": torch.tensor([float(i), float(-i)]), "n": torch.tensor(3)}, path)
            captured[str(step)] = {
                "path": path.name, "total_training_time": float(step),
                "wall_elapsed_since_manager_init": float(step), "capture_seconds": 0.0,
                "bytes": path.stat().st_size,
            }
        manifest = {
            "complete": True, "model_tag": "tiny_train", "train_num_iterations": 100,
            "snapshot_k": 4, "snapshot_spacing_steps": 10,
            "user_config": {"record_schedule_param_data_ratio": 1.0, "record_terminal_lr_clamp_frac": 0.15},
            "endpoints": endpoints, "captured": captured,
        }
        (snap / "manifest.json").write_text(json.dumps(manifest))
        source = root / "source"; source.mkdir()
        meta = {
            "step": 100, "val_bpb": None,
            "model_config": {"sequence_len": 8, "vocab_size": 16, "n_layer": 1, "n_head": 1, "n_kv_head": 1, "n_embd": 8, "window_pattern": "L"},
            "user_config": {}, "device_batch_size": 1, "max_seq_len": 8,
            "total_batch_size": 10, "dataloader_state_dict": {},
            "loop_state": {"min_val_bpb": 1.0, "smooth_train_loss": 1.0, "total_training_time": 100.0},
        }
        (source / "meta_000100.json").write_text(json.dumps(meta))
        recipes = root / "recipes.txt"; recipes.write_text("raw\nblend:0.5\nadaptive:hybrid\n")
        out_manifest = root / "candidates.csv"
        args = make_parser().parse_args([
            "--snapshot-dir", str(snap), "--source-checkpoint-dir", str(source),
            "--base-checkpoints", str(root / "base_checkpoints"),
            "--output-prefix", "tiny", "--recipes-file", str(recipes),
            "--output-manifest", str(out_manifest),
        ])
        run(args)
        rows = list(csv.DictReader(out_manifest.open()))
        assert len(rows) == 3
        for row in rows:
            assert Path(row["model_path"]).exists()
            assert Path(row["meta_path"]).exists()
    print("test_end_to_end PASS")


if __name__ == "__main__":
    main()
