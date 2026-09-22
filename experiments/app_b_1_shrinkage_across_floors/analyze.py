#!/usr/bin/env python3
"""CPU-only reconstruction for paper section B.1; no training or cloud calls."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "_tools" / "reproduce.py"
    sys.argv = [str(target), "--section", "app_b_1_shrinkage_across_floors", *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")
