#!/usr/bin/env python3
"""CPU-only reconstruction for paper section B.3; B.4; no training or cloud calls."""
from pathlib import Path
import runpy
import sys

if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "_tools" / "reproduce.py"
    sys.argv = [str(target), "--section", "app_b_3_b_4_alternative_estimators", *sys.argv[1:]]
    runpy.run_path(str(target), run_name="__main__")
