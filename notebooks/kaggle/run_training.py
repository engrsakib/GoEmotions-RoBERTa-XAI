#!/usr/bin/env python3
"""Kaggle entry point — runs full pipeline with --deploy."""

import subprocess
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
cmd = [sys.executable, str(NOTEBOOKS_DIR / "scripts" / "run_pipeline.py"), "--deploy", *sys.argv[1:]]

if __name__ == "__main__":
    subprocess.run(cmd, check=True, cwd=str(NOTEBOOKS_DIR))
