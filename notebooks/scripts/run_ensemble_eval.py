#!/usr/bin/env python3
"""CLI for weighted soft-voting ensemble evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from src.eval.ensemble_eval import main

if __name__ == "__main__":
    main()
