#!/usr/bin/env python3
"""Stage 1 — Data engineering only."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bootstrap.environment import bootstrap
from src.data.pipeline import load_config, run_data_pipeline

if __name__ == "__main__":
    bootstrap()
    run_data_pipeline(load_config())
