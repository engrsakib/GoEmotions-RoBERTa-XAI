#!/usr/bin/env python3
"""Select best Track A teacher (E2/E7/E8/E9) and run E4 distillation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from src.data.pipeline import load_config
from src.paths import CHECKPOINTS_DIR, EXPORTS_DIR, ensure_artifact_dirs
from src.training.model_profiles import find_best_teacher_experiment
from scripts.run_experiments import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E4 distillation from best teacher")
    parser.add_argument(
        "--student",
        default="m5_distilroberta",
        choices=["m5_distilroberta", "m3_roberta_base"],
        help="Student model registry ID",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Smoke test: cap rows (epochs=1)",
    )
    args = parser.parse_args()

    ensure_artifact_dirs()
    best = find_best_teacher_experiment()
    if best is None and not args.skip_train:
        raise SystemExit(
            "No teacher experiments found. Run E2, E7, E8, or E9 first."
        )

    config = load_config()
    if args.max_samples:
        config["max_samples"] = args.max_samples
        config.setdefault("epochs", 1)
        config["fp16"] = False
    if best:
        model_id = best["model_id"]
        teacher_path = CHECKPOINTS_DIR / model_id
        config["teacher_model_path"] = str(teacher_path)
        config["teacher_experiment"] = best["experiment_id"]
        print(f"Teacher: {best['experiment_id']} ({model_id}) macro-F1={best['macro_f1']}")

    config["model_id"] = args.student
    config["distill"] = True

    from src.training.model_registry import apply_model_to_config

    exp_config = apply_model_to_config(config, args.student)
    payload = run_experiment("E4", exp_config, skip_train=args.skip_train)

    out_path = EXPORTS_DIR / "experiment_E4_distill.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
