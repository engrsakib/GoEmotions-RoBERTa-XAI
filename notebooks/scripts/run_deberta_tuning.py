#!/usr/bin/env python3
"""DeBERTa-v3-base tuning grid (E2) — validation macro-F1 search only."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from src.data.pipeline import load_config
from src.paths import EXPORTS_DIR, ensure_artifact_dirs
from src.training.model_profiles import get_deberta_tuning_grid
from scripts.run_experiments import merge_config, run_experiment


def run_tuning_grid(skip_train: bool = False, max_samples: int | None = None) -> list[dict]:
    base = load_config()
    if max_samples:
        base["max_samples"] = max_samples
        base.setdefault("epochs", 1)
        base["fp16"] = False
    e2_preset = {
        "track": "multilabel",
        "model_id": "m6_deberta_v3",
        "loss_type": "asymmetric",
        "split_mode": "official",
        "dedup_policy": "consensus",
        "balance_strategy": "none",
    }
    grid = get_deberta_tuning_grid()
    results = []

    for idx, overrides in enumerate(grid):
        run_id = f"E2-T{idx + 1}"
        config = merge_config(base, e2_preset)
        config.update(overrides)
        print(f"\n=== {run_id}: {overrides} ===")

        if skip_train:
            from src.training.model_registry import apply_model_to_config
            from src.data.pipeline import run_data_pipeline

            config = apply_model_to_config(config, "m6_deberta_v3")
            stats = run_data_pipeline(config)["stats"]
            results.append({"run_id": run_id, "overrides": overrides, "stats": stats})
            continue

        payload = run_experiment("E2", base, skip_train=False, extra_overrides=overrides)
        val_f1 = float(
            (payload.get("eval_metrics") or {}).get("eval_macro_f1", -1)
        )
        test_f1 = float(
            (payload.get("test_metrics") or {}).get("eval_macro_f1", -1)
        )
        results.append(
            {
                "run_id": run_id,
                "overrides": overrides,
                "val_macro_f1": val_f1,
                "test_macro_f1": test_f1,
                "eval_metrics": payload.get("eval_metrics"),
            }
        )

        out_path = EXPORTS_DIR / f"deberta_tuning_{run_id}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(results[-1], handle, indent=2, default=str)

    best = max(results, key=lambda r: r.get("val_macro_f1", -1), default=None)
    summary = {"grid_results": results, "best_by_val_macro_f1": best}
    summary_path = EXPORTS_DIR / "deberta_tuning_grid.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"\nWrote {summary_path}")
    if best:
        print(f"Best val macro-F1: {best.get('val_macro_f1')} ({best.get('run_id')})")
        if not skip_train and best.get("overrides"):
            e2_path = EXPORTS_DIR / "experiment_E2.json"
            with e2_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "experiment_id": "E2",
                        "track": "multilabel",
                        "model_id": "m6_deberta_v3",
                        "best_grid_run": best["run_id"],
                        "hyperparameters": best["overrides"],
                        "val_macro_f1": best.get("val_macro_f1"),
                        "test_macro_f1": best.get("test_macro_f1"),
                        "eval_metrics": best.get("eval_metrics"),
                    },
                    handle,
                    indent=2,
                    default=str,
                )
            print(f"Wrote best E2 summary to {e2_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="DeBERTa-v3-base hyperparameter grid")
    parser.add_argument("--skip-train", action="store_true", help="Data pipeline only")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Smoke test: cap rows per grid run (epochs=1)",
    )
    args = parser.parse_args()
    ensure_artifact_dirs()
    run_tuning_grid(skip_train=args.skip_train, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
