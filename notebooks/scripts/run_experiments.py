#!/usr/bin/env python3
"""IEEE experiment matrix runner (E0–E10 + E5 ablations)."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import yaml

from src.data.pipeline import load_config, run_data_pipeline
from src.paths import CHECKPOINTS_DIR, EXPORTS_DIR, ensure_artifact_dirs
from src.training.baselines import run_all_baselines
from src.training.model_profiles import find_best_teacher_experiment, resolve_model_checkpoint
from src.training.model_registry import apply_model_to_config, get_model
from src.training.multilabel_trainer import build_multilabel_trainer, prepare_multilabel_hf_datasets
from src.training.thresholds import save_thresholds
from src.training.trainer_setup import (
    build_trainer,
    evaluate_multilabel_with_threshold_tuning,
    evaluate_with_threshold_tuning,
    export_model,
    load_transformer_tokenizer,
    prepare_hf_datasets,
)


EXPERIMENT_PRESETS: dict[str, dict] = {
    "E0": {"stage": "baselines"},
    "E1": {
        "track": "multilabel",
        "model_id": "m3_roberta_base",
        "loss_type": "weighted_ce",
        "split_mode": "stratified",
        "dedup_policy": "consensus",
        "balance_strategy": "none",
    },
    "E2": {
        "track": "multilabel",
        "model_id": "m6_deberta_v3",
        "loss_type": "asymmetric",
        "split_mode": "official",
        "dedup_policy": "consensus",
        "balance_strategy": "none",
    },
    "E3": {
        "track": "singlelabel",
        "model_id": "m4_roberta_focal",
        "loss_type": "weighted_ce",
        "focal_gamma": 1.0,
        "dedup_policy": "consensus",
        "balance_strategy": "none",
    },
    "E4": {
        "track": "singlelabel",
        "model_id": "m5_distilroberta",
        "distill": True,
        "dedup_policy": "consensus",
        "balance_strategy": "none",
    },
    "E5a": {"track": "singlelabel", "dedup_policy": "none", "balance_strategy": "none"},
    "E5b": {"track": "singlelabel", "dedup_policy": "global_first", "balance_strategy": "none"},
    "E5c": {"track": "singlelabel", "dedup_policy": "consensus", "balance_strategy": "hybrid"},
    "E7": {
        "track": "multilabel",
        "model_id": "m9_twitter_roberta",
        "loss_type": "asymmetric",
        "split_mode": "official",
        "dedup_policy": "consensus",
        "balance_strategy": "none",
    },
    "E8": {
        "track": "multilabel",
        "model_id": "m11_twitter_roberta_emotion",
        "loss_type": "asymmetric",
        "split_mode": "official",
        "dedup_policy": "consensus",
        "balance_strategy": "none",
    },
    "E9": {
        "track": "multilabel",
        "model_id": "m10_deberta_v3_large",
        "loss_type": "asymmetric",
        "split_mode": "official",
        "dedup_policy": "consensus",
        "balance_strategy": "none",
    },
    "E10": {
        "track": "singlelabel",
        "model_id": "m9_twitter_roberta",
        "loss_type": "weighted_ce",
        "use_class_weights": True,
        "dedup_policy": "consensus",
        "balance_strategy": "none",
    },
}


def merge_config(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def resolve_teacher_path(exp_config: dict) -> str | None:
    """Resolve teacher checkpoint from config or best E2/E7/E8/E9 result."""
    explicit = exp_config.get("teacher_model_path")
    if explicit:
        return explicit

    best = find_best_teacher_experiment()
    if not best:
        return None

    model_id = best.get("model_id")
    if not model_id:
        return None

    checkpoint = resolve_model_checkpoint(model_id)
    if checkpoint:
        exp_config["teacher_model_path"] = checkpoint
        exp_config["teacher_experiment"] = best.get("experiment_id")
        return checkpoint
    return None


def run_experiment(
    exp_id: str,
    config: dict,
    skip_train: bool = False,
    extra_overrides: dict | None = None,
) -> dict:
    preset = EXPERIMENT_PRESETS.get(exp_id, {})
    if not preset:
        raise ValueError(f"Unknown experiment id '{exp_id}'")

    if preset.get("stage") == "baselines":
        result = run_data_pipeline(config)
        train_df, val_df, test_df = result["train_df"], result["val_df"], result["test_df"]
        baselines = run_all_baselines(
            train_df["text"],
            train_df["encoded_label"],
            test_df["text"],
            test_df["encoded_label"],
        )
        return {"experiment_id": exp_id, "baselines": baselines}

    exp_config = merge_config(config, preset)
    if "model_id" in preset:
        exp_config = apply_model_to_config(exp_config, preset["model_id"])
    if extra_overrides:
        exp_config.update(extra_overrides)
    if exp_config.get("max_samples"):
        exp_config["epochs"] = 1
        exp_config["fp16"] = False

    if exp_config.get("distill") and not exp_config.get("teacher_model_path"):
        teacher_path = resolve_teacher_path(exp_config)
        if teacher_path is None and not skip_train:
            raise RuntimeError(
                "E4 distillation requires a trained teacher. Run E2, E7, E8, or E9 first, "
                "or set teacher_model_path in config."
            )

    result = run_data_pipeline(exp_config)
    train_df, val_df, test_df = result["train_df"], result["val_df"], result["test_df"]
    stats = result["stats"]

    max_samples = exp_config.get("max_samples")
    if max_samples:
        train_df = train_df.head(max_samples)
        val_df = val_df.head(max(32, max_samples // 8))
        test_df = test_df.head(max(32, max_samples // 8))
        stats = {**stats, "smoke_max_samples": max_samples}

    if skip_train:
        return {"experiment_id": exp_id, "stats": stats, "config": _safe_config(exp_config)}

    track = exp_config.get("track", "singlelabel")
    model_id = exp_config.get("model_id", "m3_roberta_base")
    get_model(model_id)

    if track == "multilabel":
        tokenizer = load_transformer_tokenizer(exp_config["model_name"])
        train_ds, val_ds, test_ds, train_labels = prepare_multilabel_hf_datasets(
            train_df, val_df, test_df, tokenizer, max_length=exp_config["max_length"]
        )
        trainer, tokenizer, model = build_multilabel_trainer(
            exp_config, train_ds, val_ds, train_labels=train_labels
        )
        trainer.train()
        export_dir = CHECKPOINTS_DIR / model_id
        trainer.save_model(str(export_dir))
        tokenizer.save_pretrained(str(export_dir))

        threshold_result = evaluate_multilabel_with_threshold_tuning(
            trainer, val_ds, test_ds, config=exp_config
        )
        default_m = threshold_result["test_metrics_default"]
        tuned_m = threshold_result["test_metrics_thresholded"]
        print(
            f"{exp_id} default (0.5)     test macro-F1={default_m['macro_f1']:.4f} "
            f"P={default_m['macro_precision']:.4f} R={default_m['macro_recall']:.4f}"
        )
        print(
            f"{exp_id} thresholded       test macro-F1={tuned_m['macro_f1']:.4f} "
            f"P={tuned_m['macro_precision']:.4f} R={tuned_m['macro_recall']:.4f}"
        )
        print(f"Per-class thresholds: {threshold_result['per_class_thresholds']}")

        if threshold_result.get("thresholds"):
            save_thresholds(
                __import__("numpy").array(threshold_result["thresholds"]),
                export_dir / "thresholds.json",
                metadata={
                    **(threshold_result.get("threshold_log") or {}),
                    "per_class_thresholds": threshold_result["per_class_thresholds"],
                    "track": "multilabel",
                },
            )

        from src.training.asl_config import resolve_asl_hyperparameters

        payload = {
            "experiment_id": exp_id,
            "track": track,
            "model_id": model_id,
            "stats": stats,
            "asl_hyperparameters": resolve_asl_hyperparameters(exp_config),
            "eval_metrics": threshold_result["val_metrics_thresholded"],
            "test_metrics_default": threshold_result["test_metrics_default"],
            "test_metrics_thresholded": threshold_result["test_metrics_thresholded"],
            "test_metrics": threshold_result["test_metrics_thresholded"],
            "val_test_macro_f1_gap": threshold_result["val_test_macro_f1_gap"],
            "thresholds": threshold_result.get("thresholds"),
            "threshold_log": threshold_result.get("threshold_log"),
            "per_class_thresholds": threshold_result.get("per_class_thresholds"),
        }
    else:
        tokenizer = load_transformer_tokenizer(exp_config["model_name"])
        train_ds, val_ds, test_ds, train_labels = prepare_hf_datasets(
            train_df, val_df, test_df, tokenizer, max_length=exp_config["max_length"]
        )
        trainer, tokenizer, model = build_trainer(
            exp_config, train_ds, val_ds, train_labels=train_labels
        )
        trainer.train()
        eval_result = evaluate_with_threshold_tuning(trainer, val_ds, test_ds, config=exp_config)
        export_name = "saved_emotion_model" if model_id in ("m3_roberta_base", "m4_roberta_focal") else model_id
        metadata = {
            **(eval_result.get("threshold_log") or {}),
            "uncertain_threshold": exp_config.get("uncertain_threshold", 0.35),
        }
        if exp_config.get("distill"):
            metadata["teacher_experiment"] = exp_config.get("teacher_experiment")
            metadata["teacher_model_path"] = exp_config.get("teacher_model_path")
        export_model(
            trainer,
            tokenizer,
            model_id=export_name,
            thresholds=(
                __import__("numpy").array(eval_result["thresholds"])
                if eval_result.get("thresholds")
                else None
            ),
            threshold_metadata=metadata,
        )
        payload = {
            "experiment_id": exp_id,
            "track": track,
            "model_id": model_id,
            "stats": stats,
            "eval_result": {
                k: v
                for k, v in eval_result.items()
                if k not in ("classification_report", "classification_report_argmax")
            },
            "classification_report": eval_result.get("classification_report"),
        }

    return payload


def _safe_config(config: dict) -> dict:
    """Return JSON-serializable subset of config for skip-train exports."""
    skip_keys = {"teacher_model_path"}
    return {k: v for k, v in config.items() if k not in skip_keys or v is not None}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IEEE experiment matrix")
    parser.add_argument(
        "--experiment",
        default="E3",
        help="Experiment ID (E0-E10, E5a-E5c)",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Smoke test: cap train/val/test rows (sets epochs=1 if not overridden)",
    )
    parser.add_argument("--faithfulness", action="store_true", help="Run XAI faithfulness after train")
    args = parser.parse_args()

    ensure_artifact_dirs()
    config = load_config()
    if args.max_samples:
        config["max_samples"] = args.max_samples
        config.setdefault("epochs", 1)
        config["fp16"] = False
    payload = run_experiment(args.experiment, config, skip_train=args.skip_train)

    out_path = EXPORTS_DIR / f"experiment_{args.experiment}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
