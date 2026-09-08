#!/usr/bin/env python3
"""IEEE experiment matrix runner (E0–E5)."""

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
from src.paths import EXPORTS_DIR, ensure_artifact_dirs
from src.training.baselines import run_all_baselines
from src.training.model_registry import apply_model_to_config, get_model
from src.training.multilabel_trainer import build_multilabel_trainer, prepare_multilabel_hf_datasets
from src.training.trainer_setup import (
    build_trainer,
    evaluate_with_threshold_tuning,
    export_model,
    prepare_hf_datasets,
)
from src.xai.faithfulness import evaluate_faithfulness_batch
from src.xai.captum_ig import explain_samples


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
        "model_id": "m3_roberta_base",
        "distill": True,
        "dedup_policy": "consensus",
        "balance_strategy": "none",
    },
    "E5a": {"track": "singlelabel", "dedup_policy": "none", "balance_strategy": "none"},
    "E5b": {"track": "singlelabel", "dedup_policy": "global_first", "balance_strategy": "none"},
    "E5c": {"track": "singlelabel", "dedup_policy": "consensus", "balance_strategy": "hybrid"},
}


def merge_config(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


def run_experiment(exp_id: str, config: dict, skip_train: bool = False) -> dict:
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

    result = run_data_pipeline(exp_config)
    train_df, val_df, test_df = result["train_df"], result["val_df"], result["test_df"]
    stats = result["stats"]

    if skip_train:
        return {"experiment_id": exp_id, "stats": stats}

    track = exp_config.get("track", "singlelabel")
    model_id = exp_config.get("model_id", "m3_roberta_base")
    spec = get_model(model_id)

    if track == "multilabel":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(exp_config["model_name"])
        train_ds, val_ds, test_ds, train_labels = prepare_multilabel_hf_datasets(
            train_df, val_df, test_df, tokenizer, max_length=exp_config["max_length"]
        )
        trainer, tokenizer, model = build_multilabel_trainer(
            exp_config, train_ds, val_ds, train_labels=train_labels
        )
        trainer.train()
        eval_result = trainer.evaluate()
        payload = {
            "experiment_id": exp_id,
            "track": track,
            "model_id": model_id,
            "stats": stats,
            "eval_metrics": eval_result,
        }
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(exp_config["model_name"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IEEE experiment matrix")
    parser.add_argument("--experiment", default="E3", help="Experiment ID (E0-E5c)")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--faithfulness", action="store_true", help="Run XAI faithfulness after train")
    args = parser.parse_args()

    ensure_artifact_dirs()
    config = load_config()
    payload = run_experiment(args.experiment, config, skip_train=args.skip_train)

    out_path = EXPORTS_DIR / f"experiment_{args.experiment}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
