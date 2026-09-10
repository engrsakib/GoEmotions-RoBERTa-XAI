#!/usr/bin/env python3
"""Post-hoc multi-label threshold tuning on a trained checkpoint (e.g. m6_deberta_v3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from transformers import AutoModelForSequenceClassification, TrainingArguments

from src.data.label_mapping import ID2LABEL, LABEL2ID, NUM_LABELS
from src.data.pipeline import load_config, run_data_pipeline
from src.paths import CHECKPOINTS_DIR, EXPORTS_DIR, ensure_artifact_dirs
from src.training.model_profiles import resolve_model_checkpoint
from src.training.model_registry import apply_model_to_config, get_model
from src.training.multilabel_trainer import MultiLabelTrainer, prepare_multilabel_hf_datasets
from src.training.thresholds import save_thresholds
from src.training.trainer_setup import (
    evaluate_multilabel_with_threshold_tuning,
    load_transformer_tokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune multi-label thresholds on val; eval test")
    parser.add_argument("--model-id", default="m6_deberta_v3", help="Registry model ID")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    parser.add_argument("--experiment", default="E2", help="Experiment label for export filename")
    args = parser.parse_args()

    ensure_artifact_dirs()
    config = load_config()
    config = apply_model_to_config(config, args.model_id)
    config["track"] = "multilabel"

    checkpoint = args.checkpoint or resolve_model_checkpoint(args.model_id)
    if not checkpoint:
        raise SystemExit(f"No checkpoint found for '{args.model_id}'. Train E2 first.")

    result = run_data_pipeline(config)
    train_df, val_df, test_df = result["train_df"], result["val_df"], result["test_df"]

    spec = get_model(args.model_id)
    tokenizer = load_transformer_tokenizer(spec.huggingface_id or config["model_name"])
    train_ds, val_ds, test_ds, _ = prepare_multilabel_hf_datasets(
        train_df, val_df, test_df, tokenizer, max_length=config["max_length"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="multi_label_classification",
    )
    from transformers import DataCollatorWithPadding

    trainer = MultiLabelTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(CHECKPOINTS_DIR / f"{args.model_id}_eval"),
            report_to=[],
            per_device_eval_batch_size=config.get("eval_batch_size", 16),
        ),
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    eval_result = evaluate_multilabel_with_threshold_tuning(
        trainer, val_ds, test_ds, config=config
    )

    default_m = eval_result["test_metrics_default"]
    tuned_m = eval_result["test_metrics_thresholded"]
    print(f"\n=== {args.model_id} threshold tuning ({args.experiment}) ===")
    print(
        f"Default (0.5)     test macro-F1={default_m['macro_f1']:.4f} "
        f"P={default_m['macro_precision']:.4f} R={default_m['macro_recall']:.4f}"
    )
    print(
        f"Thresholded       test macro-F1={tuned_m['macro_f1']:.4f} "
        f"P={tuned_m['macro_precision']:.4f} R={tuned_m['macro_recall']:.4f}"
    )
    print(f"Val-test macro-F1 gap: {eval_result['val_test_macro_f1_gap']:.4f}")
    print(f"Per-class thresholds: {eval_result['per_class_thresholds']}")

    export_dir = CHECKPOINTS_DIR / args.model_id
    if eval_result.get("thresholds"):
        save_thresholds(
            __import__("numpy").array(eval_result["thresholds"]),
            export_dir / "thresholds.json",
            metadata={
                **(eval_result.get("threshold_log") or {}),
                "per_class_thresholds": eval_result["per_class_thresholds"],
                "track": "multilabel",
            },
        )

    out_path = EXPORTS_DIR / f"experiment_{args.experiment}_thresholds.json"
    payload = {
        "experiment_id": args.experiment,
        "model_id": args.model_id,
        "checkpoint": checkpoint,
        **eval_result,
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
