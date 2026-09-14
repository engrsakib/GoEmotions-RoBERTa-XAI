#!/usr/bin/env python3
"""Export val/test multilabel probabilities for ensemble members."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding

from src.data.label_mapping import ID2LABEL, LABEL2ID, NUM_LABELS
from src.data.pipeline import load_config, load_or_build_processed_splits
from src.eval.ensemble_probs import save_probs_from_trainer
from src.paths import CHECKPOINTS_DIR, ensure_artifact_dirs
from src.training.model_profiles import get_ensemble_config, resolve_model_checkpoint
from src.training.model_registry import apply_model_to_config
from src.training.multilabel_trainer import prepare_multilabel_hf_datasets
from src.training.trainer_setup import load_transformer_tokenizer
from src.training.training_args_builder import build_training_arguments
from src.training.multilabel_trainer import MultiLabelTrainer
from src.training.metrics import hf_compute_multilabel_metrics


def _load_trainer_for_inference(model_id: str, config: dict, val_ds, train_labels=None):
    config = apply_model_to_config(config, model_id)
    model_name = config["model_name"]
    checkpoint = resolve_model_checkpoint(model_id) or str(CHECKPOINTS_DIR / model_id)
    tokenizer = load_transformer_tokenizer(model_name)
    load_kwargs = {
        "num_labels": NUM_LABELS,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "problem_type": "multi_label_classification",
    }
    if config.get("ignore_mismatched_sizes"):
        load_kwargs["ignore_mismatched_sizes"] = True
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, **load_kwargs)
    training_args = build_training_arguments(config, CHECKPOINTS_DIR / model_id)
    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=val_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=hf_compute_multilabel_metrics,
        loss_type=config.get("loss_type", "asymmetric"),
    )
    return trainer, tokenizer


def export_probs_for_models(model_ids: list[str], config: dict | None = None) -> dict[str, dict]:
    ensure_artifact_dirs()
    config = config or load_config()
    config["track"] = "multilabel"
    train_df, val_df, test_df, _stats = load_or_build_processed_splits(config)

    results = {}
    for model_id in model_ids:
        print(f"\n=== Export probs: {model_id} ===")
        cfg = apply_model_to_config(dict(config), model_id)
        if not resolve_model_checkpoint(model_id) and not (CHECKPOINTS_DIR / model_id).is_dir():
            print(f"SKIP {model_id}: no checkpoint found")
            continue
        tokenizer = load_transformer_tokenizer(cfg["model_name"])
        _train_ds, val_ds, test_ds, train_labels = prepare_multilabel_hf_datasets(
            train_df, val_df, test_df, tokenizer, max_length=cfg.get("max_length", 128)
        )
        trainer, _ = _load_trainer_for_inference(model_id, cfg, val_ds)
        paths = save_probs_from_trainer(trainer, val_ds, test_ds, model_id)
        results[model_id] = paths
        print(f"Saved: {paths}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ensemble val/test probabilities")
    parser.add_argument(
        "--model-ids",
        nargs="*",
        default=None,
        help="Model registry IDs (default: ensemble block in model_profiles.yaml)",
    )
    args = parser.parse_args()
    ens = get_ensemble_config()
    model_ids = args.model_ids or list(ens.get("model_ids") or [])
    export_probs_for_models(model_ids)


if __name__ == "__main__":
    main()
