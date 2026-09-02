#!/usr/bin/env python3
"""Run the full training pipeline (notebook Sections 1-7) from the CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import torch
from transformers import AutoTokenizer

from src.data.label_mapping import ID2LABEL
from src.data.pipeline import load_config, run_data_pipeline
from src.paths import EXPORTS_DIR, ensure_artifact_dirs
from src.training.baselines import train_baseline
from src.training.trainer_setup import (
    build_trainer,
    evaluate_on_test,
    export_model,
    prepare_hf_datasets,
)
from src.xai.captum_ig import explain_samples


def deploy_to_production(export_dir: Path, repo_root: Path) -> Path:
    import shutil

    dest = repo_root / "packages" / "model" / "saved_emotion_model"
    dest.mkdir(parents=True, exist_ok=True)
    for item in export_dir.iterdir():
        target = dest / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GoEmotions-RoBERTa training pipeline")
    parser.add_argument("--skip-data", action="store_true", help="Use existing processed CSVs")
    parser.add_argument("--skip-train", action="store_true", help="Skip RoBERTa training")
    parser.add_argument("--deploy", action="store_true", help="Copy export to packages/model/")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training rows (local dev)")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation rows (local dev)")
    args = parser.parse_args()

    ensure_artifact_dirs()
    config = load_config()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if not torch.cuda.is_available():
        config["fp16"] = False
        print("CPU mode: fp16 disabled")

    print("=== Section 1: Data Engineering ===")
    if args.skip_data:
        from src.data.pipeline import load_processed_splits

        train_df, val_df, test_df = load_processed_splits()
        stats = {}
    else:
        result = run_data_pipeline(config)
        train_df, val_df, test_df = result["train_df"], result["val_df"], result["test_df"]
        stats = result["stats"]
        print(json.dumps({k: v for k, v in stats.items() if k != "audit"}, indent=2, default=str))

    if args.max_train_samples and len(train_df) > args.max_train_samples:
        train_df = train_df.sample(n=args.max_train_samples, random_state=42).reset_index(drop=True)
        print(f"Using training subset: {len(train_df)} samples")

    if args.max_val_samples and len(val_df) > args.max_val_samples:
        val_df = val_df.sample(n=args.max_val_samples, random_state=42).reset_index(drop=True)
        print(f"Using validation subset: {len(val_df)} samples")

    print("\n=== Section 3: Baseline ===")
    baseline = train_baseline(
        train_df["text"],
        train_df["encoded_label"],
        test_df["text"],
        test_df["encoded_label"],
        max_features=config["baseline_max_features"],
        ngram_range=tuple(config["baseline_ngram_range"]),
    )
    baseline_f1 = baseline["metrics"]["macro_f1"]
    print(f"Baseline macro-F1: {baseline_f1:.4f}")

    if args.skip_train:
        print("Skipping RoBERTa training.")
        return

    print("\n=== Section 4: RoBERTa Training ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    train_ds, val_ds, test_ds, train_labels = prepare_hf_datasets(
        train_df, val_df, test_df, tokenizer, max_length=config["max_length"]
    )
    trainer, tokenizer, model = build_trainer(config, train_ds, val_ds, train_labels=train_labels)
    trainer.train()

    print("\n=== Section 5: Test Evaluation ===")
    test_result = evaluate_on_test(trainer, test_ds)
    roberta_f1 = test_result["metrics"]["macro_f1"]
    print(f"RoBERTa test macro-F1: {roberta_f1:.4f} (baseline: {baseline_f1:.4f})")
    print(test_result["classification_report"])

    print("\n=== Section 6: XAI Validation ===")
    samples = test_df["text"].sample(n=min(config["xai_sample_count"], len(test_df)), random_state=42).tolist()
    xai_results = explain_samples(
        model, tokenizer, samples, device,
        max_length=config["max_length"], n_steps=config["ig_n_steps"],
    )
    for r in xai_results:
        assert len(r["tokens"]) == len(r["heatmap"])
        print(f"  [{ID2LABEL[r['target_class']]}] {r['text'][:60]}...")

    print("\n=== Section 7: Export ===")
    export_dir = export_model(trainer, tokenizer)
    print(f"Exported to: {export_dir}")

    metrics_path = EXPORTS_DIR / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "baseline_macro_f1": baseline_f1,
                "roberta_test_macro_f1": roberta_f1,
                "test_metrics": test_result["metrics"],
            },
            handle,
            indent=2,
        )

    if args.deploy:
        repo_root = NOTEBOOKS_DIR.parent
        dest = deploy_to_production(export_dir, repo_root)
        print(f"Deployed to: {dest}")


if __name__ == "__main__":
    main()
