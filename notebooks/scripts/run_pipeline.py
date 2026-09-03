#!/usr/bin/env python3
"""
GoEmotions-RoBERTa-XAI — End-to-End Training Pipeline

Replaces lab_final.ipynb with a modular, reproducible CLI workflow.

Stages:
  0. bootstrap   — environment setup (Kaggle / local)
  1. data        — load, map, clean, split, export
  2. eda         — class distribution figure + length stats
  3. baselines   — M1 LogReg, M2 Linear SVM
  4. train       — transformer fine-tuning (M3–M7)
  5. evaluate    — test macro-F1, classification report
  6. xai         — Captum Integrated Gradients (M8)
  7. export      — HuggingFace checkpoint + optional deploy
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import torch
from transformers import AutoTokenizer

from src.bootstrap.environment import bootstrap, is_kaggle
from src.data.pipeline import load_config, load_processed_splits, run_data_pipeline
from src.paths import EXPORTS_DIR, ensure_artifact_dirs
from src.training.baselines import run_all_baselines
from src.training.model_registry import apply_model_to_config, default_transformer_id, get_model
from src.training.trainer_setup import build_trainer, evaluate_on_test, export_model, prepare_hf_datasets
from src.visualization.eda import run_eda
from src.xai.captum_ig import explain_samples


def deploy_to_production(export_dir: Path, repo_root: Path) -> Path:
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


def stage_bootstrap() -> None:
    nb = bootstrap(install_deps=True)
    print(f"Stage 0 | notebooks={nb} | kaggle={is_kaggle()}")


def stage_data(config: dict, skip: bool) -> tuple:
    if skip:
        train_df, val_df, test_df = load_processed_splits()
        return train_df, val_df, test_df, {}
    result = run_data_pipeline(config)
    return result["train_df"], result["val_df"], result["test_df"], result["stats"]


def maybe_subsample(train_df, val_df, args):
    if args.max_train_samples and len(train_df) > args.max_train_samples:
        train_df = train_df.sample(n=args.max_train_samples, random_state=42).reset_index(drop=True)
        print(f"Train subset: {len(train_df)}")
    if args.max_val_samples and len(val_df) > args.max_val_samples:
        val_df = val_df.sample(n=args.max_val_samples, random_state=42).reset_index(drop=True)
        print(f"Val subset: {len(val_df)}")
    return train_df, val_df


def main() -> None:
    parser = argparse.ArgumentParser(description="GoEmotions-RoBERTa training pipeline")
    parser.add_argument("--stage", choices=["all", "data", "eda", "baselines", "train", "evaluate", "xai", "export"], default="all")
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--deploy", action="store_true", help="Copy best export to packages/model/")
    parser.add_argument("--model-id", default=None, help="Transformer model id from registry (default: m4_roberta_focal)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    if not args.skip_bootstrap:
        stage_bootstrap()

    ensure_artifact_dirs()
    config = load_config()
    model_id = args.model_id or config.get("primary_model_id", default_transformer_id())
    config = apply_model_to_config(config, model_id)

    if args.epochs is not None:
        config["epochs"] = args.epochs
    if not torch.cuda.is_available():
        config["fp16"] = False

    run_all = args.stage == "all"

    # Stage 1: Data Engineering
    if run_all or args.stage == "data":
        print("\n=== Stage 1: Data Engineering ===")
        train_df, val_df, test_df, stats = stage_data(config, args.skip_data)
        if stats:
            print(json.dumps({k: v for k, v in stats.items() if k != "audit"}, indent=2, default=str))
        print(f"Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")
    else:
        train_df, val_df, test_df, _ = stage_data(config, skip=True)

    train_df, val_df = maybe_subsample(train_df, val_df, args)

    # Stage 2: EDA
    if run_all or args.stage == "eda":
        print("\n=== Stage 2: EDA ===")
        run_eda(train_df)

    # Stage 3: Baselines (M1, M2)
    baseline_results = {}
    if run_all or args.stage == "baselines":
        print("\n=== Stage 3: Baselines (M1–M2) ===")
        baseline_results = run_all_baselines(
            train_df["text"], train_df["encoded_label"],
            test_df["text"], test_df["encoded_label"],
            max_features=config["baseline_max_features"],
            ngram_range=tuple(config["baseline_ngram_range"]),
        )
        for mid, res in baseline_results.items():
            print(f"{mid} macro-F1: {res['metrics']['macro_f1']:.4f}")

    if args.skip_train and (run_all or args.stage == "train"):
        print("Training skipped.")
        return

    # Stage 4–5: Transformer train + evaluate
    if run_all or args.stage in ("train", "evaluate"):
        spec = get_model(model_id)
        print(f"\n=== Stage 4: Train {spec.name} ({model_id}) ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device} | HF: {config['model_name']}")

        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        train_ds, val_ds, test_ds, train_labels = prepare_hf_datasets(
            train_df, val_df, test_df, tokenizer, max_length=config["max_length"]
        )
        trainer, tokenizer, model = build_trainer(config, train_ds, val_ds, train_labels=train_labels)

        if run_all or args.stage == "train":
            trainer.train()

        print("\n=== Stage 5: Evaluation ===")
        test_result = evaluate_on_test(trainer, test_ds)
        roberta_f1 = test_result["metrics"]["macro_f1"]
        print(f"{model_id} test macro-F1: {roberta_f1:.4f}")
        print(test_result["classification_report"])

        metrics_path = EXPORTS_DIR / "training_metrics.json"
        payload = {
            "model_id": model_id,
            "model_name": config["model_name"],
            "transformer_test_macro_f1": roberta_f1,
            "test_metrics": test_result["metrics"],
        }
        if baseline_results:
            payload["baselines"] = {k: v["metrics"]["macro_f1"] for k, v in baseline_results.items()}
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    # Stage 6: XAI (M8)
    if run_all or args.stage == "xai":
        print("\n=== Stage 6: XAI (M8 Captum IG) ===")
        samples = test_df["text"].sample(n=min(config["xai_sample_count"], len(test_df)), random_state=42).tolist()
        xai_results = explain_samples(
            model, tokenizer, samples, device,
            max_length=config["max_length"], n_steps=config["ig_n_steps"],
        )
        for r in xai_results:
            assert len(r["tokens"]) == len(r["heatmap"])
            print(f"  class={r['target_class']} | {r['text'][:60]}...")

    # Stage 7: Export
    if run_all or args.stage == "export":
        print("\n=== Stage 7: Export ===")
        export_name = "saved_emotion_model" if model_id in ("m3_roberta_base", "m4_roberta_focal") else model_id
        export_dir = export_model(trainer, tokenizer, model_id=export_name)
        print(f"Exported: {export_dir}")
        if args.deploy:
            dest = deploy_to_production(export_dir, NOTEBOOKS_DIR.parent)
            print(f"Deployed: {dest}")


if __name__ == "__main__":
    main()
