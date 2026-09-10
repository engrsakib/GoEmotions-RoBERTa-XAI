"""Shared HuggingFace TrainingArguments factory for transformer trainers."""

from __future__ import annotations

from typing import Any

import torch
from transformers import TrainingArguments

from src.paths import LOGS_DIR


def resolve_mixed_precision(config: dict[str, Any]) -> dict[str, bool]:
    """
    Resolve mutually exclusive fp16/bf16 flags from config and hardware.

    When neither is explicitly forced, fp16 is enabled if requested and CUDA
    is available; bf16 only when explicitly set and supported.
    """
    cuda_available = torch.cuda.is_available()
    use_bf16 = bool(config.get("bf16", False)) and cuda_available
    use_fp16 = bool(config.get("fp16", False)) and cuda_available and not use_bf16

    if config.get("force_fp32"):
        return {"fp16": False, "bf16": False}

    return {"fp16": use_fp16, "bf16": use_bf16}


def precision_mode_from_config(config: dict[str, Any]) -> str:
    """Return active precision label: fp16, bf16, or fp32."""
    flags = resolve_mixed_precision(config)
    if flags.get("bf16"):
        return "bf16"
    if flags.get("fp16"):
        return "fp16"
    return "fp32"


def build_training_arguments(
    config: dict[str, Any],
    checkpoint_dir,
    *,
    num_train_epochs: int | None = None,
    report_to: list | None = None,
    run_name: str | None = None,
) -> TrainingArguments:
    """Build TrainingArguments from YAML config with DeBERTa-v3 defaults."""
    precision = resolve_mixed_precision(config)
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=num_train_epochs or config.get("epochs", 5),
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("eval_batch_size", 16),
        learning_rate=config.get("learning_rate", 1.5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.10),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        optim=config.get("optim", "adamw_torch"),
        adam_beta1=config.get("adam_beta1", 0.9),
        adam_beta2=config.get("adam_beta2", 0.999),
        adam_epsilon=config.get("adam_epsilon", 1e-6),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        load_best_model_at_end=True,
        metric_for_best_model=config.get("metric_for_best_model", "eval_macro_f1"),
        greater_is_better=config.get("greater_is_better", True),
        logging_dir=str(LOGS_DIR),
        logging_steps=config.get("logging_steps", 50),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        report_to=report_to if report_to is not None else config.get("report_to", []),
        run_name=run_name or config.get("run_name"),
        **precision,
    )


def compute_train_val_overfit_report(
    trainer,
    train_dataset,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare train vs best validation macro-F1; recommend higher weight_decay if gap exceeds threshold.
    """
    train_metrics = trainer.evaluate(train_dataset)
    train_f1 = float(train_metrics.get("eval_macro_f1", 0.0))
    val_f1 = max(
        (
            float(log["eval_macro_f1"])
            for log in trainer.state.log_history
            if log.get("eval_macro_f1") is not None
        ),
        default=0.0,
    )
    gap = round(train_f1 - val_f1, 4)
    threshold = float(config.get("overfit_gap_threshold", 0.03))
    recommended_wd = None
    if gap > threshold:
        recommended_wd = float(config.get("weight_decay_overfit", 0.05))
        print(
            f"WARNING: Train-val macro-F1 gap {gap:.4f} > {threshold:.2f}; "
            f"re-run with weight_decay: {recommended_wd}"
        )
    return {
        "train_macro_f1": round(train_f1, 4),
        "best_val_macro_f1": round(val_f1, 4),
        "train_val_macro_f1_gap": gap,
        "recommended_weight_decay": recommended_wd,
    }


def resolve_optimizer_hyperparameters(config: dict[str, Any]) -> dict[str, float | str]:
    """Return active optimizer/scheduler hyperparameters for logging and exports."""
    return {
        "learning_rate": float(config.get("learning_rate", 1.5e-5)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.10)),
        "lr_scheduler_type": str(config.get("lr_scheduler_type", "cosine")),
        "adam_beta1": float(config.get("adam_beta1", 0.9)),
        "adam_beta2": float(config.get("adam_beta2", 0.999)),
        "adam_epsilon": float(config.get("adam_epsilon", 1e-6)),
        "max_grad_norm": float(config.get("max_grad_norm", 1.0)),
    }
