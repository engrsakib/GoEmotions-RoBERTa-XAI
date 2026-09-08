"""HuggingFace Trainer setup, training, and export."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)

from src.data.label_mapping import ID2LABEL, LABEL2ID, NUM_LABELS
from src.paths import CHECKPOINTS_DIR, EXPORTS_DIR, LOGS_DIR, PROCESSED_DIR
from src.training.focal_loss import (
    FocalLossTrainer,
    WeightedCETrainer,
    compute_class_weights,
)
from src.training.metrics import build_classification_report, build_confusion_matrix, hf_compute_metrics
from src.training.thresholds import predict_with_thresholds, save_thresholds, softmax, tune_thresholds

try:
    from src.training.distill import DistillationTrainer, load_teacher_model
except ImportError:
    DistillationTrainer = None
    load_teacher_model = None


def load_label_map(processed_dir: Path | None = None) -> tuple[dict, dict]:
    processed_dir = processed_dir or PROCESSED_DIR
    with (processed_dir / "label_map.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    id2label = {int(k): v for k, v in payload["id2label"].items()}
    label2id = {str(k): int(v) for k, v in payload["label2id"].items()}
    return id2label, label2id


def prepare_hf_datasets(train_df, val_df, test_df, tokenizer, max_length: int = 128):
    def to_dataset(frame):
        renamed = frame.rename(columns={"encoded_label": "labels"})
        return Dataset.from_pandas(renamed[["text", "labels"]])

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)
    test_ds = to_dataset(test_df)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    train_labels = [int(x) for x in train_ds["labels"]]

    for ds in (train_ds, val_ds, test_ds):
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_ds, val_ds, test_ds, train_labels


def _resolve_loss_config(config: dict) -> tuple[str, bool, str]:
    """Return (loss_type, use_class_weights, class_weight_mode)."""
    loss_type = config.get("loss_type")
    if loss_type is None:
        loss_type = "focal" if config.get("use_focal_loss", True) else "weighted_ce"

    balance_strategy = config.get("balance_strategy", "none")
    use_class_weights = config.get("use_class_weights")
    if use_class_weights is None:
        use_class_weights = balance_strategy == "none"

    weight_mode = config.get("class_weight_mode", "sqrt_inverse")
    if not use_class_weights:
        weight_mode = "none"

    return loss_type, use_class_weights, weight_mode


def build_trainer(
    config: dict,
    train_dataset,
    val_dataset,
    train_labels: list[int] | None = None,
    id2label: dict | None = None,
    label2id: dict | None = None,
):
    id2label = id2label or ID2LABEL
    label2id = label2id or LABEL2ID
    model_name = config.get("model_name", "roberta-base")
    model_id = config.get("model_id", "transformer")
    checkpoint_dir = CHECKPOINTS_DIR / model_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config.get("epochs", 4),
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("eval_batch_size", 16),
        learning_rate=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        optim=config.get("optim", "adamw_torch"),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        eval_strategy=config.get("eval_strategy", "epoch"),
        save_strategy=config.get("save_strategy", "epoch"),
        load_best_model_at_end=True,
        metric_for_best_model=config.get("metric_for_best_model", "eval_macro_f1"),
        greater_is_better=config.get("greater_is_better", True),
        logging_dir=str(LOGS_DIR),
        logging_steps=config.get("logging_steps", 50),
        fp16=config.get("fp16", False) and torch.cuda.is_available(),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        report_to=[],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metrics_fn = lambda p: hf_compute_metrics(p, id2label)

    loss_type, use_class_weights, weight_mode = _resolve_loss_config(config)

    trainer_cls = Trainer
    trainer_kwargs: dict = {}
    callbacks = []

    if config.get("early_stopping", False):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.get("early_stopping_patience", 2),
            )
        )

    if train_labels is None:
        train_labels = [int(x) for x in train_dataset["labels"]]

    class_weights = compute_class_weights(train_labels, NUM_LABELS, mode=weight_mode)
    if class_weights is not None and torch.cuda.is_available():
        class_weights = class_weights.cuda()

    if loss_type == "focal":
        trainer_cls = FocalLossTrainer
        trainer_kwargs = {
            "focal_gamma": config.get("focal_gamma", 1.5),
            "class_weights": class_weights,
        }
    elif loss_type == "weighted_ce":
        trainer_cls = WeightedCETrainer
        trainer_kwargs = {"class_weights": class_weights}
    elif loss_type == "asymmetric":
        raise ValueError("loss_type 'asymmetric' requires track=multilabel; use build_multilabel_trainer().")
    else:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Choose: focal, weighted_ce")

    if config.get("distill") and config.get("teacher_model_path"):
        if DistillationTrainer is None or load_teacher_model is None:
            raise ImportError("DistillationTrainer not available")
        teacher = load_teacher_model(config["teacher_model_path"], NUM_LABELS)
        if torch.cuda.is_available():
            teacher = teacher.cuda()
        trainer_cls = DistillationTrainer
        trainer_kwargs = {
            "teacher_model": teacher,
            "distill_alpha": config.get("distill_alpha", 0.5),
            "distill_temperature": config.get("distill_temperature", 2.0),
            "class_weights": class_weights,
        }

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
        callbacks=callbacks,
        **trainer_kwargs,
    )
    return trainer, tokenizer, model


def _predict_probs(trainer, dataset) -> tuple[np.ndarray, np.ndarray]:
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    probs = softmax(logits)
    labels = predictions.label_ids
    return probs, labels


def evaluate_on_test(
    trainer,
    test_dataset,
    id2label: dict | None = None,
    thresholds: np.ndarray | None = None,
) -> dict:
    id2label = id2label or ID2LABEL
    probs, labels = _predict_probs(trainer, test_dataset)
    preds = (
        predict_with_thresholds(probs, thresholds)
        if thresholds is not None
        else np.argmax(probs, axis=-1)
    )

    return {
        "metrics": hf_compute_metrics((probs, labels), id2label),
        "classification_report": build_classification_report(labels, preds, id2label),
        "confusion_matrix": build_confusion_matrix(labels, preds, labels=sorted(id2label.keys())),
        "predictions": preds,
        "labels": labels,
        "probabilities": probs,
    }


def evaluate_with_threshold_tuning(
    trainer,
    val_dataset,
    test_dataset,
    id2label: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Tune thresholds on validation, evaluate test with argmax and thresholded preds."""
    id2label = id2label or ID2LABEL
    config = config or {}

    val_probs, val_labels = _predict_probs(trainer, val_dataset)
    test_probs, test_labels = _predict_probs(trainer, test_dataset)

    val_argmax_preds = np.argmax(val_probs, axis=-1)
    val_metrics = hf_compute_metrics((val_probs, val_labels), id2label)

    thresholds = None
    threshold_log: dict = {}
    if config.get("threshold_tuning", True):
        step = config.get("threshold_search_step", 0.05)
        thresholds, threshold_log = tune_thresholds(
            val_probs,
            val_labels,
            num_classes=NUM_LABELS,
            step=step,
            min_precision=config.get("threshold_min_precision"),
        )

    test_argmax_preds = np.argmax(test_probs, axis=-1)
    test_metrics_argmax = hf_compute_metrics((test_probs, test_labels), id2label)

    if thresholds is not None:
        test_threshold_preds = predict_with_thresholds(test_probs, thresholds)
        from src.training.metrics import compute_sklearn_metrics

        test_metrics_thresholded = compute_sklearn_metrics(test_labels, test_threshold_preds)
    else:
        test_threshold_preds = test_argmax_preds
        test_metrics_thresholded = test_metrics_argmax

    val_test_gap = abs(
        val_metrics["macro_f1"] - test_metrics_thresholded["macro_f1"]
    )

    return {
        "val_metrics": val_metrics,
        "test_metrics_argmax": test_metrics_argmax,
        "test_metrics_thresholded": test_metrics_thresholded,
        "test_metrics": test_metrics_thresholded,
        "val_test_macro_f1_gap": round(val_test_gap, 4),
        "thresholds": thresholds.tolist() if thresholds is not None else None,
        "threshold_log": threshold_log,
        "classification_report": build_classification_report(
            test_labels,
            test_threshold_preds if thresholds is not None else test_argmax_preds,
            id2label,
        ),
        "classification_report_argmax": build_classification_report(
            test_labels, test_argmax_preds, id2label
        ),
        "confusion_matrix": build_confusion_matrix(
            test_labels,
            test_threshold_preds if thresholds is not None else test_argmax_preds,
            labels=sorted(id2label.keys()),
        ),
        "predictions": test_threshold_preds if thresholds is not None else test_argmax_preds,
        "labels": test_labels,
    }


def export_model(
    trainer,
    tokenizer,
    export_dir: Path | None = None,
    processed_dir: Path | None = None,
    model_id: str = "saved_emotion_model",
    thresholds: np.ndarray | None = None,
    threshold_metadata: dict | None = None,
) -> Path:
    export_dir = Path(export_dir or EXPORTS_DIR / model_id)
    processed_dir = processed_dir or PROCESSED_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))

    label_map_src = processed_dir / "label_map.json"
    if label_map_src.exists():
        shutil.copy2(label_map_src, export_dir / "label_map.json")

    if thresholds is not None:
        save_thresholds(thresholds, export_dir / "thresholds.json", threshold_metadata)

    return export_dir
