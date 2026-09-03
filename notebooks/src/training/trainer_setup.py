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
    TrainingArguments,
    Trainer,
)

from src.data.label_mapping import ID2LABEL, LABEL2ID, NUM_LABELS
from src.paths import CHECKPOINTS_DIR, EXPORTS_DIR, LOGS_DIR, PROCESSED_DIR
from src.training.focal_loss import FocalLossTrainer, compute_class_weights
from src.training.metrics import build_classification_report, build_confusion_matrix, hf_compute_metrics


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

    trainer_cls = Trainer
    trainer_kwargs = {}

    if config.get("use_focal_loss", True):
        if train_labels is None:
            train_labels = [int(x) for x in train_dataset["labels"]]
        class_weights = compute_class_weights(train_labels, NUM_LABELS)
        if torch.cuda.is_available():
            class_weights = class_weights.cuda()
        trainer_cls = FocalLossTrainer
        trainer_kwargs = {
            "focal_gamma": config.get("focal_gamma", 2.0),
            "class_weights": class_weights,
        }

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=metrics_fn,
        **trainer_kwargs,
    )
    return trainer, tokenizer, model


def evaluate_on_test(trainer, test_dataset, id2label: dict | None = None) -> dict:
    id2label = id2label or ID2LABEL
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    return {
        "metrics": hf_compute_metrics((predictions.predictions, labels), id2label),
        "classification_report": build_classification_report(labels, preds, id2label),
        "confusion_matrix": build_confusion_matrix(labels, preds, labels=sorted(id2label.keys())),
        "predictions": preds,
        "labels": labels,
    }


def export_model(
    trainer,
    tokenizer,
    export_dir: Path | None = None,
    processed_dir: Path | None = None,
    model_id: str = "saved_emotion_model",
) -> Path:
    export_dir = Path(export_dir or EXPORTS_DIR / model_id)
    processed_dir = processed_dir or PROCESSED_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))

    label_map_src = processed_dir / "label_map.json"
    if label_map_src.exists():
        shutil.copy2(label_map_src, export_dir / "label_map.json")

    return export_dir
