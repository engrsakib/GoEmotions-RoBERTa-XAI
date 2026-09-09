"""Multi-label HuggingFace Trainer with asymmetric / weighted BCE."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.data.label_mapping import ID2LABEL, LABEL2ID, NUM_LABELS
from src.data.multi_label_mapping import str_to_multi_hot
from src.paths import CHECKPOINTS_DIR, LOGS_DIR
from src.training.asymmetric_loss import AsymmetricLoss
from src.training.focal_loss import compute_class_weights
from src.training.metrics import hf_compute_multilabel_metrics
from src.training.trainer_setup import load_transformer_tokenizer


def prepare_multilabel_hf_datasets(train_df, val_df, test_df, tokenizer, max_length: int = 128):
    def to_dataset(frame):
        labels = frame["multi_hot_labels"].apply(str_to_multi_hot).tolist()
        ds = Dataset.from_dict({"text": frame["text"].tolist(), "labels": labels})
        return ds

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)
    test_ds = to_dataset(test_df)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    def format_labels(batch):
        batch["labels"] = [torch.tensor(row, dtype=torch.float) for row in batch["labels"]]
        return batch

    train_ds = train_ds.map(format_labels, batched=True)
    val_ds = val_ds.map(format_labels, batched=True)
    test_ds = test_ds.map(format_labels, batched=True)

    for ds in (train_ds, val_ds, test_ds):
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_labels = [row for row in train_ds["labels"]]
    return train_ds, val_ds, test_ds, train_labels


class MultiLabelTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_type: str = "asymmetric",
        pos_weights: torch.Tensor | None = None,
        asymmetric_gamma_pos: float = 0.0,
        asymmetric_gamma_neg: float = 4.0,
        asymmetric_clip: float = 0.05,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.pos_weights = pos_weights
        self.asymmetric_loss = AsymmetricLoss(
            gamma_pos=asymmetric_gamma_pos,
            gamma_neg=asymmetric_gamma_neg,
            clip=asymmetric_clip,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        if labels.dtype != torch.float32:
            labels = labels.float()
        outputs = model(**inputs)
        logits = outputs.logits

        if self.loss_type == "asymmetric":
            loss = self.asymmetric_loss(logits, labels)
        elif self.loss_type == "weighted_ce":
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=self.pos_weights
            )
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return (loss, outputs) if return_outputs else loss


def _multilabel_pos_weights(train_labels: list, num_classes: int) -> torch.Tensor:
    """Compute pos_weight for BCE from multi-hot label vectors."""
    counts = torch.zeros(num_classes)
    for label_tensor in train_labels:
        vec = label_tensor if isinstance(label_tensor, torch.Tensor) else torch.tensor(label_tensor)
        counts += vec.float()
    total = len(train_labels)
    pos = counts.clamp(min=1.0)
    neg = (total - pos).clamp(min=1.0)
    return (neg / pos).sqrt()


def build_multilabel_trainer(
    config: dict,
    train_dataset,
    val_dataset,
    train_labels: list | None = None,
):
    model_name = config.get("model_name", "microsoft/deberta-v3-base")
    model_id = config.get("model_id", "multilabel")
    checkpoint_dir = CHECKPOINTS_DIR / model_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_transformer_tokenizer(model_name)
    load_kwargs = {
        "num_labels": NUM_LABELS,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "problem_type": "multi_label_classification",
    }
    if config.get("ignore_mismatched_sizes"):
        load_kwargs["ignore_mismatched_sizes"] = True
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_kwargs)

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=config.get("epochs", 5),
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
        greater_is_better=True,
        logging_dir=str(LOGS_DIR),
        logging_steps=config.get("logging_steps", 50),
        fp16=config.get("fp16", False) and torch.cuda.is_available(),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        report_to=[],
    )

    loss_type = config.get("loss_type", "asymmetric")
    pos_weights = None
    if train_labels is None:
        train_labels = [row for row in train_dataset["labels"]]
    if loss_type == "weighted_ce" or config.get("use_class_weights", True):
        pos_weights = _multilabel_pos_weights(train_labels, NUM_LABELS)
        if torch.cuda.is_available():
            pos_weights = pos_weights.cuda()

    callbacks = []
    if config.get("early_stopping", False):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.get("early_stopping_patience", 2),
            )
        )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=hf_compute_multilabel_metrics,
        callbacks=callbacks,
        loss_type=loss_type,
        pos_weights=pos_weights,
        asymmetric_gamma_pos=config.get("asymmetric_gamma_pos", 0.0),
        asymmetric_gamma_neg=config.get("asymmetric_gamma_neg", 4.0),
        asymmetric_clip=config.get("asymmetric_clip", 0.05),
    )
    return trainer, tokenizer, model
