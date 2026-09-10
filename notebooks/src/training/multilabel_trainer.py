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
)

from src.data.label_mapping import ID2LABEL, LABEL2ID, NUM_LABELS
from src.data.multi_label_mapping import str_to_multi_hot
from src.paths import CHECKPOINTS_DIR
from src.training.asl_config import (
    DEFAULT_CLIP,
    DEFAULT_GAMMA_NEG,
    DEFAULT_GAMMA_POS,
    resolve_asl_hyperparameters,
)
from src.training.asymmetric_loss import AsymmetricLoss
from src.training.focal_loss import compute_class_weights
from src.training.loss_logging import AsymmetricLossHyperparamCallback, OptimizerHyperparamCallback
from src.training.metrics import hf_compute_multilabel_metrics
from src.training.optimizer_utils import build_adamw_param_groups
from src.training.trainer_setup import load_transformer_tokenizer
from src.training.training_args_builder import (
    build_training_arguments,
    resolve_optimizer_hyperparameters,
)


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
        asymmetric_gamma_pos: float = DEFAULT_GAMMA_POS,
        asymmetric_gamma_neg: float = DEFAULT_GAMMA_NEG,
        asymmetric_clip: float = DEFAULT_CLIP,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.pos_weights = pos_weights
        self.asymmetric_gamma_pos = asymmetric_gamma_pos
        self.asymmetric_gamma_neg = asymmetric_gamma_neg
        self.asymmetric_clip = asymmetric_clip
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

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        param_groups = build_adamw_param_groups(opt_model, self.args.weight_decay)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer


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

    training_args = build_training_arguments(config, checkpoint_dir)

    loss_type = config.get("loss_type", "asymmetric")
    asymmetric_gamma_pos = config.get("asymmetric_gamma_pos", DEFAULT_GAMMA_POS)
    asymmetric_gamma_neg = config.get("asymmetric_gamma_neg", DEFAULT_GAMMA_NEG)
    asymmetric_clip = config.get("asymmetric_clip", DEFAULT_CLIP)
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
    callbacks.append(OptimizerHyperparamCallback(resolve_optimizer_hyperparameters(config)))
    if loss_type == "asymmetric":
        callbacks.append(AsymmetricLossHyperparamCallback(resolve_asl_hyperparameters(config)))

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
        asymmetric_gamma_pos=asymmetric_gamma_pos,
        asymmetric_gamma_neg=asymmetric_gamma_neg,
        asymmetric_clip=asymmetric_clip,
    )
    return trainer, tokenizer, model
