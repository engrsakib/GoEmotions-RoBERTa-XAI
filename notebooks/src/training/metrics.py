"""Evaluation metrics with macro-F1 as primary metric."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_sklearn_metrics(y_true, y_pred, labels: list[int] | None = None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=labels
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(weighted_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
    }


def build_classification_report(y_true, y_pred, id2label: dict) -> str:
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    return classification_report(
        y_true, y_pred, target_names=target_names, zero_division=0, digits=4
    )


def build_confusion_matrix(y_true, y_pred, labels: list[int] | None = None) -> list[list[int]]:
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()


def hf_compute_metrics(eval_pred, id2label: dict | None = None):
    """HuggingFace Trainer compute_metrics callback."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)

    metrics = compute_sklearn_metrics(labels, preds)
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "weighted_f1": metrics["weighted_f1"],
    }
