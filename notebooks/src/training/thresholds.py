"""Per-class probability threshold tuning on validation set."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=-1, keepdims=True)


def predict_with_thresholds(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Assign class with highest prob above its threshold; fallback to argmax.
    """
    adjusted = probs - thresholds.reshape(1, -1)
    preds = np.argmax(adjusted, axis=-1)
    no_winner = (probs[np.arange(len(probs)), preds] < thresholds[preds])
    if no_winner.any():
        preds[no_winner] = np.argmax(probs[no_winner], axis=-1)
    return preds


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
            labels=list(range(num_classes)),
        )
    )


def tune_thresholds(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int = 7,
    step: float = 0.05,
    min_precision: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Coordinate-ascent grid search per class to maximize validation macro-F1.
    Optionally enforce per-class minimum precision (e.g. desire >= 0.45).
    """
    from sklearn.metrics import precision_score

    default_threshold = 1.0 / num_classes
    thresholds = np.full(num_classes, default_threshold, dtype=np.float64)
    grid = np.arange(step, 1.0, step)

    def score_with_constraints(trial_thresholds: np.ndarray) -> float:
        preds = predict_with_thresholds(val_probs, trial_thresholds)
        if min_precision is not None:
            precisions = precision_score(
                val_labels, preds, average=None, zero_division=0, labels=list(range(num_classes))
            )
            if any(precisions[i] < min_precision for i in range(num_classes) if (val_labels == i).sum() > 0):
                return -1.0
        return _macro_f1(val_labels, preds, num_classes)

    best_f1 = score_with_constraints(thresholds)
    if best_f1 < 0:
        best_f1 = _macro_f1(val_labels, predict_with_thresholds(val_probs, thresholds), num_classes)

    improved = True
    while improved:
        improved = False
        for class_id in range(num_classes):
            best_class_threshold = thresholds[class_id]
            best_class_f1 = best_f1
            for candidate in grid:
                trial = thresholds.copy()
                trial[class_id] = candidate
                score = score_with_constraints(trial)
                if score > best_class_f1:
                    best_class_f1 = score
                    best_class_threshold = candidate
            if best_class_threshold != thresholds[class_id]:
                thresholds[class_id] = best_class_threshold
                best_f1 = best_class_f1
                improved = True

    argmax_f1 = _macro_f1(val_labels, np.argmax(val_probs, axis=-1), num_classes)
    return thresholds, {
        "val_macro_f1_argmax": argmax_f1,
        "val_macro_f1_thresholded": best_f1,
        "improvement": round(best_f1 - argmax_f1, 4),
        "thresholds": thresholds.tolist(),
        "min_precision_constraint": min_precision,
    }


def predict_multilabel(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Multi-hot predictions: pred[i, c] = 1 if probs[i, c] >= thresholds[c]."""
    return (probs >= thresholds.reshape(1, -1)).astype(int)


def _multilabel_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def tune_multilabel_thresholds(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    num_classes: int = 7,
    step: float = 0.05,
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    min_precision: float | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Coordinate-ascent grid search per class (0.1–0.9) to maximize validation macro-F1.
    Uses independent sigmoid thresholds per label (multi-hot semantics).
    """
    from sklearn.metrics import precision_score

    default_threshold = 0.5
    thresholds = np.full(num_classes, default_threshold, dtype=np.float64)
    grid = np.arange(threshold_min, threshold_max + step / 2, step)

    def score_with_constraints(trial_thresholds: np.ndarray) -> float:
        preds = predict_multilabel(val_probs, trial_thresholds)
        if min_precision is not None:
            precisions = precision_score(
                val_labels, preds, average=None, zero_division=0
            )
            # Apply precision floor only to minority macro labels (desire, fear)
            minority_classes = (4, 5)
            if any(precisions[i] < min_precision for i in minority_classes):
                return -1.0
        return _multilabel_macro_f1(val_labels, preds)

    default_preds = predict_multilabel(val_probs, np.full(num_classes, default_threshold))
    default_f1 = _multilabel_macro_f1(val_labels, default_preds)

    best_f1 = score_with_constraints(thresholds)
    if best_f1 < 0:
        best_f1 = default_f1

    improved = True
    while improved:
        improved = False
        for class_id in range(num_classes):
            best_class_threshold = thresholds[class_id]
            best_class_f1 = best_f1
            for candidate in grid:
                trial = thresholds.copy()
                trial[class_id] = candidate
                score = score_with_constraints(trial)
                if score > best_class_f1:
                    best_class_f1 = score
                    best_class_threshold = candidate
            if best_class_threshold != thresholds[class_id]:
                thresholds[class_id] = best_class_threshold
                best_f1 = best_class_f1
                improved = True

    return thresholds, {
        "val_macro_f1_default_0.5": round(default_f1, 4),
        "val_macro_f1_thresholded": round(best_f1, 4),
        "improvement": round(best_f1 - default_f1, 4),
        "thresholds": thresholds.tolist(),
        "min_precision_constraint": min_precision,
        "threshold_search_range": [threshold_min, threshold_max],
        "threshold_search_step": step,
    }


def save_thresholds(thresholds: np.ndarray, path: Path, metadata: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": thresholds.tolist(),
        **(metadata or {}),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_thresholds(path: Path) -> np.ndarray:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return np.array(payload["thresholds"], dtype=np.float64)
