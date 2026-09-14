"""Weighted soft-voting ensemble for multilabel emotion classification."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.data.label_mapping import ID2LABEL, NUM_LABELS
from src.eval.ensemble_probs import load_model_probs
from src.paths import EXPORTS_DIR
from src.training.metrics import compute_multilabel_metrics, multilabel_metrics_from_probs
from src.training.model_profiles import get_ensemble_config
from src.training.thresholds import predict_multilabel, tune_multilabel_thresholds


def weighted_soft_vote(prob_list: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Fuse sigmoid probabilities with normalized weights."""
    if not prob_list:
        raise ValueError("prob_list is empty")
    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()
    stacked = np.stack(prob_list, axis=0)
    return np.tensordot(w, stacked, axes=(0, 0))


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    m = compute_multilabel_metrics(y_true, y_pred)
    return {
        "macro_f1": m["macro_f1"],
        "macro_precision": m["macro_precision"],
        "macro_recall": m["macro_recall"],
        "micro_f1": m["micro_f1"],
        "subset_accuracy": m["subset_accuracy"],
        "hamming_loss": m["hamming_loss"],
    }


def evaluate_single_model_on_split(
    model_id: str,
    split: str,
    exports_dir: Path | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    bundle = load_model_probs(model_id, split, exports_dir)
    thresholds = np.full(NUM_LABELS, threshold, dtype=np.float64)
    preds = predict_multilabel(bundle["probs"], thresholds)
    metrics = _metrics_dict(bundle["labels"], preds)
    return {"model_id": model_id, "split": split, "metrics": metrics}


def ensemble_predict(
    model_ids: list[str],
    weights: dict[str, float],
    exports_dir: Path | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Fuse val/test probabilities, optionally tune thresholds on fused val probs.
    """
    config = config or get_ensemble_config()
    exports_dir = exports_dir or EXPORTS_DIR / "ensemble_probs"

    val_probs_list = []
    test_probs_list = []
    weight_list = []
    labels_val = None
    labels_test = None

    for mid in model_ids:
        w = float(weights.get(mid, 1.0))
        val_b = load_model_probs(mid, "val", exports_dir)
        test_b = load_model_probs(mid, "test", exports_dir)
        val_probs_list.append(val_b["probs"])
        test_probs_list.append(test_b["probs"])
        weight_list.append(w)
        labels_val = val_b["labels"]
        labels_test = test_b["labels"]

    val_fused = weighted_soft_vote(val_probs_list, weight_list)
    test_fused = weighted_soft_vote(test_probs_list, weight_list)

    default_t = float(config.get("default_threshold", 0.5))
    thresholds = np.full(NUM_LABELS, default_t, dtype=np.float64)
    threshold_log: dict = {}

    if config.get("tune_thresholds_on_val", True):
        thresholds, threshold_log = tune_multilabel_thresholds(
            val_fused,
            labels_val,
            num_classes=NUM_LABELS,
            step=0.05,
            threshold_min=0.1,
            threshold_max=0.9,
        )

    val_preds = predict_multilabel(val_fused, thresholds)
    test_preds = predict_multilabel(test_fused, thresholds)

    per_model_test: dict[str, dict] = {}
    for mid in model_ids:
        per_model_test[mid] = evaluate_single_model_on_split(
            mid, "test", exports_dir, threshold=default_t
        )["metrics"]

    return {
        "model_ids": model_ids,
        "weights": {mid: float(weights.get(mid, 1.0)) for mid in model_ids},
        "thresholds": thresholds.tolist(),
        "threshold_log": threshold_log,
        "per_class_thresholds": {ID2LABEL[i]: float(thresholds[i]) for i in range(NUM_LABELS)},
        "val_metrics_fused": _metrics_dict(labels_val, val_preds),
        "test_metrics_fused": _metrics_dict(labels_test, test_preds),
        "test_metrics_default_0.5": multilabel_metrics_from_probs(
            test_fused, labels_test, np.full(NUM_LABELS, 0.5)
        ),
        "per_model_test_macro_f1": {
            mid: per_model_test[mid]["macro_f1"] for mid in model_ids
        },
        "per_model_test": per_model_test,
    }


def format_ensemble_report(result: dict[str, Any]) -> str:
    t = result["test_metrics_fused"]
    lines = [
        "=== Ensemble (weighted soft-voting) ===",
        f"Models: {', '.join(result['model_ids'])}",
        f"Weights: {result['weights']}",
        f"Test Macro-F1:    {t['macro_f1']:.4f}",
        f"Test Micro-F1:    {t['micro_f1']:.4f}",
        f"Subset Accuracy:  {t['subset_accuracy']:.4f}",
        f"Hamming Loss:     {t['hamming_loss']:.4f}",
    ]
    for mid, f1 in result.get("per_model_test_macro_f1", {}).items():
        lines.append(f"  {mid} solo macro-F1: {f1:.4f}")
    return "\n".join(lines)


def run_ensemble_evaluation(
    model_ids: list[str] | None = None,
    weights: dict[str, float] | None = None,
    exports_dir: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    ens_cfg = get_ensemble_config()
    model_ids = model_ids or list(ens_cfg.get("model_ids") or [])
    if not model_ids:
        raise ValueError("No ensemble model_ids configured")

    weight_map = dict(ens_cfg.get("weights") or {})
    if weights:
        weight_map.update(weights)

    result = ensemble_predict(model_ids, weight_map, exports_dir=exports_dir, config=ens_cfg)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    report = format_ensemble_report(result)
    print(report)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"Wrote {output_path}")

    return result


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Weighted soft-voting ensemble evaluation")
    parser.add_argument(
        "--model-ids",
        nargs="*",
        default=None,
        help="Override ensemble model IDs",
    )
    parser.add_argument(
        "--probs-dir",
        type=Path,
        default=None,
        help="Directory with {model_id}_{val|test}.npz files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPORTS_DIR / "ensemble_eval.json",
    )
    args = parser.parse_args(argv)
    run_ensemble_evaluation(
        model_ids=args.model_ids,
        exports_dir=args.probs_dir,
        output_path=args.output,
    )
