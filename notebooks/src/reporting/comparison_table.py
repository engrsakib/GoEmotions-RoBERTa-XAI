"""Generate IEEE-ready comparison tables for m4 vs m6 (and related models)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.label_mapping import ID2LABEL, LABEL2ID, NUM_LABELS
from src.data.multi_label_mapping import str_to_multi_hot
from src.data.pipeline import load_config, run_data_pipeline
from src.paths import CHECKPOINTS_DIR, EXPORTS_DIR
from src.training.metrics import compute_multilabel_metrics
from src.training.model_profiles import resolve_model_checkpoint
from src.training.model_registry import apply_model_to_config
from src.training.multilabel_trainer import MultiLabelTrainer, prepare_multilabel_hf_datasets
from src.training.thresholds import predict_with_thresholds, tune_thresholds
from src.training.trainer_setup import (
    _predict_multilabel_probs,
    _predict_probs,
    evaluate_multilabel_with_threshold_tuning,
    load_transformer_tokenizer,
    prepare_hf_datasets,
)

MACRO_PRIORITY = list(range(NUM_LABELS))


MODEL_META: dict[str, dict] = {
    "m4_roberta_focal": {
        "display_name": "RoBERTa-base",
        "loss_latex": r"Focal Loss ($\gamma=1.0$)",
        "loss_md": "Focal Loss (gamma=1.0)",
        "experiment": "E3",
        "threshold_default_latex": "Default (argmax)",
        "threshold_default_md": "Default (argmax)",
        "threshold_optimized_latex": "Optimized (val)",
        "threshold_optimized_md": "Optimized (val)",
    },
    "m6_deberta_v3": {
        "display_name": "DeBERTa-v3-base",
        "loss_latex": "Asymmetric (clipped)",
        "loss_md": "Asymmetric (clipped)",
        "experiment": "E2",
        "threshold_default_latex": r"Default ($\tau=0.5$)",
        "threshold_default_md": "Default (tau=0.5)",
        "threshold_optimized_latex": "Optimized (val)",
        "threshold_optimized_md": "Optimized (val)",
    },
}


@dataclass
class ModelComparisonRow:
    model_name: str
    loss_function: str
    threshold_mode: str
    macro_f1: float
    micro_f1: float
    subset_accuracy: float
    hamming_loss: float
    loss_is_latex: bool = False
    threshold_is_latex: bool = False


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def multihot_matrix_from_df(df: pd.DataFrame) -> np.ndarray:
    if "multi_hot_labels" in df.columns:
        return np.array([str_to_multi_hot(v) for v in df["multi_hot_labels"]], dtype=int)
    if "encoded_label" in df.columns:
        labels = df["encoded_label"].astype(int).tolist()
        return np.eye(NUM_LABELS, dtype=int)[labels]
    raise ValueError("DataFrame must contain multi_hot_labels or encoded_label")


def multihot_to_single_priority(multihot: np.ndarray) -> np.ndarray:
    """Map multi-hot rows to single class via last-active priority (Track B rule)."""
    out = np.zeros(len(multihot), dtype=int)
    for i, row in enumerate(multihot):
        active = [j for j in MACRO_PRIORITY if row[j] == 1]
        out[i] = active[-1] if active else 0
    return out


def singlelabel_preds_to_multilabel_metrics(
    y_true_multihot: np.ndarray,
    pred_labels: np.ndarray,
) -> dict:
    pred_onehot = np.eye(NUM_LABELS, dtype=int)[pred_labels.astype(int)]
    return compute_multilabel_metrics(y_true_multihot, pred_onehot)


def _metrics_to_row(
    model_id: str,
    threshold_key: str,
    metrics: dict,
    *,
    latex: bool = False,
) -> ModelComparisonRow:
    meta = MODEL_META[model_id]
    if threshold_key == "default":
        threshold = meta["threshold_default_latex" if latex else "threshold_default_md"]
    else:
        threshold = meta["threshold_optimized_latex" if latex else "threshold_optimized_md"]
    return ModelComparisonRow(
        model_name=meta["display_name"],
        loss_function=meta["loss_latex" if latex else "loss_md"],
        threshold_mode=threshold,
        macro_f1=float(metrics["macro_f1"]),
        micro_f1=float(metrics["micro_f1"]),
        subset_accuracy=float(metrics["subset_accuracy"]),
        hamming_loss=float(metrics["hamming_loss"]),
        loss_is_latex=latex,
        threshold_is_latex=latex,
    )


def extract_m6_rows(payload: dict, *, latex: bool = False) -> list[ModelComparisonRow]:
    default_m = payload.get("test_metrics_default") or payload.get("test_metrics", {})
    tuned_m = payload.get("test_metrics_thresholded") or default_m
    if not default_m or "macro_f1" not in default_m:
        eval_metrics = payload.get("eval_metrics") or {}
        default_m = {
            "macro_f1": eval_metrics.get("eval_macro_f1", eval_metrics.get("macro_f1", 0)),
            "micro_f1": eval_metrics.get("eval_micro_f1", eval_metrics.get("micro_f1", 0)),
            "subset_accuracy": eval_metrics.get(
                "eval_subset_accuracy", eval_metrics.get("subset_accuracy", 0)
            ),
            "hamming_loss": eval_metrics.get(
                "eval_hamming_loss", eval_metrics.get("hamming_loss", 0)
            ),
        }
        tuned_m = default_m
    return [
        _metrics_to_row("m6_deberta_v3", "default", default_m, latex=latex),
        _metrics_to_row("m6_deberta_v3", "optimized", tuned_m, latex=latex),
    ]


def extract_m4_rows(payload: dict, *, latex: bool = False) -> list[ModelComparisonRow]:
    if payload.get("test_metrics_default") and payload.get("test_metrics_thresholded"):
        return [
            _metrics_to_row("m4_roberta_focal", "default", payload["test_metrics_default"], latex=latex),
            _metrics_to_row(
                "m4_roberta_focal", "optimized", payload["test_metrics_thresholded"], latex=latex
            ),
        ]

    eval_result = payload.get("eval_result") or payload
    argmax_m = eval_result.get("test_metrics_argmax") or eval_result.get("test_metrics") or {}
    tuned_m = eval_result.get("test_metrics_thresholded") or argmax_m

    if "micro_f1" in argmax_m:
        return [
            _metrics_to_row("m4_roberta_focal", "default", argmax_m, latex=latex),
            _metrics_to_row("m4_roberta_focal", "optimized", tuned_m, latex=latex),
        ]

    raise ValueError(
        "m4 experiment JSON lacks multi-label harmonized metrics; re-run with --evaluate"
    )


def load_experiment_metrics(exports_dir: Path, experiment_id: str) -> dict | None:
    exports_dir = Path(exports_dir)
    candidates = [
        exports_dir / f"experiment_{experiment_id}_thresholds.json",
        exports_dir / f"experiment_{experiment_id}.json",
    ]
    if experiment_id == "E3":
        candidates.append(exports_dir / "training_metrics.json")
    for path in candidates:
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                return json.load(handle)
    return None


def _resolve_m4_checkpoint() -> str | None:
    for candidate in (
        EXPORTS_DIR / "saved_emotion_model",
        CHECKPOINTS_DIR / "m4_roberta_focal",
        CHECKPOINTS_DIR / "saved_emotion_model",
    ):
        if candidate.is_dir() and any(candidate.iterdir()):
            weight_names = ("model.safetensors", "pytorch_model.bin")
            if any((candidate / name).is_file() for name in weight_names):
                return str(candidate)
            nested = resolve_model_checkpoint("m4_roberta_focal")
            if nested:
                return nested
    return resolve_model_checkpoint("m4_roberta_focal")


def evaluate_m6_rows(config: dict | None = None) -> list[ModelComparisonRow]:
    config = apply_model_to_config(load_config() if config is None else dict(config), "m6_deberta_v3")
    config["track"] = "multilabel"

    checkpoint = resolve_model_checkpoint("m6_deberta_v3")
    if not checkpoint:
        raise FileNotFoundError("No m6_deberta_v3 checkpoint found. Run E2 first.")

    result = run_data_pipeline(config)
    train_df, val_df, test_df = result["train_df"], result["val_df"], result["test_df"]
    tokenizer = load_transformer_tokenizer(config["model_name"])
    _, val_ds, test_ds, _ = prepare_multilabel_hf_datasets(
        train_df, val_df, test_df, tokenizer, max_length=config["max_length"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        problem_type="multi_label_classification",
    )
    trainer = MultiLabelTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(CHECKPOINTS_DIR / "m6_deberta_v3_eval"),
            report_to=[],
            per_device_eval_batch_size=config.get("eval_batch_size", 16),
        ),
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    payload = evaluate_multilabel_with_threshold_tuning(trainer, val_ds, test_ds, config=config)
    return extract_m6_rows(payload)


def evaluate_m4_rows(config: dict | None = None) -> list[ModelComparisonRow]:
    """Evaluate m4 single-label model on unified multi-label test (one-hot preds)."""
    config = apply_model_to_config(load_config() if config is None else dict(config), "m4_roberta_focal")
    ml_config = dict(config)
    ml_config["track"] = "multilabel"

    checkpoint = _resolve_m4_checkpoint()
    if not checkpoint:
        raise FileNotFoundError("No m4_roberta_focal checkpoint found. Run E3 or pipeline train first.")

    result = run_data_pipeline(ml_config)
    train_df, val_df, test_df = result["train_df"], result["val_df"], result["test_df"]
    val_multihot = multihot_matrix_from_df(val_df)
    test_multihot = multihot_matrix_from_df(test_df)

    tokenizer = load_transformer_tokenizer(config["model_name"])
    val_labels_int = multihot_to_single_priority(val_multihot)
    val_df_sl = val_df.copy()
    val_df_sl["encoded_label"] = val_labels_int
    test_df_sl = test_df.copy()
    test_df_sl["encoded_label"] = multihot_to_single_priority(test_multihot)

    train_df_sl = train_df.copy()
    if "encoded_label" not in train_df_sl.columns:
        train_df_sl["encoded_label"] = multihot_to_single_priority(multihot_matrix_from_df(train_df))

    train_ds, val_ds, test_ds, _ = prepare_hf_datasets(
        train_df_sl, val_df_sl, test_df_sl, tokenizer, max_length=config["max_length"]
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(CHECKPOINTS_DIR / "m4_roberta_focal_eval"),
            report_to=[],
            per_device_eval_batch_size=config.get("eval_batch_size", 16),
        ),
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    val_probs, _ = _predict_probs(trainer, val_ds)
    test_probs, _ = _predict_probs(trainer, test_ds)

    test_argmax = np.argmax(test_probs, axis=-1)
    default_metrics = singlelabel_preds_to_multilabel_metrics(test_multihot, test_argmax)

    thresholds, _ = tune_thresholds(
        val_probs,
        val_labels_int,
        num_classes=NUM_LABELS,
        step=config.get("threshold_search_step", 0.05),
        min_precision=config.get("threshold_min_precision"),
    )
    test_threshold = predict_with_thresholds(test_probs, thresholds)
    tuned_metrics = singlelabel_preds_to_multilabel_metrics(test_multihot, test_threshold)

    payload = {
        "test_metrics_default": default_metrics,
        "test_metrics_thresholded": tuned_metrics,
    }
    return extract_m4_rows(payload)


def _load_m4_rows(m4_payload: dict | None, evaluate: bool) -> list[ModelComparisonRow]:
    if evaluate or m4_payload is None:
        return evaluate_m4_rows()
    try:
        return extract_m4_rows(m4_payload)
    except ValueError:
        return evaluate_m4_rows()


def _load_m6_rows(m6_payload: dict | None, evaluate: bool) -> list[ModelComparisonRow]:
    if evaluate or m6_payload is None:
        return evaluate_m6_rows()
    return extract_m6_rows(m6_payload)


def build_comparison_rows(
    m4_payload: dict | None = None,
    m6_payload: dict | None = None,
    *,
    evaluate: bool = False,
) -> list[ModelComparisonRow]:
    m4_rows = _load_m4_rows(m4_payload, evaluate)
    m6_rows = _load_m6_rows(m6_payload, evaluate)
    return m4_rows + m6_rows


def format_markdown(rows: list[ModelComparisonRow]) -> str:
    header = (
        "| Model | Loss Function | Threshold Tuning | Macro-F1 | Micro-F1 | "
        "Subset Exact Match | Hamming Loss |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: |\n"
    )
    lines = [header]
    for row in rows:
        loss = row.loss_function
        threshold = row.threshold_mode
        lines.append(
            f"| {row.model_name} | {loss} | {threshold} | "
            f"{_fmt(row.macro_f1)} | {_fmt(row.micro_f1)} | "
            f"{_fmt(row.subset_accuracy)} | {_fmt(row.hamming_loss)} |\n"
        )
    return "".join(lines)


def format_latex(
    rows: list[ModelComparisonRow],
    caption: str | None = None,
    label: str = "tab:roberta_deberta_comparison",
) -> str:
    caption = caption or (
        "Comparison of RoBERTa-Focal and DeBERTa-v3 on GoEmotions seven-macro "
        "multi-label classification (test set). RoBERTa predictions are one-hot "
        "encodings derived from single-label softmax outputs."
    )
    lines = [
        "% Requires: \\usepackage{booktabs}\n",
        "\\begin{table}[t]\n",
        "\\centering\n",
        f"\\caption{{{caption}}}\n",
        f"\\label{{{label}}}\n",
        "\\begin{tabular}{lllcccc}\n",
        "\\toprule\n",
        "Model & Loss & Threshold & Macro-F1 & Micro-F1 & Subset Acc. & Hamming \\\\\n",
        "\\midrule\n",
    ]
    meta = MODEL_META
    for row in rows:
        model_id = "m4_roberta_focal" if row.model_name == meta["m4_roberta_focal"]["display_name"] else "m6_deberta_v3"
        loss = meta[model_id]["loss_latex"]
        if "Default" in row.threshold_mode or "argmax" in row.threshold_mode or "0.5" in row.threshold_mode:
            threshold = meta[model_id]["threshold_default_latex"]
        else:
            threshold = meta[model_id]["threshold_optimized_latex"]
        lines.append(
            f"{row.model_name} & {loss} & {threshold} & "
            f"{_fmt(row.macro_f1)} & {_fmt(row.micro_f1)} & "
            f"{_fmt(row.subset_accuracy)} & {_fmt(row.hamming_loss)} \\\\\n"
        )
    lines.extend(["\\bottomrule\n", "\\end{tabular}\n", "\\end{table}\n"])
    return "".join(lines)


def write_comparison_tables(
    rows: list[ModelComparisonRow],
    output_dir: Path | None = None,
    stem: str = "model_comparison_m4_m6",
) -> tuple[Path, Path]:
    output_dir = Path(output_dir or EXPORTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"{stem}.md"
    tex_path = output_dir / f"{stem}.tex"
    md_path.write_text(format_markdown(rows), encoding="utf-8")
    tex_path.write_text(format_latex(rows), encoding="utf-8")
    return tex_path, md_path
