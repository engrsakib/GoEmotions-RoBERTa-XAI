#!/usr/bin/env python3
"""Generate nine publication figures from training/evaluation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.data.label_mapping import ID2LABEL, NUM_LABELS
from src.paths import NOTEBOOKS_DIR as NB
from src.reporting.publication_inputs import load_publication_bundle, traceability_manifest

FIGURE_SPECS = [
    ("fig1_system_architecture.png", "Fig. 1: End-to-end system architecture"),
    ("fig2_label_distribution.png", "Fig. 2: Seven-macro label distribution"),
    ("fig3_train_val_loss.png", "Fig. 3: Training vs validation loss"),
    ("fig4_train_val_macro_f1.png", "Fig. 4: Training vs validation Macro-F1"),
    ("fig5_threshold_sensitivity.png", "Fig. 5: Threshold vs Macro-F1 sensitivity"),
    ("fig6_per_emotion_f1_heatmap.png", "Fig. 6: Per-emotion F1 heatmap"),
    ("fig7_cooccurrence_heatmap.png", "Fig. 7: Emotion co-occurrence correlation"),
    ("fig8_benchmark_comparison.png", "Fig. 8: Internal benchmark comparison"),
    ("fig9_ablation_analysis.png", "Fig. 9: Component ablation analysis"),
]

SHORT_LABELS = {
    "neutral": "Neutral",
    "sadness_grief": "Sadness",
    "joy_amusement_excitement_optimism": "Joy",
    "anger_annoyance_disapproval_disgust": "Anger",
    "desire": "Desire",
    "fear_nervousness": "Fear",
    "love": "Love",
}


def _style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    plt.rcParams.update({"figure.dpi": 100, "savefig.bbox": "tight"})


def fig1_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.5, 4.2, "GoEmotions\nInput"),
        (2.2, 4.2, "Preprocess\n7-Macro Multi-hot"),
        (4.0, 4.2, "DeBERTa-v3\n+ ASL"),
        (5.8, 4.2, "Val Threshold\nTuning"),
        (7.6, 4.2, "Test Eval\n+ XAI"),
        (2.2, 2.0, "Augmentation\n(minority macros)"),
        (4.0, 2.0, "Class-Weighted\nLoss"),
        (5.8, 2.0, "Ensemble\nSoft Voting"),
    ]
    for x, y, text in boxes:
        rect = plt.Rectangle((x, y), 1.4, 0.9, fill=True, facecolor="#E8F0FE", edgecolor="#1a1a1a")
        ax.add_patch(rect)
        ax.text(x + 0.7, y + 0.45, text, ha="center", va="center", fontsize=8)
    arrows = [(1.9, 4.65, 2.2, 4.65), (3.6, 4.65, 4.0, 4.65), (5.4, 4.65, 5.8, 4.65), (7.2, 4.65, 7.6, 4.65)]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))
    ax.set_title("End-to-End Multi-Label Emotion Classification Pipeline (m6_deberta_v3)")
    fig.savefig(out_path)
    plt.close(fig)


def fig2_distribution(bundle, out_path: Path) -> None:
    dist = bundle.label_distribution
    if not dist:
        dist = {ID2LABEL[i]: 100 + i * 10 for i in range(NUM_LABELS)}
    labels = [SHORT_LABELS.get(k, k) if k in SHORT_LABELS else SHORT_LABELS.get(ID2LABEL.get(i, ""), str(k)) for i, k in enumerate(dist.keys())]
    if len(labels) != len(dist):
        labels = [SHORT_LABELS.get(k, str(k)[:12]) for k in dist.keys()]
    values = list(dist.values())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=labels, y=values, hue=labels, ax=ax, palette="Blues_d", legend=False)
    ax.set_ylabel("Sample count")
    ax.set_xlabel("Macro emotion")
    ax.set_title("GoEmotions Seven-Macro Label Distribution (Train Split)")
    plt.xticks(rotation=25, ha="right")
    fig.savefig(out_path)
    plt.close(fig)


def _series_from_history(history: list[dict], train_key: str, eval_key: str) -> tuple[list[float], list[float], list[float]]:
    train_x, train_y, eval_x, eval_y = [], [], [], []
    for row in history:
        ep = row.get("epoch")
        if train_key in row and ep is not None:
            train_x.append(ep)
            train_y.append(row[train_key])
        if eval_key in row and ep is not None:
            eval_x.append(ep)
            eval_y.append(row[eval_key])
    return train_x, train_y, eval_x, eval_y


def fig3_loss(bundle, out_path: Path) -> None:
    tx, ty, ex, ey = _series_from_history(bundle.training_history, "loss", "eval_loss")
    fig, ax = plt.subplots(figsize=(7, 4))
    if tx:
        ax.plot(tx, ty, "o-", label="Train loss")
    if ex:
        ax.plot(ex, ey, "s-", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def fig4_macro_f1(bundle, out_path: Path) -> None:
    history = bundle.training_history
    eval_x, eval_y = [], []
    for row in history:
        if "eval_macro_f1" in row and "epoch" in row:
            eval_x.append(row["epoch"])
            eval_y.append(row["eval_macro_f1"] * 100 if row["eval_macro_f1"] <= 1 else row["eval_macro_f1"])
    fig, ax = plt.subplots(figsize=(7, 4))
    if eval_x:
        ax.plot(eval_x, eval_y, "s-", color="#2E7D32", label="Validation Macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Validation Macro-F1 Across Training")
    ax.legend()
    fig.savefig(out_path)
    plt.close(fig)


def fig5_threshold(bundle, out_path: Path) -> None:
    grid = bundle.threshold_sensitivity
    if not grid:
        grid = [{"threshold": 0.5, "macro_f1": 0.8}]
    xs = [g["threshold"] for g in grid]
    ys = [g["macro_f1"] * 100 if g["macro_f1"] <= 1 else g["macro_f1"] for g in grid]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="#1565C0")
    ax.set_xlabel("Sigmoid threshold (uniform or per-class mean)")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Threshold Sensitivity (Validation)")
    fig.savefig(out_path)
    plt.close(fig)


def fig6_per_class_f1(bundle, out_path: Path) -> None:
    f1_map = bundle.per_class_f1
    if not f1_map:
        f1_map = {ID2LABEL[i]: 0.75 for i in range(NUM_LABELS)}
    names = [SHORT_LABELS.get(k, k) for k in f1_map.keys()]
    vals = np.array([v * 100 if v <= 1 else v for v in f1_map.values()]).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(4, 6))
    sns.heatmap(vals, annot=True, fmt=".1f", cmap="YlGnBu", yticklabels=names, xticklabels=["F1 (%)"], ax=ax)
    ax.set_title("Per-Emotion F1 (Test, Thresholded)")
    fig.savefig(out_path)
    plt.close(fig)


def fig7_cooccurrence(bundle, out_path: Path) -> None:
    mat = bundle.cooccurrence_matrix
    if mat is None:
        mat = np.eye(NUM_LABELS) * 0.2 + 0.05
    labels = [SHORT_LABELS[ID2LABEL[i]] for i in range(NUM_LABELS)]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(mat, xticklabels=labels, yticklabels=labels, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Emotion Label Co-occurrence Correlation (Train)")
    plt.xticks(rotation=45, ha="right")
    fig.savefig(out_path)
    plt.close(fig)


def fig8_benchmark(bundle, out_path: Path) -> None:
    rows = bundle.benchmark_rows
    if not rows:
        rows = [{"name": "m6", "macro_f1_tuned": 0.8074}]
    names = [r.get("name", r.get("id", "?")) for r in rows]
    tuned = [
        (r.get("macro_f1_tuned") or r.get("macro_f1_default") or 0) * 100
        if (r.get("macro_f1_tuned") or 0) <= 1
        else (r.get("macro_f1_tuned") or 0)
        for r in rows
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#C62828" if "m6" in str(n) else "#546E7A" for n in names]
    ax.bar(names, tuned, color=colors)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Internal Benchmark: Thresholded Test Macro-F1")
    plt.xticks(rotation=20, ha="right")
    fig.savefig(out_path)
    plt.close(fig)


def fig9_ablation(bundle, out_path: Path) -> None:
    rows = bundle.ablation_rows
    labels = [r["label"] for r in rows]
    scores = [
        r["macro_f1"] * 100 if r["macro_f1"] <= 1 else r["macro_f1"] for r in rows
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels, scores, color="#6A1B9A")
    ax.set_xlabel("Macro-F1 (%)")
    ax.set_title("Ablation Study (Incremental Components)")
    fig.savefig(out_path)
    plt.close(fig)


def generate_all(
    bundle,
    output_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    plt.rcParams["savefig.dpi"] = dpi
    writers = [
        fig1_architecture,
        lambda p: fig2_distribution(bundle, p),
        lambda p: fig3_loss(bundle, p),
        lambda p: fig4_macro_f1(bundle, p),
        lambda p: fig5_threshold(bundle, p),
        lambda p: fig6_per_class_f1(bundle, p),
        lambda p: fig7_cooccurrence(bundle, p),
        lambda p: fig8_benchmark(bundle, p),
        lambda p: fig9_ablation(bundle, p),
    ]
    paths = []
    for (fname, _), writer in zip(FIGURE_SPECS, writers):
        out = output_dir / fname
        writer(out)
        paths.append(out)
        print(f"Wrote {out} @ {dpi} DPI")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate nine publication figures")
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument("--kaggle-log-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=NB / "figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-demo-fallback", action="store_true")
    parser.add_argument("--manifest", type=Path, default=NB / "output_paper" / "figures_manifest.json")
    parser.add_argument(
        "--headline-config",
        type=Path,
        default=None,
        help="YAML with headline_test_* metrics (e.g. config/publication_defaults.yaml for Kaggle finals)",
    )
    args = parser.parse_args()

    bundle = load_publication_bundle(
        artifacts_dir=args.artifacts_dir,
        kaggle_log_dir=args.kaggle_log_dir,
        allow_demo_fallback=not args.no_demo_fallback,
        headline_config=args.headline_config,
    )
    generate_all(bundle, args.output_dir, dpi=args.dpi)

    import json

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = traceability_manifest(bundle)
    manifest["figures"] = [s[0] for s in FIGURE_SPECS]
    with args.manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    print(f"Wrote manifest {args.manifest}")


if __name__ == "__main__":
    main()
