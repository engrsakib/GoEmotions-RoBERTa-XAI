"""Exploratory data analysis and figure export."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.label_mapping import ID2LABEL
from src.paths import ARTIFACTS_DIR


def plot_class_distribution(train_df: pd.DataFrame, output_path: Path | None = None) -> Path:
    counts = train_df["encoded_label"].value_counts().sort_index()
    labels = [ID2LABEL[int(i)] for i in counts.index]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(labels, counts.values, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_title("Training Set Class Distribution (GoEmotions 7-Class Mapping)")
    ax.set_xlabel("Emotion Category")
    ax.set_ylabel("Sample Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    output_path = output_path or ARTIFACTS_DIR / "figures" / "class_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def summarize_text_lengths(train_df: pd.DataFrame) -> dict:
    return {
        "avg_char_length": round(float(train_df["char_length"].mean()), 2),
        "avg_token_length_approx": round(float(train_df["token_length_approx"].mean()), 2),
        "max_char_length": int(train_df["char_length"].max()),
        "max_token_length_approx": int(train_df["token_length_approx"].max()),
    }


def run_eda(train_df: pd.DataFrame) -> dict:
    figure_path = plot_class_distribution(train_df)
    length_stats = summarize_text_lengths(train_df)
    print(f"EDA figure saved: {figure_path}")
    print(f"Avg char length: {length_stats['avg_char_length']}")
    print(f"Avg token length (approx): {length_stats['avg_token_length_approx']}")
    return {"figure_path": str(figure_path), **length_stats}
