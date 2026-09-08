"""Stratified train/val/test splitting, official splits, and leakage checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.clean import dedup_split_internal


def stratified_split(
    df: pd.DataFrame,
    label_column: str = "encoded_label",
    random_seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    dedup_policy: str = "consensus",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    temp_size = 1.0 - train_ratio
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=random_seed,
        stratify=df[label_column],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=random_seed,
        stratify=temp_df[label_column],
    )

    split_internal_dropped = {"train": 0, "val": 0, "test": 0}
    if dedup_policy == "split_internal":
        train_df, split_internal_dropped["train"] = dedup_split_internal(train_df)
        val_df, split_internal_dropped["val"] = dedup_split_internal(val_df)
        test_df, split_internal_dropped["test"] = dedup_split_internal(test_df)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_log = {
        "split_mode": "stratified",
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "split_internal_dropped": split_internal_dropped,
        "train_distribution": (
            train_df[label_column].value_counts(normalize=True).sort_index() * 100
        ).round(4).to_dict(),
        "val_distribution": (
            val_df[label_column].value_counts(normalize=True).sort_index() * 100
        ).round(4).to_dict(),
        "test_distribution": (
            test_df[label_column].value_counts(normalize=True).sort_index() * 100
        ).round(4).to_dict(),
    }
    return train_df, val_df, test_df, split_log


def official_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_column: str = "encoded_label",
    dedup_policy: str = "consensus",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    split_internal_dropped = {"train": 0, "val": 0, "test": 0}
    if dedup_policy == "split_internal":
        train_df, split_internal_dropped["train"] = dedup_split_internal(train_df)
        val_df, split_internal_dropped["val"] = dedup_split_internal(val_df)
        test_df, split_internal_dropped["test"] = dedup_split_internal(test_df)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    split_log = {
        "split_mode": "official",
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "split_internal_dropped": split_internal_dropped,
        "train_distribution": (
            train_df[label_column].value_counts(normalize=True).sort_index() * 100
        ).round(4).to_dict(),
        "val_distribution": (
            val_df[label_column].value_counts(normalize=True).sort_index() * 100
        ).round(4).to_dict(),
        "test_distribution": (
            test_df[label_column].value_counts(normalize=True).sort_index() * 100
        ).round(4).to_dict(),
    }
    return train_df, val_df, test_df, split_log


def check_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    train_texts = set(train_df["text"].str.lower())
    val_texts = set(val_df["text"].str.lower())
    test_texts = set(test_df["text"].str.lower())

    train_val_overlap = len(train_texts & val_texts)
    train_test_overlap = len(train_texts & test_texts)
    val_test_overlap = len(val_texts & test_texts)

    return {
        "train_val_overlap": train_val_overlap,
        "train_test_overlap": train_test_overlap,
        "val_test_overlap": val_test_overlap,
        "no_leakage": train_val_overlap == 0 and train_test_overlap == 0 and val_test_overlap == 0,
    }


def check_class_balance(split_log: dict, tolerance: float = 0.5) -> dict:
    distributions = [
        split_log["train_distribution"],
        split_log["val_distribution"],
        split_log["test_distribution"],
    ]
    all_labels = sorted({k for d in distributions for k in d})
    max_spread = {}
    for label in all_labels:
        values = [d.get(label, 0.0) for d in distributions]
        max_spread[str(label)] = round(max(values) - min(values), 4)

    balanced = all(spread <= tolerance for spread in max_spread.values())
    return {"max_spread_pct": max_spread, "within_tolerance": balanced, "tolerance_pct": tolerance}


def discover_official_tsv_dir(search_dirs: list[Path]) -> Path | None:
    for base in search_dirs:
        if base is None or not base.is_dir():
            continue
        if all((base / name).is_file() for name in ("train.tsv", "dev.tsv", "test.tsv")):
            return base
        for sub in base.rglob("*"):
            if sub.is_dir() and all((sub / name).is_file() for name in ("train.tsv", "dev.tsv", "test.tsv")):
                return sub
    return None
