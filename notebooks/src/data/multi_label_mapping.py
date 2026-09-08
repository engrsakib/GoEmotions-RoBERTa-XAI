"""Seven-macro multi-hot label construction (Track A — no priority last-wins)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.label_mapping import (
    EMOTION_TO_TARGET,
    GOEMOTIONS_ALL_EMOTION_COLUMNS,
    ID2LABEL,
    NUM_LABELS,
    SCHEMA_VERSION,
    TARGET_ID_TO_EMOTIONS,
)


def macro_active_targets(row: pd.Series) -> list[int]:
    """Return all macro IDs with at least one active fine-grained emotion."""
    active: list[int] = []
    for target_id, emotions in TARGET_ID_TO_EMOTIONS.items():
        if any(row.get(emotion, 0) == 1 for emotion in emotions if emotion in row.index):
            active.append(target_id)
    return active


def assign_multi_hot_vector(row: pd.Series) -> list[int]:
    """Build 7-dim multi-hot vector (1 if any emotion in macro group is active)."""
    vector = [0] * NUM_LABELS
    for target_id in macro_active_targets(row):
        vector[target_id] = 1
    return vector


def assign_single_from_multi_hot(row: pd.Series) -> float:
    """
    Single-label for Track B: use multi-hot; if exactly one macro active, use it.
    If multiple macros active, return NaN (clean subset) unless encoded_label already set.
    """
    active = macro_active_targets(row)
    if len(active) == 1:
        return float(active[0])
    if len(active) == 0 and row.get("neutral", 0) == 1:
        others = [c for c in GOEMOTIONS_ALL_EMOTION_COLUMNS if c != "neutral" and c in row.index]
        if all(row.get(c, 0) == 0 for c in others):
            return 0.0
    if row.get("example_very_unclear", False):
        return np.nan
    if len(active) > 1:
        return np.nan
    return np.nan


def multi_hot_to_str(vector: list[int]) -> str:
    return json.dumps(vector)


def str_to_multi_hot(value: str) -> list[int]:
    return json.loads(value)


def apply_multi_label_mapping(df: pd.DataFrame, track: str = "multilabel") -> tuple[pd.DataFrame, dict]:
    from src.data.label_mapping import assign_single_label

    working = df.copy()
    initial_rows = len(working)

    vectors = working.apply(assign_multi_hot_vector, axis=1)
    working["multi_hot_labels"] = vectors.apply(multi_hot_to_str)
    working["n_active_macros"] = vectors.apply(sum)

    # Primary label for stratified split (legacy priority rule)
    working["encoded_label"] = working.apply(assign_single_label, axis=1)

    # Keep rows with at least one active macro in multi-hot
    has_macro = working["n_active_macros"] > 0
    working = working[has_macro | working["encoded_label"].notna()]
    working = working.dropna(subset=["encoded_label"])
    working = working[working["text"].astype(str).str.strip() != ""]
    working["encoded_label"] = working["encoded_label"].astype(int)

    if track == "singlelabel":
        working = filter_clean_single_label(working)

    working = working.reset_index(drop=True)

    multi_label_rows = int((working["n_active_macros"] > 1).sum())
    class_counts = working["encoded_label"].value_counts().sort_index().to_dict()

    log = {
        "schema_version": SCHEMA_VERSION,
        "track": track,
        "initial_rows": initial_rows,
        "remaining_rows": len(working),
        "dropped_rows": initial_rows - len(working),
        "multi_label_rows_kept": multi_label_rows,
        "single_macro_rows": int((working["n_active_macros"] == 1).sum()),
        "class_counts": class_counts,
        "emotion_to_target_size": len(EMOTION_TO_TARGET),
    }
    return working, log


def filter_clean_single_label(df: pd.DataFrame) -> pd.DataFrame:
    """Track B: keep only rows with exactly one active macro group."""
    if "n_active_macros" in df.columns:
        return df[df["n_active_macros"] == 1].reset_index(drop=True)
    return df.reset_index(drop=True)
