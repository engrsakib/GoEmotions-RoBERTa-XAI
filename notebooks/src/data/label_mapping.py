"""Single source of truth for 7-class label mapping (aligned with production)."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID2LABEL = {
    0: "neutral",
    1: "sadness_grief",
    2: "joy_amusement_excitement_optimism",
    3: "anger_annoyance_disapproval_disgust",
    4: "desire",
    5: "fear_nervousness",
    6: "love",
}

LABEL2ID = {label: idx for idx, label in ID2LABEL.items()}

NUM_LABELS = len(ID2LABEL)

EMOTION_TO_TARGET = {
    "neutral": 0,
    "sadness": 1,
    "grief": 1,
    "joy": 2,
    "amusement": 2,
    "excitement": 2,
    "optimism": 2,
    "anger": 3,
    "annoyance": 3,
    "disapproval": 3,
    "disgust": 3,
    "desire": 4,
    "fear": 5,
    "nervousness": 5,
    "love": 6,
}

TARGET_ID_TO_EMOTIONS = {
    0: ["neutral"],
    1: ["sadness", "grief"],
    2: ["joy", "amusement", "excitement", "optimism"],
    3: ["anger", "annoyance", "disapproval", "disgust"],
    4: ["desire"],
    5: ["fear", "nervousness"],
    6: ["love"],
}

PRIORITY_TARGET_IDS = [0, 1, 2, 3, 4, 5, 6]

GOEMOTIONS_EMOTION_COLUMNS = list(EMOTION_TO_TARGET.keys())


def label_map_payload() -> dict:
    return {
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "label2id": LABEL2ID,
    }


def assign_single_label(row: pd.Series) -> float:
    active_labels: list[int] = []
    for target_id in PRIORITY_TARGET_IDS:
        relevant = TARGET_ID_TO_EMOTIONS[target_id]
        if any(row.get(col, 0) == 1 for col in relevant if col in row.index):
            active_labels.append(target_id)

    if len(active_labels) == 1:
        return float(active_labels[0])
    if len(active_labels) > 1:
        return float(active_labels[-1])

    if row.get("neutral", 0) == 1:
        emotion_cols = [c for c in GOEMOTIONS_EMOTION_COLUMNS if c != "neutral" and c in row.index]
        if all(row.get(col, 0) == 0 for col in emotion_cols):
            return 0.0

    if row.get("example_very_unclear", False):
        return np.nan
    return np.nan


def apply_label_mapping(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    initial_rows = len(df)
    working = df.copy()

    working["encoded_label"] = working.apply(assign_single_label, axis=1)
    working = working.dropna(subset=["encoded_label"])
    working = working[working["text"].astype(str).str.strip() != ""]
    working["encoded_label"] = working["encoded_label"].astype(int)
    working = working.reset_index(drop=True)

    log = {
        "initial_rows": initial_rows,
        "remaining_rows": len(working),
        "dropped_rows": initial_rows - len(working),
        "class_counts": working["encoded_label"].value_counts().sort_index().to_dict(),
        "class_percentages": (
            working["encoded_label"].value_counts(normalize=True).sort_index() * 100
        ).round(4).to_dict(),
    }
    return working, log
