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

SCHEMA_VERSION = "2.0"

# All 28 GoEmotions emotion columns (excluding id, text, example_very_unclear)
GOEMOTIONS_ALL_EMOTION_COLUMNS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

EMOTION_TO_TARGET = {
    "neutral": 0,
    "confusion": 0,
    "curiosity": 0,
    "realization": 0,
    "surprise": 0,
    "sadness": 1,
    "grief": 1,
    "disappointment": 1,
    "remorse": 1,
    "joy": 2,
    "amusement": 2,
    "excitement": 2,
    "optimism": 2,
    "admiration": 2,
    "approval": 2,
    "gratitude": 2,
    "pride": 2,
    "relief": 2,
    "anger": 3,
    "annoyance": 3,
    "disapproval": 3,
    "disgust": 3,
    "desire": 4,
    "fear": 5,
    "nervousness": 5,
    "embarrassment": 5,
    "love": 6,
    "caring": 6,
}

TARGET_ID_TO_EMOTIONS = {
    0: ["neutral", "confusion", "curiosity", "realization", "surprise"],
    1: ["sadness", "grief", "disappointment", "remorse"],
    2: [
        "joy",
        "amusement",
        "excitement",
        "optimism",
        "admiration",
        "approval",
        "gratitude",
        "pride",
        "relief",
    ],
    3: ["anger", "annoyance", "disapproval", "disgust"],
    4: ["desire"],
    5: ["fear", "nervousness", "embarrassment"],
    6: ["love", "caring"],
}

PRIORITY_TARGET_IDS = [0, 1, 2, 3, 4, 5, 6]

GOEMOTIONS_EMOTION_COLUMNS = list(EMOTION_TO_TARGET.keys())


def validate_mapping() -> None:
    """Assert every GoEmotions emotion maps to exactly one macro target."""
    missing = set(GOEMOTIONS_ALL_EMOTION_COLUMNS) - set(EMOTION_TO_TARGET.keys())
    if missing:
        raise ValueError(f"Unmapped GoEmotions columns: {sorted(missing)}")

    extra = set(EMOTION_TO_TARGET.keys()) - set(GOEMOTIONS_ALL_EMOTION_COLUMNS)
    if extra:
        raise ValueError(f"Unknown emotion keys in EMOTION_TO_TARGET: {sorted(extra)}")

    covered: set[str] = set()
    for emotions in TARGET_ID_TO_EMOTIONS.values():
        for emotion in emotions:
            if emotion in covered:
                raise ValueError(f"Emotion '{emotion}' appears in multiple target groups")
            covered.add(emotion)

    if covered != set(GOEMOTIONS_ALL_EMOTION_COLUMNS):
        raise ValueError(
            "TARGET_ID_TO_EMOTIONS and GOEMOTIONS_ALL_EMOTION_COLUMNS mismatch: "
            f"missing={sorted(set(GOEMOTIONS_ALL_EMOTION_COLUMNS) - covered)}, "
            f"extra={sorted(covered - set(GOEMOTIONS_ALL_EMOTION_COLUMNS))}"
        )

    for emotion, target_id in EMOTION_TO_TARGET.items():
        expected = next(
            tid for tid, cols in TARGET_ID_TO_EMOTIONS.items() if emotion in cols
        )
        if target_id != expected:
            raise ValueError(
                f"EMOTION_TO_TARGET['{emotion}']={target_id} "
                f"conflicts with TARGET_ID_TO_EMOTIONS group {expected}"
            )


def build_mapping_audit(df: pd.DataFrame) -> dict:
    """Summarize label mapping coverage on a raw dataframe."""
    emotion_cols = [c for c in GOEMOTIONS_ALL_EMOTION_COLUMNS if c in df.columns]
    unmapped_in_df = [c for c in emotion_cols if c not in EMOTION_TO_TARGET]
    per_emotion_counts = {
        col: int((df[col] == 1).sum()) for col in emotion_cols if col in EMOTION_TO_TARGET
    }
    labels = df.apply(assign_single_label, axis=1)
    multi_label_rows = 0
    for _, row in df.iterrows():
        active_targets = []
        for target_id in PRIORITY_TARGET_IDS:
            relevant = TARGET_ID_TO_EMOTIONS[target_id]
            if any(row.get(col, 0) == 1 for col in relevant if col in row.index):
                active_targets.append(target_id)
        if len(active_targets) > 1:
            multi_label_rows += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "total_rows": len(df),
        "mapped_emotion_columns": len(emotion_cols) - len(unmapped_in_df),
        "unmapped_emotion_columns": unmapped_in_df,
        "nan_label_rate": float(labels.isna().mean()),
        "multi_label_conflict_rows": multi_label_rows,
        "per_emotion_positive_counts": per_emotion_counts,
    }


def counts_with_label_names(counts: dict) -> dict[str, int]:
    return {
        ID2LABEL[int(class_id)]: int(count)
        for class_id, count in counts.items()
    }


def percentages_with_label_names(percentages: dict) -> dict[str, float]:
    return {
        ID2LABEL[int(class_id)]: float(pct)
        for class_id, pct in percentages.items()
    }


def label_map_payload() -> dict:
    validate_mapping()
    return {
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
        "label2id": LABEL2ID,
        "target_id_to_emotions": {str(k): v for k, v in TARGET_ID_TO_EMOTIONS.items()},
        "emotion_to_target": EMOTION_TO_TARGET,
        "schema_version": SCHEMA_VERSION,
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
        emotion_cols = [
            c for c in GOEMOTIONS_ALL_EMOTION_COLUMNS if c != "neutral" and c in row.index
        ]
        if all(row.get(col, 0) == 0 for col in emotion_cols):
            return 0.0

    if row.get("example_very_unclear", False):
        return np.nan
    return np.nan


def apply_label_mapping(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    validate_mapping()
    initial_rows = len(df)
    working = df.copy()

    mapping_audit = build_mapping_audit(working)
    working["encoded_label"] = working.apply(assign_single_label, axis=1)
    working = working.dropna(subset=["encoded_label"])
    working = working[working["text"].astype(str).str.strip() != ""]
    working["encoded_label"] = working["encoded_label"].astype(int)
    working = working.reset_index(drop=True)

    class_counts = working["encoded_label"].value_counts().sort_index().to_dict()
    class_percentages = (
        working["encoded_label"].value_counts(normalize=True).sort_index() * 100
    ).round(4).to_dict()

    log = {
        "initial_rows": initial_rows,
        "remaining_rows": len(working),
        "dropped_rows": initial_rows - len(working),
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "class_counts_named": counts_with_label_names(class_counts),
        "class_percentages_named": percentages_with_label_names(class_percentages),
        "mapping_audit": mapping_audit,
        "schema_version": SCHEMA_VERSION,
    }
    return working, log
