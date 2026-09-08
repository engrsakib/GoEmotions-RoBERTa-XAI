#!/usr/bin/env python3
"""Audit GoEmotions 28→7 label mapping (schema v1 vs v2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

import pandas as pd

from src.data.label_mapping import (
    EMOTION_TO_TARGET,
    GOEMOTIONS_ALL_EMOTION_COLUMNS,
    SCHEMA_VERSION,
    assign_single_label,
    build_mapping_audit,
    validate_mapping,
)
from src.paths import PROCESSED_DIR, ensure_artifact_dirs

# Legacy v1 mapping (15 emotions only) for comparison
V1_EMOTION_TO_TARGET = {
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


def assign_v1(row: pd.Series) -> float:
    import numpy as np

    active = []
    priority_groups = {
        0: ["neutral"],
        1: ["sadness", "grief"],
        2: ["joy", "amusement", "excitement", "optimism"],
        3: ["anger", "annoyance", "disapproval", "disgust"],
        4: ["desire"],
        5: ["fear", "nervousness"],
        6: ["love"],
    }
    for tid in range(7):
        if any(row.get(c, 0) == 1 for c in priority_groups[tid] if c in row.index):
            active.append(tid)
    if len(active) == 1:
        return float(active[0])
    if len(active) > 1:
        return float(active[-1])
    if row.get("neutral", 0) == 1:
        return 0.0
    return np.nan


def main() -> None:
    dataset_path = NOTEBOOKS_DIR / "dataset" / "go_emotions_dataset.csv"
    if not dataset_path.is_file():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    validate_mapping()
    df = pd.read_csv(dataset_path)

    v1_labels = df.apply(assign_v1, axis=1)
    v2_labels = df.apply(assign_single_label, axis=1)
    audit = build_mapping_audit(df)

    print(f"=== Label Mapping Audit (schema {SCHEMA_VERSION}) ===")
    print(f"Dataset rows: {len(df)}")
    print(f"Mapped emotions: {len(EMOTION_TO_TARGET)}/{len(GOEMOTIONS_ALL_EMOTION_COLUMNS)}")
    print(f"v1 NaN rate: {v1_labels.isna().mean():.4f} ({int(v1_labels.isna().sum())} rows)")
    print(f"v2 NaN rate: {v2_labels.isna().mean():.4f} ({int(v2_labels.isna().sum())} rows)")
    print(f"Rows recovered v1->v2: {int(v1_labels.isna().sum() - v2_labels.isna().sum())}")
    print(f"Multi-label conflicts: {audit['multi_label_conflict_rows']}")

    print("\n=== v2 Class Distribution ===")
    dist = v2_labels.dropna().astype(int).value_counts(normalize=True).sort_index() * 100
    for class_id, pct in dist.items():
        print(f"  class {class_id}: {pct:.2f}%")

    ensure_artifact_dirs()
    out_path = PROCESSED_DIR / "mapping_audit.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_path": str(dataset_path),
        "v1_nan_rate": float(v1_labels.isna().mean()),
        "v2_nan_rate": float(v2_labels.isna().mean()),
        "rows_recovered": int(v1_labels.isna().sum() - v2_labels.isna().sum()),
        **audit,
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
