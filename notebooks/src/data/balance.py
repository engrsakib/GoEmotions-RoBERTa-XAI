"""Train-split class balancing strategies."""

from __future__ import annotations

import pandas as pd

from src.data.label_mapping import ID2LABEL, NUM_LABELS


def _distribution(df: pd.DataFrame) -> dict:
    counts = df["encoded_label"].value_counts().sort_index()
    total = len(df)
    return {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "counts_named": {ID2LABEL[int(k)]: int(v) for k, v in counts.items()},
        "percentages": {
            int(k): round(float(v) / total * 100, 4) for k, v in counts.items()
        },
        "total": total,
    }


def _undersample_neutral(
    df: pd.DataFrame,
    target_neutral_pct: float,
    random_seed: int,
) -> pd.DataFrame:
    neutral_id = 0
    neutral_df = df[df["encoded_label"] == neutral_id]
    other_df = df[df["encoded_label"] != neutral_id]

    if other_df.empty:
        return df

    other_count = len(other_df)
    target_neutral_count = int(other_count * target_neutral_pct / (1.0 - target_neutral_pct))
    target_neutral_count = min(target_neutral_count, len(neutral_df))

    if target_neutral_count >= len(neutral_df):
        return df

    sampled_neutral = neutral_df.sample(n=target_neutral_count, random_state=random_seed)
    return pd.concat([sampled_neutral, other_df], ignore_index=True).sample(
        frac=1.0, random_state=random_seed
    ).reset_index(drop=True)


def _oversample_minority(df: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    counts = df["encoded_label"].value_counts()
    target_count = int(counts.median())
    parts = []

    for class_id in range(NUM_LABELS):
        class_df = df[df["encoded_label"] == class_id]
        if class_df.empty:
            continue
        if len(class_df) < target_count:
            extra = class_df.sample(
                n=target_count - len(class_df),
                replace=True,
                random_state=random_seed + class_id,
            )
            parts.append(pd.concat([class_df, extra], ignore_index=True))
        else:
            parts.append(class_df)

    return pd.concat(parts, ignore_index=True).sample(
        frac=1.0, random_state=random_seed
    ).reset_index(drop=True)


def _oversample_classes(
    df: pd.DataFrame,
    class_ids: list[int],
    random_seed: int,
) -> pd.DataFrame:
    counts = df["encoded_label"].value_counts()
    target_count = int(counts.median())
    parts = []

    for class_id in range(NUM_LABELS):
        class_df = df[df["encoded_label"] == class_id]
        if class_df.empty:
            continue
        if class_id in class_ids and len(class_df) < target_count:
            extra = class_df.sample(
                n=target_count - len(class_df),
                replace=True,
                random_state=random_seed + class_id,
            )
            parts.append(pd.concat([class_df, extra], ignore_index=True))
        else:
            parts.append(class_df)

    return pd.concat(parts, ignore_index=True).sample(
        frac=1.0, random_state=random_seed
    ).reset_index(drop=True)


def balance_train_df(
    train_df: pd.DataFrame,
    strategy: str = "none",
    target_neutral_pct: float = 0.27,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply train-only balancing. Val/test splits must never be modified.

    Strategies:
      - none: no change
      - undersample_neutral: cap neutral class to target_neutral_pct
      - oversample_minority: oversample all classes below median to median
      - hybrid: undersample neutral + oversample desire/fear
    """
    before = _distribution(train_df)

    if strategy == "none":
        return train_df, {"strategy": "none", "before": before, "after": before}

    balanced = train_df.copy()

    if strategy == "undersample_neutral":
        balanced = _undersample_neutral(balanced, target_neutral_pct, random_seed)
    elif strategy == "oversample_minority":
        balanced = _oversample_minority(balanced, random_seed)
    elif strategy == "hybrid":
        balanced = _undersample_neutral(balanced, target_neutral_pct, random_seed)
        balanced = _oversample_classes(balanced, class_ids=[4, 5], random_seed=random_seed)
    else:
        raise ValueError(
            f"Unknown balance_strategy '{strategy}'. "
            "Choose: none, undersample_neutral, oversample_minority, hybrid"
        )

    after = _distribution(balanced)
    log = {
        "strategy": strategy,
        "target_neutral_pct": target_neutral_pct,
        "random_seed": random_seed,
        "before": before,
        "after": after,
        "rows_before": before["total"],
        "rows_after": after["total"],
    }
    return balanced, log
