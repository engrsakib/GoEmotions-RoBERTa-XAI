"""Text normalization, length filtering, and deduplication."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

WHITESPACE_RE = re.compile(r"\s+")

VALID_DEDUP_POLICIES = ("none", "consensus", "split_internal", "global_first")


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = unicodedata.normalize("NFKC", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def apply_dedup_policy(
    df: pd.DataFrame,
    policy: str = "consensus",
    label_column: str = "encoded_label",
) -> tuple[pd.DataFrame, dict]:
    """
    Deduplicate rows by normalized lowercase text.

    Policies:
      - none: keep all rows
      - consensus: keep one row per text only when all duplicates share the same label
      - split_internal: no dedup here (applied per-split after train/val/test split)
      - global_first: legacy keep-first (not recommended — label noise)
    """
    if policy not in VALID_DEDUP_POLICIES:
        raise ValueError(f"Unknown dedup_policy '{policy}'. Choose from {VALID_DEDUP_POLICIES}")

    working = df.copy()
    if "_norm_key" not in working.columns:
        working["_norm_key"] = working["text"].astype(str).str.lower()

    before = len(working)
    if policy == "none" or policy == "split_internal":
        log = {
            "dedup_policy": policy,
            "dedup_dropped": 0,
            "ambiguous_duplicates_dropped": 0,
            "rows_before_dedup": before,
            "rows_after_dedup": before,
        }
        return working, log

    if policy == "global_first":
        deduped = working.drop_duplicates(subset=["_norm_key"], keep="first")
        log = {
            "dedup_policy": policy,
            "dedup_dropped": before - len(deduped),
            "ambiguous_duplicates_dropped": 0,
            "rows_before_dedup": before,
            "rows_after_dedup": len(deduped),
        }
        return deduped.reset_index(drop=True), log

    # consensus: keep rows where all duplicates agree on label
    label_nunique = working.groupby("_norm_key")[label_column].nunique()
    consensus_keys = set(label_nunique[label_nunique == 1].index)
    ambiguous_keys = set(label_nunique[label_nunique > 1].index)

    consensus_df = working[working["_norm_key"].isin(consensus_keys)]
    deduped = consensus_df.drop_duplicates(subset=["_norm_key"], keep="first")
    ambiguous_rows_dropped = int(working["_norm_key"].isin(ambiguous_keys).sum())

    log = {
        "dedup_policy": policy,
        "dedup_dropped": before - len(deduped),
        "ambiguous_duplicates_dropped": ambiguous_rows_dropped,
        "ambiguous_unique_texts": len(ambiguous_keys),
        "consensus_unique_texts": len(consensus_keys),
        "rows_before_dedup": before,
        "rows_after_dedup": len(deduped),
    }
    return deduped.reset_index(drop=True), log


def dedup_split_internal(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Dedup within a single split only (after train/val/test assignment)."""
    working = df.copy()
    if "_norm_key" not in working.columns:
        working["_norm_key"] = working["text"].astype(str).str.lower()
    before = len(working)
    deduped = working.drop_duplicates(subset=["_norm_key"], keep="first")
    return deduped.reset_index(drop=True), before - len(deduped)


def clean_dataframe(
    df: pd.DataFrame,
    min_char_length: int = 3,
    max_token_length_approx: int = 128,
    dedup_policy: str = "consensus",
    label_column: str = "encoded_label",
) -> tuple[pd.DataFrame, dict]:
    initial_rows = len(df)
    working = df.copy()

    working["text"] = working["text"].astype(str).apply(normalize_text)
    working = working[working["text"] != ""]
    working["_norm_key"] = working["text"].str.lower()

    before_dedup = len(working)
    working, dedup_log = apply_dedup_policy(working, policy=dedup_policy, label_column=label_column)

    working["char_length"] = working["text"].str.len()
    working["token_length_approx"] = working["text"].str.split().str.len().fillna(0).astype(int)

    length_before = len(working)
    working = working[working["char_length"] >= min_char_length]
    working = working[working["token_length_approx"] <= max_token_length_approx]
    length_dropped = length_before - len(working)

    working = working.drop(columns=["_norm_key"], errors="ignore").reset_index(drop=True)

    log = {
        "initial_rows": initial_rows,
        "after_empty_filter": before_dedup,
        "length_filter_dropped": length_dropped,
        "remaining_rows": len(working),
        "avg_char_length": round(float(working["char_length"].mean()), 2) if len(working) else 0,
        "avg_token_length_approx": round(float(working["token_length_approx"].mean()), 2) if len(working) else 0,
        "class_counts": working[label_column].value_counts().sort_index().to_dict()
        if label_column in working.columns
        else {},
        **dedup_log,
    }
    return working, log
