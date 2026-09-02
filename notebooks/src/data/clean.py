"""Text normalization, length filtering, and deduplication."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = unicodedata.normalize("NFKC", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_dataframe(
    df: pd.DataFrame,
    min_char_length: int = 3,
    max_token_length_approx: int = 128,
) -> tuple[pd.DataFrame, dict]:
    initial_rows = len(df)
    working = df.copy()

    working["text"] = working["text"].astype(str).apply(normalize_text)
    working = working[working["text"] != ""]
    working["_norm_key"] = working["text"].str.lower()

    before_dedup = len(working)
    working = working.drop_duplicates(subset=["_norm_key"], keep="first")
    dedup_dropped = before_dedup - len(working)

    working["char_length"] = working["text"].str.len()
    working["token_length_approx"] = working["text"].str.split().str.len().fillna(0).astype(int)

    length_before = len(working)
    working = working[working["char_length"] >= min_char_length]
    working = working[working["token_length_approx"] <= max_token_length_approx]
    length_dropped = length_before - len(working)

    working = working.drop(columns=["_norm_key"]).reset_index(drop=True)

    log = {
        "initial_rows": initial_rows,
        "after_empty_filter": before_dedup,
        "dedup_dropped": dedup_dropped,
        "length_filter_dropped": length_dropped,
        "remaining_rows": len(working),
        "avg_char_length": round(float(working["char_length"].mean()), 2) if len(working) else 0,
        "avg_token_length_approx": round(float(working["token_length_approx"].mean()), 2) if len(working) else 0,
        "class_counts": working["encoded_label"].value_counts().sort_index().to_dict(),
    }
    return working, log
