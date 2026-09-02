"""Load raw GoEmotions data from Kaggle input, KaggleHub, or local path."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.paths import DATA_RAW_DIR, kaggle_input_dirs


def _find_csv_in_dir(directory: Path) -> Path | None:
    csv_files = sorted(directory.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    tsv_files = sorted(directory.glob("*.tsv"))
    if tsv_files:
        return tsv_files[0]
    return None


def _load_from_kaggle_input() -> pd.DataFrame | None:
    for input_dir in kaggle_input_dirs():
        csv_path = _find_csv_in_dir(input_dir)
        if csv_path is None:
            for sub in input_dir.iterdir():
                if sub.is_dir():
                    csv_path = _find_csv_in_dir(sub)
                    if csv_path:
                        break
        if csv_path:
            sep = "\t" if csv_path.suffix == ".tsv" else ","
            df = pd.read_csv(csv_path, sep=sep)
            print(f"Loaded from Kaggle input: {csv_path} shape={df.shape}")
            return df
    return None


def _load_from_local() -> pd.DataFrame | None:
    if not DATA_RAW_DIR.exists():
        return None
    csv_path = _find_csv_in_dir(DATA_RAW_DIR)
    if csv_path is None:
        return None
    sep = "\t" if csv_path.suffix == ".tsv" else ","
    df = pd.read_csv(csv_path, sep=sep)
    print(f"Loaded from local: {csv_path} shape={df.shape}")
    return df


def _load_from_kagglehub() -> pd.DataFrame:
    import kagglehub

    path = kagglehub.dataset_download("shivamb/go-emotions-google-emotions-dataset")
    print(f"KaggleHub path: {path}")
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in KaggleHub dataset at {path}")
    csv_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_path)
    print(f"Loaded from KaggleHub: shape={df.shape}")
    return df


def load_official_splits(base_dir: Path) -> dict[str, pd.DataFrame] | None:
    splits = {}
    for name in ("train", "dev", "test"):
        tsv_path = base_dir / f"{name}.tsv"
        if not tsv_path.exists():
            return None
        splits[name if name != "dev" else "validation"] = pd.read_csv(tsv_path, sep="\t")
    return splits


def load_raw_dataframe() -> pd.DataFrame:
    df = _load_from_kaggle_input()
    if df is not None:
        return df

    df = _load_from_local()
    if df is not None:
        return df

    return _load_from_kagglehub()


def audit_dataframe(df: pd.DataFrame) -> dict:
    stats = {
        "rows": len(df),
        "columns": list(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "unclear_rate": float(df["example_very_unclear"].mean()) if "example_very_unclear" in df.columns else None,
        "avg_text_length": float(df["text"].astype(str).str.len().mean()) if "text" in df.columns else None,
    }
    return stats
