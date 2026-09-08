"""Load raw GoEmotions data from Kaggle input, local path, or KaggleHub (local only)."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from src.paths import DATA_RAW_DIR, IS_KAGGLE, NOTEBOOKS_DIR

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
GOEMOTIONS_DATASET_SLUG = "shivamb/go-emotions-google-emotions-dataset"

# Preferred Kaggle Input mount paths (checked before a full /kaggle/input walk).
KAGGLE_KNOWN_INPUT_DIRS = (
    KAGGLE_INPUT_ROOT / "notebooks" / "shivamb" / "list-of-emotions",
    KAGGLE_INPUT_ROOT / "list-of-emotions",
    KAGGLE_INPUT_ROOT / "go-emotions-google-emotions-dataset",
)

# Heuristic markers for GoEmotions CSV discovery under /kaggle/input/
GOEMOTIONS_PATH_KEYWORDS = ("list-of-emotions", "go-emotion", "goemotion", "go_emotion")
GOEMOTIONS_EMOTION_MARKERS = (
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "love",
    "disgust",
    "surprise",
)


def _read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    return pd.read_csv(path, sep=sep)


def _has_text_column(columns: set[str]) -> bool:
    return "text" in columns or "comment_text" in columns


def _emotion_column_count(columns: set[str]) -> int:
    return sum(1 for marker in GOEMOTIONS_EMOTION_MARKERS if marker in columns)


def _looks_like_goemotions_columns(columns: set[str]) -> bool:
    if not _has_text_column(columns):
        return False
    if _emotion_column_count(columns) >= 3:
        return True
    return "id" in columns and "example_very_unclear" in columns


def _looks_like_goemotions(df: pd.DataFrame) -> bool:
    return _looks_like_goemotions_columns(set(df.columns))


def _peek_columns(path: Path) -> set[str] | None:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        header = pd.read_csv(path, sep=sep, nrows=0)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    return set(header.columns)


def _schema_rejection_reason(columns: set[str] | None) -> str:
    if columns is None:
        return "unreadable"
    if not _has_text_column(columns):
        return "missing text/comment_text column"
    if _emotion_column_count(columns) < 3 and not (
        "id" in columns and "example_very_unclear" in columns
    ):
        return "missing GoEmotions emotion/id columns"
    return "unknown schema mismatch"


def _normalize_goemotions_df(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    if "text" not in working.columns and "comment_text" in working.columns:
        working = working.rename(columns={"comment_text": "text"})
    if "id" not in working.columns:
        working["id"] = range(len(working))
    if "example_very_unclear" not in working.columns:
        working["example_very_unclear"] = False
    return working


def _path_priority(path: Path) -> tuple[int, str]:
    """Lower sort key = higher priority."""
    lowered = str(path).lower()
    if "list-of-emotions" in lowered:
        return (0, lowered)
    if any(keyword in lowered for keyword in GOEMOTIONS_PATH_KEYWORDS):
        return (1, lowered)
    return (2, lowered)


def _find_csv_in_dir(directory: Path) -> Path | None:
    csv_files = sorted(directory.glob("*.csv"))
    if csv_files:
        return csv_files[0]
    tsv_files = sorted(directory.glob("*.tsv"))
    if tsv_files:
        return tsv_files[0]
    return None


def _discover_table_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files: list[Path] = []
    for pattern in ("*.csv", "*.tsv"):
        files.extend(root.rglob(pattern))
    return sorted(files, key=_path_priority)


def _load_from_path(path: Path, source_label: str) -> pd.DataFrame | None:
    columns = _peek_columns(path)
    if columns is None:
        print(f"Skipping unreadable file {path}")
        return None
    if not _looks_like_goemotions_columns(columns):
        return None

    try:
        df = _read_table(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        print(f"Skipping unreadable file {path}: {exc}")
        return None

    df = _normalize_goemotions_df(df)
    print(f"Loaded from {source_label}: {path} shape={df.shape}")
    return df


def _log_kaggle_discovery_summary(candidates: list[Path]) -> None:
    if not candidates:
        print(f"No CSV/TSV files found under {KAGGLE_INPUT_ROOT}")
        return

    print(f"Discovered {len(candidates)} CSV/TSV file(s) under {KAGGLE_INPUT_ROOT}:")
    for path in candidates:
        print(f"  - {path}")

    print("Rejected candidates:")
    for path in candidates:
        columns = _peek_columns(path)
        reason = _schema_rejection_reason(columns)
        print(f"  - {path}: {reason}")

    print("Expected mount paths (attach GoEmotions as Input):")
    for directory in KAGGLE_KNOWN_INPUT_DIRS:
        print(f"  - {directory}/")


def _load_from_known_kaggle_dirs() -> pd.DataFrame | None:
    for directory in KAGGLE_KNOWN_INPUT_DIRS:
        if not directory.exists():
            continue
        candidates = _discover_table_files(directory)
        for candidate in candidates:
            df = _load_from_path(candidate, f"Kaggle input ({directory.name})")
            if df is not None:
                return df
    return None


def _load_from_kaggle_input() -> pd.DataFrame | None:
    if not IS_KAGGLE or not KAGGLE_INPUT_ROOT.exists():
        return None

    df = _load_from_known_kaggle_dirs()
    if df is not None:
        return df

    candidates = _discover_table_files(KAGGLE_INPUT_ROOT)
    if not candidates:
        _log_kaggle_discovery_summary([])
        return None

    # Prefer go-emotions paths, then validate schema before loading full file.
    for candidate in candidates:
        df = _load_from_path(candidate, "Kaggle input")
        if df is not None:
            return df

    # Last resort: first CSV in any attached input folder (legacy behaviour).
    for input_dir in sorted(KAGGLE_INPUT_ROOT.iterdir()):
        if not input_dir.is_dir():
            continue
        csv_path = _find_csv_in_dir(input_dir)
        if csv_path is None:
            for sub in input_dir.iterdir():
                if sub.is_dir():
                    csv_path = _find_csv_in_dir(sub)
                    if csv_path:
                        break
        if csv_path:
            df = _load_from_path(csv_path, "Kaggle input (first CSV)")
            if df is not None:
                return df

    _log_kaggle_discovery_summary(candidates)
    return None


def _load_from_notebooks_dataset() -> pd.DataFrame | None:
    dataset_csv = NOTEBOOKS_DIR / "dataset" / "go_emotions_dataset.csv"
    if not dataset_csv.is_file():
        return None
    return _load_from_path(dataset_csv, "notebooks/dataset")


def _load_from_local() -> pd.DataFrame | None:
    if not DATA_RAW_DIR.exists():
        return None

    candidates = _discover_table_files(DATA_RAW_DIR)
    for candidate in candidates:
        df = _load_from_path(candidate, "local")
        if df is not None:
            return df

    csv_path = _find_csv_in_dir(DATA_RAW_DIR)
    if csv_path is None:
        return None

    df = _load_from_path(csv_path, "local")
    return df


def _load_from_kagglehub() -> pd.DataFrame:
    """Local-dev fallback only — never called on Kaggle."""
    import kagglehub

    path = kagglehub.dataset_download(GOEMOTIONS_DATASET_SLUG)
    print(f"KaggleHub path: {path}")
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in KaggleHub dataset at {path}")
    csv_path = os.path.join(path, csv_files[0])
    df = pd.read_csv(csv_path)
    print(f"Loaded from KaggleHub: shape={df.shape}")
    return df


def _kaggle_dataset_missing_error() -> FileNotFoundError:
    expected_paths = "\n".join(f"  - {directory}/" for directory in KAGGLE_KNOWN_INPUT_DIRS)
    return FileNotFoundError(
        "GoEmotions dataset not found on Kaggle.\n"
        "Fix: In the notebook sidebar, click Add Input and attach either:\n"
        f"  {GOEMOTIONS_DATASET_SLUG}\n"
        "  shivamb/list-of-emotions (notebook with GoEmotions input)\n"
        "Then Save Version and re-run.\n"
        "Expected mount paths:\n"
        f"{expected_paths}\n"
        "Alternatively, place a GoEmotions CSV under:\n"
        f"  {DATA_RAW_DIR}"
    )


def load_official_splits(base_dir: Path) -> dict[str, pd.DataFrame] | None:
    splits = {}
    for name in ("train", "dev", "test"):
        tsv_path = base_dir / f"{name}.tsv"
        if not tsv_path.exists():
            return None
        splits[name if name != "dev" else "validation"] = pd.read_csv(tsv_path, sep="\t")
    return splits


def load_raw_dataframe() -> pd.DataFrame:
    """
    Load order:
      1. Known Kaggle paths (e.g. /kaggle/input/notebooks/shivamb/list-of-emotions)
      2. Full /kaggle/input/ walk (prefer go-emotions files)
      3. data/raw/goemotions/
      4. notebooks/dataset/go_emotions_dataset.csv
      5. KaggleHub — local environments only (never on Kaggle)
    """
    df = _load_from_kaggle_input()
    if df is not None:
        return df

    df = _load_from_local()
    if df is not None:
        return df

    df = _load_from_notebooks_dataset()
    if df is not None:
        return df

    if IS_KAGGLE:
        raise _kaggle_dataset_missing_error()

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
