"""Path resolution for Kaggle and local environments."""

from __future__ import annotations

import os
from pathlib import Path


def _detect_kaggle() -> bool:
    return os.path.exists("/kaggle/input") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


IS_KAGGLE = _detect_kaggle()


def _find_repo_root() -> Path:
    if IS_KAGGLE:
        candidates = [
            Path("/kaggle/working/repo"),
            Path("/kaggle/working/GoEmotions-RoBERTa-XAI"),
        ]
        cwd = Path.cwd()
        if (cwd / "src").exists():
            return cwd.parent if cwd.name == "notebooks" else cwd
        for candidate in candidates:
            if (candidate / "notebooks" / "src").exists():
                return candidate
        return Path("/kaggle/working/repo")

    # Local: this file is notebooks/src/paths.py
    return Path(__file__).resolve().parent.parent.parent


REPO_ROOT = _find_repo_root()
NOTEBOOKS_DIR = REPO_ROOT / "notebooks" if (REPO_ROOT / "notebooks").exists() else REPO_ROOT
SRC_DIR = NOTEBOOKS_DIR / "src"
CONFIG_DIR = NOTEBOOKS_DIR / "config"
ARTIFACTS_DIR = NOTEBOOKS_DIR / "artifacts"
PROCESSED_DIR = ARTIFACTS_DIR / "processed"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
LOGS_DIR = ARTIFACTS_DIR / "logs"
EXPORTS_DIR = ARTIFACTS_DIR / "exports"
DATA_RAW_DIR = REPO_ROOT / "data" / "raw" / "goemotions"
DATA_PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def ensure_artifact_dirs() -> None:
    for directory in (PROCESSED_DIR, CHECKPOINTS_DIR, LOGS_DIR, EXPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def kaggle_input_dirs() -> list[Path]:
    if not IS_KAGGLE:
        return []
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        return []
    return [path for path in input_root.iterdir() if path.is_dir()]
