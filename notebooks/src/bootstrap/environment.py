"""Environment bootstrap for Kaggle and local execution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/engrsakib/GoEmotions-RoBERTa-XAI.git"
KAGGLE_REPO_DIR = "/kaggle/working/repo"


def is_kaggle() -> bool:
    return os.path.exists("/kaggle/input") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def notebooks_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def bootstrap(install_deps: bool = True) -> Path:
    """Configure working directory and Python path. Returns notebooks root."""
    nb_dir = notebooks_dir()

    if is_kaggle():
        repo_dir = Path(KAGGLE_REPO_DIR)
        if not repo_dir.exists():
            subprocess.run(["git", "clone", "--branch", "main", "--depth", "1", REPO_URL, str(repo_dir)], check=True)
        nb_dir = repo_dir / "notebooks"
        os.chdir(nb_dir)
    elif Path.cwd().name != "notebooks" and (Path.cwd() / "notebooks").is_dir():
        os.chdir(Path.cwd() / "notebooks")
        nb_dir = Path.cwd()

    nb_str = str(nb_dir)
    if nb_str not in sys.path:
        sys.path.insert(0, nb_str)

    if install_deps:
        req = nb_dir / "requirements-train.txt"
        if req.exists():
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                check=True,
            )

    return nb_dir
