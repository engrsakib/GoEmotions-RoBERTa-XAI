"""Save and load multilabel validation/test probabilities for ensembling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.paths import EXPORTS_DIR
from src.training.trainer_setup import _predict_multilabel_probs

ENSEMBLE_PROBS_DIR = EXPORTS_DIR / "ensemble_probs"


def probs_path(model_id: str, split: str, exports_dir: Path | None = None) -> Path:
    root = exports_dir or ENSEMBLE_PROBS_DIR
    return root / f"{model_id}_{split}.npz"


def save_probs_from_trainer(
    trainer,
    val_dataset,
    test_dataset,
    model_id: str,
    exports_dir: Path | None = None,
) -> dict[str, str]:
    """Persist val/test sigmoid probabilities and labels for one model."""
    root = exports_dir or ENSEMBLE_PROBS_DIR
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    for split, dataset in (("val", val_dataset), ("test", test_dataset)):
        probs, labels = _predict_multilabel_probs(trainer, dataset)
        logits = np.log(probs / np.clip(1.0 - probs, 1e-8, 1.0))
        out = probs_path(model_id, split, root)
        np.savez_compressed(
            out,
            probs=probs.astype(np.float32),
            labels=labels.astype(np.float32),
            logits=logits.astype(np.float32),
            model_id=model_id,
            split=split,
        )
        paths[split] = str(out)
    return paths


def load_model_probs(
    model_id: str,
    split: str,
    exports_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    path = probs_path(model_id, split, exports_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Missing probability file: {path}")
    data = np.load(path)
    return {
        "probs": data["probs"],
        "labels": data["labels"],
        "logits": data["logits"] if "logits" in data else None,
        "model_id": str(data.get("model_id", model_id)),
    }
