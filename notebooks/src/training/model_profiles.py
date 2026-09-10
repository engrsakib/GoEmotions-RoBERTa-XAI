"""Load per-model hyperparameter profiles from config/model_profiles.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.paths import CONFIG_DIR

_PROFILES_CACHE: dict | None = None


def _load_profiles_file() -> dict:
    global _PROFILES_CACHE
    if _PROFILES_CACHE is not None:
        return _PROFILES_CACHE

    path = CONFIG_DIR / "model_profiles.yaml"
    if not path.is_file():
        _PROFILES_CACHE = {"defaults": {}, "profiles": {}, "deberta_tuning_grid": []}
        return _PROFILES_CACHE

    with path.open(encoding="utf-8") as handle:
        _PROFILES_CACHE = yaml.safe_load(handle) or {}
    return _PROFILES_CACHE


def apply_model_profile(config: dict, model_id: str) -> dict:
    """Merge model-specific profile overrides into config."""
    data = _load_profiles_file()
    merged = dict(config)
    defaults = data.get("defaults") or {}
    profile = (data.get("profiles") or {}).get(model_id) or {}

    for key, value in defaults.items():
        merged.setdefault(key, value)
    merged.update(profile)
    return merged


def get_deberta_tuning_grid() -> list[dict]:
    data = _load_profiles_file()
    return list(data.get("deberta_tuning_grid") or [])


def get_asl_tuning_grid(config: dict | None = None) -> list[dict]:
    """Cartesian product of ASL hyperparameter grids for Macro-F1 search."""
    from src.training.asl_config import build_asl_tuning_grid

    merged_config = dict(config or {})
    data = _load_profiles_file()
    profile = (data.get("profiles") or {}).get("m6_deberta_v3") or {}
    lr = merged_config.get("learning_rate", profile.get("learning_rate", 1.5e-5))
    return build_asl_tuning_grid(merged_config, extra_overrides={"learning_rate": lr})


def resolve_model_checkpoint(model_id: str, checkpoints_dir: Path | None = None) -> str | None:
    """Return path to saved model weights for a registry model_id."""
    from src.paths import CHECKPOINTS_DIR

    root = (checkpoints_dir or CHECKPOINTS_DIR) / model_id
    if not root.is_dir():
        return None

    weight_names = (
        "model.safetensors",
        "pytorch_model.bin",
        "tf_model.h5",
        "model.ckpt.index",
        "flax_model.msgpack",
    )
    if any((root / name).is_file() for name in weight_names):
        return str(root)

    checkpoints = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    for path in reversed(checkpoints):
        if any((path / name).is_file() for name in weight_names):
            return str(path)
    return None


def find_best_teacher_experiment(exports_dir: Path | None = None) -> dict | None:
    """Pick highest val/test macro-F1 among E2, E7, E8, E9 experiment JSON files."""
    from src.paths import EXPORTS_DIR

    exports_dir = exports_dir or EXPORTS_DIR
    candidates = ["E2", "E7", "E8", "E9"]
    best = None
    best_f1 = -1.0

    for exp_id in candidates:
        path = exports_dir / f"experiment_{exp_id}.json"
        if not path.is_file():
            continue
        import json

        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        metrics = payload.get("eval_metrics") or {}
        f1 = float(metrics.get("eval_macro_f1", metrics.get("macro_f1", -1)))
        if f1 > best_f1:
            best_f1 = f1
            best = {"experiment_id": exp_id, "macro_f1": f1, "model_id": payload.get("model_id")}

    return best
