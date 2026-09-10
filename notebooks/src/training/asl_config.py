"""Asymmetric Loss (ASL) hyperparameter resolution and tuning grids."""

from __future__ import annotations

import itertools
from typing import Any

DEFAULT_GAMMA_NEG = 4.0
DEFAULT_GAMMA_POS = 1.0
DEFAULT_CLIP = 0.05

ASL_CONFIG_KEYS = (
    "asymmetric_gamma_neg",
    "asymmetric_gamma_pos",
    "asymmetric_clip",
)

DEFAULT_ASL_TUNING_GRIDS: dict[str, list[float]] = {
    "asymmetric_gamma_neg": [2.0, 3.0, 4.0],
    "asymmetric_gamma_pos": [0.0, 1.0],
    "asymmetric_clip": [0.0, 0.05, 0.1],
}


def resolve_asl_hyperparameters(config: dict[str, Any]) -> dict[str, float]:
    """Return active ASL hyperparameters from a trainer config dict."""
    return {
        "gamma_neg": float(config.get("asymmetric_gamma_neg", DEFAULT_GAMMA_NEG)),
        "gamma_pos": float(config.get("asymmetric_gamma_pos", DEFAULT_GAMMA_POS)),
        "clip": float(config.get("asymmetric_clip", DEFAULT_CLIP)),
    }


def asl_tuning_grids_from_config(config: dict[str, Any] | None = None) -> dict[str, list[float]]:
    """Merge YAML `asl_tuning_grids` with repository defaults."""
    if not config:
        return dict(DEFAULT_ASL_TUNING_GRIDS)
    merged = dict(DEFAULT_ASL_TUNING_GRIDS)
    merged.update(config.get("asl_tuning_grids") or {})
    return merged


def build_asl_tuning_grid(
    config: dict[str, Any] | None = None,
    *,
    extra_overrides: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """
    Cartesian product of ASL search grids for Macro-F1 tuning.

    Grid axes (defaults): gamma_neg [2,3,4], gamma_pos [0,1], clip [0, 0.05, 0.1].
    """
    grids = asl_tuning_grids_from_config(config)
    keys = list(DEFAULT_ASL_TUNING_GRIDS.keys())
    value_lists = [grids[key] for key in keys]
    runs: list[dict[str, float]] = []
    for combo in itertools.product(*value_lists):
        entry = dict(zip(keys, combo))
        if extra_overrides:
            entry.update(extra_overrides)
        runs.append(entry)
    return runs
