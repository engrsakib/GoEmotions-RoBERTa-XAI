"""Mixed-precision training with NaN-loss fallback (fp16 -> bf16 -> fp32)."""

from __future__ import annotations

import copy
import logging
from typing import Any, Callable

import torch
from transformers import Trainer, TrainerCallback

from src.training.training_args_builder import precision_mode_from_config

logger = logging.getLogger(__name__)

PRECISION_MODES = ("fp16", "bf16", "fp32")


def gpu_supports_bf16() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        return False


def apply_precision_mode(config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return a config copy with fp16/bf16/force_fp32 set for the given mode."""
    updated = copy.deepcopy(config)
    updated["fp16"] = mode == "fp16"
    updated["bf16"] = mode == "bf16"
    updated["force_fp32"] = mode == "fp32"
    return updated


def precision_fallback_chain(config: dict[str, Any]) -> list[str]:
    """Ordered precision modes to try when fp16_fallback_enabled."""
    if not config.get("fp16_fallback_enabled", True):
        return [precision_mode_from_config(config)]

    start = precision_mode_from_config(config)
    chain: list[str] = [start]
    if start == "fp16":
        if gpu_supports_bf16():
            chain.append("bf16")
        chain.append("fp32")
    elif start == "bf16":
        chain.append("fp32")
    return chain


class NanLossGuardCallback(TrainerCallback):
    """Stop training when loss becomes NaN or Inf; record flag for fallback retry."""

    def __init__(self):
        self.nan_detected = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control
        loss = logs.get("loss")
        if loss is not None and not _is_finite(loss):
            self.nan_detected = True
            logger.warning("Non-finite loss detected (loss=%s); stopping for precision fallback.", loss)
            print(f"WARNING: Non-finite loss detected (loss={loss}); stopping for precision fallback.")
            control.should_training_stop = True
        return control


def _is_finite(value) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def train_with_precision_fallback(
    trainer_factory: Callable[[dict[str, Any]], Trainer],
    config: dict[str, Any],
) -> tuple[Trainer, str, bool]:
    """
    Train with automatic mixed-precision fallback on NaN loss.

    Returns (trainer, precision_used, nan_detected_on_final_run).
    """
    chain = precision_fallback_chain(config)
    trainer: Trainer | None = None
    precision_used = chain[0]
    nan_detected = False

    for idx, mode in enumerate(chain):
        mode_config = apply_precision_mode(config, mode)
        if idx > 0:
            print(f"Retrying training with precision mode: {mode}")

        nan_cb = NanLossGuardCallback()
        trainer = trainer_factory(mode_config)
        trainer.add_callback(nan_cb)
        trainer.train()

        precision_used = mode
        nan_detected = nan_cb.nan_detected
        if not nan_detected:
            break
        if idx < len(chain) - 1:
            print(f"NaN loss with {mode}; falling back to next precision mode.")

    assert trainer is not None
    print(f"Training completed with precision: {precision_used}")
    return trainer, precision_used, nan_detected
