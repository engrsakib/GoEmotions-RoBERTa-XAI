"""Log active loss hyperparameters to console, W&B, and MLflow."""

from __future__ import annotations

import logging
from typing import Any

from transformers import TrainerCallback

from src.training.asl_config import resolve_asl_hyperparameters

logger = logging.getLogger(__name__)


def _log_to_wandb(params: dict[str, float]) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.config.update({f"asl/{k}": v for k, v in params.items()})


def _log_to_mlflow(params: dict[str, float]) -> None:
    try:
        import mlflow
    except ImportError:
        return
    if mlflow.active_run() is None:
        return
    mlflow.log_params({f"asl_{k}": v for k, v in params.items()})


def log_asymmetric_loss_hyperparameters(
    params: dict[str, float],
    *,
    prefix: str = "Asymmetric Loss",
) -> None:
    """Emit ASL hyperparameters to console and any active experiment trackers."""
    message = (
        f"{prefix}: gamma_neg={params['gamma_neg']}, "
        f"gamma_pos={params['gamma_pos']}, clip={params['clip']}"
    )
    logger.info(message)
    print(message)
    _log_to_wandb(params)
    _log_to_mlflow(params)


def log_asymmetric_loss_from_config(config: dict[str, Any]) -> dict[str, float]:
    """Resolve and log ASL hyperparameters from trainer config."""
    params = resolve_asl_hyperparameters(config)
    log_asymmetric_loss_hyperparameters(params)
    return params


class AsymmetricLossHyperparamCallback(TrainerCallback):
    """Log ASL hyperparameters once at the start of HuggingFace Trainer training."""

    def __init__(self, params: dict[str, float]):
        self.params = params

    def on_train_begin(self, args, state, control, **kwargs):
        log_asymmetric_loss_hyperparameters(self.params)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step > 1:
            return control
        # Ensure ASL values appear in the first Trainer log payload (TensorBoard, etc.).
        if logs is not None:
            logs.update({f"asl/{k}": v for k, v in self.params.items()})
        return control
