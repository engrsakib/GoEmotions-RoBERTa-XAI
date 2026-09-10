"""Log active loss hyperparameters to console, W&B, and MLflow."""

from __future__ import annotations

import logging
from typing import Any

from transformers import TrainerCallback

from src.training.asl_config import resolve_asl_hyperparameters
from src.training.training_args_builder import resolve_optimizer_hyperparameters

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


def log_optimizer_hyperparameters(
    params: dict[str, float | str],
    *,
    prefix: str = "Optimizer",
) -> None:
    """Emit optimizer/scheduler hyperparameters to console and experiment trackers."""
    message = (
        f"{prefix}: lr={params['learning_rate']}, weight_decay={params['weight_decay']}, "
        f"scheduler={params['lr_scheduler_type']}, warmup_ratio={params['warmup_ratio']}, "
        f"adam_eps={params['adam_epsilon']}, max_grad_norm={params['max_grad_norm']}"
    )
    logger.info(message)
    print(message)
    try:
        import wandb

        if wandb.run is not None:
            wandb.config.update({f"optim/{k}": v for k, v in params.items()})
    except ImportError:
        pass
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_params({f"optim_{k}": v for k, v in params.items()})
    except ImportError:
        pass


class OptimizerHyperparamCallback(TrainerCallback):
    """Log optimizer/scheduler hyperparameters at training start."""

    def __init__(self, params: dict[str, float | str]):
        self.params = params

    def on_train_begin(self, args, state, control, **kwargs):
        log_optimizer_hyperparameters(self.params)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state.global_step > 1:
            return control
        logs.update({f"optim/{k}": v for k, v in self.params.items()})
        return control


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
