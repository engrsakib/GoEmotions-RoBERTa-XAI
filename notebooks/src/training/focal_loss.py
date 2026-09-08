"""Focal Loss, Weighted CE, and custom HuggingFace Trainers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import Trainer


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal = alpha_t * focal
        return focal.mean()


def compute_class_weights(
    labels,
    num_classes: int,
    mode: str = "inverse_freq",
) -> torch.Tensor | None:
    """
    Compute per-class weights for loss functions.

    Modes:
      - inverse_freq: standard balanced weights (N / (K * n_c))
      - sqrt_inverse: sqrt of inverse_freq (less aggressive)
      - none: return None (uniform weighting)
    """
    if mode == "none":
        return None

    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    counts = counts.clamp(min=1.0)
    weights = counts.sum() / (num_classes * counts)

    if mode == "sqrt_inverse":
        weights = torch.sqrt(weights)

    return weights / weights.sum() * num_classes


class FocalLossTrainer(Trainer):
    def __init__(
        self,
        *args,
        focal_gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.focal_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


class WeightedCETrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights,
        )
        return (loss, outputs) if return_outputs else loss
