"""Class-weighted multilabel losses (public API for Track A training)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.training.asymmetric_loss import AsymmetricLoss
from src.training.asl_config import DEFAULT_CLIP, DEFAULT_GAMMA_NEG, DEFAULT_GAMMA_POS


def compute_multilabel_class_weights(
    train_multi_hot: np.ndarray,
    *,
    mode: str = "inverse_freq",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Per-class positive-frequency weights for multilabel training.

    inverse_freq: w_c = N / (C * max(freq_c, 1))
    sqrt_inverse: sqrt of inverse_freq
    """
    if train_multi_hot.ndim == 1:
        train_multi_hot = train_multi_hot.reshape(1, -1)
    y = train_multi_hot.astype(np.float64)
    n_samples, num_classes = y.shape
    freq = y.sum(axis=0)
    freq = np.maximum(freq, 1.0)
    weights = n_samples / (num_classes * freq)
    if mode == "sqrt_inverse":
        weights = np.sqrt(weights)
    elif mode != "inverse_freq":
        raise ValueError(f"Unknown multilabel class weight mode: {mode}")
    if normalize:
        weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def multi_hot_from_train_labels(train_labels: list) -> np.ndarray:
    rows = []
    for label_tensor in train_labels:
        if isinstance(label_tensor, torch.Tensor):
            rows.append(label_tensor.detach().cpu().numpy())
        else:
            rows.append(np.asarray(label_tensor, dtype=np.float32))
    return np.stack(rows, axis=0)


class WeightedAsymmetricLoss(nn.Module):
    """Asymmetric loss with per-class weights on each (sample, class) term."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        gamma_pos: float = DEFAULT_GAMMA_POS,
        gamma_neg: float = DEFAULT_GAMMA_NEG,
        clip: float = DEFAULT_CLIP,
    ):
        super().__init__()
        self.base_loss = AsymmetricLoss(
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            clip=clip,
        )
        self.register_buffer("class_weights", class_weights.view(1, -1))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        p = torch.sigmoid(logits).clamp(min=self.base_loss.eps, max=1.0 - self.base_loss.eps)
        p_neg = (p + self.base_loss.clip).clamp(max=1.0)

        log_p = torch.log(p)
        log_one_minus_p_neg = torch.log((1.0 - p_neg).clamp(min=self.base_loss.eps))

        loss_pos = targets * log_p
        loss_neg = (1.0 - targets) * log_one_minus_p_neg
        base_loss = loss_pos + loss_neg

        pos_weight = (1.0 - p).pow(self.base_loss.gamma_pos)
        neg_weight = p_neg.pow(self.base_loss.gamma_neg)
        focal_weight = targets * pos_weight + (1.0 - targets) * neg_weight

        per_elem = -(base_loss * focal_weight)
        w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
        weighted = per_elem * w
        return weighted.mean()


class WeightedMultilabelFocalLoss(nn.Module):
    """Sigmoid focal loss for multilabel heads with optional class weights."""

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: torch.Tensor | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.view(1, -1))
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal = (1.0 - pt).pow(self.gamma)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal * bce
        if self.class_weights is not None:
            w = self.class_weights.to(device=logits.device, dtype=logits.dtype)
            loss = loss * w
        return loss.mean()
