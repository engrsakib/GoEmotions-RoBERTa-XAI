"""Clipped asymmetric loss for multi-label imbalanced classification (Paper 3)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.asl_config import DEFAULT_CLIP, DEFAULT_GAMMA_NEG, DEFAULT_GAMMA_POS


class AsymmetricLoss(nn.Module):
    """
    Clipped asymmetric loss for multi-label classification.

    Reference: Ramakrishnan & Babu, IEEE Access 2025 (Paper 3).

    For each logit:
      - p = sigmoid(logit), clamped to [eps, 1-eps]
      - p_neg = (p + clip).clamp(max=1) for negative-label probability shifting
      - positive focal weight: (1 - p) ** gamma_pos
      - negative focal weight: p_neg ** gamma_neg
    """

    def __init__(
        self,
        gamma_pos: float = DEFAULT_GAMMA_POS,
        gamma_neg: float = DEFAULT_GAMMA_NEG,
        clip: float = DEFAULT_CLIP,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def hyperparameters(self) -> dict[str, float]:
        return {
            "gamma_neg": self.gamma_neg,
            "gamma_pos": self.gamma_pos,
            "clip": self.clip,
        }

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        p = torch.sigmoid(logits).clamp(min=self.eps, max=1.0 - self.eps)

        p_neg = (p + self.clip).clamp(max=1.0)

        log_p = torch.log(p)
        log_one_minus_p_neg = torch.log((1.0 - p_neg).clamp(min=self.eps))

        loss_pos = targets * log_p
        loss_neg = (1.0 - targets) * log_one_minus_p_neg
        base_loss = loss_pos + loss_neg

        pos_weight = (1.0 - p).pow(self.gamma_pos)
        neg_weight = p_neg.pow(self.gamma_neg)
        focal_weight = targets * pos_weight + (1.0 - targets) * neg_weight

        return (-(base_loss * focal_weight)).mean()
