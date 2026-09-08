"""Clipped asymmetric loss for multi-label imbalanced classification (Paper 3)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Clipped asymmetric loss for multi-label classification.
    Reference: Ramakrishnan & Babu, IEEE Access 2025 (Paper 3).
    """

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Asymmetric clipping on negative examples
        probs_neg = (probs + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(probs.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=self.eps))

        loss = loss_pos + loss_neg

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            pt_pos = probs * targets + (1 - probs) * (1 - targets)
            pt_neg = probs_neg * (1 - targets) + (1 - probs_neg) * targets
            asymmetric_weight = (
                targets * pt_pos.pow(self.gamma_pos) + (1 - targets) * pt_neg.pow(self.gamma_neg)
            )
            loss = loss * asymmetric_weight

        return (-loss).mean()
