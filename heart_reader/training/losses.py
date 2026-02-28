"""
Loss functions for multi-label ECG classification.

- BCEWithLogitsLoss (default, matching PTB-XL benchmark)
- Focal Loss (for handling class imbalance on rare classes like HYP/MI)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification.

    Addresses class imbalance by down-weighting well-classified examples.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    Applied per-class with BCEWithLogits (sigmoid) formulation.

    Args:
        alpha: Balancing factor (default 0.25).
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C) raw model outputs (before sigmoid).
            targets: (B, C) multi-hot ground truth labels.
        """
        # Standard BCE with logits (numerically stable)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Probability of correct class
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Focal modulation
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(cfg: dict) -> nn.Module:
    """Build loss function from config.

    Args:
        cfg: Training config section.

    Returns:
        Loss module.
    """
    loss_name = cfg.get("loss", "bce")

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "focal":
        return FocalLoss(
            alpha=cfg.get("focal_alpha", 0.25),
            gamma=cfg.get("focal_gamma", 2.0),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
