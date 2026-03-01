"""
Loss functions for multi-label ECG classification.

- BCEWithLogitsLoss (default, matching PTB-XL benchmark)
- Focal Loss (for handling class imbalance on rare classes like HYP/MI)
- Label smoothing wrapper
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


class LabelSmoothingBCE(nn.Module):
    """Binary cross-entropy with label smoothing for multi-label tasks.

    Smooths targets: 1 → 1 - smoothing, 0 → smoothing.
    Prevents overconfident predictions and improves calibration.

    Args:
        smoothing: Label smoothing factor (0.05 recommended).
        base_loss: Underlying loss module (BCEWithLogitsLoss or FocalLoss).
    """

    def __init__(self, smoothing: float = 0.05, base_loss: nn.Module = None):
        super().__init__()
        self.smoothing = smoothing
        self.base_loss = base_loss or nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = targets * (1.0 - self.smoothing) + (1.0 - targets) * self.smoothing
        return self.base_loss(logits, smoothed)


def mixup_data(signal, labels, features=None, alpha=0.3):
    """Apply Mixup augmentation to a batch.

    Interpolates between random pairs of samples in the batch.

    Args:
        signal: (B, C, T) signal tensor.
        labels: (B, num_classes) label tensor.
        features: (B, F) feature tensor or None.
        alpha: Beta distribution parameter.

    Returns:
        Tuple of (mixed_signal, mixed_labels, mixed_features_or_None, lam).
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = signal.size(0)
    index = torch.randperm(batch_size, device=signal.device)

    mixed_signal = lam * signal + (1 - lam) * signal[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]

    mixed_features = None
    if features is not None:
        mixed_features = lam * features + (1 - lam) * features[index]

    return mixed_signal, mixed_labels, mixed_features, lam


def build_loss(cfg: dict) -> nn.Module:
    """Build loss function from config.

    Args:
        cfg: Training config section.

    Returns:
        Loss module.
    """
    loss_name = cfg.get("loss", "bce")
    smoothing = cfg.get("label_smoothing", 0.0)

    if loss_name == "bce":
        base = nn.BCEWithLogitsLoss()
    elif loss_name == "focal":
        base = FocalLoss(
            alpha=cfg.get("focal_alpha", 0.25),
            gamma=cfg.get("focal_gamma", 2.0),
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    if smoothing > 0:
        return LabelSmoothingBCE(smoothing=smoothing, base_loss=base)

    return base
