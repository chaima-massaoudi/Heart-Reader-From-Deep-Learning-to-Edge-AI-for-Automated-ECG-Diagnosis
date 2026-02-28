"""
Shared building blocks for 1D ECG model architectures.

Contains:
- AdaptiveConcatPool1d (avg + max pooling)
- SqueezeExcite1d (SE channel attention)
- ClassifierHead (pooling → MLP → output)
- Weight initialization utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveConcatPool1d(nn.Module):
    """Concatenation of adaptive average and max pooling.

    Output size = 2 × input_channels (avg ‖ max concatenated).
    This is the standard head used in the PTB-XL benchmarking codebase.
    """

    def __init__(self, output_size: int = 1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool1d(output_size)
        self.mp = nn.AdaptiveMaxPool1d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L) → (B, 2*C, output_size)
        return torch.cat([self.mp(x), self.ap(x)], dim=1)


class SqueezeExcite1d(nn.Module):
    """Squeeze-and-Excitation block for 1D signals.

    Learns channel-wise attention weights via:
    GlobalAvgPool → FC(C → C//r) → ReLU → FC(C//r → C) → Sigmoid → scale

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for the bottleneck (default 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)     # (B, C)
        w = self.excitation(w).view(b, c, 1)  # (B, C, 1)
        return x * w


class ClassifierHead(nn.Module):
    """Classification head: AdaptiveConcatPool → flatten → MLP → output.

    Args:
        in_channels: Number of channels from the backbone.
        num_classes: Number of output classes.
        concat_pooling: If True, use avg+max concat (2× channels).
        lin_ftrs: List of hidden layer sizes. Default [128].
        ps: Dropout probability for hidden layers.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        concat_pooling: bool = True,
        lin_ftrs: list = None,
        ps: float = 0.5,
    ):
        super().__init__()

        if lin_ftrs is None:
            lin_ftrs = [128]

        pool_out = in_channels * 2 if concat_pooling else in_channels

        layers = []
        if concat_pooling:
            layers.append(AdaptiveConcatPool1d(1))
        else:
            layers.append(nn.AdaptiveAvgPool1d(1))

        layers.append(nn.Flatten())
        layers.append(nn.BatchNorm1d(pool_out))
        layers.append(nn.Dropout(ps))

        # Hidden layers
        prev_dim = pool_out
        for dim in lin_ftrs:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(ps))
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def init_weights_kaiming(module: nn.Module):
    """Apply Kaiming normal initialization (used by XResNet, Inception, etc.)."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> int:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
