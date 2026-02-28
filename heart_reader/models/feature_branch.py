"""
Feature branch MLP for structured PTB-XL+ features.

A small feed-forward network that processes pre-extracted clinical ECG
features (P/QRS/T amplitudes, durations, intervals, axis, etc.) into
a dense embedding vector for fusion with signal-based features.
"""

import torch
import torch.nn as nn

from .heads import init_weights_kaiming


class FeatureBranchMLP(nn.Module):
    """MLP that encodes structured ECG features into a dense embedding.

    Architecture:
        Linear → BN → ReLU → Dropout → ... → Linear(output_dim)

    Args:
        input_dim: Number of input features (from PTB-XL+).
        hidden_dims: List of hidden layer sizes [256, 128, 64].
        output_dim: Final embedding dimension (for fusion).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        output_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        # Final projection to output_dim (no activation — will be fused)
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)

        self.apply(init_weights_kaiming)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, F) structured features.
        Returns:
            (B, output_dim) embedding.
        """
        return self.mlp(x)
