"""
InceptionTime 1D for ECG classification.

Based on the InceptionTime paper (Fawaz et al., 2020) and the PTB-XL
benchmarking implementation, with an optional Squeeze-and-Excitation
channel attention module after each inception block (novel improvement).

Architecture:
    Multi-scale parallel convolutions (k, k/2, k/4) with bottleneck →
    Residual shortcuts every 3 blocks → AdaptiveConcatPool → MLP head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads import (
    AdaptiveConcatPool1d,
    ClassifierHead,
    SqueezeExcite1d,
    init_weights_kaiming,
)


class InceptionBlock1d(nn.Module):
    """Single Inception block with multi-scale convolutions.

    Input → Bottleneck (1×1 conv) → parallel [Conv(k), Conv(k/2), Conv(k/4)]
                                  ‖ MaxPool → 1×1 Conv
    → Concatenate → BatchNorm → ReLU

    Args:
        in_channels: Input channels.
        nb_filters: Number of filters per branch.
        kernel_size: Base kernel size (will also use k//2 and k//4).
        bottleneck_size: Bottleneck dimension (1×1 conv before parallel branches).
        use_se: Whether to add SE attention after this block.
        se_reduction: SE reduction ratio.
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_size: int = 40,
        bottleneck_size: int = 32,
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()

        # Compute kernel sizes (ensure odd for same-padding)
        ks = [kernel_size, kernel_size // 2, kernel_size // 4]
        ks = [k - 1 if k % 2 == 0 else k for k in ks]

        # Bottleneck 1×1 convolution
        self.bottleneck = nn.Conv1d(
            in_channels, bottleneck_size, kernel_size=1, bias=False
        )

        # Parallel convolution branches
        self.convs = nn.ModuleList()
        for k in ks:
            self.convs.append(
                nn.Conv1d(
                    bottleneck_size, nb_filters, kernel_size=k,
                    padding=k // 2, bias=False,
                )
            )

        # MaxPool branch
        self.mp = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.mp_conv = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        # Total output channels = (len(ks) + 1) * nb_filters
        out_channels = (len(ks) + 1) * nb_filters
        self.bn = nn.BatchNorm1d(out_channels)

        # Optional SE attention
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcite1d(out_channels, reduction=se_reduction)

    @property
    def out_channels(self):
        """Number of output channels (for residual connections)."""
        # 4 branches × nb_filters
        return self.bn.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck
        x_bn = self.bottleneck(x)

        # Parallel convolutions on bottleneck output
        conv_outs = [conv(x_bn) for conv in self.convs]

        # MaxPool branch on original input
        mp_out = self.mp_conv(self.mp(x))

        # Concatenate all branches
        out = torch.cat(conv_outs + [mp_out], dim=1)
        out = F.relu(self.bn(out), inplace=True)

        # Optional SE attention
        if self.use_se:
            out = self.se(out)

        return out


class Shortcut1d(nn.Module):
    """1×1 convolution + BN for residual dimension matching."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class InceptionBackbone(nn.Module):
    """InceptionTime backbone: stacked InceptionBlocks with residual shortcuts.

    Residual shortcuts are placed every `shortcut_every` blocks.
    Depth must be divisible by shortcut_every.

    Args:
        input_channels: Number of input channels (12 for ECG).
        depth: Number of inception blocks (must be divisible by 3).
        nb_filters: Filters per branch.
        kernel_size: Base kernel size.
        bottleneck_size: Bottleneck dimension.
        use_residual: Whether to use residual shortcuts.
        use_se: Whether to add SE attention to each block.
        se_reduction: SE reduction ratio.
    """

    def __init__(
        self,
        input_channels: int = 12,
        depth: int = 6,
        nb_filters: int = 32,
        kernel_size: int = 40,
        bottleneck_size: int = 32,
        use_residual: bool = True,
        use_se: bool = False,
        se_reduction: int = 16,
    ):
        super().__init__()
        assert depth % 3 == 0, "depth must be divisible by 3"

        self.use_residual = use_residual
        self.depth = depth

        # Build inception blocks
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        in_ch = input_channels
        for i in range(depth):
            block = InceptionBlock1d(
                in_channels=in_ch,
                nb_filters=nb_filters,
                kernel_size=kernel_size,
                bottleneck_size=bottleneck_size,
                use_se=use_se,
                se_reduction=se_reduction,
            )
            self.blocks.append(block)
            out_ch = block.out_channels

            # Add shortcut every 3 blocks
            if use_residual and (i + 1) % 3 == 0:
                # The shortcut bridges from 3 blocks ago to current output
                shortcut_in = input_channels if i < 3 else self.blocks[i - 3].out_channels
                self.shortcuts.append(Shortcut1d(shortcut_in, out_ch))
            else:
                self.shortcuts.append(None)

            in_ch = out_ch

        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for i, block in enumerate(self.blocks):
            out = block(x if i == 0 or not self.use_residual else out_acc)

            if self.use_residual and (i + 1) % 3 == 0:
                shortcut = self.shortcuts[i]
                if shortcut is not None:
                    out = out + shortcut(residual)
                out = F.relu(out, inplace=True)
                residual = out

            out_acc = out

        return out_acc


class Inception1d(nn.Module):
    """Full InceptionTime 1D model: backbone + classifier head.

    Args:
        num_classes: Number of output classes.
        input_channels: ECG lead count (12).
        kernel_size: Base kernel (default 40).
        depth: Number of inception blocks (default 6, must be div by 3).
        nb_filters: Filters per branch (default 32).
        bottleneck_size: Bottleneck conv channels (default 32).
        use_residual: Enable residual shortcuts every 3 blocks.
        use_se: Enable SE channel attention (novel improvement).
        se_reduction: SE bottleneck reduction ratio.
        ps_head: Dropout in classifier head.
        lin_ftrs_head: Hidden dims in classifier head.
        concat_pooling: Use avg+max concat pooling in head.
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 12,
        kernel_size: int = 40,
        depth: int = 6,
        nb_filters: int = 32,
        bottleneck_size: int = 32,
        use_residual: bool = True,
        use_se: bool = True,
        se_reduction: int = 16,
        ps_head: float = 0.5,
        lin_ftrs_head: list = None,
        concat_pooling: bool = True,
    ):
        super().__init__()

        self.backbone = InceptionBackbone(
            input_channels=input_channels,
            depth=depth,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            bottleneck_size=bottleneck_size,
            use_residual=use_residual,
            use_se=use_se,
            se_reduction=se_reduction,
        )

        self.head = ClassifierHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            concat_pooling=concat_pooling,
            lin_ftrs=lin_ftrs_head or [128],
            ps=ps_head,
        )

        # Initialize weights
        self.apply(init_weights_kaiming)

    @property
    def embedding_dim(self) -> int:
        """Dimension of the feature embedding before the final classifier."""
        return self.backbone.out_channels

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features (before pooling/head)."""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: backbone → head → logits."""
        features = self.backbone(x)
        return self.head(features)
