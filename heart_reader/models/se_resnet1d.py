"""
SE-ResNet 1D for ECG classification.

Combines the resnet1d_wang architecture (best single model on PTB-XL
superdiagnostic at 0.930 AUC) with Squeeze-and-Excitation channel
attention blocks.

Architecture:
    3 residual stages with SE blocks, kernel_sizes=[5,3], inplanes=128.
    Based on the Wang et al. (2017) FCN-ResNet adapted to 1D ECG signals.
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


class SEBasicBlock1d(nn.Module):
    """Residual block with Squeeze-and-Excitation attention.

    Conv(k1) → BN → ReLU → Conv(k2) → BN → SE → + residual → ReLU.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_sizes: List of 2 kernel sizes for the two convolutions.
        stride: Stride for downsampling.
        se_reduction: SE bottleneck ratio.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list = None,
        stride: int = 1,
        se_reduction: int = 16,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [5, 3]

        k1, k2 = kernel_sizes

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, k1, stride=stride,
            padding=k1 // 2, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, k2, stride=1,
            padding=k2 // 2, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Squeeze-and-Excitation
        self.se = SqueezeExcite1d(out_channels, reduction=se_reduction)

        # Shortcut for dimension matching
        self.shortcut = nn.Identity()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        return F.relu(out + residual, inplace=True)


class SEResNet1dBackbone(nn.Module):
    """SE-ResNet1d backbone following the Wang-style architecture.

    No pooling in stem, stride=1 throughout (like resnet1d_wang).
    Uses kernel_sizes=[5,3] and inplanes=128.

    Args:
        input_channels: Number of input channels (12).
        inplanes: Base channel width (128).
        kernel_sizes: Kernel sizes for each residual block.
        layers: Number of blocks per stage.
        se_reduction: SE reduction ratio.
    """

    def __init__(
        self,
        input_channels: int = 12,
        inplanes: int = 128,
        kernel_sizes: list = None,
        layers: list = None,
        se_reduction: int = 16,
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [5, 3]
        if layers is None:
            layers = [1, 1, 1]

        # Stem: simple conv without pooling (Wang-style)
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, inplanes, kernel_size=7, stride=1,
                      padding=3, bias=False),
            nn.BatchNorm1d(inplanes),
            nn.ReLU(inplace=True),
        )

        # Residual stages
        self.stages = nn.ModuleList()
        in_ch = inplanes
        stage_channels = [inplanes, inplanes * 2, inplanes * 2]

        for stage_idx, (n_blocks, out_ch) in enumerate(zip(layers, stage_channels)):
            blocks = []
            for block_idx in range(n_blocks):
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                blocks.append(SEBasicBlock1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_sizes=kernel_sizes,
                    stride=stride,
                    se_reduction=se_reduction,
                ))
                in_ch = out_ch
            self.stages.append(nn.Sequential(*blocks))

        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class SEResNet1d(nn.Module):
    """Full SE-ResNet 1D model: backbone + classifier head.

    Combines the proven resnet1d_wang architecture with
    Squeeze-and-Excitation channel attention for improved
    multi-label ECG classification.

    Args:
        num_classes: Output classes (5 superclasses).
        input_channels: ECG leads (12).
        inplanes: Base channel width (128).
        kernel_sizes: Kernel sizes [5, 3].
        layers: Blocks per stage [1, 1, 1].
        se_reduction: SE bottleneck ratio (16).
        ps_head: Dropout in head (0.5).
        concat_pooling: Use avg+max concat pooling.
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 12,
        inplanes: int = 128,
        kernel_sizes: list = None,
        layers: list = None,
        se_reduction: int = 16,
        ps_head: float = 0.5,
        lin_ftrs_head: list = None,
        concat_pooling: bool = True,
    ):
        super().__init__()

        self.backbone = SEResNet1dBackbone(
            input_channels=input_channels,
            inplanes=inplanes,
            kernel_sizes=kernel_sizes,
            layers=layers,
            se_reduction=se_reduction,
        )

        self.head = ClassifierHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            concat_pooling=concat_pooling,
            lin_ftrs=lin_ftrs_head or [128],
            ps=ps_head,
        )

        self.apply(init_weights_kaiming)

    @property
    def embedding_dim(self) -> int:
        return self.backbone.out_channels

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
