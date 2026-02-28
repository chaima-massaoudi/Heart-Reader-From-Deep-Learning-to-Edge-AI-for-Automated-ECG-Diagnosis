"""
XResNet 1D for ECG classification.

Based on the "Bag of Tricks for Image Classification" paper (He et al., 2019)
adapted to 1D time series, as used in the PTB-XL benchmarking (Strodthoff et al.).

Key improvements over standard ResNet:
    - 3-conv stem (instead of single large conv)
    - BatchNorm zero-init on last BN in each residual block
    - Average pooling on identity path (instead of strided conv)
    - Kaiming normal initialization throughout
"""

import torch
import torch.nn as nn

from .heads import (
    AdaptiveConcatPool1d,
    ClassifierHead,
    init_weights_kaiming,
)


class ConvBnRelu(nn.Sequential):
    """Conv1d → BatchNorm1d → (optional) ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        act: bool = True,
        zero_bn: bool = False,
    ):
        padding = kernel_size // 2
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))

        super().__init__(*layers)

        # Zero-init BN for residual blocks (BatchNormZero trick)
        if zero_bn:
            nn.init.zeros_(self[1].weight)


class ResBlock1d(nn.Module):
    """Residual block for XResNet1d.

    Two-path design:
        Conv path: ConvBnRelu → ConvBnRelu (with zero-init final BN)
        Identity path: optional 1×1 conv + AvgPool for dim/stride matching

    For bottleneck (expansion=4):
        Conv path: 1×1 reduce → k×1 conv → 1×1 expand

    Args:
        expansion: Channel expansion factor (1 for basic, 4 for bottleneck).
        in_channels: Input channels.
        mid_channels: Middle channels (output = mid_channels × expansion).
        stride: Stride for downsampling.
        kernel_size: Kernel size for the main conv.
    """

    def __init__(
        self,
        expansion: int,
        in_channels: int,
        mid_channels: int,
        stride: int = 1,
        kernel_size: int = 5,
    ):
        super().__init__()
        out_channels = mid_channels * expansion

        # ── Conv path ──
        if expansion == 1:
            # Basic block: 2 convolutions
            self.conv_path = nn.Sequential(
                ConvBnRelu(in_channels, mid_channels, kernel_size, stride=stride),
                ConvBnRelu(mid_channels, out_channels, kernel_size, zero_bn=True, act=False),
            )
        else:
            # Bottleneck block: 1×1 → k×1 → 1×1
            self.conv_path = nn.Sequential(
                ConvBnRelu(in_channels, mid_channels, 1),
                ConvBnRelu(mid_channels, mid_channels, kernel_size, stride=stride),
                ConvBnRelu(mid_channels, out_channels, 1, zero_bn=True, act=False),
            )

        # ── Identity path (with optional projection + pooling) ──
        id_layers = []
        if in_channels != out_channels:
            id_layers.append(ConvBnRelu(in_channels, out_channels, 1, act=False))
        if stride > 1:
            id_layers.append(nn.AvgPool1d(stride, ceil_mode=True))

        self.id_path = nn.Sequential(*id_layers) if id_layers else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv_path(x) + self.id_path(x))


class XResNet1dBackbone(nn.Module):
    """XResNet1d backbone (stem + residual stages).

    Args:
        expansion: 1 (ResNet18/34) or 4 (ResNet50/101/152).
        layers: List of block counts per stage, e.g., [3, 4, 23, 3] for 101.
        input_channels: Number of input channels (12 for ECG).
        kernel_size: Kernel for residual convolutions.
        kernel_size_stem: Kernel for stem convolutions.
        stem_szs: Channel sizes for the 3-conv stem.
        block_szs: Base channel sizes per stage.
    """

    def __init__(
        self,
        expansion: int = 4,
        layers: list = None,
        input_channels: int = 12,
        kernel_size: int = 5,
        kernel_size_stem: int = 5,
        stem_szs: list = None,
        block_szs: list = None,
    ):
        super().__init__()

        if layers is None:
            layers = [3, 4, 23, 3]
        if stem_szs is None:
            stem_szs = [32, 32, 64]
        if block_szs is None:
            block_szs = [64, 128, 256, 512]

        # ── 3-conv stem ──
        stem_channels = [input_channels] + list(stem_szs)
        stem = []
        for i in range(len(stem_channels) - 1):
            stride = 2 if i == 0 else 1
            stem.append(ConvBnRelu(
                stem_channels[i], stem_channels[i + 1],
                kernel_size_stem, stride=stride,
            ))
        stem.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        self.stem = nn.Sequential(*stem)

        # ── Residual stages ──
        self.stages = nn.ModuleList()
        in_ch = stem_szs[-1]
        for stage_idx, (n_blocks, base_ch) in enumerate(zip(layers, block_szs)):
            blocks = []
            for block_idx in range(n_blocks):
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                block_in = in_ch if block_idx == 0 else base_ch * expansion
                blocks.append(ResBlock1d(
                    expansion=expansion,
                    in_channels=block_in,
                    mid_channels=base_ch,
                    stride=stride,
                    kernel_size=kernel_size,
                ))
                in_ch = base_ch * expansion
            self.stages.append(nn.Sequential(*blocks))

        self.out_channels = block_szs[-1] * expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class XResNet1d(nn.Module):
    """Full XResNet 1D model: backbone + classifier head.

    Args:
        num_classes: Output classes.
        input_channels: ECG leads (12).
        expansion: 1 or 4.
        layers: Block counts per stage.
        kernel_size: Residual conv kernel.
        stem_szs: Stem channel sizes.
        block_szs: Stage base channel sizes.
        ps_head: Dropout in head.
        concat_pooling: Use avg+max concat pooling.
    """

    def __init__(
        self,
        num_classes: int = 5,
        input_channels: int = 12,
        expansion: int = 4,
        layers: list = None,
        kernel_size: int = 5,
        kernel_size_stem: int = 5,
        stem_szs: list = None,
        block_szs: list = None,
        ps_head: float = 0.5,
        lin_ftrs_head: list = None,
        concat_pooling: bool = True,
    ):
        super().__init__()

        self.backbone = XResNet1dBackbone(
            expansion=expansion,
            layers=layers,
            input_channels=input_channels,
            kernel_size=kernel_size,
            kernel_size_stem=kernel_size_stem,
            stem_szs=stem_szs,
            block_szs=block_szs,
        )

        self.head = ClassifierHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            concat_pooling=concat_pooling,
            lin_ftrs=lin_ftrs_head or [128],
            ps=ps_head,
        )

        # Kaiming normal init
        self.apply(init_weights_kaiming)

    @property
    def embedding_dim(self) -> int:
        return self.backbone.out_channels

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


# ── Factory functions ────────────────────────────────────────────────────────

def xresnet1d18(num_classes=5, input_channels=12, **kwargs):
    return XResNet1d(num_classes, input_channels, expansion=1,
                     layers=[2, 2, 2, 2], **kwargs)

def xresnet1d34(num_classes=5, input_channels=12, **kwargs):
    return XResNet1d(num_classes, input_channels, expansion=1,
                     layers=[3, 4, 6, 3], **kwargs)

def xresnet1d50(num_classes=5, input_channels=12, **kwargs):
    return XResNet1d(num_classes, input_channels, expansion=4,
                     layers=[3, 4, 6, 3], **kwargs)

def xresnet1d101(num_classes=5, input_channels=12, **kwargs):
    return XResNet1d(num_classes, input_channels, expansion=4,
                     layers=[3, 4, 23, 3], **kwargs)

def xresnet1d152(num_classes=5, input_channels=12, **kwargs):
    return XResNet1d(num_classes, input_channels, expansion=4,
                     layers=[3, 8, 36, 3], **kwargs)
