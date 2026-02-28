"""
Multimodal Fusion Model for ECG classification.

Combines a signal backbone (CNN on raw 12-lead ECG) with a feature
branch (MLP on structured PTB-XL+ features) via late concatenation fusion.

Signal backbone → AdaptiveConcatPool → signal_embed
Feature MLP → feature_embed
[signal_embed ‖ feature_embed] → fusion MLP → 5-class logits
"""

import torch
import torch.nn as nn

from .heads import AdaptiveConcatPool1d, init_weights_kaiming
from .feature_branch import FeatureBranchMLP
from .inception1d import Inception1d
from .xresnet1d import xresnet1d101, XResNet1d
from .se_resnet1d import SEResNet1d


def build_backbone(name: str, cfg: dict, num_classes: int, input_channels: int):
    """Instantiate a signal backbone by name.

    Args:
        name: One of "inception1d", "xresnet1d101", "se_resnet1d".
        cfg: Model config section from YAML.
        num_classes: Number of output classes (used for standalone head, not for fusion).
        input_channels: Number of ECG leads.

    Returns:
        backbone module (with .backbone and .embedding_dim attributes).
    """
    if name == "inception1d":
        inc_cfg = cfg.get("inception1d", {})
        model = Inception1d(
            num_classes=num_classes,
            input_channels=input_channels,
            kernel_size=inc_cfg.get("kernel_size", 40),
            depth=inc_cfg.get("depth", 6),
            nb_filters=inc_cfg.get("nb_filters", 32),
            bottleneck_size=inc_cfg.get("bottleneck_size", 32),
            use_residual=inc_cfg.get("use_residual", True),
            use_se=inc_cfg.get("use_se", True),
            se_reduction=inc_cfg.get("se_reduction", 16),
        )
        return model

    elif name == "xresnet1d101":
        xr_cfg = cfg.get("xresnet1d101", {})
        model = XResNet1d(
            num_classes=num_classes,
            input_channels=input_channels,
            expansion=xr_cfg.get("expansion", 4),
            layers=xr_cfg.get("layers", [3, 4, 23, 3]),
            kernel_size=xr_cfg.get("kernel_size", 5),
            stem_szs=xr_cfg.get("stem_szs", [32, 32, 64]),
            block_szs=xr_cfg.get("block_szs", [64, 128, 256, 512]),
        )
        return model

    elif name == "se_resnet1d":
        se_cfg = cfg.get("se_resnet1d", {})
        model = SEResNet1d(
            num_classes=num_classes,
            input_channels=input_channels,
            inplanes=se_cfg.get("inplanes", 128),
            kernel_sizes=se_cfg.get("kernel_sizes", [5, 3]),
            layers=se_cfg.get("layers", [1, 1, 1]),
            se_reduction=se_cfg.get("se_reduction", 16),
        )
        return model

    else:
        raise ValueError(f"Unknown backbone: {name}")


class FusionModel(nn.Module):
    """Multimodal fusion: signal backbone + feature branch MLP.

    Late concatenation fusion strategy:
        1. Signal backbone → AdaptiveConcatPool1d → flatten → Linear → signal_embed
        2. Feature MLP → feature_embed
        3. [signal_embed ‖ feature_embed] → fusion head → logits

    When no structured features are available (num_features=0),
    falls back to a signal-only model.

    Args:
        backbone_name: Name of the signal backbone.
        model_cfg: Model config section from YAML.
        num_classes: Number of output classes.
        input_channels: Number of ECG leads.
        num_features: Number of structured input features (0 to disable).
    """

    def __init__(
        self,
        backbone_name: str,
        model_cfg: dict,
        num_classes: int = 5,
        input_channels: int = 12,
        num_features: int = 0,
    ):
        super().__init__()

        fusion_cfg = model_cfg.get("fusion", {})
        signal_embed_dim = fusion_cfg.get("signal_embed_dim", 256)
        feature_embed_dim = fusion_cfg.get("feature_embed_dim", 64)
        fusion_hidden = fusion_cfg.get("fusion_hidden", 128)
        dropout = fusion_cfg.get("dropout", 0.5)

        self.num_features = num_features
        self.backbone_name = backbone_name

        # ── Signal backbone (we use its .backbone attribute, skip its head) ──
        full_model = build_backbone(backbone_name, model_cfg, num_classes, input_channels)
        self.signal_backbone = full_model.backbone  # extract just the backbone
        backbone_out_ch = full_model.embedding_dim

        # Signal embedding: pool + project
        self.signal_pool = AdaptiveConcatPool1d(1)
        pool_dim = backbone_out_ch * 2  # concat pooling doubles channels
        self.signal_proj = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(pool_dim),
            nn.Dropout(dropout),
            nn.Linear(pool_dim, signal_embed_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(signal_embed_dim),
        )

        # ── Feature branch (only if features available) ──
        if num_features > 0:
            feat_cfg = model_cfg.get("feature_branch", {})
            self.feature_branch = FeatureBranchMLP(
                input_dim=num_features,
                hidden_dims=feat_cfg.get("hidden_dims", [256, 128]),
                output_dim=feature_embed_dim,
                dropout=feat_cfg.get("dropout", 0.3),
            )
            fusion_input_dim = signal_embed_dim + feature_embed_dim
        else:
            self.feature_branch = None
            fusion_input_dim = signal_embed_dim

        # ── Fusion head ──
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(fusion_hidden),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

        # Init the new layers
        self.signal_proj.apply(init_weights_kaiming)
        self.fusion_head.apply(init_weights_kaiming)

    def forward(self, signal: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            signal: (B, 12, seq_len) raw ECG signal.
            features: (B, F) structured features, or None.

        Returns:
            (B, num_classes) logits.
        """
        # Signal path
        x = self.signal_backbone(signal)       # (B, C, L')
        x = self.signal_pool(x)                 # (B, 2*C, 1)
        signal_embed = self.signal_proj(x)       # (B, signal_embed_dim)

        # Feature path
        if self.feature_branch is not None and features is not None:
            feat_embed = self.feature_branch(features)  # (B, feature_embed_dim)
            fused = torch.cat([signal_embed, feat_embed], dim=1)
        else:
            fused = signal_embed

        # Fusion head
        return self.fusion_head(fused)


class StandaloneModel(nn.Module):
    """Wrapper for using a single backbone without fusion (signal-only).

    This is a thin wrapper that routes the forward call properly
    for the training loop which passes dict batches.

    Args:
        backbone_name: Name of the signal backbone.
        model_cfg: Model config.
        num_classes: Number of output classes.
        input_channels: Number of ECG leads.
    """

    def __init__(
        self,
        backbone_name: str,
        model_cfg: dict,
        num_classes: int = 5,
        input_channels: int = 12,
    ):
        super().__init__()
        self.model = build_backbone(backbone_name, model_cfg, num_classes, input_channels)

    def forward(self, signal: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        return self.model(signal)
