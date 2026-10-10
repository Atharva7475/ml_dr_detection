"""
CNN + Transformer Hybrid Model for Diabetic Retinopathy Classification.

Uses an ImageNet-pretrained EfficientNet-B0 as the CNN backbone to produce
spatial feature maps, projects them into token embeddings, adds learnable
positional encodings, passes them through a Transformer encoder, and
classifies using the pooled global representation.

Architecture:
    Image [B, 3, 224, 224]
         │
         ▼
    EfficientNet-B0 features      → [B, 1280, 7, 7]
         │
         ▼
    1×1 Conv projection           → [B, embed_dim, 7, 7]
         │
         ▼
    Flatten spatial → tokens      → [B, 49, embed_dim]
         │
         ▼
    + Learnable positional encoding
         │
         ▼
    Transformer Encoder (N layers) → [B, 49, embed_dim]
         │
         ▼
    Global average pooling         → [B, embed_dim]
         │
         ▼
    Classification head            → [B, 5]

Usage:
    from src.models.cnn_transformer import build_cnn_transformer, count_parameters

    model = build_cnn_transformer(num_classes=5, pretrained=True, freeze_cnn_backbone=True)
    total, trainable = count_parameters(model)
"""

import math

import torch
import torch.nn as nn
from torchvision import models


class CNNTransformerHybrid(nn.Module):
    """
    CNN + Transformer hybrid that treats spatial locations from a CNN feature
    map as tokens for a Transformer encoder.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Load ImageNet pretrained CNN backbone.
        freeze_cnn_backbone (bool): Freeze all CNN backbone parameters.
        embed_dim (int): Transformer embedding dimension.
        num_transformer_layers (int): Number of Transformer encoder layers.
        num_attention_heads (int): Number of multi-head attention heads.
        dropout (float): Dropout probability for the Transformer and classifier.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_cnn_backbone: bool = True,
        embed_dim: int = 256,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ---- CNN backbone: EfficientNet-B0 feature extractor ----
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            backbone = models.efficientnet_b0(weights=weights)
        else:
            backbone = models.efficientnet_b0(weights=None)

        # Keep only the feature-extraction layers (no avgpool / classifier)
        self.cnn_features = backbone.features  # Output: [B, 1280, 7, 7] for 224×224 input
        self.cnn_feature_dim = 1280  # EfficientNet-B0 last feature-map channels

        if freeze_cnn_backbone:
            for param in self.cnn_features.parameters():
                param.requires_grad = False

        # ---- 1×1 convolution to project CNN channels → embed_dim ----
        self.projection = nn.Sequential(
            nn.Conv2d(self.cnn_feature_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # ---- Learnable positional encoding ----
        # For 224×224 input through EfficientNet-B0 features → 7×7 spatial map → 49 tokens
        self.num_tokens = 7 * 7  # 49
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.num_tokens, embed_dim) * 0.02
        )

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attention_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # ---- Layer norm before classification ----
        self.norm = nn.LayerNorm(embed_dim)

        # ---- Classification head ----
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Initialise projection and classifier weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier/Kaiming initialisation to non-pretrained layers."""
        for module in [self.projection, self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: CNN features → tokens → Transformer → classification.

        Args:
            x (torch.Tensor): Input images of shape [B, 3, 224, 224].

        Returns:
            torch.Tensor: Class logits of shape [B, num_classes].
        """
        # CNN feature extraction → [B, 1280, H, W]
        features = self.cnn_features(x)

        # 1×1 conv projection → [B, embed_dim, H, W]
        projected = self.projection(features)

        B, C, H, W = projected.shape

        # Flatten spatial dimensions into a token sequence → [B, H*W, embed_dim]
        tokens = projected.flatten(2).transpose(1, 2)  # [B, num_tokens, embed_dim]

        # Add learnable positional encoding
        # Handle case where spatial resolution differs from expected 7×7
        if tokens.size(1) != self.positional_encoding.size(1):
            pos_enc = nn.functional.interpolate(
                self.positional_encoding.transpose(1, 2),
                size=tokens.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            pos_enc = self.positional_encoding

        tokens = tokens + pos_enc

        # Transformer encoder → [B, num_tokens, embed_dim]
        encoded = self.transformer_encoder(tokens)

        # Global average pooling over tokens → [B, embed_dim]
        global_repr = self.norm(encoded.mean(dim=1))

        # Classification → [B, num_classes]
        logits = self.classifier(global_repr)

        return logits


def build_cnn_transformer(
    num_classes: int = 5,
    pretrained: bool = True,
    freeze_cnn_backbone: bool = True,
    embed_dim: int = 256,
    num_transformer_layers: int = 4,
    num_attention_heads: int = 8,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Build a CNN + Transformer hybrid model for classification.

    Args:
        num_classes (int): Number of output classes (default: 5 for DR grading).
        pretrained (bool): If True, load ImageNet pretrained CNN backbone (default: True).
        freeze_cnn_backbone (bool): If True, freeze CNN backbone parameters (default: True).
        embed_dim (int): Transformer embedding dimension (default: 256).
        num_transformer_layers (int): Number of Transformer encoder layers (default: 4).
        num_attention_heads (int): Number of attention heads (default: 8).
        dropout (float): Dropout rate (default: 0.1).

    Returns:
        nn.Module: CNN + Transformer hybrid model.
    """
    return CNNTransformerHybrid(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_cnn_backbone=freeze_cnn_backbone,
        embed_dim=embed_dim,
        num_transformer_layers=num_transformer_layers,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> tuple:
    """
    Count total and trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        tuple: (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_summary(model: nn.Module) -> None:
    """
    Print a human-readable summary of the CNN+Transformer model.

    Args:
        model (nn.Module): The CNNTransformerHybrid model to summarize.
    """
    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params

    print("=" * 60)
    print("        CNN + TRANSFORMER HYBRID MODEL SUMMARY         ")
    print("=" * 60)
    print(f"  Total Parameters:      {total_params:>12,}")
    print(f"  Trainable Parameters:  {trainable_params:>12,}")
    print(f"  Frozen Parameters:     {frozen_params:>12,}")
    print("-" * 60)
    if hasattr(model, "cnn_feature_dim"):
        print(f"  CNN Feature Channels:  {model.cnn_feature_dim}")
        print(f"  Num Spatial Tokens:    {model.num_tokens}")
    if hasattr(model, "classifier"):
        linear = model.classifier[-1]
        print(f"  Embed Dim:             {linear.in_features}")
        print(f"  Output Classes:        {linear.out_features}")
    print("=" * 60)
