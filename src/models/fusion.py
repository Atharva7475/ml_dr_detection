"""
EfficientNet-B0 + DenseNet121 Feature-Fusion Model for Diabetic Retinopathy.

Extracts feature vectors from two pretrained backbones, concatenates them,
and passes the combined representation through a fusion MLP for 5-class
DR classification.

Architecture:
                ┌── EfficientNet-B0 (1280-d) ──┐
    Image ──────┤                                ├─→ concatenate (2304-d)
                └── DenseNet121    (1024-d) ──┘          │
                                                         ▼
                                                    Fusion MLP
                                                         │
                                                         ▼
                                                      5 classes

Usage:
    from src.models.fusion import build_fusion_model, count_parameters

    model = build_fusion_model(num_classes=5, pretrained=True, freeze_backbones=True)
    total, trainable = count_parameters(model)
"""

import torch
import torch.nn as nn
from torchvision import models


class FusionModel(nn.Module):
    """
    Dual-backbone feature-fusion model combining EfficientNet-B0 and
    DenseNet121 feature vectors for classification.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Load ImageNet pretrained weights for both backbones.
        freeze_backbones (bool): Freeze all backbone parameters, keeping only
            the fusion head trainable.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        freeze_backbones: bool = True,
    ) -> None:
        super().__init__()

        # ---- EfficientNet-B0 backbone ----
        if pretrained:
            eff_weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.efficientnet = models.efficientnet_b0(weights=eff_weights)
        else:
            self.efficientnet = models.efficientnet_b0(weights=None)

        # Extract feature dimension dynamically before removing classifier
        self.eff_feature_dim = self.efficientnet.classifier[1].in_features
        # Remove original classifier — use features + avgpool only
        self.efficientnet.classifier = nn.Identity()

        # ---- DenseNet121 backbone ----
        if pretrained:
            dense_weights = models.DenseNet121_Weights.IMAGENET1K_V1
            self.densenet = models.densenet121(weights=dense_weights)
        else:
            self.densenet = models.densenet121(weights=None)

        # Extract feature dimension dynamically
        self.dense_feature_dim = self.densenet.classifier.in_features
        # Remove original classifier
        self.densenet.classifier = nn.Identity()

        # ---- Freeze backbones if requested ----
        if freeze_backbones:
            for param in self.efficientnet.parameters():
                param.requires_grad = False
            for param in self.densenet.parameters():
                param.requires_grad = False

        # ---- Fusion MLP head ----
        fused_dim = self.eff_feature_dim + self.dense_feature_dim
        hidden_dim = 512

        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        # Fusion head is always trainable
        for param in self.fusion_head.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through both backbones and fusion head.

        Args:
            x (torch.Tensor): Input images of shape [B, 3, 224, 224].

        Returns:
            torch.Tensor: Class logits of shape [B, num_classes].
        """
        # EfficientNet-B0 feature extraction
        eff_features = self.efficientnet(x)  # [B, eff_feature_dim]

        # DenseNet121 feature extraction
        dense_features = self.densenet(x)  # [B, dense_feature_dim]

        # Concatenate feature vectors
        fused = torch.cat([eff_features, dense_features], dim=1)  # [B, fused_dim]

        # Classification through fusion MLP
        logits = self.fusion_head(fused)  # [B, num_classes]

        return logits


def build_fusion_model(
    num_classes: int = 5,
    pretrained: bool = True,
    freeze_backbones: bool = True,
) -> nn.Module:
    """
    Build an EfficientNet-B0 + DenseNet121 feature-fusion model.

    Args:
        num_classes (int): Number of output classes (default: 5 for DR grading).
        pretrained (bool): If True, load ImageNet pretrained weights for both
            backbones (default: True).
        freeze_backbones (bool): If True, freeze all backbone parameters and keep
            only the fusion MLP head trainable (default: True).

    Returns:
        nn.Module: Fusion model.
    """
    return FusionModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbones=freeze_backbones,
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
    Print a human-readable summary of the fusion model's parameter counts.

    Args:
        model (nn.Module): The FusionModel to summarize.
    """
    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params

    print("=" * 60)
    print("     EFFICIENTNET-B0 + DENSENET121 FUSION SUMMARY      ")
    print("=" * 60)
    print(f"  Total Parameters:      {total_params:>12,}")
    print(f"  Trainable Parameters:  {trainable_params:>12,}")
    print(f"  Frozen Parameters:     {frozen_params:>12,}")
    print("-" * 60)
    if hasattr(model, "eff_feature_dim"):
        print(f"  EfficientNet-B0 dim:   {model.eff_feature_dim}")
        print(f"  DenseNet121 dim:       {model.dense_feature_dim}")
        print(f"  Fused dimension:       {model.eff_feature_dim + model.dense_feature_dim}")
    print("=" * 60)
