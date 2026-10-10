"""
EfficientNet-B0 Transfer-Learning Model for Diabetic Retinopathy Classification.

Builds an EfficientNet-B0 backbone (pretrained on ImageNet) with a replaced
classifier head for 5-class DR grading (No DR, Mild, Moderate, Severe,
Proliferative DR).

Usage:
    from src.models.efficientnet import build_efficientnet_b0, count_parameters

    model = build_efficientnet_b0(num_classes=5, pretrained=True, freeze_backbone=True)
    total, trainable = count_parameters(model)
"""

import torch.nn as nn
from torchvision import models


def build_efficientnet_b0(
    num_classes: int = 5,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Construct an EfficientNet-B0 model configured for transfer learning.

    Args:
        num_classes (int): Number of output classes (default: 5 for DR grading).
        pretrained (bool): If True, load ImageNet pretrained weights (default: True).
        freeze_backbone (bool): If True, freeze all backbone (feature-extractor)
            parameters and keep only the classifier head trainable (default: True).

    Returns:
        nn.Module: EfficientNet-B0 model with the classifier replaced.
    """
    # Load EfficientNet-B0 with current torchvision weights API
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    else:
        model = models.efficientnet_b0(weights=None)

    # Freeze backbone (features) parameters if requested
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the classifier head
    # EfficientNet-B0 classifier: Sequential(Dropout, Linear(1280, 1000))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    # Ensure classifier is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


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
    Print a human-readable summary of the model's parameter counts
    and classifier configuration.

    Args:
        model (nn.Module): The model to summarize.
    """
    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params

    # Extract output features from the final Linear layer in the classifier
    classifier_linear = model.classifier[-1]

    print("=" * 60)
    print("           EFFICIENTNET-B0 MODEL SUMMARY              ")
    print("=" * 60)
    print(f"  Total Parameters:      {total_params:>12,}")
    print(f"  Trainable Parameters:  {trainable_params:>12,}")
    print(f"  Frozen Parameters:     {frozen_params:>12,}")
    print("-" * 60)
    print(f"  Classifier:            Linear(in={classifier_linear.in_features}, "
          f"out={classifier_linear.out_features})")
    print(f"  Output Classes:        {classifier_linear.out_features}")
    print("=" * 60)
