"""
ResNet50 Transfer-Learning Model for Diabetic Retinopathy Classification.

Builds a ResNet50 backbone (pretrained on ImageNet) with a replaced final
fully connected layer for 5-class DR grading (No DR, Mild, Moderate, Severe,
Proliferative DR).

Usage:
    from src.models.resnet50 import build_resnet50, count_parameters

    model = build_resnet50(num_classes=5, pretrained=True, freeze_backbone=True)
    total, trainable = count_parameters(model)
"""

import torch
import torch.nn as nn
from torchvision import models


def build_resnet50(
    num_classes: int = 5,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """
    Construct a ResNet50 model configured for transfer learning.

    Args:
        num_classes (int): Number of output classes (default: 5 for DR grading).
        pretrained (bool): If True, load ImageNet pretrained weights (default: True).
        freeze_backbone (bool): If True, freeze all backbone parameters and keep
            only the final classification layer trainable (default: True).

    Returns:
        nn.Module: ResNet50 model with the final FC layer replaced.
    """
    # Load ResNet50 with or without ImageNet pretrained weights
    if pretrained:
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    else:
        model = models.resnet50(weights=None)

    # Freeze backbone parameters if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # The new FC layer is always trainable (even when backbone is frozen)
    for param in model.fc.parameters():
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
    and final classification layer configuration.

    Args:
        model (nn.Module): The model to summarize.
    """
    total_params, trainable_params = count_parameters(model)
    frozen_params = total_params - trainable_params

    print("=" * 60)
    print("               RESNET50 MODEL SUMMARY                ")
    print("=" * 60)
    print(f"  Total Parameters:      {total_params:>12,}")
    print(f"  Trainable Parameters:  {trainable_params:>12,}")
    print(f"  Frozen Parameters:     {frozen_params:>12,}")
    print("-" * 60)
    print(f"  Final FC Layer:        Linear(in={model.fc.in_features}, out={model.fc.out_features})")
    print(f"  Output Classes:        {model.fc.out_features}")
    print("=" * 60)
