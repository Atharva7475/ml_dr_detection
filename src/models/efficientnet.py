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


def freeze_backbone(model: nn.Module) -> None:
    """
    Freeze all parameters in the EfficientNet-B0 feature-extraction backbone.
    """
    if hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreeze all parameters in the EfficientNet-B0 feature-extraction backbone
    for full fine-tuning.
    """
    if hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = True


def freeze_batchnorm_layers(model: nn.Module) -> None:
    """
    Set all BatchNorm layers in the backbone to evaluation mode to prevent
    their running statistics (mean, variance) from updating during head warmup.
    """
    if hasattr(model, "features"):
        for m in model.features.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def get_parameter_groups(
    model: nn.Module,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> list:
    """
    Construct separate parameter groups for Stage 2 full fine-tuning.
    Ensures every intended trainable parameter is assigned to exactly one group.

    Args:
        model (nn.Module): EfficientNet-B0 model.
        backbone_lr (float): Learning rate for the backbone features (default: 1e-5).
        head_lr (float): Learning rate for the classification head (default: 1e-4).
        weight_decay (float): L2 regularization weight decay (default: 1e-4).

    Returns:
        list[dict]: List containing two parameter group dictionaries for AdamW.
    """
    backbone_params = [p for p in model.features.parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]

    # Integrity assertions: disjoint sets and complete coverage
    backbone_ids = set(id(p) for p in backbone_params)
    head_ids = set(id(p) for p in head_params)

    assert backbone_ids.isdisjoint(head_ids), (
        "Parameter overlap detected between backbone and classification head!"
    )

    all_trainable_ids = set(id(p) for p in model.parameters() if p.requires_grad)
    combined_ids = backbone_ids.union(head_ids)

    assert combined_ids == all_trainable_ids, (
        f"Trainable parameter mismatch: expected {len(all_trainable_ids)} total trainable parameters, "
        f"got {len(combined_ids)} across groups."
    )

    return [
        {
            "name": "backbone",
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
        },
        {
            "name": "head",
            "params": head_params,
            "lr": head_lr,
            "weight_decay": weight_decay,
        },
    ]

