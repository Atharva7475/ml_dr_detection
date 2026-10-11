"""
Loss functions for Diabetic Retinopathy classification under class imbalance.

Supports:
1. Standard CrossEntropyLoss (uniform or class-weighted).
2. Multi-class Focal Loss (Lin et al., 2017) with configurable gamma (focusing factor)
   and alpha (class weighting).
"""

from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss for handling severe class imbalance.

    Formulation:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
        - p_t is the predicted probability for the ground truth class.
        - gamma (>= 0) reduces the relative loss for well-classified examples,
          directing gradient updates toward hard/minority instances.
        - alpha_t is an optional class weight vector.
        - When gamma = 0 and alpha = None, FocalLoss is mathematically identical
          to standard CrossEntropyLoss.

    Args:
        gamma (float): Focusing parameter. Default: 2.0. Higher values downweight easy examples.
        alpha (Tensor | list | None): Per-class weighting factors of shape (C,). Default: None.
        reduction (str): 'mean', 'sum', or 'none'. Default: 'mean'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[torch.Tensor, List[float], float]] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0.0:
            raise ValueError(f"Focal loss gamma must be >= 0, got {gamma}")
        self.gamma = float(gamma)

        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction mode: {reduction}. Use 'mean', 'sum', or 'none'.")
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
            elif isinstance(alpha, torch.Tensor):
                alpha_tensor = alpha.clone().detach().to(dtype=torch.float32)
            elif isinstance(alpha, (int, float)):
                alpha_tensor = torch.tensor([float(alpha)], dtype=torch.float32)
            else:
                raise TypeError(f"alpha must be a Tensor, list, or None, got {type(alpha)}")
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-class Focal Loss.

        Args:
            inputs (torch.Tensor): Raw model logits of shape (B, C).
            targets (torch.Tensor): Ground-truth class labels of shape (B,) with integer values in [0, C-1].

        Returns:
            torch.Tensor: Computed scalar loss (or vector if reduction='none').
        """
        if inputs.ndim != 2:
            raise ValueError(f"Expected 2D inputs (B, C), got tensor with shape {inputs.shape}")
        if targets.ndim != 1:
            raise ValueError(f"Expected 1D targets (B,), got tensor with shape {targets.shape}")
        if inputs.size(0) != targets.size(0):
            raise ValueError(f"Batch size mismatch: inputs has {inputs.size(0)}, targets has {targets.size(0)}")

        num_classes = inputs.size(1)
        if (targets < 0).any() or (targets >= num_classes).any():
            raise ValueError(f"Target values out of range [0, {num_classes - 1}]. Got targets: {targets}")

        # Compute log probabilities and probabilities in a numerically stable manner
        log_p = F.log_softmax(inputs, dim=-1)  # (B, C)
        p = torch.exp(log_p)                   # (B, C)

        # Gather probability of the target class: p_t and log(p_t)
        targets_unsqueezed = targets.unsqueeze(1)  # (B, 1)
        log_pt = log_p.gather(dim=1, index=targets_unsqueezed).squeeze(1)  # (B,)
        pt = p.gather(dim=1, index=targets_unsqueezed).squeeze(1)          # (B,)

        # Compute focal weighting factor: (1 - p_t)^gamma
        focal_weight = torch.pow(1.0 - pt, self.gamma)

        # Base focal loss per sample
        loss = -focal_weight * log_pt  # (B,)

        # Optional class weighting alpha
        if self.alpha is not None:
            if self.alpha.numel() == num_classes:
                alpha_t = self.alpha.to(device=inputs.device).gather(dim=0, index=targets)
                loss = alpha_t * loss
            elif self.alpha.numel() == 1:
                loss = self.alpha.to(device=inputs.device) * loss
            else:
                raise ValueError(
                    f"Alpha tensor length ({self.alpha.numel()}) does not match number of classes ({num_classes})"
                )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

    def __repr__(self) -> str:
        alpha_info = f", alpha={self.alpha.tolist()}" if self.alpha is not None else ""
        return f"FocalLoss(gamma={self.gamma}{alpha_info}, reduction='{self.reduction}')"


def get_loss_criterion(
    loss_type: str = "cross_entropy",
    class_weights: Optional[Union[torch.Tensor, List[float]]] = None,
    focal_gamma: float = 2.0,
    focal_alpha: Optional[Union[torch.Tensor, List[float]]] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Factory function to instantiate loss criterion based on configuration.

    Args:
        loss_type (str): 'cross_entropy' or 'focal_loss'.
        class_weights (Tensor | list | None): Weights for CrossEntropyLoss.
        focal_gamma (float): Focusing parameter for FocalLoss. Default: 2.0.
        focal_alpha (Tensor | list | None): Class weighting for FocalLoss.
        device (torch.device | None): Device to place class weights on.

    Returns:
        nn.Module: Configured loss criterion.
    """
    loss_type_lower = loss_type.lower().strip()
    if loss_type_lower == "cross_entropy":
        weights_tensor = None
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            else:
                weights_tensor = class_weights.float()
            if device is not None:
                weights_tensor = weights_tensor.to(device)
        return nn.CrossEntropyLoss(weight=weights_tensor)

    elif loss_type_lower == "focal_loss":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha or class_weights,
            reduction="mean",
        )
    else:
        raise ValueError(
            f"Unsupported loss_type: '{loss_type}'. Expected 'cross_entropy' or 'focal_loss'."
        )
