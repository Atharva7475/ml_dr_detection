"""
Learning rate schedulers for multi-stage fine-tuning.
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 3,
    total_epochs: int = 30,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """
    Creates a learning-rate scheduler that executes a linear warmup
    followed by cosine annealing decay.

    When used with an optimizer containing multiple parameter groups
    (e.g., backbone at 1e-5, head at 1e-4), each group's learning rate
    is scaled proportionally by the same schedule factor:
      - Epochs 0 to (warmup_epochs - 1): scales linearly from (1 / warmup_epochs) to 1.0.
      - Epochs warmup_epochs to (total_epochs - 1): scales following a cosine curve
        from 1.0 down to min_lr_ratio.

    Args:
        optimizer (Optimizer): PyTorch optimizer.
        warmup_epochs (int): Number of warmup epochs (default: 3).
        total_epochs (int): Total epochs in Stage 2 (default: 30).
        min_lr_ratio (float): Ratio of minimum learning rate to base learning rate (default: 0.01).

    Returns:
        LambdaLR: Configured PyTorch scheduler.
    """
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            # Linear warmup
            return float(current_epoch + 1) / float(max(1, warmup_epochs))

        # Cosine annealing decay
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
