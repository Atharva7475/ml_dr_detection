"""
Two-Stage Fine-Tuning Pipeline for EfficientNet-B0.

Stage 1 (Head Warmup):
  - Backbone frozen (parameters have requires_grad=False).
  - BatchNorm layers kept in eval mode to preserve pretrained running statistics.
  - Trains only the classification head with AdamW (lr=1e-3, weight_decay=1e-4).
  - Trains for 3-5 epochs, saving the best checkpoint to:
    checkpoints/efficientnet_b0_stage1_best.pth.

Stage 2 (Full Fine-Tuning):
  - Restores best weights from Stage 1.
  - Unfreezes the entire backbone.
  - Uses AdamW with differential learning rates:
      * Backbone lr: 1e-5
      * Classification head lr: 1e-4
      * Weight decay: 1e-4
  - Linear warmup (configurable, default 3 epochs) followed by cosine decay.
  - Maximum 30 epochs with early-stopping patience of 7 (monitored on val_loss).
  - Saves best checkpoint to:
    checkpoints/efficientnet_b0_stage2_best.pth.
  - Restores best Stage 2 weights upon completion.
"""

import os
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.efficientnet import (
    build_efficientnet_b0,
    freeze_backbone,
    unfreeze_backbone,
    get_parameter_groups,
    count_parameters,
)
from src.training.trainer import Trainer, set_seed
from src.training.scheduler import get_cosine_warmup_scheduler


def train_stage1_warmup(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_dir: str = "checkpoints",
    checkpoint_filename: str = "efficientnet_b0_stage1_best.pth",
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    criterion: Optional[nn.Module] = None,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Tuple[nn.Module, Dict]:
    """
    Execute Stage 1: Classification Head Warmup.

    Args:
        model (nn.Module): EfficientNet-B0 model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        checkpoint_dir (str): Directory where checkpoints are saved.
        checkpoint_filename (str): Name of the Stage 1 checkpoint file.
        epochs (int): Number of warmup epochs (3 to 5). Default: 5.
        lr (float): Learning rate for classification head. Default: 1e-3.
        weight_decay (float): Weight decay. Default: 1e-4.
        criterion (nn.Module | None): Custom loss criterion. Default: None (CrossEntropyLoss).
        use_amp (bool): Whether to enable automatic mixed precision on CUDA. Default: True.
        device (torch.device | None): Training device.
        seed (int): Random seed. Default: 42.

    Returns:
        Tuple[nn.Module, Dict]: Trained model with best Stage 1 weights restored,
            and the training history dictionary.
    """
    print("\n" + "=" * 70)
    print("      STAGE 1: CLASSIFICATION HEAD WARMUP (FROZEN BACKBONE)      ")
    print("=" * 70)

    # 1. Ensure backbone is frozen and only classifier is trainable
    freeze_backbone(model)
    for p in model.classifier.parameters():
        p.requires_grad = True

    total_params, trainable_params = count_parameters(model)
    print(f"  Total Parameters:      {total_params:,}")
    print(f"  Trainable Parameters:  {trainable_params:,} (Head only)")
    print(f"  Frozen Parameters:     {total_params - trainable_params:,} (Backbone)")

    # 2. Setup AdamW optimizer for trainable parameters only (head)
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=lr,
        weight_decay=weight_decay,
    )

    # 3. Configure Trainer with freeze_bn=True to lock BatchNorm running stats
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        criterion=criterion,
        device=device,
        num_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=checkpoint_filename,
        early_stopping_patience=0,  # Run full warmup epochs
        seed=seed,
        stage_name="stage1_warmup",
        freeze_bn=True,  # Protect BatchNorm statistics
        use_amp=use_amp,
        model_config={"architecture": "efficientnet_b0", "stage": 1, "pretrained": True},
    )

    history = trainer.fit()
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_filename)
    print(f"\n[Stage 1 Complete] Best checkpoint saved and restored: {ckpt_path}")
    return model, history


def train_stage2_finetuning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    stage1_checkpoint_path: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    checkpoint_filename: str = "efficientnet_b0_stage2_best.pth",
    max_epochs: int = 30,
    warmup_epochs: int = 3,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 7,
    criterion: Optional[nn.Module] = None,
    use_amp: bool = True,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Tuple[nn.Module, Dict]:
    """
    Execute Stage 2: Full End-to-End Fine-Tuning.

    Args:
        model (nn.Module): EfficientNet-B0 model.
        train_loader (DataLoader): Training DataLoader.
        val_loader (DataLoader): Validation DataLoader.
        stage1_checkpoint_path (str | None): Path to Stage 1 checkpoint. If provided,
            loads weights before starting Stage 2.
        checkpoint_dir (str): Directory where checkpoints are saved.
        checkpoint_filename (str): Name of the Stage 2 checkpoint file.
        max_epochs (int): Maximum training epochs for Stage 2. Default: 30.
        warmup_epochs (int): Number of warmup epochs for cosine schedule. Default: 3.
        backbone_lr (float): Learning rate for backbone parameters. Default: 1e-5.
        head_lr (float): Learning rate for classification head. Default: 1e-4.
        weight_decay (float): Weight decay for both groups. Default: 1e-4.
        early_stopping_patience (int): Consecutive epochs without val_loss improvement
            before stopping. Default: 7.
        criterion (nn.Module | None): Custom loss criterion. Default: None (CrossEntropyLoss).
        use_amp (bool): Whether to enable automatic mixed precision on CUDA. Default: True.
        device (torch.device | None): Training device.
        seed (int): Random seed. Default: 42.

    Returns:
        Tuple[nn.Module, Dict]: Trained model with best Stage 2 weights restored,
            and the training history dictionary.
    """
    print("\n" + "=" * 70)
    print("      STAGE 2: FULL FINE-TUNING (DIFFERENTIAL LEARNING RATES)    ")
    print("=" * 70)

    # 1. Load Stage 1 checkpoint if specified
    if stage1_checkpoint_path is not None:
        if not os.path.isfile(stage1_checkpoint_path):
            raise FileNotFoundError(f"Stage 1 checkpoint missing at: {stage1_checkpoint_path}")
        checkpoint = torch.load(stage1_checkpoint_path, map_location=device or "cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"  Successfully loaded Stage 1 checkpoint: {stage1_checkpoint_path}")

    # 2. Unfreeze entire model (backbone + head)
    unfreeze_backbone(model)
    for p in model.classifier.parameters():
        p.requires_grad = True

    total_params, trainable_params = count_parameters(model)
    print(f"  Total Parameters:      {total_params:,}")
    print(f"  Trainable Parameters:  {trainable_params:,} (100% trainable)")

    # 3. Construct 2 parameter groups with distinct learning rates
    param_groups = get_parameter_groups(
        model=model,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
    )

    optimizer = torch.optim.AdamW(param_groups)

    # 4. Construct Warmup + Cosine Decay scheduler
    scheduler = get_cosine_warmup_scheduler(
        optimizer=optimizer,
        warmup_epochs=warmup_epochs,
        total_epochs=max_epochs,
        min_lr_ratio=0.01,
    )

    # 5. Configure Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        num_epochs=max_epochs,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=checkpoint_filename,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
        stage_name="stage2_finetune",
        freeze_bn=False,  # In Stage 2, BatchNorm adapts to retinal imagery
        use_amp=use_amp,
        model_config={
            "architecture": "efficientnet_b0",
            "stage": 2,
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "warmup_epochs": warmup_epochs,
        },
    )

    history = trainer.fit()
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_filename)
    print(f"\n[Stage 2 Complete] Best checkpoint saved and restored: {ckpt_path}")
    return model, history


def run_efficientnet_two_stage_pipeline(
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_dir: str = "checkpoints",
    stage1_filename: str = "efficientnet_b0_stage1_best.pth",
    stage2_filename: str = "efficientnet_b0_stage2_best.pth",
    stage1_epochs: int = 5,
    stage2_epochs: int = 30,
    stage2_warmup_epochs: int = 3,
    stage1_lr: float = 1e-3,
    stage2_backbone_lr: float = 1e-5,
    stage2_head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    early_stopping_patience: int = 7,
    criterion: Optional[nn.Module] = None,
    use_amp: bool = True,
    skip_stage1: bool = False,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict:
    """
    End-to-End Orchestrator:
      1. Initializes EfficientNet-B0 with ImageNet pretrained weights.
      2. Executes Stage 1 (Head Warmup) unless skip_stage1 is True and checkpoint exists.
      3. Passes best Stage 1 weights to Stage 2.
      4. Executes Stage 2 (Full Fine-Tuning).
      5. Restores best Stage 2 weights.

    Returns:
        dict: Pipeline results containing model, histories, and checkpoint paths.
    """
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Initialize model
    model = build_efficientnet_b0(num_classes=5, pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # Step 2: Stage 1 Warmup (or resume)
    stage1_ckpt_path = os.path.join(checkpoint_dir, stage1_filename)
    history_stage1 = None
    if skip_stage1 and os.path.isfile(stage1_ckpt_path):
        print(f"\n[Resume Notice] Skipping Stage 1. Reusing existing Stage 1 checkpoint: {stage1_ckpt_path}")
    else:
        model, history_stage1 = train_stage1_warmup(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=checkpoint_dir,
            checkpoint_filename=stage1_filename,
            epochs=stage1_epochs,
            lr=stage1_lr,
            weight_decay=weight_decay,
            criterion=criterion,
            use_amp=use_amp,
            device=device,
            seed=seed,
        )

    # Step 3: Stage 2 Full Fine-Tuning
    stage2_ckpt_path = os.path.join(checkpoint_dir, stage2_filename)
    model, history_stage2 = train_stage2_finetuning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        stage1_checkpoint_path=stage1_ckpt_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_filename=stage2_filename,
        max_epochs=stage2_epochs,
        warmup_epochs=stage2_warmup_epochs,
        backbone_lr=stage2_backbone_lr,
        head_lr=stage2_head_lr,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        criterion=criterion,
        use_amp=use_amp,
        device=device,
        seed=seed,
    )

    return {
        "model": model,
        "history_stage1": history_stage1,
        "history_stage2": history_stage2,
        "stage1_checkpoint_path": stage1_ckpt_path,
        "stage2_checkpoint_path": stage2_ckpt_path,
    }
