"""
Reusable Training Engine for Diabetic Retinopathy Classification.

Provides a model-agnostic Trainer class that handles the complete training
loop including validation, mixed-precision training, checkpointing, and
early stopping.  Designed to work with any PyTorch classification model
(ResNet50, EfficientNet-B0, DenseNet121 fusion, CNN+Transformer hybrid, etc.).

Usage:
    from src.training.trainer import Trainer

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        class_weights=class_weights,
        device=device,
        num_epochs=30,
        checkpoint_dir="checkpoints",
        early_stopping_patience=5,
    )
    history = trainer.fit()
"""

import os
import copy
import time
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed (int): Seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic behaviour (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training History
# ---------------------------------------------------------------------------

class TrainingHistory:
    """
    Container that records per-epoch training metrics.

    Tracked metrics:
        - train_loss, val_loss
        - train_accuracy, val_accuracy
        - learning_rate (primary/group-0 lr)
        - learning_rates (all parameter groups)
        - stage (training stage name)
    """

    def __init__(self) -> None:
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_accuracy: List[float] = []
        self.val_accuracy: List[float] = []
        self.learning_rate: List[float] = []
        self.learning_rates: List[Dict[str, float]] = []
        self.stage: List[str] = []

    def record(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        lrs: Optional[Dict[str, float]] = None,
        stage: Optional[str] = None,
    ) -> None:
        """Append one epoch's metrics."""
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_accuracy.append(train_acc)
        self.val_accuracy.append(val_acc)
        self.learning_rate.append(lr)
        if lrs is not None:
            self.learning_rates.append(lrs)
        if stage is not None:
            self.stage.append(stage)

    def to_dict(self) -> Dict[str, list]:
        """Return the full history as a serialisable dictionary."""
        d = {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
        }
        if self.learning_rates:
            d["learning_rates"] = self.learning_rates
        if self.stage:
            d["stage"] = self.stage
        return d


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Model-agnostic training engine for PyTorch classification models.

    Features:
        * Multi-stage fine-tuning support with custom checkpoint names
        * Frozen BatchNorm running statistics during warmup
        * Multi-parameter group learning-rate tracking
        * Automatic mixed-precision (AMP) on CUDA
        * Best-model checkpointing (lowest validation loss)
        * Early stopping based on validation loss
        * Full training history tracking

    Args:
        model (nn.Module): The classification model to train.
        train_loader (DataLoader): DataLoader for the training split.
        val_loader (DataLoader): DataLoader for the validation split.
        optimizer (torch.optim.Optimizer): Optimiser instance.
        scheduler: Optional learning-rate scheduler (must implement .step()).
        class_weights (torch.Tensor | np.ndarray | list | None):
            Per-class weights for CrossEntropyLoss. Pass None for uniform weights.
        device (torch.device | str | None):
            Device to train on. Defaults to CUDA if available, else CPU.
        num_epochs (int): Maximum number of training epochs (default: 30).
        checkpoint_dir (str): Directory to save model checkpoints
            (default: "checkpoints").
        checkpoint_filename (str): Name of the checkpoint file
            (default: "best_model.pth").
        early_stopping_patience (int): Number of epochs with no val-loss
            improvement before training stops (default: 5). Set to 0 to
            disable early stopping.
        seed (int): Random seed for reproducibility (default: 42).
        stage_name (str): Label for the current training stage (default: "stage1").
        freeze_bn (bool): If True, freeze BatchNorm running statistics during
            training to prevent covariate shift on frozen backbones (default: False).
        model_config (dict | None): Optional model configuration metadata to store in checkpoint.
        class_names (list[str] | None): Class names for mapping in checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        criterion: Optional[nn.Module] = None,
        class_weights=None,
        device=None,
        num_epochs: int = 30,
        checkpoint_dir: str = "checkpoints",
        checkpoint_filename: str = "best_model.pth",
        early_stopping_patience: int = 5,
        seed: int = 42,
        stage_name: str = "stage1",
        freeze_bn: bool = False,
        use_amp: bool = True,
        model_config: Optional[dict] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        # Reproducibility
        self.seed = seed
        set_seed(seed)

        # Device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = checkpoint_filename
        self.early_stopping_patience = early_stopping_patience
        self.stage_name = stage_name
        self.freeze_bn = freeze_bn
        self.model_config = model_config or {}
        self.class_names = class_names or [
            "0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative DR"
        ]

        # Build loss function (custom criterion or CrossEntropyLoss with optional class weights)
        if criterion is not None:
            self.criterion = criterion.to(self.device)
        else:
            if class_weights is not None:
                if not isinstance(class_weights, torch.Tensor):
                    class_weights = torch.tensor(class_weights, dtype=torch.float32)
                class_weights = class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Mixed-precision scaler (enabled only on CUDA if use_amp is True)
        self.use_amp = (self.device.type == "cuda") and use_amp
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # History & early-stopping state
        self.history = TrainingHistory()
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        self.epochs_without_improvement = 0

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-epoch training
    # ------------------------------------------------------------------

    def _train_one_epoch(self) -> Tuple[float, float]:
        """
        Run one full training epoch.
        Maintains BatchNorm layers in evaluation mode if freeze_bn is True.

        Returns:
            Tuple[float, float]: (average_train_loss, train_accuracy)
        """
        self.model.train()

        # If freeze_bn is requested, keep BatchNorm layers in eval mode
        # to prevent corrupting running statistics with frozen convolutional features
        if self.freeze_bn:
            for m in self.model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # ------------------------------------------------------------------
    # Single-epoch validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate_one_epoch(self) -> Tuple[float, float]:
        """
        Run one full validation epoch.

        Returns:
            Tuple[float, float]: (average_val_loss, val_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float) -> str:
        """
        Save the current model state with full reproducibility metadata.

        Args:
            epoch (int): Current epoch number (0-indexed).
            val_loss (float): Current validation loss.
            val_acc (float): Current validation accuracy.

        Returns:
            str: Path to the saved checkpoint file.
        """
        checkpoint = {
            "epoch": epoch + 1,
            "stage": self.stage_name,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "best_validation_loss": val_loss,
            "best_validation_accuracy": val_acc,
            "class_names": self.class_names,
            "class_to_idx": {name: idx for idx, name in enumerate(self.class_names)},
            "model_config": self.model_config,
            "training_config": {
                "num_epochs": self.num_epochs,
                "early_stopping_patience": self.early_stopping_patience,
                "learning_rates": {
                    pg.get("name", f"group_{i}"): pg["lr"]
                    for i, pg in enumerate(self.optimizer.param_groups)
                },
                "weight_decay": self.optimizer.param_groups[0].get("weight_decay", 0.0),
                "freeze_bn": self.freeze_bn,
            },
            "random_seed": self.seed,
            "training_history": self.history.to_dict(),
        }

        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def _load_best_checkpoint(self) -> None:
        """
        Restore model weights from the saved best checkpoint.
        Falls back to the in-memory best_model_state copy.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            val_loss = checkpoint.get("best_validation_loss", self.best_val_loss)
            epoch_info = checkpoint.get("epoch", "unknown")
            print(f"  Restored best model from {self.checkpoint_filename} (epoch {epoch_info}, "
                  f"val_loss={val_loss:.4f})")
        elif self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("  Restored best model from in-memory state.")
        else:
            print("  WARNING: No best model checkpoint found. Using final epoch weights.")

    def load_checkpoint_weights(
        self,
        checkpoint_path: str,
        restore_optimizer: bool = False,
        restore_scheduler: bool = False,
    ) -> dict:
        """
        Safely load weights from an existing checkpoint into self.model.
        Optionally restores optimizer and scheduler state.

        Args:
            checkpoint_path (str): Path to checkpoint file.
            restore_optimizer (bool): If True and optimizer state is present, restore optimizer.
            restore_scheduler (bool): If True and scheduler state is present, restore scheduler.

        Returns:
            dict: The loaded checkpoint dictionary.
        """
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        print(f"  Loaded model weights from: {checkpoint_path}")

        if restore_optimizer and "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("  Restored optimizer state from checkpoint.")
            except Exception as e:
                print(f"  Notice: Could not restore optimizer state ({e}). Using fresh optimizer.")

        if restore_scheduler and "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("  Restored scheduler state from checkpoint.")
            except Exception as e:
                print(f"  Notice: Could not restore scheduler state ({e}). Using fresh scheduler.")

        return checkpoint

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, List[float]]:
        """
        Execute the full training loop with validation, checkpointing,
        and early stopping.

        Returns:
            Dict[str, List[float]]: Full training history.
        """
        print("=" * 65)
        print(f"            TRAINING STARTED [{self.stage_name.upper()}]            ")
        print("=" * 65)
        print(f"  Device:              {self.device}")
        print(f"  Mixed Precision:     {self.use_amp}")
        print(f"  Max Epochs:          {self.num_epochs}")
        print(f"  Early Stop Patience: {self.early_stopping_patience}")
        print(f"  Checkpoint File:     {self.checkpoint_filename}")
        print(f"  Freeze BatchNorm:    {self.freeze_bn}")

        # Display initial learning rates
        for i, pg in enumerate(self.optimizer.param_groups):
            name = pg.get("name", f"Group {i}")
            print(f"  LR ({name}):          {pg['lr']:.2e}")
        print("=" * 65)

        training_start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()

            # --- Train ---
            train_loss, train_acc = self._train_one_epoch()

            # --- Validate ---
            val_loss, val_acc = self._validate_one_epoch()

            # --- Learning rates ---
            all_lrs = {
                pg.get("name", f"group_{i}"): pg["lr"]
                for i, pg in enumerate(self.optimizer.param_groups)
            }
            primary_lr = self.optimizer.param_groups[0]["lr"]

            # Step the scheduler (if provided)
            if self.scheduler is not None:
                # ReduceLROnPlateau requires the monitored metric
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # --- Record history ---
            self.history.record(
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                lr=primary_lr,
                lrs=all_lrs,
                stage=self.stage_name,
            )

            # --- Epoch summary ---
            epoch_elapsed = time.time() - epoch_start_time
            if len(self.optimizer.param_groups) > 1:
                lr_repr = " | ".join([f"{k}: {v:.2e}" for k, v in all_lrs.items()])
            else:
                lr_repr = f"LR: {primary_lr:.6f}"

            print(
                f"  [{self.stage_name}] Epoch [{epoch + 1:>2}/{self.num_epochs:>2}]  "
                f"Train Loss: {train_loss:.4f}  "
                f"Train Acc: {train_acc:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val Acc: {val_acc:.4f}  |  "
                f"{lr_repr}  ({epoch_elapsed:.1f}s)"
            )

            # --- Checkpointing (best model) ---
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                ckpt_path = self._save_checkpoint(epoch, val_loss, val_acc)
                self.epochs_without_improvement = 0
                print(
                    f"         -> New best {self.stage_name} model saved! "
                    f"(val_loss improved by {improvement:.4f}) -> {ckpt_path}"
                )
            else:
                self.epochs_without_improvement += 1

            # --- Early stopping ---
            if (
                self.early_stopping_patience > 0
                and self.epochs_without_improvement >= self.early_stopping_patience
            ):
                print(
                    f"\n  Early stopping triggered after {self.early_stopping_patience} "
                    f"consecutive epochs without validation loss improvement."
                )
                break

        # --- Post-training ---
        total_time = time.time() - training_start_time
        total_epochs = len(self.history.train_loss)

        print("\n" + "=" * 65)
        print(f"            TRAINING COMPLETE [{self.stage_name.upper()}]            ")
        print("=" * 65)
        print(f"  Total Epochs Run:    {total_epochs}")
        print(f"  Total Time:          {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  Best Val Loss:       {self.best_val_loss:.4f}")
        print(f"  Best Val Accuracy:   {max(self.history.val_accuracy):.4f}")
        # Restore the best model weights for this stage
        self._load_best_checkpoint()

        return self.history.to_dict()
