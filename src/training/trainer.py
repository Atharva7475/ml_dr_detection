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
        - learning_rate
    """

    def __init__(self) -> None:
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_accuracy: List[float] = []
        self.val_accuracy: List[float] = []
        self.learning_rate: List[float] = []

    def record(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
    ) -> None:
        """Append one epoch's metrics."""
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_accuracy.append(train_acc)
        self.val_accuracy.append(val_acc)
        self.learning_rate.append(lr)

    def to_dict(self) -> Dict[str, List[float]]:
        """Return the full history as a serialisable dictionary."""
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "learning_rate": self.learning_rate,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Model-agnostic training engine for PyTorch classification models.

    Features:
        * Weighted CrossEntropyLoss (handles class imbalance)
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
            Per-class weights for CrossEntropyLoss.  Pass None for uniform weights.
        device (torch.device | str | None):
            Device to train on.  Defaults to CUDA if available, else CPU.
        num_epochs (int): Maximum number of training epochs (default: 30).
        checkpoint_dir (str): Directory to save model checkpoints
            (default: "checkpoints").
        early_stopping_patience (int): Number of epochs with no val-loss
            improvement before training stops (default: 5).  Set to 0 to
            disable early stopping.
        seed (int): Random seed for reproducibility (default: 42).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        class_weights=None,
        device=None,
        num_epochs: int = 30,
        checkpoint_dir: str = "checkpoints",
        early_stopping_patience: int = 5,
        seed: int = 42,
    ) -> None:
        # Reproducibility
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
        self.early_stopping_patience = early_stopping_patience

        # Build loss function with optional class weights
        if class_weights is not None:
            if not isinstance(class_weights, torch.Tensor):
                class_weights = torch.tensor(class_weights, dtype=torch.float32)
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Mixed-precision scaler (enabled only on CUDA)
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # History & early-stopping state
        self.history = TrainingHistory()
        self.best_val_loss = float("inf")
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

        Returns:
            Tuple[float, float]: (average_train_loss, train_accuracy)
        """
        self.model.train()
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

    def _save_checkpoint(self, epoch: int, val_loss: float) -> str:
        """
        Save the current model state as the best checkpoint.

        Args:
            epoch (int): Current epoch number (0-indexed).
            val_loss (float): Current validation loss.

        Returns:
            str: Path to the saved checkpoint file.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_validation_loss": val_loss,
            "training_history": self.history.to_dict(),
        }

        # Include scheduler state if available
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def _load_best_checkpoint(self) -> None:
        """
        Restore model weights from the saved best checkpoint.
        Falls back to the in-memory best_model_state copy.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  Restored best model from checkpoint (epoch {checkpoint['epoch'] + 1}, "
                  f"val_loss={checkpoint['best_validation_loss']:.4f})")
        elif self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("  Restored best model from in-memory state.")
        else:
            print("  WARNING: No best model checkpoint found. Using final epoch weights.")

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> Dict[str, List[float]]:
        """
        Execute the full training loop with validation, checkpointing,
        and early stopping.

        Returns:
            Dict[str, List[float]]: Full training history containing
                train_loss, val_loss, train_accuracy, val_accuracy,
                and learning_rate per epoch.
        """
        print("=" * 65)
        print("                    TRAINING STARTED                    ")
        print("=" * 65)
        print(f"  Device:              {self.device}")
        print(f"  Mixed Precision:     {self.use_amp}")
        print(f"  Max Epochs:          {self.num_epochs}")
        print(f"  Early Stop Patience: {self.early_stopping_patience}")
        print(f"  Checkpoint Dir:      {self.checkpoint_dir}")
        print("=" * 65)

        training_start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()

            # --- Train ---
            train_loss, train_acc = self._train_one_epoch()

            # --- Validate ---
            val_loss, val_acc = self._validate_one_epoch()

            # --- Learning rate ---
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step the scheduler (if provided)
            if self.scheduler is not None:
                # ReduceLROnPlateau requires the monitored metric
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # --- Record history ---
            self.history.record(train_loss, val_loss, train_acc, val_acc, current_lr)

            # --- Epoch summary ---
            epoch_elapsed = time.time() - epoch_start_time
            print(
                f"  Epoch [{epoch + 1:>3}/{self.num_epochs}]  "
                f"Train Loss: {train_loss:.4f}  "
                f"Train Acc: {train_acc:.4f}  |  "
                f"Val Loss: {val_loss:.4f}  "
                f"Val Acc: {val_acc:.4f}  |  "
                f"LR: {current_lr:.6f}  "
                f"({epoch_elapsed:.1f}s)"
            )

            # --- Checkpointing (best model) ---
            if val_loss < self.best_val_loss:
                improvement = self.best_val_loss - val_loss
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                ckpt_path = self._save_checkpoint(epoch, val_loss)
                self.epochs_without_improvement = 0
                print(
                    f"         ↳ New best model saved! "
                    f"(val_loss improved by {improvement:.4f}) → {ckpt_path}"
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
                    f"epochs without improvement."
                )
                break

        # --- Post-training ---
        total_time = time.time() - training_start_time
        total_epochs = len(self.history.train_loss)

        print("\n" + "=" * 65)
        print("                    TRAINING COMPLETE                   ")
        print("=" * 65)
        print(f"  Total Epochs Run:    {total_epochs}")
        print(f"  Total Time:          {total_time:.1f}s ({total_time / 60:.1f} min)")
        print(f"  Best Val Loss:       {self.best_val_loss:.4f}")
        print(f"  Best Val Accuracy:   {max(self.history.val_accuracy):.4f}")
        print("=" * 65)

        # Restore the best model weights
        self._load_best_checkpoint()

        return self.history.to_dict()
