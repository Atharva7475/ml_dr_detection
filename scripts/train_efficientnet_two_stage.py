"""
Execution script for EfficientNet-B0 Two-Stage Fine-Tuning.

Usage:
    python -m scripts.train_efficientnet_two_stage
    # or with custom arguments:
    python -m scripts.train_efficientnet_two_stage --stage1-epochs 5 --stage2-epochs 30 --patience 7
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.dataloader import get_dataloaders
from src.training.two_stage_pipeline import run_efficientnet_two_stage_pipeline
from src.models.efficientnet import build_efficientnet_b0
from src.evaluation.evaluator import evaluate_model
from src.evaluation.metrics import DR_CLASS_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet-B0 Two-Stage Fine-Tuning")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Input resolution (default: 224)")
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Stage 1 warmup epochs (3 to 5)")
    parser.add_argument("--stage2-epochs", type=int, default=30, help="Stage 2 max epochs")
    parser.add_argument("--stage2-warmup-epochs", type=int, default=3, help="Stage 2 warmup epochs")
    parser.add_argument("--stage1-lr", type=float, default=1e-3, help="Stage 1 head learning rate")
    parser.add_argument("--stage2-backbone-lr", type=float, default=1e-5, help="Stage 2 backbone learning rate")
    parser.add_argument("--stage2-head-lr", type=float, default=1e-4, help="Stage 2 head learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience for Stage 2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=2 if os.name != 'nt' else 0, help="DataLoader workers")
    return parser.parse_args()


def plot_training_curves(history: dict, stage_name: str, loss_path: str, acc_path: str):
    """Plot and save training and validation loss and accuracy curves for a stage."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # 1. Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "s-", color="#ff7f0e", label="Validation Loss", linewidth=2)
    plt.title(f"EfficientNet-B0 {stage_name} - Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (CrossEntropy)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()

    # 2. Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], "o-", color="#2ca02c", label="Train Accuracy", linewidth=2)
    plt.plot(epochs, history["val_accuracy"], "s-", color="#d62728", label="Validation Accuracy", linewidth=2)
    plt.title(f"EfficientNet-B0 {stage_name} - Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(acc_path, dpi=300)
    plt.close()


def plot_combined_two_stage_curves(h1: dict, h2: dict, combined_path: str):
    """Plot unified two-stage training progression with stage boundary."""
    s1_len = len(h1["train_loss"])
    s2_len = len(h2["train_loss"])
    total_epochs = s1_len + s2_len
    x = list(range(1, total_epochs + 1))

    train_loss = list(h1["train_loss"]) + list(h2["train_loss"])
    val_loss = list(h1["val_loss"]) + list(h2["val_loss"])
    train_acc = list(h1["train_accuracy"]) + list(h2["train_accuracy"])
    val_acc = list(h1["val_accuracy"]) + list(h2["val_accuracy"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: Loss
    ax1.plot(x, train_loss, "o-", color="#1f77b4", label="Train Loss", linewidth=2)
    ax1.plot(x, val_loss, "s-", color="#ff7f0e", label="Val Loss", linewidth=2)
    ax1.axvline(x=s1_len + 0.5, color="gray", linestyle="--", linewidth=1.5, label="Stage Transition")
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("EfficientNet-B0 Two-Stage Fine-Tuning: Loss & Accuracy Progression", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=10)

    # Bottom: Accuracy
    ax2.plot(x, train_acc, "o-", color="#2ca02c", label="Train Accuracy", linewidth=2)
    ax2.plot(x, val_acc, "s-", color="#d62728", label="Val Accuracy", linewidth=2)
    ax2.axvline(x=s1_len + 0.5, color="gray", linestyle="--", linewidth=1.5, label="Stage Transition")
    ax2.set_xlabel("Epoch (Stage 1 Warmup -> Stage 2 Full Fine-Tuning)", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(combined_path, dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(cm: np.ndarray, class_names: list, save_path: str):
    """Plot and save confusion matrix heatmap with counts and percentages."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Validation Confusion Matrix (EfficientNet-B0 Stage 2)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        for j in range(cm.shape[1]):
            pct = (cm[i, j] / row_sum * 100) if row_sum > 0 else 0
            text = f"{cm[i, j]}\n({pct:.1f}%)"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    args = parse_args()

    project_root = os.path.abspath(args.project_root)
    splits_dir = os.path.join(project_root, "src", "data", "splits")
    raw_img_dir = os.path.join(project_root, "src", "data", "raw", "train_images")
    ckpt_dir = os.path.join(project_root, "checkpoints")
    metrics_dir = os.path.join(project_root, "results", "metrics")
    plots_dir = os.path.join(project_root, "results", "plots")

    for d in [ckpt_dir, metrics_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    print("=" * 70)
    print("      EFFICIENTNET-B0 TWO-STAGE FINE-TUNING PIPELINE RUNNER       ")
    print("=" * 70)
    print(f"  Project Root:        {project_root}")
    print(f"  Splits Dir:          {splits_dir}")
    print(f"  Raw Images Dir:      {raw_img_dir}")
    print(f"  Stage 1 Epochs:      {args.stage1_epochs} (lr={args.stage1_lr:.1e})")
    print(f"  Stage 2 Epochs:      {args.stage2_epochs} (backbone_lr={args.stage2_backbone_lr:.1e}, head_lr={args.stage2_head_lr:.1e})")
    print(f"  Early Stop Patience: {args.patience}")
    print("=" * 70)

    # 1. Load DataLoaders (Strictly Train & Val; Test split is never touched)
    dataloaders = get_dataloaders(
        splits_dir=splits_dir,
        img_dir=raw_img_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    print(f"\nDataLoaders initialized:")
    print(f"  Train: {len(train_loader.dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_loader.dataset)} samples ({len(val_loader)} batches)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Active Device: {device}")

    start_total_time = time.time()

    # 2. Execute Two-Stage Pipeline
    results = run_efficientnet_two_stage_pipeline(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=ckpt_dir,
        stage1_filename="efficientnet_b0_stage1_best.pth",
        stage2_filename="efficientnet_b0_stage2_best.pth",
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        stage2_warmup_epochs=args.stage2_warmup_epochs,
        stage1_lr=args.stage1_lr,
        stage2_backbone_lr=args.stage2_backbone_lr,
        stage2_head_lr=args.stage2_head_lr,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        device=device,
        seed=args.seed,
    )

    total_training_duration = time.time() - start_total_time

    # 3. Save Histories to CSV
    h1_path = os.path.join(metrics_dir, "efficientnet_b0_stage1_training_history.csv")
    h2_path = os.path.join(metrics_dir, "efficientnet_b0_stage2_training_history.csv")

    df_h1 = pd.DataFrame(results["history_stage1"])
    df_h1.insert(0, "epoch", range(1, len(df_h1) + 1))
    df_h1.to_csv(h1_path, index=False)

    df_h2 = pd.DataFrame(results["history_stage2"])
    df_h2.insert(0, "epoch", range(1, len(df_h2) + 1))
    df_h2.to_csv(h2_path, index=False)

    # 4. Generate & Save Training Plots
    plot_training_curves(
        results["history_stage1"],
        "Stage 1 Warmup",
        os.path.join(plots_dir, "efficientnet_b0_stage1_loss.png"),
        os.path.join(plots_dir, "efficientnet_b0_stage1_accuracy.png"),
    )
    plot_training_curves(
        results["history_stage2"],
        "Stage 2 Full Fine-Tuning",
        os.path.join(plots_dir, "efficientnet_b0_stage2_loss.png"),
        os.path.join(plots_dir, "efficientnet_b0_stage2_accuracy.png"),
    )
    plot_combined_two_stage_curves(
        results["history_stage1"],
        results["history_stage2"],
        os.path.join(plots_dir, "efficientnet_b0_two_stage_loss_acc.png"),
    )

    # 5. Evaluate Best Stage 2 Model on Validation Set (Strictly Validation, NOT Test!)
    print("\n" + "=" * 70)
    print("      EVALUATING BEST STAGE 2 MODEL ON VALIDATION SET (549 SAMPLES)      ")
    print("=" * 70)
    best_stage2_model = build_efficientnet_b0(num_classes=5, pretrained=False)
    ckpt = torch.load(results["stage2_checkpoint_path"], map_location=device, weights_only=False)
    best_stage2_model.load_state_dict(ckpt["model_state_dict"])

    val_eval_results = evaluate_model(
        model=best_stage2_model,
        dataloader=val_loader,
        device=device,
        num_classes=5,
        verbose=True,
    )

    val_metrics = val_eval_results["metrics"]

    # 6. Save Confusion Matrix Plot
    cm_path = os.path.join(plots_dir, "efficientnet_b0_stage2_confusion_matrix.png")
    plot_confusion_matrix_heatmap(
        np.array(val_metrics["confusion_matrix"]),
        DR_CLASS_NAMES,
        cm_path,
    )

    # 7. Save Validation Metrics JSON & Text Summary
    val_json_path = os.path.join(metrics_dir, "efficientnet_b0_stage2_val_metrics.json")
    with open(val_json_path, "w") as f:
        json.dump(val_metrics, f, indent=4)

    # 8. Print Executive Diagnostics Report
    h2 = results["history_stage2"]
    best_val_loss = min(h2["val_loss"])
    best_epoch_idx = h2["val_loss"].index(best_val_loss)
    best_epoch_s2 = best_epoch_idx + 1
    total_epochs_s2 = len(h2["val_loss"])
    stopping_reason = (
        f"Early stopping triggered (patience={args.patience}) at Stage 2 epoch {total_epochs_s2}"
        if total_epochs_s2 < args.stage2_epochs
        else f"Completed full schedule ({args.stage2_epochs} epochs)"
    )

    final_train_loss = h2["train_loss"][-1]
    final_val_loss = h2["val_loss"][-1]
    loss_gap = final_val_loss - final_train_loss
    overfitting_alert = "HIGH (val_loss >> train_loss)" if loss_gap > 0.4 else "MODERATE/LOW"

    min_sens_cls = np.argmin(val_metrics["sensitivity_per_class"])
    min_sens_val = val_metrics["sensitivity_per_class"][min_sens_cls]

    print("\n" + "=" * 70)
    print("               TWO-STAGE PIPELINE RUN SUMMARY & DIAGNOSTICS             ")
    print("=" * 70)
    print(f"  Stage 1 Best Checkpoint: {results['stage1_checkpoint_path']}")
    print(f"  Stage 2 Best Checkpoint: {results['stage2_checkpoint_path']}")
    print(f"  Stage 2 Best Epoch:      {best_epoch_s2} (Val Loss: {best_val_loss:.4f})")
    print(f"  Stage 2 Stopping Reason: {stopping_reason}")
    print(f"  Total Duration:          {total_training_duration:.1f}s ({total_training_duration/60:.2f} min)")
    print(f"  Validation Accuracy:     {val_metrics['accuracy']:.4f}")
    print(f"  Validation Macro F1:     {val_metrics['f1_macro']:.4f}")
    print(f"  Validation QWK:          {val_metrics['quadratic_weighted_kappa']:.4f}")
    print(f"  Overfitting Check:       {overfitting_alert} (Train Loss: {final_train_loss:.4f}, Val Loss: {final_val_loss:.4f})")
    print(f"  Minority Class Check:    Class {min_sens_cls} ({DR_CLASS_NAMES[min_sens_cls]}) sensitivity is {min_sens_val:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()

