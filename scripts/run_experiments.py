"""
Controlled Master Experiment Runner for APTOS 2019 Diabetic Retinopathy Pipeline.

Executes experiments independently with strict isolation:
  - E0: Existing Two-Stage 224x224 Baseline
  - E1: Retinal Cropped 384x384
  - E2: Retinal Cropped 384x384 + Ben Graham Preprocessing
  - E3: Best Preprocessing + Balanced Sampler (WeightedRandomSampler)
  - E4: Best Preprocessing + Focal Loss (gamma=2.0)

Usage:
  # Run a single experiment:
  python scripts/run_experiments.py --experiment E1

  # Run all experiments sequentially:
  python scripts/run_experiments.py --experiment all

  # Run rapid smoke test (verifies end-to-end forward/backward, checkpointing, and evaluation):
  python scripts/run_experiments.py --experiment E2 --smoke-test

  # Custom batch size or worker count for specific workstation:
  python scripts/run_experiments.py --experiment E3 --batch-size 8 --num-workers 4
"""

import os
import sys
import argparse
import json
import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    sns = None
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.experiments import EXPERIMENTS, ExperimentConfig, get_experiment_config
from src.data.dataloader_v2 import get_crop_dataloaders
from src.training.losses import get_loss_criterion
from src.training.trainer import set_seed
from src.training.two_stage_pipeline import run_efficientnet_two_stage_pipeline
from src.evaluation.evaluator import evaluate_model
from torch.utils.data import DataLoader, Subset


def plot_and_save_experiment_results(
    history_s1: Optional[Dict],
    history_s2: Dict,
    val_results: Dict,
    output_dir: str,
    exp_cfg: ExperimentConfig,
) -> None:
    """Generate training curve plots and confusion matrix for an experiment run."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Training & Validation Loss curves
    s1_train_loss = history_s1.get("train_loss", []) if history_s1 else []
    s1_val_loss = history_s1.get("val_loss", []) if history_s1 else []
    s2_train_loss = history_s2.get("train_loss", [])
    s2_val_loss = history_s2.get("val_loss", [])

    total_train_loss = s1_train_loss + s2_train_loss
    total_val_loss = s1_val_loss + s2_val_loss
    epochs = list(range(1, len(total_train_loss) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, total_train_loss, "b-o", label="Train Loss", linewidth=1.8, markersize=4)
    plt.plot(epochs, total_val_loss, "r-s", label="Val Loss", linewidth=1.8, markersize=4)
    if s1_train_loss:
        plt.axvline(x=len(s1_train_loss) + 0.5, color="gray", linestyle="--", label="Stage 2 Transition")
    plt.title(f"[{exp_cfg.experiment_id}] {exp_cfg.name} - Training & Validation Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curves.png"), dpi=200)
    plt.close()

    # 2. Confusion Matrix
    cm = val_results.get("metrics", {}).get("confusion_matrix", None)
    if cm is not None:
        cm_arr = np.array(cm)
        class_names = ["0-No DR", "1-Mild", "2-Mod", "3-Severe", "4-Prolif"]
        plt.figure(figsize=(7, 6))
        if sns is not None:
            sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names)
        else:
            plt.imshow(cm_arr, cmap="Blues")
            plt.colorbar()
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    plt.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color="black")
            plt.xticks(range(len(class_names)), class_names)
            plt.yticks(range(len(class_names)), class_names)
        acc_title = val_results.get("metrics", {}).get("accuracy", 0.0)
        qwk_title = val_results.get("metrics", {}).get("qwk", val_results.get("metrics", {}).get("quadratic_weighted_kappa", 0.0))
        plt.title(f"[{exp_cfg.experiment_id}] Validation Confusion Matrix\n"
                  f"Acc: {acc_title:.2%} | QWK: {qwk_title:.4f}",
                  fontsize=11, fontweight="bold")
        plt.xlabel("Predicted Grade")
        plt.ylabel("Ground Truth Grade")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=200)
        plt.close()


def run_single_experiment(
    exp_cfg: ExperimentConfig,
    splits_dir: str = "src/data/splits",
    img_dir: str = "src/data/raw/train_images",
    results_dir: str = "results",
    batch_size_override: Optional[int] = None,
    num_workers_override: Optional[int] = None,
    device_str: Optional[str] = None,
    use_amp: bool = True,
    smoke_test: bool = False,
    resume: bool = False,
) -> Dict:
    """Execute a single experiment under controlled conditions."""
    print("\n" + "#" * 80)
    print(f"      STARTING EXPERIMENT: [{exp_cfg.experiment_id}] {exp_cfg.name.upper()}      ")
    print("#" * 80)
    print(f"  Description:         {exp_cfg.description}")
    print(f"  Input Resolution:    {exp_cfg.img_size}x{exp_cfg.img_size}")
    print(f"  Retinal Cropping:    {exp_cfg.do_crop}")
    print(f"  Ben Graham Norm:     {exp_cfg.use_ben_graham}")
    print(f"  Sampling Strategy:   {'WeightedRandomSampler (Balanced)' if exp_cfg.use_balanced_sampler else 'Uniform Random'}")
    print(f"  Loss Function:       {exp_cfg.loss_type} (gamma={exp_cfg.focal_gamma if exp_cfg.loss_type == 'focal_loss' else 'N/A'})")
    print(f"  Checkpoint Dir:      {exp_cfg.checkpoint_dir}")

    # Set device
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device:    {device} (AMP={use_amp and device.type == 'cuda'})")

    # Determine batch size & worker counts
    batch_size = batch_size_override or exp_cfg.batch_size
    num_workers = num_workers_override if num_workers_override is not None else exp_cfg.num_workers

    # Configure epochs for smoke-test vs full run
    if smoke_test:
        stage1_epochs = 1
        stage2_epochs = 1
        stage2_warmup = 0
        patience = 0
        print("  MODE:                >>> RAPID SMOKE TEST (1 Epoch per Stage on small subset) <<<")
    else:
        stage1_epochs = exp_cfg.stage1_epochs
        stage2_epochs = exp_cfg.stage2_epochs
        stage2_warmup = exp_cfg.stage2_warmup_epochs
        patience = exp_cfg.early_stopping_patience

    # 1. Build DataLoaders
    print("\n[Step 1/4] Constructing DataLoaders...")
    loaders = get_crop_dataloaders(
        splits_dir=splits_dir,
        img_dir=img_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=exp_cfg.img_size,
        do_crop=exp_cfg.do_crop,
        preserve_aspect_ratio=exp_cfg.preserve_aspect_ratio,
        interpolation=exp_cfg.interpolation,
        crop_tolerance=exp_cfg.crop_tolerance,
        crop_pad_percent=exp_cfg.crop_pad_percent,
        use_ben_graham=exp_cfg.use_ben_graham,
        ben_graham_sigma=exp_cfg.ben_graham_sigma,
        ben_graham_sigma_ratio=exp_cfg.ben_graham_sigma_ratio,
        ben_graham_mask_background=exp_cfg.ben_graham_mask_background,
        use_balanced_sampler=exp_cfg.use_balanced_sampler,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # In smoke-test mode, restrict to tiny subset for instant verification
    if smoke_test:
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(min(16, len(train_loader.dataset)))),
            batch_size=min(batch_size, 8),
            shuffle=False,
            num_workers=0,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(min(8, len(val_loader.dataset)))),
            batch_size=min(batch_size, 8),
            shuffle=False,
            num_workers=0,
        )

    # 2. Build Loss Criterion
    print("[Step 2/4] Initializing Loss Criterion...")
    criterion = get_loss_criterion(
        loss_type=exp_cfg.loss_type,
        focal_gamma=exp_cfg.focal_gamma,
        focal_alpha=exp_cfg.focal_alpha,
        device=device,
    )
    print(f"  Configured Criterion: {criterion}")

    # 3. Execute Two-Stage Pipeline
    print("[Step 3/4] Launching Two-Stage Fine-Tuning Pipeline...")
    os.makedirs(exp_cfg.checkpoint_dir, exist_ok=True)
    start_time = time.time()

    pipeline_results = run_efficientnet_two_stage_pipeline(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=exp_cfg.checkpoint_dir,
        stage1_filename=exp_cfg.stage1_filename,
        stage2_filename=exp_cfg.stage2_filename,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        stage2_warmup_epochs=stage2_warmup,
        stage1_lr=exp_cfg.stage1_lr,
        stage2_backbone_lr=exp_cfg.stage2_backbone_lr,
        stage2_head_lr=exp_cfg.stage2_head_lr,
        weight_decay=exp_cfg.weight_decay,
        early_stopping_patience=patience,
        criterion=criterion,
        use_amp=use_amp,
        skip_stage1=resume,
        device=device,
        seed=exp_cfg.seed,
    )
    duration_minutes = (time.time() - start_time) / 60.0

    # 4. Save Histories
    metrics_dir = os.path.join(results_dir, "metrics")
    plots_dir = os.path.join(results_dir, "plots", f"{exp_cfg.experiment_id}_{exp_cfg.name}")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    s1_hist = pipeline_results["history_stage1"]
    if hasattr(s1_hist, "to_dict"):
        s1_hist = s1_hist.to_dict()
    s2_hist = pipeline_results["history_stage2"]
    if hasattr(s2_hist, "to_dict"):
        s2_hist = s2_hist.to_dict()

    combined_history = {
        "experiment_id": exp_cfg.experiment_id,
        "name": exp_cfg.name,
        "config": exp_cfg.to_dict(),
        "duration_minutes": duration_minutes,
        "stage1": s1_hist,
        "stage2": s2_hist,
    }
    history_file = os.path.join(metrics_dir, exp_cfg.history_filename)
    with open(history_file, "w") as f:
        json.dump(combined_history, f, indent=4)
    print(f"\n  Saved training history to: {history_file}")

    # 5. Evaluate Best Model on Validation Split (Never Test Split)
    print("\n[Step 4/4] Computing Clinical Metrics on Validation Split...")
    best_model = pipeline_results["model"]
    val_results = evaluate_model(
        model=best_model,
        dataloader=val_loader,
        device=device,
        num_classes=5,
        verbose=True,
    )

    # Save validation metrics
    val_metrics_file = os.path.join(metrics_dir, exp_cfg.val_metrics_filename)
    metrics_dict = val_results.get("metrics", {})
    acc_val = float(metrics_dict.get("accuracy", 0.0))
    macro_f1_val = float(metrics_dict.get("macro_f1", metrics_dict.get("f1_macro", 0.0)))
    qwk_val = float(metrics_dict.get("qwk", metrics_dict.get("quadratic_weighted_kappa", 0.0)))
    sens_val = metrics_dict.get("sensitivity_per_class", [])
    if isinstance(sens_val, np.ndarray):
        sens_val = sens_val.tolist()
    cm_val = metrics_dict.get("confusion_matrix", [])
    if hasattr(cm_val, "tolist"):
        cm_val = cm_val.tolist()

    val_summary = {
        "experiment_id": exp_cfg.experiment_id,
        "name": exp_cfg.name,
        "duration_minutes": duration_minutes,
        "accuracy": acc_val,
        "macro_f1": macro_f1_val,
        "f1_macro": macro_f1_val,
        "qwk": qwk_val,
        "quadratic_weighted_kappa": qwk_val,
        "sensitivity_per_class": sens_val,
        "confusion_matrix": cm_val,
    }
    with open(val_metrics_file, "w") as f:
        json.dump(val_summary, f, indent=4)
    print(f"  Saved validation metrics to: {val_metrics_file}")

    # Generate plots
    plot_and_save_experiment_results(
        history_s1=s1_hist,
        history_s2=s2_hist,
        val_results=val_results,
        output_dir=plots_dir,
        exp_cfg=exp_cfg,
    )
    print(f"  Saved diagnostic plots to: {plots_dir}")

    print("\n" + "=" * 80)
    print(f"  EXPERIMENT [{exp_cfg.experiment_id}] COMPLETE: "
          f"Acc={val_summary['accuracy']:.2%}, Macro F1={val_summary['macro_f1']:.4f}, QWK={val_summary['qwk']:.4f}")
    print("=" * 80 + "\n")
    return val_summary


def main():
    parser = argparse.ArgumentParser(description="Master Experiment Runner for APTOS DR Detection")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["E0", "E1", "E2", "E3", "E4", "all"],
        help="Experiment to execute (E0, E1, E2, E3, E4, or all)",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch sanity check")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Worker subprocess count")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cuda:0', 'cpu')")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--resume", action="store_true", help="Skip Stage 1 if checkpoint already exists")
    parser.add_argument("--splits-dir", type=str, default="src/data/splits")
    parser.add_argument("--img-dir", type=str, default="src/data/raw/train_images")
    parser.add_argument("--results-dir", type=str, default="results")

    args = parser.parse_args()

    selected_experiments = ["E0", "E1", "E2", "E3", "E4"] if args.experiment == "all" else [args.experiment]

    all_results = {}
    for exp_id in selected_experiments:
        cfg = get_experiment_config(exp_id)
        res = run_single_experiment(
            exp_cfg=cfg,
            splits_dir=args.splits_dir,
            img_dir=args.img_dir,
            results_dir=args.results_dir,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
            device_str=args.device,
            use_amp=not args.no_amp,
            smoke_test=args.smoke_test,
            resume=args.resume,
        )
        all_results[exp_id] = res

    # Summary report
    print("\n" + "=" * 80)
    print("                   ALL REQUESTED EXPERIMENTS COMPLETE                   ")
    print("=" * 80)
    print(f"{'Exp ID':<8} {'Name':<32} {'Accuracy':<10} {'Macro F1':<10} {'QWK':<10}")
    print("-" * 80)
    for exp_id, res in all_results.items():
        cfg = get_experiment_config(exp_id)
        print(f"{exp_id:<8} {cfg.name:<32} {res['accuracy']:<10.4f} {res['macro_f1']:<10.4f} {res['qwk']:<10.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
