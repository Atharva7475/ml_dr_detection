"""
Evaluation & Test Set Benchmarking Script for Diabetic Retinopathy Models.

Evaluates trained model checkpoints strictly on the holdout test split (test.csv: 550 samples)
or validation split (val.csv: 549 samples) to produce:
  1. Full classification report (Accuracy, QWK, Macro F1, Precision, Recall).
  2. Per-class Sensitivity and Specificity across Grades 0-4.
  3. High-resolution Confusion Matrix heatmap saved to results/plots/.
  4. Formatted clinical leaderboard JSON in results/metrics/.

Usage:
    # Evaluate the winning CNN-Vision-Transformer on holdout test set:
    python scripts/evaluate_test_set.py --model cnn_transformer --split test

    # Evaluate all 4 trained models on holdout test set:
    python scripts/evaluate_test_set.py --model all --split test

    # Evaluate on validation set:
    python scripts/evaluate_test_set.py --model cnn_transformer --split val
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.resnet50 import build_resnet50
from src.models.efficientnet import build_efficientnet_b0
from src.models.fusion import build_fusion_model
from src.models.cnn_transformer import build_cnn_transformer
from src.evaluation.evaluator import evaluate_model
from src.evaluation.metrics import DR_CLASS_NAMES


class EvalRetinaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, torch.tensor(int(row["diagnosis"]), dtype=torch.long)


def find_checkpoint(model_key: str) -> str:
    """Finds best_model.pth across possible checkpoint folder variations."""
    key_variants = [
        model_key,
        model_key.replace("_", "-"),
        model_key.replace("_", ""),
        "resnet-50" if "resnet" in model_key else model_key,
        "efficientnet-b0" if "efficientnet" in model_key else model_key,
        "dual-backbone-fusion" if "fusion" in model_key else model_key,
        "cnn-vision-transformer" if "transformer" in model_key else model_key,
    ]
    base_dirs = [
        os.path.join(REPO_ROOT, "checkpoints"),
        os.path.join(REPO_ROOT, "notebooks", "checkpoints"),
    ]

    for base in base_dirs:
        for v in key_variants:
            candidate = os.path.join(base, v, "best_model.pth")
            if os.path.isfile(candidate):
                return candidate

    return ""


def plot_confusion_matrix(cm: np.ndarray, model_name: str, split_name: str, output_path: str):
    """Saves a publication-quality confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{model_name} Confusion Matrix ({split_name.upper()} Set)", fontsize=13, fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(DR_CLASS_NAMES))
    plt.xticks(tick_marks, DR_CLASS_NAMES, rotation=35, ha="right")
    plt.yticks(tick_marks, DR_CLASS_NAMES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(
                j, i, f"{val:,}",
                horizontalalignment="center",
                color="white" if val > thresh else "black",
                fontweight="bold"
            )

    plt.ylabel("True Clinical Grade", fontweight="bold")
    plt.xlabel("Predicted Grade", fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained DR Models on Test/Val Split")
    parser.add_argument(
        "--model",
        type=str,
        default="cnn_transformer",
        choices=["resnet50", "efficientnet_b0", "fusion", "cnn_transformer", "all"],
        help="Architecture to evaluate (default: cnn_transformer)",
    )
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nEvaluation Device: {device}")

    # Dataset Paths
    split_csv = os.path.join(REPO_ROOT, "src", "data", "splits", f"{args.split}.csv")
    processed_dir = os.path.join(REPO_ROOT, "src", "data", "processed", "images_384")
    raw_dir = os.path.join(REPO_ROOT, "src", "data", "raw", "train_images")
    data_dir = processed_dir if os.path.isdir(processed_dir) and len(os.listdir(processed_dir)) > 0 else raw_dir

    if not os.path.exists(split_csv):
        print(f"Error: Split file not found at {split_csv}")
        return

    df = pd.read_csv(split_csv)
    print(f"Loaded {args.split.upper()} split ({len(df):,} samples) from: {split_csv}")
    print(f"Image Source: {data_dir}\n")

    dataset = EvalRetinaDataset(df, data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model_registry = {
        "resnet50": ("ResNet-50", build_resnet50),
        "efficientnet_b0": ("EfficientNet-B0", build_efficientnet_b0),
        "fusion": ("Dual-Backbone-Fusion", build_fusion_model),
        "cnn_transformer": ("CNN-Vision-Transformer", build_cnn_transformer),
    }

    selected = list(model_registry.keys()) if args.model == "all" else [args.model]
    results_summary = {}

    for key in selected:
        display_name, builder = model_registry[key]
        ckpt_path = find_checkpoint(key)

        if not ckpt_path:
            print(f"[-] Checkpoint for {display_name} not found. Skipping.")
            continue

        print("=" * 80)
        print(f"  EVALUATING: {display_name} ({args.split.upper()} SET)")
        print(f"  Checkpoint: {ckpt_path}")
        print("=" * 80)

        model = builder(num_classes=5, pretrained=False).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

        eval_output = evaluate_model(model, loader, device=device, num_classes=5, verbose=True)
        metrics = eval_output["metrics"]
        results_summary[display_name] = metrics

        # Save confusion matrix plot
        plot_path = os.path.join(REPO_ROOT, "results", "plots", f"{key}_{args.split}_confusion_matrix.png")
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            plot_confusion_matrix(cm, display_name, args.split, plot_path)
            print(f"  Saved confusion matrix plot: {plot_path}")

        # Save metrics JSON
        json_path = os.path.join(REPO_ROOT, "results", "metrics", f"{key}_{args.split}_eval_metrics.json")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        # Convert non-serializable objects
        clean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                clean_metrics[k] = v.tolist()
            elif isinstance(v, (np.floating, float)):
                clean_metrics[k] = round(float(v), 4)
            elif isinstance(v, (np.integer, int)):
                clean_metrics[k] = int(v)
            else:
                clean_metrics[k] = v

        with open(json_path, "w") as f:
            json.dump(clean_metrics, f, indent=4)
        print(f"  Saved evaluation metrics JSON: {json_path}\n")

    if results_summary:
        print("=" * 85)
        print(f"            FINAL BENCHMARK SUMMARY ({args.split.upper()} SPLIT)")
        print("=" * 85)
        print(f"  {'Model Name':<28} {'Accuracy':<12} {'QWK':<12} {'Macro F1':<12} {'ROC-AUC':<12}")
        print("-" * 85)
        for name, m in results_summary.items():
            auc_str = f"{m.get('roc_auc_macro', 0.0):.4f}" if m.get('roc_auc_macro') is not None else "N/A"
            print(f"  {name:<28} {m['accuracy']:>10.4f}  {m['quadratic_weighted_kappa']:>10.4f}  {m['f1_macro']:>10.4f}  {auc_str:>10}")
        print("=" * 85)


if __name__ == "__main__":
    main()
