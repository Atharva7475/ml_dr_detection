"""
E1-specific APTOS evaluator.
Defaults to validation. The held-out test set is used only with --split test.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, classification_report, cohen_kappa_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from src.data.transforms_v2 import get_crop_val_transforms
from src.models.efficientnet import build_efficientnet_b0

NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
N = len(NAMES)


class RetinaEvalDataset(Dataset):
    def __init__(self, csv_path, image_dir, transform):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        if not {"id_code", "diagnosis"}.issubset(self.df.columns):
            raise ValueError(f"{csv_path} must contain id_code and diagnosis columns")
        self.image_dir, self.transform = Path(image_dir), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = self.image_dir / f"{row['id_code']}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            x = self.transform(im.convert("RGB"))
        y = int(row["diagnosis"])
        if y not in range(N):
            raise ValueError(f"Invalid label {y}: {path.name}")
        return x, y, str(row["id_code"])


def compute_metrics(y, pred, probs):
    cm = confusion_matrix(y, pred, labels=list(range(N)))
    total = int(cm.sum())
    per_class = {}
    for i, name in enumerate(NAMES):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = total - tp - fn - fp
        per_class[name] = {
            "support": int(cm[i, :].sum()),
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
        }
    try:
        auc_macro = float(roc_auc_score(y, probs, labels=list(range(N)), multi_class="ovr", average="macro"))
        auc_weighted = float(roc_auc_score(y, probs, labels=list(range(N)), multi_class="ovr", average="weighted"))
    except ValueError:
        auc_macro, auc_weighted = None, None
    return {
        "num_samples": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y, pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "recall_weighted": float(recall_score(y, pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "quadratic_weighted_kappa": float(cohen_kappa_score(y, pred, weights="quadratic")),
        "roc_auc_macro_ovr": auc_macro,
        "roc_auc_weighted_ovr": auc_weighted,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y, pred, labels=list(range(N)), target_names=NAMES,
            output_dict=True, zero_division=0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate E1 EfficientNet-B0 on APTOS.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/E1_cropped_384/stage2_best.pth")
    parser.add_argument("--image-dir", default="src/data/raw/train_images")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be >= 1 and num-workers must be >= 0")

    csv_path = ROOT / "src" / "data" / "splits" / f"{args.split}.csv"
    ckpt_path = ROOT / args.checkpoint
    image_dir = ROOT / args.image_dir
    for p in (csv_path, ckpt_path, image_dir):
        if not p.exists():
            raise FileNotFoundError(p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Split: {args.split}; device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    transform = get_crop_val_transforms(
        img_size=384, do_crop=True, preserve_aspect_ratio=True,
        interpolation="bicubic", crop_tolerance=15, crop_pad_percent=0.01,
        use_ben_graham=False,
    )
    ds = RetinaEvalDataset(csv_path, image_dir, transform)
    if len(ds) == 0:
        raise ValueError(f"Empty split: {csv_path}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = build_efficientnet_b0(num_classes=N, pretrained=False)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Checkpoint is missing model_state_dict")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    labels, predictions, probabilities, ids = [], [], [], []
    with torch.inference_mode():
        for batch_no, (x, y, image_ids) in enumerate(loader, start=1):
            x = x.to(device, non_blocking=(device.type == "cuda"))
            logits = model(x)
            if logits.ndim != 2 or logits.shape[1] != N:
                raise RuntimeError(f"Expected [batch, 5] logits, got {tuple(logits.shape)}")
            p = torch.softmax(logits.float(), dim=1)
            labels.extend(y.numpy().tolist())
            predictions.extend(p.argmax(1).cpu().numpy().tolist())
            probabilities.extend(p.cpu().numpy().tolist())
            ids.extend(image_ids)
            if batch_no == 1:
                print("Input shape:", tuple(x.shape), "| Output shape:", tuple(logits.shape))
            if batch_no % 10 == 0 or batch_no == len(loader):
                print(f"Processed {min(batch_no * args.batch_size, len(ds))}/{len(ds)}")

    y, pred, probs = np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)
    metrics = compute_metrics(y, pred, probs)
    metrics_path = ROOT / "results" / "metrics" / f"E1_{args.split}_final_metrics.json"
    pred_path = ROOT / "results" / "metrics" / f"E1_{args.split}_predictions.csv"
    cm_path = ROOT / "results" / "plots" / f"E1_{args.split}_final_confusion_matrix.png"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cm_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "experiment": "E1_cropped_384",
        "architecture": "efficientnet_b0",
        "split": args.split,
        "checkpoint": str(ckpt_path.relative_to(ROOT)),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_monitor_metric": ckpt.get("monitor_metric"),
        "preprocessing": {
            "size": [384, 384], "retinal_crop": True,
            "preserve_aspect_ratio": True, "interpolation": "bicubic",
            "ben_graham": False, "normalization": "ImageNet",
        },
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame({
        "id_code": ids, "true_diagnosis": y, "predicted_diagnosis": pred,
        **{f"probability_grade_{i}": probs[:, i] for i in range(N)},
    }).to_csv(pred_path, index=False)

    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(N), yticks=np.arange(N), xticklabels=NAMES,
           yticklabels=NAMES, xlabel="Predicted grade", ylabel="True grade",
           title=f"E1 EfficientNet-B0 | {args.split.upper()} confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for i in range(N):
        for j in range(N):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)

    print("\n=== E1 EVALUATION SUMMARY ===")
    print(f"Samples: {metrics['num_samples']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"Quadratic weighted kappa: {metrics['quadratic_weighted_kappa']:.4f}")
    print(f"Macro ROC-AUC (OvR): {metrics['roc_auc_macro_ovr']}")
    print("Per-class sensitivity / specificity:")
    for name, m in metrics["per_class"].items():
        print(f"  {name}: sensitivity={m['sensitivity']}, specificity={m['specificity']}, support={m['support']}")
    print("Saved metrics:", metrics_path)
    print("Saved predictions:", pred_path)
    print("Saved confusion matrix:", cm_path)
    if args.split == "test":
        print("NOTE: Do not use held-out test results to tune the model.")


if __name__ == "__main__":
    main()
