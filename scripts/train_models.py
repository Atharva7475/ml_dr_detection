"""
Turnkey Training & Multi-Architecture Benchmarking Script for Diabetic Retinopathy.

Trains candidate models with:
- Ultra-fast cached data loader (src/data/processed/images_384, ~25s/epoch)
- Differential learning rates (backbone lr=1e-5, head lr=1e-4)
- CosineAnnealingLR decay scheduler
- Damped class weights + label smoothing (prevents accuracy collapse)
- Best checkpointing monitored on validation accuracy

Usage:
    # Train only ResNet-50 (fastest test, ~6 minutes for 15 epochs):
    python scripts/train_models.py --model resnet50 --epochs 15

    # Train all 4 architectures sequentially and generate leaderboard:
    python scripts/train_models.py --model all --epochs 15

    # Rapid smoke test (1 epoch on tiny subset):
    python scripts/train_models.py --model resnet50 --smoke-test
"""

import os
import sys
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.resnet50 import build_resnet50
from src.models.efficientnet import build_efficientnet_b0
from src.models.fusion import build_fusion_model
from src.models.cnn_transformer import build_cnn_transformer
from src.training.trainer import Trainer, set_seed
from src.evaluation.evaluator import evaluate_model


class FastRetinaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, is_training: bool = False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        if is_training:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomRotation(degrees=20),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
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


def get_model_param_groups(model: nn.Module, model_name: str):
    """Partition model parameters into backbone and classification head."""
    if "resnet" in model_name.lower():
        head_names = ["fc"]
    elif "efficientnet" in model_name.lower():
        head_names = ["classifier"]
    elif "fusion" in model_name.lower():
        head_names = ["fusion_head"]
    else:
        head_names = ["classifier", "transformer", "projection", "positional_encoding"]

    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if any(h in name for h in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return backbone_params, head_params


def set_backbone_freeze(model: nn.Module, model_name: str, freeze: bool = True):
    """Freeze or unfreeze the feature backbone."""
    backbone_params, head_params = get_model_param_groups(model, model_name)
    for p in backbone_params:
        p.requires_grad = not freeze
    for p in head_params:
        p.requires_grad = True


def prepare_differential_optimizer(
    model: nn.Module,
    model_name: str,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
) -> optim.Optimizer:
    set_backbone_freeze(model, model_name, freeze=False)
    backbone_params, head_params = get_model_param_groups(model, model_name)
    param_groups = [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"},
        {"params": head_params, "lr": head_lr, "weight_decay": weight_decay, "name": "head"},
    ]
    return optim.AdamW(param_groups)


def main():
    parser = argparse.ArgumentParser(description="Train Diabetic Retinopathy Models")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "fusion", "cnn_transformer", "all"],
        help="Architecture to train (default: resnet50)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of Stage 2 fine-tuning epochs (default: 15)")
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Number of Stage 1 warmup epochs (default: 5)")
    parser.add_argument("--stage1-lr", type=float, default=1e-3, help="Stage 1 classifier warmup learning rate (default: 1e-3)")
    parser.add_argument("--no-two-stage", action="store_true", help="Disable Stage 1 warmup and run single-stage fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (default: 32)")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Backbone learning rate (default: 1e-5)")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="Classifier head learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-epoch quick sanity check")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")

    # Paths
    splits_dir = os.path.join(REPO_ROOT, "src", "data", "splits")
    processed_dir = os.path.join(REPO_ROOT, "src", "data", "processed", "images_384")
    raw_dir = os.path.join(REPO_ROOT, "src", "data", "raw", "train_images")

    data_dir = processed_dir if os.path.isdir(processed_dir) and len(os.listdir(processed_dir)) > 0 else raw_dir
    print(f"Dataset source: {data_dir}")

    df_train = pd.read_csv(os.path.join(splits_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(splits_dir, "val.csv"))

    if args.smoke_test:
        df_train = df_train.head(64)
        df_val = df_val.head(32)
        args.epochs = 1
        args.stage1_epochs = 1

    train_ds = FastRetinaDataset(df_train, data_dir, is_training=True)
    val_ds = FastRetinaDataset(df_val, data_dir, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    # Compute smoothed class weights
    labels = df_train["diagnosis"].values
    counts = np.bincount(labels, minlength=5)
    raw_weights = len(labels) / (5.0 * np.clip(counts, 1, None))
    smooth_weights = np.sqrt(raw_weights)
    smooth_weights = smooth_weights / smooth_weights.mean()
    class_weights_t = torch.tensor(smooth_weights, dtype=torch.float32).to(device)

    model_registry = {
        "resnet50": ("ResNet-50", build_resnet50),
        "efficientnet_b0": ("EfficientNet-B0", build_efficientnet_b0),
        "fusion": ("Dual-Backbone-Fusion", build_fusion_model),
        "cnn_transformer": ("CNN-Vision-Transformer", build_cnn_transformer),
    }

    selected = list(model_registry.keys()) if args.model == "all" else [args.model]
    results = {}

    for key in selected:
        name, builder = model_registry[key]
        print("\n" + "=" * 75)
        print(f"  STARTING TRAINING: {name.upper()}")
        print("=" * 75)
        model = builder(num_classes=5, pretrained=True).to(device)
        ckpt_dir = os.path.join(REPO_ROOT, "checkpoints", key)
        os.makedirs(ckpt_dir, exist_ok=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.05)

        if not args.no_two_stage:
            # -------------------------------------------------------------
            # Stage 1: Head Warmup (Backbone and BatchNorm frozen)
            # -------------------------------------------------------------
            print(f"\n  [Stage 1 Warmup] Freezing backbone and BatchNorm statistics for {args.stage1_epochs} epochs...")
            set_backbone_freeze(model, name, freeze=True)
            _, head_params = get_model_param_groups(model, name)
            opt_s1 = optim.AdamW(head_params, lr=args.stage1_lr, weight_decay=args.weight_decay)

            trainer_s1 = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=opt_s1,
                scheduler=None,
                criterion=criterion,
                device=device,
                num_epochs=args.stage1_epochs,
                checkpoint_dir=ckpt_dir,
                checkpoint_filename="stage1_best.pth",
                early_stopping_patience=0,
                stage_name="stage1_warmup",
                freeze_bn=True,
                monitor_metric="val_acc",
            )
            trainer_s1.fit()

            # -------------------------------------------------------------
            # Stage 2: Full End-to-End Fine-Tuning (Differential LR)
            # -------------------------------------------------------------
            print(f"\n  [Stage 2 Fine-Tuning] Unfreezing backbone for {args.epochs} epochs...")
            set_backbone_freeze(model, name, freeze=False)
            opt_s2 = prepare_differential_optimizer(
                model=model,
                model_name=name,
                backbone_lr=args.backbone_lr,
                head_lr=args.head_lr,
                weight_decay=args.weight_decay,
            )
            sched_s2 = optim.lr_scheduler.CosineAnnealingLR(opt_s2, T_max=args.epochs, eta_min=1e-6)

            trainer_s2 = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=opt_s2,
                scheduler=sched_s2,
                criterion=criterion,
                device=device,
                num_epochs=args.epochs,
                checkpoint_dir=ckpt_dir,
                checkpoint_filename="best_model.pth",
                early_stopping_patience=args.patience,
                stage_name="stage2_finetune",
                freeze_bn=False,
                monitor_metric="val_acc",
            )
            history = trainer_s2.fit()
        else:
            # Single-stage full fine-tuning
            optimizer = prepare_differential_optimizer(
                model=model,
                model_name=name,
                backbone_lr=args.backbone_lr,
                head_lr=args.head_lr,
                weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=device,
                num_epochs=args.epochs,
                checkpoint_dir=ckpt_dir,
                checkpoint_filename="best_model.pth",
                early_stopping_patience=args.patience,
                stage_name="full_finetune",
                freeze_bn=False,
                monitor_metric="val_acc",
            )
            history = trainer.fit()

        eval_res = evaluate_model(model, val_loader, device=device, num_classes=5, verbose=False)
        m = eval_res["metrics"]
        results[name] = m
        print(f"\n  -> [{name}] Final Val QWK: {m['quadratic_weighted_kappa']:.4f} | Accuracy: {m['accuracy']:.4f}")

    if len(results) > 1:
        print("\n" + "=" * 75)
        print("             ARCHITECTURE BENCHMARKING LEADERBOARD             ")
        print("=" * 75)
        print(f"  {'Model Name':<28} {'Accuracy':<12} {'QWK':<12} {'Macro F1':<12}")
        print("-" * 75)
        for name, m in results.items():
            print(f"  {name:<28} {m['accuracy']:>10.4f}  {m['quadratic_weighted_kappa']:>10.4f}  {m['f1_macro']:>10.4f}")
        print("=" * 75)


if __name__ == "__main__":
    import numpy as np
    main()
