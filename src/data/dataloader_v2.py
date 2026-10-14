"""
Configurable DataLoader factory supporting retinal cropping, multi-resolution inputs (224x224 and 384x384),
Ben Graham illumination normalization, and class-balanced sampling.

Completely isolated from the baseline dataloader.py to avoid altering any currently active training.
"""

import os
from typing import Dict, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import APTOSDataset
from src.data.transforms_v2 import get_crop_train_transforms, get_crop_val_transforms


def create_balanced_sampler(
    csv_path: str,
    label_col: str = "diagnosis",
    num_classes: int = 5,
) -> Tuple[WeightedRandomSampler, torch.Tensor]:
    """
    Constructs a PyTorch WeightedRandomSampler for class imbalance compensation.

    Computes inverse-frequency weights per class:
        w_c = 1.0 / N_c
    Assigns each sample i weight:
        W_i = w_{c_i}

    Args:
        csv_path (str): Path to split CSV file (must be train.csv).
        label_col (str): Column name containing class indices [0..4]. Default: 'diagnosis'.
        num_classes (int): Total number of distinct classes. Default: 5.

    Returns:
        Tuple[WeightedRandomSampler, torch.Tensor]: The sampler instance and normalized per-class weights tensor.
    """
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found in {csv_path}")

    labels = df[label_col].values.astype(int)
    class_counts = torch.zeros(num_classes, dtype=torch.float32)
    for c in range(num_classes):
        class_counts[c] = float((labels == c).sum())

    # Protect against any empty class
    class_counts = torch.clamp(class_counts, min=1.0)
    class_weights = 1.0 / class_counts

    # Per-sample weights
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights.double(),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_weights


def get_crop_dataloaders(
    splits_dir: str,
    img_dir: str,
    batch_size: int = 32,
    num_workers: Optional[int] = None,
    img_size: int = 224,
    do_crop: bool = True,
    preserve_aspect_ratio: bool = True,
    interpolation: str = "bicubic",
    crop_tolerance: int = 15,
    crop_pad_percent: float = 0.01,
    use_ben_graham: bool = False,
    ben_graham_sigma: Optional[float] = None,
    ben_graham_sigma_ratio: float = 1.0 / 30.0,
    ben_graham_mask_background: bool = True,
    use_balanced_sampler: bool = False,
    pin_memory: Optional[bool] = None,
) -> Dict[str, DataLoader]:
    """
    Creates and returns a dictionary of PyTorch DataLoaders for train, val, and test splits
    using the new configurable cropping, resolution, illumination, and sampling pipeline.

    Args:
        splits_dir (str): Directory containing train.csv, val.csv, test.csv splits.
        img_dir (str): Directory containing raw images (e.g. src/data/raw/train_images).
        batch_size (int): Mini-batch size (e.g. 32 for 224x224, 16 for 384x384).
        num_workers (int | None): Subprocess worker count. Defaults to 0 on Windows, 2 on Linux/CUDA.
        img_size (int): Target image resolution height and width (e.g. 224 or 384).
        do_crop (bool): Whether to detect and remove black margins around retina.
        preserve_aspect_ratio (bool): Whether to pad cropped region to square before resizing.
        interpolation (str): Image resizing interpolation mode ('bicubic' or 'bilinear').
        crop_tolerance (int): Foreground threshold for retinal border detection.
        crop_pad_percent (float): Margin padding around detected retina bounding box.
        use_ben_graham (bool): If True, apply Ben Graham Gaussian blur subtraction.
        ben_graham_sigma (float | None): Explicit Gaussian filter standard deviation.
        ben_graham_sigma_ratio (float): Image size fraction for blur scale. Default: 1/30.
        ben_graham_mask_background (bool): If True, mask background outside circular retina.
        use_balanced_sampler (bool): If True, uses WeightedRandomSampler for training split only.
        pin_memory (bool | None): If True, pin memory in CPU tensors for faster CUDA transfer.

    Returns:
        Dict[str, DataLoader]: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")

    # Worker process default: Windows safe default is 0 or 2, Linux default is 2
    if num_workers is None:
        num_workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 2)

    # Pin memory default: True if CUDA is available, else False
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Auto-detect preprocessed cached images directory if available
    # Note: images_384 in processed cache contains both Retinal Cropping AND Ben Graham normalization.
    # Therefore, it should ONLY be used when both do_crop and use_ben_graham are requested.
    proc_candidate = os.path.join(os.path.dirname(img_dir), "..", "processed", f"images_{img_size}")
    if (
        os.path.isdir(proc_candidate)
        and len(os.listdir(proc_candidate)) > 0
        and do_crop
        and use_ben_graham
    ):
        active_img_dir = proc_candidate
        active_do_crop = False
        active_use_bg = False
    else:
        active_img_dir = img_dir
        active_do_crop = do_crop
        active_use_bg = use_ben_graham

    # Build torchvision pipelines
    train_transforms = get_crop_train_transforms(
        img_size=img_size,
        do_crop=active_do_crop,
        preserve_aspect_ratio=preserve_aspect_ratio,
        interpolation=interpolation,
        crop_tolerance=crop_tolerance,
        crop_pad_percent=crop_pad_percent,
        use_ben_graham=active_use_bg,
        ben_graham_sigma=ben_graham_sigma,
        ben_graham_sigma_ratio=ben_graham_sigma_ratio,
        ben_graham_mask_background=ben_graham_mask_background,
    )

    val_transforms = get_crop_val_transforms(
        img_size=img_size,
        do_crop=active_do_crop,
        preserve_aspect_ratio=preserve_aspect_ratio,
        interpolation=interpolation,
        crop_tolerance=crop_tolerance,
        crop_pad_percent=crop_pad_percent,
        use_ben_graham=active_use_bg,
        ben_graham_sigma=ben_graham_sigma,
        ben_graham_sigma_ratio=ben_graham_sigma_ratio,
        ben_graham_mask_background=ben_graham_mask_background,
    )

    train_dataset = APTOSDataset(csv_path=train_csv, img_dir=active_img_dir, transform=train_transforms)
    val_dataset = APTOSDataset(csv_path=val_csv, img_dir=active_img_dir, transform=val_transforms)
    test_dataset = APTOSDataset(csv_path=test_csv, img_dir=active_img_dir, transform=val_transforms)

    # Handle train loader sampling strategy
    if use_balanced_sampler:
        sampler, _ = create_balanced_sampler(train_csv)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,  # PyTorch requires shuffle=False when sampler is provided
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # Validation and test DataLoaders ALWAYS remain deterministic, sequential, and unweighted
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
