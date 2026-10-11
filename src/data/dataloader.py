"""
DataLoader factory functions for building PyTorch DataLoaders for train, val, and test splits.
"""

import os
from typing import Dict, Tuple
from torch.utils.data import DataLoader

from src.data.dataset import APTOSDataset
from src.data.transforms import get_train_transforms, get_val_transforms


def create_dataloader(
    csv_path: str,
    img_dir: str,
    transform,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True
) -> DataLoader:
    """
    Helper function to instantiate APTOSDataset and DataLoader.
    """
    dataset = APTOSDataset(csv_path=csv_path, img_dir=img_dir, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    return loader


def get_dataloaders(
    splits_dir: str,
    img_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    img_size: int = 224,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Creates and returns a dictionary of PyTorch DataLoaders for train, val, and test splits.

    Args:
        splits_dir (str): Directory containing train.csv, val.csv, test.csv splits.
        img_dir (str): Directory containing raw images (e.g. src/data/raw/train_images).
        batch_size (int): Mini-batch size (default: 32).
        num_workers (int): Number of worker subprocesses for data loading (default: 2).
        img_size (int): Target image resolution height and width (default: 224).
        pin_memory (bool): If True, memory pin tensors before returning (default: True).

    Returns:
        Dict[str, DataLoader]: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")

    train_transforms = get_train_transforms(img_size=img_size)
    val_transforms = get_val_transforms(img_size=img_size)

    train_loader = create_dataloader(
        csv_path=train_csv,
        img_dir=img_dir,
        transform=train_transforms,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = create_dataloader(
        csv_path=val_csv,
        img_dir=img_dir,
        transform=val_transforms,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = create_dataloader(
        csv_path=test_csv,
        img_dir=img_dir,
        transform=val_transforms,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
