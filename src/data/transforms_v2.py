"""
Configurable image transformation pipelines supporting retinal cropping and multi-resolution inputs (224x224 and 384x384).

Completely isolated from the baseline transforms.py to avoid altering any currently active training.
"""

from typing import Optional, Union
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.data.retinal_crop import RetinalCropTransform
from src.data.ben_graham import BenGrahamTransform


def _get_interpolation_mode(interpolation: str) -> InterpolationMode:
    interp_lower = interpolation.lower().strip()
    if interp_lower == "bicubic":
        return InterpolationMode.BICUBIC
    elif interp_lower == "bilinear":
        return InterpolationMode.BILINEAR
    elif interp_lower == "nearest":
        return InterpolationMode.NEAREST
    elif interp_lower == "lanczos":
        return InterpolationMode.LANCZOS
    else:
        return InterpolationMode.BICUBIC


def get_crop_train_transforms(
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
) -> transforms.Compose:
    """
    Returns torchvision data augmentation pipeline for training with optional retinal cropping
    and optional Ben Graham illumination normalization.

    Args:
        img_size (int): Target height and width (e.g. 224 or 384).
        do_crop (bool): If True, apply circular retinal cropping before resizing.
        preserve_aspect_ratio (bool): If True, pad cropped box to square before resizing.
        interpolation (str): Interpolation mode ('bicubic' or 'bilinear').
        crop_tolerance (int): Foreground intensity threshold for margin removal.
        crop_pad_percent (float): Margin padding around detected retina bounding box.
        use_ben_graham (bool): If True, apply Ben Graham Gaussian blur subtraction. Default: False.
        ben_graham_sigma (float | None): Explicit Gaussian filter standard deviation.
        ben_graham_sigma_ratio (float): Image size fraction for blur scale. Default: 1/30.
        ben_graham_mask_background (bool): If True, zero out pixels outside circular retina.

    Returns:
        transforms.Compose: Training transformation pipeline.
    """
    interp_mode = _get_interpolation_mode(interpolation)
    transform_list = []

    # 1. Optional Retinal Margin Cropping
    if do_crop:
        transform_list.append(
            RetinalCropTransform(
                tolerance=crop_tolerance,
                pad_percent=crop_pad_percent,
                preserve_aspect_ratio=preserve_aspect_ratio,
            )
        )

    # 2. High-Quality Resize
    transform_list.append(
        transforms.Resize(
            (img_size, img_size),
            interpolation=interp_mode,
            antialias=True,
        )
    )

    # 3. Optional Ben Graham Preprocessing (deterministic local contrast enhancement)
    if use_ben_graham:
        transform_list.append(
            BenGrahamTransform(
                sigma=ben_graham_sigma,
                sigma_ratio=ben_graham_sigma_ratio,
                mask_background=ben_graham_mask_background,
            )
        )

    # 4. Clinical Data Augmentations (Rotation, Flips, Subtle Color Jitter)
    transform_list.extend([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, interpolation=interp_mode),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(transform_list)


def get_crop_val_transforms(
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
) -> transforms.Compose:
    """
    Returns deterministic torchvision transformation pipeline for validation/testing.

    Args:
        img_size (int): Target height and width (e.g. 224 or 384).
        do_crop (bool): If True, apply circular retinal cropping before resizing.
        preserve_aspect_ratio (bool): If True, pad cropped box to square before resizing.
        interpolation (str): Interpolation mode ('bicubic' or 'bilinear').
        crop_tolerance (int): Foreground intensity threshold for margin removal.
        crop_pad_percent (float): Margin padding around detected retina bounding box.
        use_ben_graham (bool): If True, apply Ben Graham Gaussian blur subtraction. Default: False.
        ben_graham_sigma (float | None): Explicit Gaussian filter standard deviation.
        ben_graham_sigma_ratio (float): Image size fraction for blur scale. Default: 1/30.
        ben_graham_mask_background (bool): If True, zero out pixels outside circular retina.

    Returns:
        transforms.Compose: Deterministic validation transformation pipeline.
    """
    interp_mode = _get_interpolation_mode(interpolation)
    transform_list = []

    # 1. Optional Retinal Margin Cropping
    if do_crop:
        transform_list.append(
            RetinalCropTransform(
                tolerance=crop_tolerance,
                pad_percent=crop_pad_percent,
                preserve_aspect_ratio=preserve_aspect_ratio,
            )
        )

    # 2. High-Quality Resize
    transform_list.append(
        transforms.Resize(
            (img_size, img_size),
            interpolation=interp_mode,
            antialias=True,
        )
    )

    # 3. Optional Ben Graham Preprocessing (deterministic across splits)
    if use_ben_graham:
        transform_list.append(
            BenGrahamTransform(
                sigma=ben_graham_sigma,
                sigma_ratio=ben_graham_sigma_ratio,
                mask_background=ben_graham_mask_background,
            )
        )

    # 4. Tensor Conversion and Standard ImageNet Normalization
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(transform_list)


def get_crop_test_transforms(
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
) -> transforms.Compose:
    """
    Alias for get_crop_val_transforms for evaluation on the held-out test set.
    """
    return get_crop_val_transforms(
        img_size=img_size,
        do_crop=do_crop,
        preserve_aspect_ratio=preserve_aspect_ratio,
        interpolation=interpolation,
        crop_tolerance=crop_tolerance,
        crop_pad_percent=crop_pad_percent,
        use_ben_graham=use_ben_graham,
        ben_graham_sigma=ben_graham_sigma,
        ben_graham_sigma_ratio=ben_graham_sigma_ratio,
        ben_graham_mask_background=ben_graham_mask_background,
    )

