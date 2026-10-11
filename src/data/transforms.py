"""
Image transformation pipelines for Diabetic Retinopathy Detection.
"""

from torchvision import transforms


def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Returns torchvision data augmentation pipeline for training.

    Args:
        img_size (int): Target height and width for resizing (default: 224).

    Returns:
        transforms.Compose: Training transformation pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Returns torchvision transformation pipeline for validation and testing.

    Args:
        img_size (int): Target height and width for resizing (default: 224).

    Returns:
        transforms.Compose: Validation/Test transformation pipeline.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transforms(img_size: int = 224) -> transforms.Compose:
    """
    Alias for get_val_transforms for evaluation on the test set.
    """
    return get_val_transforms(img_size=img_size)
