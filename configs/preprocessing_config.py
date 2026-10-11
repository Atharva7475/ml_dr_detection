"""
Configuration specifications for Retinal Cropping and Multi-Resolution Preprocessing (STEP 2).

Maintains complete isolation from active training runs.
"""

import json
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class PreprocessingConfig:
    """
    Configuration parameters for retinal image preprocessing.
    """
    # Image resolution: 224 (baseline) or 384 (high resolution)
    img_size: int = 224

    # Retinal margin cropping
    do_crop: bool = True
    crop_tolerance: int = 15
    crop_pad_percent: float = 0.01
    preserve_aspect_ratio: bool = True

    # Resizing method
    interpolation: str = "bicubic"

    # Ben Graham illumination normalization (Gaussian-blur subtraction)
    use_ben_graham: bool = False
    ben_graham_sigma: Optional[float] = None
    ben_graham_sigma_ratio: float = 1.0 / 30.0
    ben_graham_mask_background: bool = True

    # Batch sizes adapted for resolution (32 for 224x224, 16 for 384x384 to fit GPU VRAM)
    batch_size: int = 32

    # Normalization constants (Standard ImageNet)
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    # Worker processes
    num_workers: int = 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["mean"] = list(self.mean)
        d["std"] = list(self.std)
        return d

    def save_json(self, json_path: str) -> None:
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, json_path: str) -> "PreprocessingConfig":
        with open(json_path, "r") as f:
            data = json.load(f)
        data["mean"] = tuple(data.get("mean", (0.485, 0.456, 0.406)))
        data["std"] = tuple(data.get("std", (0.229, 0.224, 0.225)))
        return cls(**data)


# Standard Preset Configurations
PRESETS = {
    # 1. Baseline standard (uncropped, 224x224) - for ablation comparison
    "baseline_224": PreprocessingConfig(
        img_size=224,
        do_crop=False,
        preserve_aspect_ratio=False,
        interpolation="bilinear",
        use_ben_graham=False,
        batch_size=32,
    ),
    # 2. Retinal cropped 224x224 (isolates effect of removing black margins)
    "cropped_224": PreprocessingConfig(
        img_size=224,
        do_crop=True,
        preserve_aspect_ratio=True,
        interpolation="bicubic",
        use_ben_graham=False,
        batch_size=32,
    ),
    # 3. Retinal cropped 384x384 (high resolution + cropping for fine microlesions)
    "cropped_384": PreprocessingConfig(
        img_size=384,
        do_crop=True,
        preserve_aspect_ratio=True,
        interpolation="bicubic",
        use_ben_graham=False,
        batch_size=16,  # Reduced batch size to stay safely within GPU memory
    ),
    # 4. Retinal cropped 384x384 + Ben Graham illumination normalization
    "cropped_384_ben_graham": PreprocessingConfig(
        img_size=384,
        do_crop=True,
        preserve_aspect_ratio=True,
        interpolation="bicubic",
        use_ben_graham=True,
        ben_graham_sigma_ratio=1.0 / 30.0,
        ben_graham_mask_background=True,
        batch_size=16,
    ),
}
