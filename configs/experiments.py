"""
Controlled Experiment Configurations for APTOS 2019 DR Pipeline (Phases 3 & 4).

Experiments:
  - E0: Baseline Two-Stage Fine-Tuning (224x224, standard uncropped)
  - E1: Retinal Cropped 384x384 (margin removal + square padding)
  - E2: Retinal Cropped 384x384 + Ben Graham illumination normalization
  - E3: Best Preprocessing (Cropped 384 + Ben Graham) + WeightedRandomSampler
  - E4: Best Preprocessing (Cropped 384 + Ben Graham) + Focal Loss (gamma=2.0)

Design Rules:
  - Architecture: EfficientNet-B0 (strictly preserved across all runs).
  - Pretrained initialization: ImageNet (strictly preserved).
  - Train/Val Splits: src/data/splits/train.csv and val.csv (strictly preserved).
  - Two-Stage Training Schedule:
      Stage 1: Head Warmup (lr=1e-3, 5 epochs, frozen backbone, frozen BatchNorm).
      Stage 2: End-to-End Fine-Tuning (backbone lr=1e-5, head lr=1e-4, 30 max epochs,
               warmup 3 epochs + cosine decay, patience 7).
  - Checkpoint and Metric Isolation: Each experiment has an isolated checkpoint
    directory and metrics file. Baseline files are NEVER overwritten.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict
import json
import os


@dataclass
class ExperimentConfig:
    """Specification of an isolated experiment configuration."""
    experiment_id: str
    name: str
    description: str

    # Preprocessing
    img_size: int = 384
    do_crop: bool = True
    crop_tolerance: int = 15
    crop_pad_percent: float = 0.01
    preserve_aspect_ratio: bool = True
    interpolation: str = "bicubic"
    use_ben_graham: bool = False
    ben_graham_sigma: Optional[float] = None
    ben_graham_sigma_ratio: float = 1.0 / 30.0
    ben_graham_mask_background: bool = True

    # Data Loading & Sampling
    batch_size: int = 16
    num_workers: Optional[int] = None
    use_balanced_sampler: bool = False

    # Loss Function
    loss_type: str = "cross_entropy"  # 'cross_entropy' or 'focal_loss'
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None

    # Training Schedule
    stage1_epochs: int = 5
    stage2_epochs: int = 30
    stage2_warmup_epochs: int = 3
    early_stopping_patience: int = 7
    stage1_lr: float = 1e-3
    stage2_backbone_lr: float = 1e-5
    stage2_head_lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42

    # Artifact Storage Paths
    checkpoint_dir: str = ""
    stage1_filename: str = "stage1_best.pth"
    stage2_filename: str = "stage2_best.pth"
    history_filename: str = ""
    val_metrics_filename: str = ""

    def __post_init__(self):
        if not self.checkpoint_dir:
            self.checkpoint_dir = os.path.join("checkpoints", f"{self.experiment_id}_{self.name}")
        if not self.history_filename:
            self.history_filename = f"{self.experiment_id}_{self.name}_history.json"
        if not self.val_metrics_filename:
            self.val_metrics_filename = f"{self.experiment_id}_{self.name}_val_metrics.json"

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_json(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_json(cls, filepath: str) -> "ExperimentConfig":
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)


EXPERIMENTS: Dict[str, ExperimentConfig] = {
    # -------------------------------------------------------------------------
    # E0: Baseline Two-Stage Fine-Tuning (224x224, uncropped)
    # -------------------------------------------------------------------------
    "E0": ExperimentConfig(
        experiment_id="E0",
        name="baseline_224",
        description="Two-stage EfficientNet-B0 fine-tuning on uncropped 224x224 images with standard CrossEntropyLoss.",
        img_size=224,
        do_crop=False,
        preserve_aspect_ratio=False,
        interpolation="bilinear",
        use_ben_graham=False,
        use_balanced_sampler=False,
        loss_type="cross_entropy",
        batch_size=32,
    ),

    # -------------------------------------------------------------------------
    # E1: Retinal Cropped 384x384
    # -------------------------------------------------------------------------
    "E1": ExperimentConfig(
        experiment_id="E1",
        name="cropped_384",
        description="Two-stage fine-tuning with retinal margin cropping and higher 384x384 resolution.",
        img_size=384,
        do_crop=True,
        preserve_aspect_ratio=True,
        interpolation="bicubic",
        use_ben_graham=False,
        use_balanced_sampler=False,
        loss_type="cross_entropy",
        batch_size=16,
    ),

    # -------------------------------------------------------------------------
    # E2: Cropped 384x384 + Ben Graham Preprocessing
    # -------------------------------------------------------------------------
    "E2": ExperimentConfig(
        experiment_id="E2",
        name="cropped_384_ben_graham",
        description="Two-stage fine-tuning with retinal cropping (384x384) and Ben Graham Gaussian illumination normalization.",
        img_size=384,
        do_crop=True,
        preserve_aspect_ratio=True,
        interpolation="bicubic",
        use_ben_graham=True,
        ben_graham_sigma_ratio=1.0 / 30.0,
        ben_graham_mask_background=True,
        use_balanced_sampler=False,
        loss_type="cross_entropy",
        batch_size=16,
    ),

    # -------------------------------------------------------------------------
    # E3: Best Preprocessing + Balanced Sampler
    # -------------------------------------------------------------------------
    "E3": ExperimentConfig(
        experiment_id="E3",
        name="cropped_384_bg_balanced_sampler",
        description="Best preprocessing pipeline (Cropped 384 + Ben Graham) with WeightedRandomSampler on training split.",
        img_size=384,
        do_crop=True,
        preserve_aspect_ratio=True,
        interpolation="bicubic",
        use_ben_graham=True,
        ben_graham_sigma_ratio=1.0 / 30.0,
        ben_graham_mask_background=True,
        use_balanced_sampler=True,
        loss_type="cross_entropy",
        batch_size=16,
    ),

    # -------------------------------------------------------------------------
    # E4: Best Preprocessing + Focal Loss
    # -------------------------------------------------------------------------
    "E4": ExperimentConfig(
        experiment_id="E4",
        name="cropped_384_bg_focal_loss",
        description="Best preprocessing pipeline (Cropped 384 + Ben Graham) with Focal Loss (gamma=2.0) without sampling.",
        img_size=384,
        do_crop=True,
        preserve_aspect_ratio=True,
        interpolation="bicubic",
        use_ben_graham=True,
        ben_graham_sigma_ratio=1.0 / 30.0,
        ben_graham_mask_background=True,
        use_balanced_sampler=False,
        loss_type="focal_loss",
        focal_gamma=2.0,
        batch_size=16,
    ),
}


def get_experiment_config(key: str) -> ExperimentConfig:
    """Retrieve experiment config by key ('E0', 'E1', etc. or run name)."""
    clean_key = key.strip().upper()
    if clean_key in EXPERIMENTS:
        return EXPERIMENTS[clean_key]
    # Check by run name
    for exp_id, exp_cfg in EXPERIMENTS.items():
        if exp_cfg.name.lower() == key.strip().lower():
            return exp_cfg
    valid_keys = list(EXPERIMENTS.keys()) + [cfg.name for cfg in EXPERIMENTS.values()]
    raise KeyError(f"Experiment '{key}' not found. Available: {valid_keys}")
