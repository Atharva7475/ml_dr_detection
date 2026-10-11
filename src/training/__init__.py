# src/training - Training engine for Diabetic Retinopathy classification models.
#
# Public API:
#   Trainer                             - Model-agnostic training loop with checkpointing & early stopping.
#   TrainingHistory                     - Container for per-epoch training metrics.
#   set_seed                            - Set random seeds for reproducibility.
#   get_cosine_warmup_scheduler         - Linear warmup + cosine decay LR scheduler.
#   train_stage1_warmup                 - Stage 1: Classification head warmup.
#   train_stage2_finetuning             - Stage 2: Full backbone fine-tuning.
#   run_efficientnet_two_stage_pipeline - Two-stage fine-tuning orchestrator.

from src.training.trainer import Trainer, TrainingHistory, set_seed
from src.training.scheduler import get_cosine_warmup_scheduler
from src.training.two_stage_pipeline import (
    train_stage1_warmup,
    train_stage2_finetuning,
    run_efficientnet_two_stage_pipeline,
)

__all__ = [
    "Trainer",
    "TrainingHistory",
    "set_seed",
    "get_cosine_warmup_scheduler",
    "train_stage1_warmup",
    "train_stage2_finetuning",
    "run_efficientnet_two_stage_pipeline",
]
