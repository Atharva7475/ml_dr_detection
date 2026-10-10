"""
Model Evaluator for Diabetic Retinopathy Classification.

Runs inference on a DataLoader, collects predictions and probabilities,
and computes the full classification metrics report.

CRITICAL: This evaluator is intended for the TEST SET only, executed
after training has completely finished.  Do not use test data during
training or hyperparameter tuning.

Usage:
    from src.evaluation.evaluator import evaluate_model

    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        num_classes=5,
    )
"""

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import calculate_classification_metrics, print_metrics


def count_parameters(model: nn.Module) -> tuple:
    """
    Count total and trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The model to inspect.

    Returns:
        tuple: (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    num_classes: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a trained classification model on a DataLoader.

    Sets the model to eval mode, disables gradients, runs inference,
    collects true labels / predicted labels / class probabilities,
    computes all classification metrics, and measures inference time.

    Args:
        model (nn.Module): Trained classification model.
        dataloader (DataLoader): DataLoader for the evaluation split
            (typically the held-out test set).
        device (torch.device | None): Device for inference.
            Defaults to CUDA if available, else CPU.
        num_classes (int): Number of output classes (default: 5).
        verbose (bool): If True, print the metrics report (default: True).

    Returns:
        dict: Evaluation results containing:
            - metrics (dict): All classification metrics from
              calculate_classification_metrics.
            - total_parameters (int): Total model parameters.
            - trainable_parameters (int): Trainable model parameters.
            - inference_time_seconds (float): Wall-clock inference time.
            - num_samples (int): Number of samples evaluated.
            - y_true (np.ndarray): Ground-truth labels.
            - y_pred (np.ndarray): Predicted labels.
            - y_prob (np.ndarray): Predicted class probabilities.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_true: list = []
    all_pred: list = []
    all_prob: list = []

    # ---- Inference ----
    inference_start = time.time()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Support AMP inference on CUDA
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
            else:
                outputs = model(images)

            # Class probabilities via softmax
            probs = torch.softmax(outputs, dim=1)

            # Predicted class
            _, preds = torch.max(outputs, dim=1)

            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

    inference_time = time.time() - inference_start

    # ---- Aggregate predictions ----
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)

    # ---- Compute metrics ----
    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=num_classes,
    )

    # ---- Parameter counts ----
    total_params, trainable_params = count_parameters(model)

    # ---- Assemble full results ----
    results = {
        "metrics": metrics,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "inference_time_seconds": inference_time,
        "num_samples": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    # ---- Print report ----
    if verbose:
        print("=" * 60)
        print("                  EVALUATION REPORT                    ")
        print("=" * 60)
        print(f"  Samples Evaluated:     {len(y_true):>10,}")
        print(f"  Total Parameters:      {total_params:>10,}")
        print(f"  Trainable Parameters:  {trainable_params:>10,}")
        print(f"  Inference Time:        {inference_time:>10.2f}s")
        throughput = len(y_true) / inference_time if inference_time > 0 else 0
        print(f"  Throughput:            {throughput:>10.1f} samples/s")
        print()
        print_metrics(metrics)

    return results
