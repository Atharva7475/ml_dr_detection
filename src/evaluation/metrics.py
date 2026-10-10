"""
Classification Metrics for Diabetic Retinopathy Evaluation.

Provides reusable metric functions for multiclass classification including
accuracy, precision, recall, F1, sensitivity, specificity, ROC-AUC,
quadratic weighted kappa, and confusion matrix.

All metrics are computed from NumPy arrays / lists and return a structured
dictionary suitable for logging and comparison.

Usage:
    from src.evaluation.metrics import calculate_classification_metrics

    results = calculate_classification_metrics(y_true, y_pred, y_prob)
"""

from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Class label reference
# ---------------------------------------------------------------------------
DR_CLASS_NAMES: List[str] = [
    "0 - No DR",
    "1 - Mild",
    "2 - Moderate",
    "3 - Severe",
    "4 - Proliferative DR",
]


# ---------------------------------------------------------------------------
# Per-class specificity
# ---------------------------------------------------------------------------

def _compute_per_class_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute specificity (true-negative rate) for each class using a
    one-vs-rest formulation.

    Specificity_c = TN_c / (TN_c + FP_c)

    Args:
        y_true (np.ndarray): Ground-truth integer labels.
        y_pred (np.ndarray): Predicted integer labels.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Per-class specificity of shape (num_classes,).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    specificities = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        # True negatives: everything outside row c and column c
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        # False positives: column c total minus true positives
        fp = cm[:, c].sum() - cm[c, c]
        denom = tn + fp
        specificities[c] = tn / denom if denom > 0 else 0.0

    return specificities


# ---------------------------------------------------------------------------
# Per-class sensitivity (= recall)
# ---------------------------------------------------------------------------

def _compute_per_class_sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """
    Compute sensitivity (true-positive rate / recall) for each class.

    Sensitivity_c = TP_c / (TP_c + FN_c)

    Args:
        y_true (np.ndarray): Ground-truth integer labels.
        y_pred (np.ndarray): Predicted integer labels.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Per-class sensitivity of shape (num_classes,).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    sensitivities = np.zeros(num_classes, dtype=np.float64)

    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        denom = tp + fn
        sensitivities[c] = tp / denom if denom > 0 else 0.0

    return sensitivities


# ---------------------------------------------------------------------------
# Main metrics function
# ---------------------------------------------------------------------------

def calculate_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
    num_classes: int = 5,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth integer class labels of shape (N,).
        y_pred: Predicted integer class labels of shape (N,).
        y_prob: Predicted class probabilities of shape (N, num_classes).
            Required for ROC-AUC.  Pass None if probabilities are unavailable.
        num_classes (int): Number of classes (default: 5).
        class_names (list[str] | None): Human-readable class names.
            Defaults to DR_CLASS_NAMES.

    Returns:
        dict: Structured dictionary containing all computed metrics:
            - accuracy (float)
            - precision_macro, precision_weighted (float)
            - recall_macro, recall_weighted (float)
            - f1_macro, f1_weighted (float)
            - sensitivity_per_class (list[float])
            - sensitivity_macro (float)
            - specificity_per_class (list[float])
            - specificity_macro (float)
            - quadratic_weighted_kappa (float)
            - roc_auc_macro, roc_auc_weighted (float | None)
            - confusion_matrix (list[list[int]])
            - class_names (list[str])
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    if class_names is None:
        class_names = DR_CLASS_NAMES[:num_classes]

    labels = list(range(num_classes))

    # ---- Core sklearn metrics ----
    accuracy = float(accuracy_score(y_true, y_pred))

    precision_macro = float(precision_score(y_true, y_pred, labels=labels,
                                            average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_true, y_pred, labels=labels,
                                               average="weighted", zero_division=0))

    recall_macro = float(recall_score(y_true, y_pred, labels=labels,
                                      average="macro", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, labels=labels,
                                         average="weighted", zero_division=0))

    f1_macro = float(f1_score(y_true, y_pred, labels=labels,
                              average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, labels=labels,
                                 average="weighted", zero_division=0))

    # ---- Sensitivity (per-class and macro) ----
    sensitivity_per_class = _compute_per_class_sensitivity(y_true, y_pred, num_classes)
    sensitivity_macro = float(sensitivity_per_class.mean())

    # ---- Specificity (per-class and macro) ----
    specificity_per_class = _compute_per_class_specificity(y_true, y_pred, num_classes)
    specificity_macro = float(specificity_per_class.mean())

    # ---- Quadratic weighted kappa (uses integer labels) ----
    qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    # ---- ROC-AUC (requires class probabilities) ----
    roc_auc_macro = None
    roc_auc_weighted = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        try:
            roc_auc_macro = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=labels
            ))
            roc_auc_weighted = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted", labels=labels
            ))
        except ValueError:
            # Can occur if a class is absent from y_true
            pass

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # ---- Assemble results ----
    results = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "sensitivity_per_class": sensitivity_per_class.tolist(),
        "sensitivity_macro": sensitivity_macro,
        "specificity_per_class": specificity_per_class.tolist(),
        "specificity_macro": specificity_macro,
        "quadratic_weighted_kappa": qwk,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    return results


def print_metrics(metrics: Dict) -> None:
    """
    Pretty-print a metrics dictionary produced by calculate_classification_metrics.

    Args:
        metrics (dict): Output of calculate_classification_metrics().
    """
    print("=" * 60)
    print("              CLASSIFICATION METRICS REPORT             ")
    print("=" * 60)
    print(f"  Accuracy:                    {metrics['accuracy']:.4f}")
    print(f"  Precision (macro):           {metrics['precision_macro']:.4f}")
    print(f"  Precision (weighted):        {metrics['precision_weighted']:.4f}")
    print(f"  Recall (macro):              {metrics['recall_macro']:.4f}")
    print(f"  Recall (weighted):           {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score (macro):            {metrics['f1_macro']:.4f}")
    print(f"  F1 Score (weighted):         {metrics['f1_weighted']:.4f}")
    print(f"  Sensitivity (macro):         {metrics['sensitivity_macro']:.4f}")
    print(f"  Specificity (macro):         {metrics['specificity_macro']:.4f}")
    print(f"  Quadratic Weighted Kappa:    {metrics['quadratic_weighted_kappa']:.4f}")

    roc_macro = metrics.get("roc_auc_macro")
    roc_weighted = metrics.get("roc_auc_weighted")
    print(f"  ROC-AUC (macro):             {roc_macro:.4f}" if roc_macro is not None else
          "  ROC-AUC (macro):             N/A (no probabilities)")
    print(f"  ROC-AUC (weighted):          {roc_weighted:.4f}" if roc_weighted is not None else
          "  ROC-AUC (weighted):          N/A (no probabilities)")

    # Per-class sensitivity & specificity
    class_names = metrics.get("class_names", [])
    sens = metrics.get("sensitivity_per_class", [])
    spec = metrics.get("specificity_per_class", [])

    if sens and spec:
        print("-" * 60)
        print("  Per-Class Sensitivity & Specificity:")
        for i, name in enumerate(class_names):
            s = sens[i] if i < len(sens) else 0.0
            sp = spec[i] if i < len(spec) else 0.0
            print(f"    {name:<25}  Sens={s:.4f}   Spec={sp:.4f}")

    # Confusion matrix
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        print("-" * 60)
        print("  Confusion Matrix:")
        header = "          " + "  ".join(f"P{i}" for i in range(len(cm)))
        print(header)
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>4}" for v in row)
            print(f"    T{i}    {row_str}")

    print("=" * 60)
