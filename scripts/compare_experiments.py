"""
Experiment Comparison & Metrics Summary Tool.

Scans results/metrics/ for all completed runs (E0, E1, E2, E3, E4, and baselines)
and prints a standardized comparison table of validation metrics:
  - Accuracy (%)
  - Macro F1
  - Quadratic Weighted Kappa (QWK) - Primary DR clinical metric
  - Per-Class Sensitivity (Recall for Grade 0 to Grade 4)
  - Training Duration (minutes)

Usage:
  python scripts/compare_experiments.py
"""

import os
import sys
import json
from typing import Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.experiments import EXPERIMENTS


def extract_metric_fields(data: Dict) -> Dict:
    """Safely extracts accuracy, macro_f1, qwk, per-class sensitivity, and duration."""
    acc = data.get("accuracy", 0.0)
    f1 = data.get("macro_f1", data.get("f1_macro", 0.0))
    qwk = data.get("qwk", data.get("quadratic_weighted_kappa", 0.0))
    dur = data.get("duration_minutes", 0.0)

    # Per-class sensitivity list [Grade 0..4]
    sens = [0.0] * 5
    if "sensitivity_per_class" in data and isinstance(data["sensitivity_per_class"], (list, tuple)):
        sens = [float(v) for v in data["sensitivity_per_class"][:5]]
    elif "per_class" in data and isinstance(data["per_class"], dict):
        for c in range(5):
            c_data = data["per_class"].get(str(c), data["per_class"].get(f"Class {c}", {}))
            if isinstance(c_data, dict):
                sens[c] = float(c_data.get("sensitivity", c_data.get("recall", 0.0)))

    return {
        "accuracy": float(acc),
        "macro_f1": float(f1) if f1 is not None else 0.0,
        "qwk": float(qwk) if qwk is not None else 0.0,
        "duration_minutes": float(dur),
        "sensitivity": sens,
    }


def load_all_metrics(metrics_dir: str = "results/metrics") -> Dict[str, Dict]:
    results = {}

    # 1. Original Frozen Baseline
    base_file = os.path.join(metrics_dir, "baseline_val_metrics.json")
    if os.path.exists(base_file):
        try:
            with open(base_file, "r") as f:
                results["Baseline (Frozen B0)"] = extract_metric_fields(json.load(f))
        except Exception:
            pass

    # 2. Step 1 Completed Two-Stage Fine-Tuning Run
    step1_file = os.path.join(metrics_dir, "efficientnet_b0_stage2_val_metrics.json")
    if os.path.exists(step1_file):
        try:
            with open(step1_file, "r") as f:
                results["Step 1 (Two-Stage 224px)"] = extract_metric_fields(json.load(f))
        except Exception:
            pass

    # 3. New Controlled Experiments (E0 to E4)
    for exp_id, exp_cfg in EXPERIMENTS.items():
        exp_file = os.path.join(metrics_dir, exp_cfg.val_metrics_filename)
        if os.path.exists(exp_file):
            try:
                with open(exp_file, "r") as f:
                    results[f"{exp_id}: {exp_cfg.name}"] = extract_metric_fields(json.load(f))
            except Exception:
                pass

    return results


def print_comparison_table():
    print("=" * 110)
    print("                 APTOS 2019 DR DETECTION: VALIDATION EXPERIMENT COMPARISON                 ")
    print("=" * 110)

    metrics_dir = "results/metrics"
    results = load_all_metrics(metrics_dir)

    header = (
        f"{'Experiment':<32} | {'Accuracy':<9} | {'Macro F1':<9} | {'QWK':<8} | "
        f"{'Sens 0':<6} {'Sens 1':<6} {'Sens 2':<6} {'Sens 3':<6} {'Sens 4':<6} | {'Time (m)':<8}"
    )
    print(header)
    print("-" * 110)

    best_qwk = -1.0
    best_exp = None

    for exp_label, metrics in results.items():
        acc = metrics["accuracy"]
        f1 = metrics["macro_f1"]
        qwk = metrics["qwk"]
        dur = metrics["duration_minutes"]
        sens = metrics["sensitivity"]

        row = (
            f"{exp_label:<32} | {acc * 100:>7.2f}% | {f1:>9.4f} | {qwk:>8.4f} | "
            f"{sens[0]:>6.2f} {sens[1]:>6.2f} {sens[2]:>6.2f} {sens[3]:>6.2f} {sens[4]:>6.2f} | {dur:>8.1f}"
        )
        print(row)

        if qwk > best_qwk:
            best_qwk = qwk
            best_exp = exp_label

    print("-" * 110)
    if best_exp:
        print(f"  * CURRENT VALIDATION LEADER: {best_exp} (QWK: {best_qwk:.4f})")

    # Show workstation experiments execution status
    print("\n[WORKSTATION EXPERIMENT STATUS FOR TOMORROW]:")
    for exp_id, exp_cfg in EXPERIMENTS.items():
        is_done = f"{exp_id}: {exp_cfg.name}" in results
        status_tag = "[COMPLETED]             " if is_done else "[QUEUED - Ready for GPU]"
        print(f"  {status_tag} {exp_id}: {exp_cfg.name:<32} ({exp_cfg.description})")

    print("=" * 110 + "\n")


if __name__ == "__main__":
    print_comparison_table()
