# src/evaluation - Evaluation pipeline for Diabetic Retinopathy classification.
#
# Public API:
#   calculate_classification_metrics  - Compute all metrics from labels/predictions
#   print_metrics                     - Pretty-print a metrics dictionary
#   evaluate_model                    - Full inference + metrics on a DataLoader
#
# Modules:
#   metrics.py   - Metric computation functions (sklearn-based)
#   evaluator.py - DataLoader-level model evaluation
