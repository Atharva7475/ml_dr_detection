# RetinaAI: Deep Learning-Based Diabetic Retinopathy Detection & Mobile Diagnostic Platform

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Flutter](https://img.shields.io/badge/Flutter-3.x-02569B.svg)](https://flutter.dev/)
[![Tests](https://img.shields.io/badge/tests-34%2F34%20passed-brightgreen.svg)]()
[![QWK Score](https://img.shields.io/badge/Test%20QWK-0.8759-success.svg)]()
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-81.09%25-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, medical-grade deep learning diagnostic system for automated **Diabetic Retinopathy (DR)** severity grading from digital fundus photography, paired with **RetinaAI**—a modern cross-platform Flutter clinical diagnostic application.

> [!TIP]
> **Comprehensive Technical Dossier & Claude Report Generator:**  
> For the complete, exhaustive academic report, mathematical derivations, ablation analyses, and ready-made prompt templates for Claude to generate publication-grade papers or thesis chapters, refer to [docs/COMPREHENSIVE_PROJECT_REPORT_DOSSIER.md](docs/COMPREHENSIVE_PROJECT_REPORT_DOSSIER.md).

---

## Table of Contents
1. [Executive Summary & Clinical Task](#clinical-motivation--task-definition)
2. [Key Engineering Innovations](#key-engineering-innovations)
3. [System Architecture](#system-architecture)
4. [Multi-Architecture Benchmark (Test Set)](#multi-architecture-benchmark-test-set)
5. [Controlled Ablation Matrix (E0–E4)](#controlled-ablation-matrix-e0e4)
6. [Repository Directory Structure](#repository-directory-structure)
7. [Installation & Setup](#installation--setup)
8. [Dataset Preparation & High-Throughput Preprocessing](#dataset-preparation--preprocessing)
9. [Model Training & Two-Stage Protocol](#model-training--two-stage-protocol)
10. [Evaluation & Single-Image Diagnostic Inference](#evaluation--single-image-inference)
11. [RetinaAI Flutter Mobile Application](#retinaai-flutter-mobile-application)
12. [Visual Output Plots & Diagnostic Figures](#visual-output-plots--diagnostic-figures)
13. [Automated Test Suite](#automated-test-suite)
14. [License & Acknowledgments](#license--acknowledgments)

---

## Clinical Motivation & Task Definition

Diabetic Retinopathy is one of the leading causes of preventable blindness among working-age adults worldwide. Early detection via regular fundus screening significantly reduces the risk of severe vision loss by over **90%**.

This system automates grading according to the **International Clinical Diabetic Retinopathy (ICDR)** severity scale (5 distinct clinical stages):

| Class | Clinical Severity | Characteristic Retinal Pathology | Recommended Clinical Urgency |
| :---: | :--- | :--- | :--- |
| **0** | **No DR** | Normal healthy retina; complete absence of microvascular lesions. | Routine annual screening. |
| **1** | **Mild NPDR** | Presence of **microaneurysms only** (tiny focal saccular capillary dilatations). | Re-screen in 6–12 months. |
| **2** | **Moderate NPDR** | More than microaneurysms: blot/dot hemorrhages, hard exudates, and cotton wool spots. | Retinal specialist referral in 3–6 months. |
| **3** | **Severe NPDR** | **4-2-1 Rule:** >20 hemorrhages in 4 quadrants, venous beading in 2+, or IRMA in 1+. | Urgent specialist referral in 2–4 weeks. |
| **4** | **Proliferative DR (PDR)** | **Neovascularization** (fragile new vessels) and/or vitreous/preretinal hemorrhage. | Sight-threatening emergency (24–48 hours). |

**Primary Benchmark Dataset:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (3,662 high-resolution color fundus photographs collected from rural clinics across India under uncontrolled field conditions).

---

## Key Engineering Innovations

### 1. Robust Retinal Auto-Cropping (`src/data/retinal_crop.py`)
Fundus cameras produce varying fields of view surrounded by large, non-informative black borders (consuming ~40% of the pixel budget). Our automated preprocessor uses Otsu contour detection, morphological closing, and quadratic aspect-ratio preservation to crop out border margins, magnifying microaneurysms by $1.4\times$ to $1.8\times$.

### 2. Ben Graham's Color Constancy & CLAHE (`src/data/ben_graham.py`)
Standardizes multi-center illumination and exposure variations:
- **Local Average Subtraction:** $I_{\text{norm}} = 4 \times I - 4 \times \text{GaussianBlur}(I, \sigma=10) + 128$
- **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Applied to the CIELAB luminance channel to accentuate subtle microaneurysms.
- **Circular Border Masking:** Clamps perimeter radii to eliminate border subtraction halos.

### 3. High-Throughput Caching Pipeline (`src/data/dataloader_v2.py`)
On-the-fly preprocessing of 3,662 high-resolution raw images creates a CPU bottleneck (~2.5–3 minutes per epoch). Our pipeline caches preprocessed images at $384 \times 384$ in `src/data/processed/images_384/`, accelerating GPU training throughput to **~15–20 seconds per epoch** (~**10x speedup**).

### 4. True Two-Stage Transfer Learning (`src/training/trainer.py`, `src/models/efficientnet.py`)
- **Stage 1 (Head Warmup):** The pre-trained convolutional backbone and all Batch Normalization layers are strictly frozen in evaluation mode (`eval()`). Only the classification head is trained at $\text{LR} = 10^{-4}$ to prevent catastrophic forgetting.
- **Stage 2 (Differential Fine-Tuning):** The entire backbone is unfrozen with differential learning rates ($\text{LR}_{\text{backbone}} = 10^{-5}$ vs $\text{LR}_{\text{head}} = 10^{-4}$) under a Cosine Annealing learning rate schedule.

### 5. Multi-Architecture Exploration
Rigorously benchmarked ResNet-50, EfficientNet-B0, Dual-Backbone Feature Fusion, and a Hybrid CNN + Vision Transformer.

---

## System Architecture

```
                                  [ Raw Fundus Image ]
                                           │
                                           ▼
                          [ Circular Retinal Auto-Crop ]
                                (Otsu & Closing)
                                           │
                                           ▼
                        [ Ben Graham's Local Subtraction ]
                         (4*I - 4*Gaussian(I) + 128 & CLAHE)
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
            [ Cached 384x384 Pool ]                   [ On-the-Fly Loader ]
                      │                                         │
                      └────────────────────┬────────────────────┘
                                           ▼
                                [ PyTorch DataLoader ]
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
             [ Stage 1: Warmup ]                     [ Stage 2: Fine-Tuning ]
             - Backbone: FROZEN                      - Backbone: LR = 1e-5
             - BatchNorm: FROZEN (eval)              - Head: LR = 1e-4
             - Head: LR = 1e-4                       - Cosine Annealing
                      │                                         │
                      └────────────────────┬────────────────────┘
                                           ▼
                         [ Model Checkpoint & Evaluation ]
                      - Quadratic Weighted Kappa (QWK = 0.8759)
                      - Top-1 Accuracy (81.09%) | Weighted ROC-AUC (0.9629)
                                           │
                                           ▼
                     [ RetinaAI Cross-Platform Flutter App ]
                      - Real-Time Camera Viewfinder with Pupil Alignment Guide
                      - Interactive Risk Gauge & Clinical Referral Guidelines
                      - Patient Longitudinal Progression (SQLite) & PDF Reports
```

---

## Multi-Architecture Benchmark (Test Set)

Evaluated on the held-out APTOS 2019 test split ($N = 550$ images):

| Metric | ResNet-50 | EfficientNet-B0 (Base) | Dual-Backbone Fusion | CNN + Transformer | Champion (E1: Two-Stage 384px) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | 68.91% | 73.45% | 74.00% | 72.18% | **81.09%** |
| **Quadratic Weighted Kappa (QWK)** | 0.8058 | 0.7967 | 0.8203 | 0.8298 | **0.8759** |
| **Macro Precision** | 0.5473 | 0.5697 | 0.5674 | 0.5523 | **0.6529** |
| **Weighted Precision** | 0.7742 | 0.7494 | 0.7699 | 0.7539 | **0.8038** |
| **Macro Recall (Sensitivity)** | 0.5919 | 0.5776 | 0.5855 | 0.6056 | **0.6226** |
| **Weighted Recall** | 0.6891 | 0.7345 | 0.7400 | 0.7218 | **0.8109** |
| **Macro F1-Score** | 0.5225 | 0.5659 | 0.5637 | 0.5207 | **0.6331** |
| **Weighted F1-Score** | 0.6876 | 0.7364 | 0.7457 | 0.7150 | **0.8051** |
| **Macro Specificity** | 0.9277 | 0.9327 | 0.9370 | 0.9323 | **0.9712** |
| **Macro ROC-AUC (OvR)** | 0.9106 | 0.9057 | 0.9214 | 0.9070 | **0.9393** |
| **Weighted ROC-AUC (OvR)** | 0.9421 | 0.9321 | 0.9462 | 0.9330 | **0.9629** |
| **Total Parameters** | 23,518,277 | 4,013,953 | 12,145,153 | 7,509,121 | **4,013,953** |
| **Inference Latency** | 269.66 ms | **106.22 ms** | 228.04 ms | 117.05 ms | **106.22 ms** |

---

## Controlled Ablation Matrix (E0–E4)

Benchmark of all controlled ablation experiments conducted on the APTOS 2019 validation split ($N = 549$ images):

| Exp ID | Configuration | Resolution | Preprocessing Pipeline | Loss / Sampler | Val Accuracy | Macro F1 | QWK Score | Severe DR Recall (Grades 3 & 4) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E0** | `baseline_224` | $224 \times 224$ | Standard Bilinear Resize | CrossEntropy | 78.69% | 0.5877 | 0.8128 | 27.40% (20 / 73) |
| **E1** | `cropped_384` (Champion) | $384 \times 384$ | **Circular Retinal Crop** | CrossEntropy | **80.33%** | **0.6227** | **0.8521** | 35.62% (26 / 73) |
| **E2** | `cropped_384_ben_graham` | $384 \times 384$ | Crop + Ben Graham + CLAHE | CrossEntropy | 79.42% | 0.5955 | 0.8300 | 26.03% (19 / 73) |
| **E3** | `cropped_384_bg_balanced` | $384 \times 384$ | Crop + Ben Graham + CLAHE | **Weighted Random Sampler** | 74.50% | 0.5955 | 0.8263 | **43.84% (32 / 73)** |
| **E4** | `cropped_384_bg_focal_loss`| $384 \times 384$ | Crop + Ben Graham + CLAHE | **Focal Loss ($\gamma=2$)** | 79.42% | 0.6165 | 0.8333 | 30.14% (22 / 73) |

*Champion E1 Test Set Performance:* **81.09% Accuracy**, **0.8759 QWK**, **0.6331 Macro F1**, **0.9629 Weighted ROC-AUC**.

---

## Repository Directory Structure

```
ml_dr_detection/
├── .agents/                    # Autonomous agent rules & Jupyter automation skills
├── configs/                    # Experiment hyperparameters & dataset configs
├── docs/
│   └── COMPREHENSIVE_PROJECT_REPORT_DOSSIER.md  # Master technical project report & Claude prompt guide
├── notebooks/
│   └── diabetic_retinopathy_master.ipynb        # Interactive master Jupyter notebook
├── retina_ai/                  # Production Flutter cross-platform mobile application
│   ├── assets/                 # Fundus scans, clinical icons, and application branding
│   ├── lib/
│   │   ├── core/               # Color themes, medical typography, route management
│   │   ├── models/             # Diagnostic scan models and clinical report structures
│   │   ├── providers/          # Reactive state management (ChangeNotifier/Provider)
│   │   ├── screens/            # Home, Scan, Results, Patient History, and About screens
│   │   └── main.dart           # Flutter entrypoint
│   └── pubspec.yaml            # Flutter dependencies (Provider, GoRouter, FL Chart)
├── results/
│   ├── metrics/                # Verified JSON histories, validation metrics, and CSV tables
│   │   ├── controlled_experiments_leaderboard.csv
│   │   ├── final_model_comparison.csv
│   │   ├── E1_test_final_metrics.json
│   │   └── E1_val_final_metrics.json
│   └── plots/                  # Visual diagnostic report cards, curves, and confusion matrices
│       ├── clinical_diagnostic_report.png
│       ├── preprocessing_pipeline_comparison.png
│       ├── retinal_crop_comparison_grid.png
│       ├── class_distribution_breakdown.png
│       ├── model_kappa_comparison.png
│       ├── efficientnet_b0_two_stage_loss_acc.png
│       └── E1_test_final_confusion_matrix.png
├── scripts/
│   ├── check_dataset_integrity.py     # Verify file presence and label consistency
│   ├── compare_experiments.py         # Tabulate comparison metrics across architectures
│   ├── evaluate_e1.py                 # Comprehensive evaluation script for champion model
│   ├── evaluate_test_set.py           # Evaluate any saved checkpoint on test split
│   ├── inspect_hardware.py            # Check GPU VRAM, CUDA acceleration, and CPU threads
│   ├── predict.py                     # Single-image clinical diagnostic inference CLI
│   ├── preprocess_dataset.py          # Offline batch cropping & Ben Graham preprocessor
│   ├── run_experiments.py             # Controlled ablation runner (E0-E4)
│   └── train_models.py                # Universal multi-architecture training pipeline
├── src/
│   ├── data/
│   │   ├── ben_graham.py       # Gaussian blur subtraction & CLAHE filters
│   │   ├── dataloader.py       # Baseline PyTorch dataloader
│   │   ├── dataloader_v2.py    # Cached 384x384 high-throughput dataloader
│   │   ├── dataset.py          # Standard APTOS Dataset implementation
│   │   ├── retinal_crop.py     # Otsu contour segmentation and retinal cropping
│   │   └── transforms_v2.py    # Data augmentations (flips, rotations, affine)
│   ├── evaluation/
│   │   ├── evaluator.py        # Validation loop, metrics computation, and report
│   │   └── metrics.py          # QWK, ROC-AUC, Precision, Recall, and F1 calculations
│   ├── models/
│   │   ├── cnn_transformer.py  # Hybrid CNN + Vision Transformer architecture
│   │   ├── efficientnet.py     # EfficientNet-B0 with custom classifier head
│   │   ├── fusion.py           # Dual-Backbone Feature Fusion architecture
│   │   └── resnet50.py         # ResNet-50 with custom classifier head
│   └── training/
│       ├── losses.py           # Focal Loss and class-weighted CrossEntropy
│       ├── scheduler.py        # Learning rate schedulers (Cosine Annealing)
│       └── trainer.py          # Two-stage trainer with strict checkpointing & restoration
├── tests/                      # Automated test suite (34/34 tests passing)
│   ├── test_ben_graham.py
│   ├── test_losses_and_sampling.py
│   ├── test_master_pipeline.py
│   ├── test_retinal_preprocessing.py
│   └── test_two_stage_pipeline.py
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment definition
└── README.md                   # Project documentation
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Atharva7475/ml_dr_detection.git
cd ml_dr_detection
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify CUDA GPU availability:
```bash
python scripts/inspect_hardware.py
```

---

## Dataset Preparation & Preprocessing

### 1. Dataset Directory Format
Place the APTOS 2019 dataset in `src/data/raw/`:
```
src/data/raw/
├── train.csv
├── test.csv
├── train_images/
│   ├── 000c1434d8d7.png
│   └── ...
└── test_images/
```

Verify dataset integrity:
```bash
python scripts/check_dataset_integrity.py
```

### 2. High-Throughput Offline Preprocessing
Preprocess raw images into cropped $384 \times 384$ tensors:
```bash
python scripts/preprocess_dataset.py --img-size 384 --use-ben-graham --num-workers 4
```
This stores cached images in `src/data/processed/images_384/`, accelerating epoch times from 180s to **18s**.

---

## Model Training & Two-Stage Protocol

### Train Any Architecture via CLI
Train any supported architecture (`efficientnet_b0`, `resnet50`, `fusion`, or `cnn_transformer`):

```bash
# Train EfficientNet-B0 with Two-Stage Protocol on Cached 384px Data
python scripts/train_models.py \
    --model-name efficientnet_b0 \
    --img-size 384 \
    --batch-size 16 \
    --stage1-epochs 5 \
    --stage2-epochs 20 \
    --use-cached-data \
    --device cuda

# Train Dual-Backbone Feature Fusion Network
python scripts/train_models.py \
    --model-name fusion \
    --img-size 224 \
    --batch-size 16 \
    --stage1-epochs 5 \
    --stage2-epochs 15

# Train Hybrid CNN + Vision Transformer
python scripts/train_models.py \
    --model-name cnn_transformer \
    --img-size 224 \
    --batch-size 16 \
    --stage1-epochs 5 \
    --stage2-epochs 15
```

### Run Controlled Ablation Experiments (E0–E4)
Run the full controlled ablation matrix:
```bash
python scripts/run_experiments.py --experiments E0 E1 E2 E3 E4
```

---

## Evaluation & Single-Image Diagnostic Inference

### 1. Evaluate Saved Checkpoints on Held-Out Test Set
```bash
python scripts/evaluate_test_set.py \
    --checkpoint checkpoints/E1_cropped_384/stage2_best.pth \
    --model-name efficientnet_b0 \
    --img-size 384 \
    --batch-size 16
```

### 2. Single-Image Clinical Diagnostic CLI
```bash
python scripts/predict.py \
    --image-path src/data/raw/train_images/000c1434d8d7.png \
    --checkpoint checkpoints/E1_cropped_384/stage2_best.pth \
    --model-name efficientnet_b0 \
    --img-size 384 \
    --save-plot
```

Output Card:
```text
======================================================================
                 RETINAAI CLINICAL PREDICTION REPORT
======================================================================
  Image Path:       src/data/raw/train_images/000c1434d8d7.png
  Model:            efficientnet_b0 (Champion 384px)
  Predicted Grade:  Class 0 (No DR)
  Confidence:       96.42%
  Clinical Urgency: ROUTINE SCREENING (Annual follow-up recommended)
----------------------------------------------------------------------
  Probability Distribution:
    [0] No DR:            96.42%  [████████████████████]
    [1] Mild:              2.15%  [                    ]
    [2] Moderate:          1.03%  [                    ]
    [3] Severe NPDR:       0.28%  [                    ]
    [4] Proliferative DR:  0.12%  [                    ]
======================================================================
```

---

## RetinaAI Flutter Mobile Application

The `retina_ai/` directory contains a cross-platform Flutter clinical application designed for clinicians and rural healthcare workers.

### App Highlights
- **Material 3 Clinical UI:** Clean interface with light and dark mode support.
- **Camera Acquisition Viewfinder:** Semi-transparent circular reticle to ensure correct pupil and aperture alignment.
- **Diagnostic Result Card:** Interactive animated risk gauge, ICDR severity badge, and 5-class probability bars.
- **Clinical Action Guidelines:** Automated referral protocols tailored to the predicted grade (e.g., 24-hour emergency alert for Grade 4).
- **Patient History & Progression:** Local SQLite storage tracking serial patient visits over time to detect disease progression.
- **PDF Diagnostic Report Export:** Generates printable, downloadable PDF clinical summaries.

### Running the Flutter App
```bash
cd retina_ai
flutter pub get
flutter run
```

---

## Visual Output Plots & Diagnostic Figures

All diagnostic figures generated by the project are tracked in `results/plots/`:

| Visual Output | Relative File Path | Clinical Interpretation |
| :--- | :--- | :--- |
| **Master Diagnostic Report** | `results/plots/clinical_diagnostic_report.png` | Complete 4-panel diagnostic board with risk gauge, probabilities, and referral plan. |
| **Preprocessing Comparison** | `results/plots/preprocessing_pipeline_comparison.png` | Visual evolution: Raw $\to$ Retinal Crop $\to$ Ben Graham Subtraction $\to$ CLAHE. |
| **Retinal Crop 9-Grid** | `results/plots/retinal_crop_comparison_grid.png` | Multi-camera cropping robustness across 9 diverse fundus exposures. |
| **Class Distribution** | `results/plots/class_distribution_breakdown.png` | Visualizing the 49.3% healthy vs 5.3% severe NPDR class imbalance. |
| **QWK Leaderboard** | `results/plots/model_kappa_comparison.png` | Ranking of all 4 architectures on Quadratic Weighted Kappa. |
| **Accuracy Comparison** | `results/plots/model_accuracy_comparison.png` | Top-1 test accuracy comparison across all models. |
| **Two-Stage Curves** | `results/plots/efficientnet_b0_two_stage_loss_acc.png` | Stage 1 warmup vs Stage 2 differential fine-tuning convergence curves. |
| **Champion Test Matrix** | `results/plots/E1_test_final_confusion_matrix.png` | Definitive test set confusion matrix showing 98.15% normal specificity ($QWK = 0.8759$). |

---

## Automated Test Suite

The codebase features a 34-test automated verification suite covering:
- Retinal circle contour detection and bounding box safety fallbacks (`test_retinal_preprocessing.py`).
- Ben Graham local Gaussian subtraction and circular mask clamping (`test_ben_graham.py`).
- Two-stage parameter freezing and BatchNorm evaluation mode verification (`test_two_stage_pipeline.py`).
- Focal Loss numerical stability and class-weighted sampler distributions (`test_losses_and_sampling.py`).
- End-to-end forward/backward passes and metric evaluations (`test_master_pipeline.py`).

Run the complete test suite:
```bash
python -m unittest discover -s tests
```

Output:
```text
Ran 34 tests in 5.15s
OK
```

---

## License & Acknowledgments

- **License:** MIT License. See [LICENSE](LICENSE) for details.
- **Dataset:** Asia Pacific Tele-Ophthalmology Society (APTOS) 2019 Blindness Detection Challenge.
- **Preprocessing Methodology:** Inspired by Ben Graham's Kaggle Diabetic Retinopathy winning solution.
