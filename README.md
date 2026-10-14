# RetinaAI: Deep Learning-Based Diabetic Retinopathy Detection & Mobile Diagnostic Platform

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Flutter](https://img.shields.io/badge/Flutter-3.x-02569B.svg)](https://flutter.dev/)
[![Tests](https://img.shields.io/badge/tests-34%2F34%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end medical deep learning system for automated **Diabetic Retinopathy (DR)** severity grading from digital fundus photography, paired with **RetinaAI**—a modern cross-platform Flutter clinical diagnostic application.

---

## Table of Contents
1. [Clinical Motivation & Task Definition](#clinical-motivation--task-definition)
2. [Key Engineering Innovations](#key-engineering-innovations)
3. [System Architecture](#system-architecture)
4. [Repository Directory Structure](#repository-directory-structure)
5. [Installation & Setup](#installation--setup)
6. [Dataset Preparation & Preprocessing](#dataset-preparation--preprocessing)
7. [Model Training & Two-Stage Protocol](#model-training--two-stage-protocol)
8. [Evaluation & Single-Image Inference](#evaluation--single-image-inference)
9. [RetinaAI Flutter Mobile Application](#retinaai-flutter-mobile-application)
10. [Experimental Leaderboard & Ablations](#experimental-leaderboard--ablations)
11. [Automated Test Suite](#automated-test-suite)
12. [License & Acknowledgments](#license--acknowledgments)

---

## Clinical Motivation & Task Definition

Diabetic Retinopathy is one of the leading causes of preventable blindness among working-age adults worldwide. Early detection via regular fundus screening significantly reduces the risk of severe vision loss.

This system automates grading according to the **International Clinical Diabetic Retinopathy (ICDR)** severity scale (5 distinct clinical stages):

| Class | Clinical Severity | Characteristic Retinal Pathology |
| :---: | :--- | :--- |
| **0** | **No DR** | Normal healthy retina; absence of microvascular lesions. |
| **1** | **Mild** | Presence of microaneurysms only. |
| **2** | **Moderate** | More than microaneurysms but less than severe non-proliferative DR; exudates and cotton wool spots. |
| **3** | **Severe NPDR** | >20 intraretinal hemorrhages in each of 4 quadrants, venous beading in 2+ quadrants, or IRMA in 1+ quadrant. |
| **4** | **Proliferative DR (PDR)** | Neovascularization (new fragile blood vessels) and/or vitreous/preretinal hemorrhage. |

**Primary Benchmark Dataset:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (3,662 high-resolution color fundus photographs collected from rural clinics across India).

---

## Key Engineering Innovations

### 1. Robust Retinal Auto-Cropping (`src/data/retinal_crop.py`)
Fundus cameras produce varying fields of view surrounded by large, non-informative black borders. Our automated preprocessor uses Otsu contour detection and morphological closing to segment the circular retina, crop out border margins, and maximize the effective pixel resolution delivered to neural backbones.

### 2. Ben Graham's Color Constancy & CLAHE (`src/data/ben_graham.py`)
Fundus photographs from multi-center clinical trials suffer from severe illumination and exposure variation:
- **Local Average Subtraction:** $I_{\text{norm}} = 4 \times I - 4 \times \text{GaussianBlur}(I, \sigma=10) + 128$
- **Contrast Limited Adaptive Histogram Equalization (CLAHE):** Enhances subtle microaneurysms and faint hemorrhages without amplifying background noise.
- **Circular Border Masking:** Enforces a clean boundary radius to eliminate edge artifacts.

### 3. High-Throughput Caching Pipeline (`src/data/dataloader_v2.py`)
On-the-fly preprocessing of 3,662 high-resolution raw images creates an I/O bottleneck (~2-3 minutes per epoch). Our pipeline caches preprocessed images at $384 \times 384$ in `src/data/processed/images_384/`, accelerating GPU training throughput to **~15-25 seconds per epoch** (~8x speedup).

### 4. True Two-Stage Transfer Learning (`src/training/trainer.py`, `scripts/train_models.py`)
- **Stage 1 (Head Warmup):** The pre-trained convolutional backbone and all Batch Normalization layers are strictly frozen in evaluation mode (`eval()`). Only the multi-layer classification head is trained at a higher learning rate ($\text{LR} = 10^{-4}$) to prevent catastrophic forgetting.
- **Stage 2 (Differential Fine-Tuning):** The entire backbone is unfrozen and fine-tuned with a lower learning rate ($\text{LR}_{\text{backbone}} = 10^{-5}$) while the head continues at $\text{LR}_{\text{head}} = 10^{-4}$ under a Cosine Annealing learning rate schedule.

### 5. Multi-Architecture Support
- **EfficientNet-B0:** Compound-scaled depthwise separable convolutions for high efficiency.
- **ResNet-50:** Residual network baseline with deep bottleneck feature maps.
- **Dual-Backbone Fusion:** Parallel feature extraction concatenating complementary spatial representations.
- **CNN-Vision Transformer:** Convolutional stem combined with multi-head self-attention encoders.

### 6. Imbalance Mitigation (Weighted Sampling & Focal Loss)
Severe DR classes (Class 1, 3, 4) represent a minority of clinical cases. The framework supports `WeightedRandomSampler` and multi-class `FocalLoss` ($\gamma = 2.0$) to focus gradient updates on challenging, hard-to-classify borderline examples.

---

## System Architecture

```
                                 [ Raw Fundus Image ]
                                          │
                                          ▼
                         [ Circular Retinal Auto-Crop ]
                                          │
                                          ▼
                       [ Ben Graham's Local Subtraction ]
                                          │
                                          ▼
                             [ CLAHE & Circular Mask ]
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
                     - Quadratic Weighted Kappa (QWK)
                     - Multiclass ROC-AUC & Confusion Matrix
                     - Severity Recall (Grades 3 & 4)
                                          │
                                          ▼
                    [ RetinaAI Cross-Platform Flutter App ]
                     - Real-Time Camera / Gallery Classification
                     - Interactive Risk Gauge & Clinical Reports
                     - Patient History & Progression Tracking
```

---

## Repository Directory Structure

```
ml_dr_detection/
├── .agents/                    # Autonomous agent rules & Jupyter automation skills
├── notebooks/
│   └── diabetic_retinopathy_master.ipynb  # Comprehensive interactive master notebook
├── retina_ai/                  # Flutter cross-platform mobile application
│   ├── assets/                 # App icons, clinical imagery, and sample fundus scans
│   ├── lib/
│   │   ├── core/               # App theme (colors, typography, dark/light modes)
│   │   ├── models/             # Diagnostic scan models & data classes
│   │   ├── providers/          # State management (ChangeNotifier/Provider)
│   │   ├── screens/            # Home, Scan, Results, History, and About screens
│   │   └── main.dart           # Flutter application entrypoint
│   └── pubspec.yaml            # Flutter dependencies (GoRouter, Provider, FL Chart)
├── results/
│   ├── metrics/
│   │   └── controlled_experiments_leaderboard.csv  # Verified experimental results
│   └── plots/                  # Diagnostic plots, confusion matrices, and comparisons
├── scripts/
│   ├── check_dataset_integrity.py     # Verify file presence and label consistency
│   ├── compare_experiments.py         # Generate comparison metrics across models
│   ├── evaluate_test_set.py           # Evaluate saved checkpoints on test split
│   ├── inspect_hardware.py            # Check CUDA GPU, VRAM, and CPU threads
│   ├── predict.py                     # Single-image diagnostic CLI with probabilities
│   ├── preprocess_dataset.py          # Batch offline cropping & Ben Graham preprocessing
│   ├── run_experiments.py             # Controlled ablation runner (E0-E4)
│   ├── train_efficientnet_two_stage.py# Standalone two-stage EfficientNet runner
│   └── train_models.py                # Universal multi-architecture training pipeline
├── src/
│   ├── data/
│   │   ├── ben_graham.py       # Gaussian blur subtraction & CLAHE filter
│   │   ├── dataloader.py       # Baseline dataloader
│   │   ├── dataloader_v2.py    # Cached 384x384 high-throughput dataloader
│   │   ├── dataset.py          # PyTorch Dataset implementation
│   │   ├── retinal_crop.py     # Contour segmentation and retinal cropping
│   │   └── transforms_v2.py    # Augmentation pipelines (flips, rotations, affine)
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
│       ├── scheduler.py        # Learning rate schedulers (Cosine, StepLR)
│       └── trainer.py          # Base trainer with checkpointing & early stopping
├── tests/
│   ├── test_ben_graham.py             # Unit tests for Ben Graham preprocessing
│   ├── test_losses_and_sampling.py    # Unit tests for Focal Loss & sampler
│   ├── test_master_pipeline.py        # End-to-end integration tests
│   ├── test_retinal_preprocessing.py  # Unit tests for retinal cropping
│   └── test_two_stage_pipeline.py     # Verification tests for frozen layers
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Atharva7475/ml_dr_detection.git
cd ml_dr_detection
```

### 2. Create and Activate a Python Virtual Environment
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

Verify GPU availability:
```bash
python scripts/inspect_hardware.py
```

---

## Dataset Preparation & Preprocessing

### 1. Download the APTOS 2019 Dataset
Place the dataset inside `src/data/raw/`:
```
src/data/raw/
├── train.csv
├── test.csv
├── train_images/
│   ├── 000c1434d8d7.png
│   ├── 001639a390f0.png
│   └── ...
└── test_images/
```

Verify dataset integrity:
```bash
python scripts/check_dataset_integrity.py
```

### 2. Run High-Throughput Offline Preprocessing
Preprocess raw fundus images into cropped $384 \times 384$ images with Ben Graham color enhancement:
```bash
python scripts/preprocess_dataset.py --img-size 384 --use-ben-graham --num-workers 4
```
This generates preprocessed images in `src/data/processed/images_384/`, enabling high-speed training.

---

## Model Training & Two-Stage Protocol

### Train Any Architecture via CLI
Train any supported architecture (`efficientnet_b0`, `resnet50`, `fusion`, or `cnn_transformer`):

```bash
# Train EfficientNet-B0 with Two-Stage Protocol on Cached 384px data
python scripts/train_models.py \
    --model-name efficientnet_b0 \
    --img-size 384 \
    --batch-size 16 \
    --stage1-epochs 5 \
    --stage2-epochs 15 \
    --use-cached-data \
    --device cuda

# Train ResNet-50
python scripts/train_models.py \
    --model-name resnet50 \
    --img-size 224 \
    --batch-size 32 \
    --stage1-epochs 5 \
    --stage2-epochs 15
```

### Run Controlled Ablation Experiments (E0–E4)
Run the full ablation matrix across resolutions, cropping, Ben Graham filtering, sampling, and loss functions:
```bash
python scripts/run_experiments.py --experiments E0 E1 E2 E3 E4
```

---

## Evaluation & Single-Image Inference

### 1. Evaluate Saved Checkpoints on the Test Set
Compute accuracy, Quadratic Weighted Kappa (QWK), ROC-AUC, macro F1, and per-class recall:
```bash
python scripts/evaluate_test_set.py \
    --checkpoint checkpoints/best_model_efficientnet_b0.pth \
    --model-name efficientnet_b0 \
    --img-size 384 \
    --batch-size 16 \
    --output-dir results/evaluation_run
```

### 2. Single-Image Diagnostic Inference
Classify a single retinal fundus image and view confidence distributions and clinical action recommendations:
```bash
python scripts/predict.py \
    --image-path src/data/raw/train_images/000c1434d8d7.png \
    --checkpoint checkpoints/best_model_efficientnet_b0.pth \
    --model-name efficientnet_b0 \
    --img-size 384
```

Output:
```text
======================================================================
                 RETINAAI CLINICAL PREDICTION REPORT
======================================================================
  Image Path:       src/data/raw/train_images/000c1434d8d7.png
  Model:            efficientnet_b0
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

The `retina_ai/` directory contains a production-ready Flutter mobile app designed for clinicians and field healthcare workers.

### App Features
- **Intuitive Clinical Interface:** Material 3 design with medical light and dark modes.
- **Image Acquisition:** Camera capture with positioning overlays or gallery upload.
- **Diagnostic Result Screen:** Interactive risk gauge, severity badge, probability bars, and clinical guidelines.
- **Historical Records:** SQLite/SharedPreferences persistence for tracking patient disease progression over time.
- **Export & Share:** Export diagnostic summaries as PDF or share directly with specialist ophthalmologists.

### Running the Flutter App
```bash
cd retina_ai
flutter pub get
flutter run
```

---

## Experimental Leaderboard & Ablations

All metrics are benchmarked on the APTOS 2019 validation split using **Quadratic Weighted Kappa (QWK)** and **Severe DR Recall (Grades 3 & 4)**:

| Status | Configuration | Resolution | Preprocessing | Loss / Sampler | Validation Acc (%) | Macro F1 | QWK Score | Severe DR Recall |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **COMPLETED** | **Frozen B0 (Baseline)** | $224 \times 224$ | Standard Resize | CrossEntropy | 73.59% | 0.5610 | 0.7995 | 34.5% |
| **COMPLETED** | **Two-Stage Fine-Tuning** | $224 \times 224$ | Standard Resize | CrossEntropy | 78.69% | 0.5679 | 0.8176 | 34.5% |
| *Queued* | E0: Baseline 224 | $224 \times 224$ | Standard Resize | CrossEntropy | Pending | Pending | Pending | Pending |
| *Queued* | E1: Retinal Cropping | $384 \times 384$ | Circular Crop | CrossEntropy | Pending | Pending | Pending | Pending |
| *Queued* | E2: Cropping + Ben Graham | $384 \times 384$ | Crop + Ben Graham | CrossEntropy | Pending | Pending | Pending | Pending |
| *Queued* | E3: Balanced Sampling | $384 \times 384$ | Crop + Ben Graham | Weighted Sampler | Pending | Pending | Pending | Pending |
| *Queued* | E4: Focal Loss | $384 \times 384$ | Crop + Ben Graham | Focal Loss ($\gamma=2$) | Pending | Pending | Pending | Pending |

*The updated leaderboard is tracked in [results/metrics/controlled_experiments_leaderboard.csv](results/metrics/controlled_experiments_leaderboard.csv).*

---

## Automated Test Suite

The codebase includes comprehensive unit and integration tests covering:
- Retinal circle cropping accuracy and contour edge cases (`test_retinal_preprocessing.py`).
- Ben Graham Gaussian subtraction and CLAHE transforms (`test_ben_graham.py`).
- Focal Loss gradient behavior and class weighted sampling (`test_losses_and_sampling.py`).
- Two-stage frozen backbone/BatchNorm parameter checks (`test_two_stage_pipeline.py`).
- End-to-end forward/backward passes and evaluation metrics (`test_master_pipeline.py`).

Run the complete test suite:
```bash
python -m unittest discover -s tests
```

Output:
```text
Ran 34 tests in 5.89s
OK
```

---

## License & Acknowledgments

- **License:** MIT License. See [LICENSE](LICENSE) for details.
- **Dataset:** Asia Pacific Tele-Ophthalmology Society (APTOS) 2019 Blindness Detection Challenge.
- **Preprocessing Methodology:** Inspired by Ben Graham's Kaggle Diabetic Retinopathy winning solution.
