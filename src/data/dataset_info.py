"""
Dataset Metadata for APTOS 2019 Blindness Detection.

This module contains metadata constants and class mapping definitions for the
APTOS 2019 Blindness Detection dataset used in Diabetic Retinopathy classification.

Note:
    This file contains metadata definitions only and does not load, preprocess,
    or download any dataset files.
"""

DATASET_NAME = "APTOS 2019 Blindness Detection"

NUM_CLASSES = 5

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

IMAGE_COLUMN = "id"
LABEL_COLUMN = "diagnosis"
