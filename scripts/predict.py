"""
Single-Image & Patient Inference Script for Diabetic Retinopathy Detection.

Runs clinical inference on a single retinal fundus image or image directory:
  1. Applies Ben Graham & retinal cropping preprocessing pipeline.
  2. Passes image through trained champion model (default: CNN-Vision-Transformer).
  3. Outputs predicted DR grade (0-4), clinical severity stage, and class probability distribution.
  4. Optionally generates a visual diagnosis card.

Usage:
    # Predict on a specific raw or preprocessed image:
    python scripts/predict.py --image src/data/raw/train_images/000c1449937f.png

    # Predict using a specific model architecture:
    python scripts/predict.py --image src/data/raw/train_images/000c1449937f.png --model resnet50

    # Save a visual diagnostic plot:
    python scripts/predict.py --image src/data/raw/train_images/000c1449937f.png --save-plot
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.resnet50 import build_resnet50
from src.models.efficientnet import build_efficientnet_b0
from src.models.fusion import build_fusion_model
from src.models.cnn_transformer import build_cnn_transformer
from src.data.retinal_crop import crop_retina
from src.data.ben_graham import ben_graham_filter
from src.evaluation.metrics import DR_CLASS_NAMES

SEVERITY_LEVELS = [
    ("No DR", "Normal retina. No microaneurysms or diabetic lesions detected. Annual screening recommended."),
    ("Mild NPDR", "Microaneurysms only. Early non-proliferative diabetic retinopathy. Follow up in 6-12 months."),
    ("Moderate NPDR", "Microaneurysms, dot/blot hemorrhages, and/or hard exudates. Retinal specialist referral indicated."),
    ("Severe NPDR", "4-2-1 rule met: hemorrhages in 4 quadrants, venous beading, or IRMA. High risk of progression."),
    ("Proliferative DR", "Neovascularization or vitreous/preretinal hemorrhage. Urgent ophthalmologic intervention required."),
]


def find_checkpoint(model_key: str) -> str:
    key_variants = [
        model_key,
        model_key.replace("_", "-"),
        model_key.replace("_", ""),
        "cnn-vision-transformer" if "transformer" in model_key else model_key,
        "resnet-50" if "resnet" in model_key else model_key,
        "dual-backbone-fusion" if "fusion" in model_key else model_key,
        "efficientnet-b0" if "efficientnet" in model_key else model_key,
    ]
    base_dirs = [
        os.path.join(REPO_ROOT, "checkpoints"),
        os.path.join(REPO_ROOT, "notebooks", "checkpoints"),
    ]
    for base in base_dirs:
        for v in key_variants:
            candidate = os.path.join(base, v, "best_model.pth")
            if os.path.isfile(candidate):
                return candidate
    return ""


def preprocess_image(image_path: str, img_size: int = 384, is_already_preprocessed: bool = False) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    if not is_already_preprocessed:
        img = crop_retina(img, tolerance=15, preserve_aspect_ratio=True)
        img = img.resize((img_size, img_size), Image.BICUBIC)
        img = ben_graham_filter(img, sigma_ratio=1.0/30.0, mask_background=True, return_pil=True)
    else:
        img = img.resize((img_size, img_size), Image.BICUBIC)

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return normalize(img).unsqueeze(0), img


def main():
    parser = argparse.ArgumentParser(description="Clinical Inference on Retinal Fundus Image")
    parser.add_argument("--image", type=str, required=True, help="Path to retinal image PNG/JPG")
    parser.add_argument("--model", type=str, default="cnn_transformer",
                        choices=["resnet50", "efficientnet_b0", "fusion", "cnn_transformer"],
                        help="Model architecture (default: cnn_transformer)")
    parser.add_argument("--save-plot", action="store_true", help="Save visualization to results/plots/")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = find_checkpoint(args.model)
    if not ckpt_path:
        print(f"Error: No checkpoint found for model '{args.model}'.")
        return

    model_registry = {
        "resnet50": ("ResNet-50", build_resnet50),
        "efficientnet_b0": ("EfficientNet-B0", build_efficientnet_b0),
        "fusion": ("Dual-Backbone-Fusion", build_fusion_model),
        "cnn_transformer": ("CNN-Vision-Transformer", build_cnn_transformer),
    }

    display_name, builder = model_registry[args.model]
    model = builder(num_classes=5, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    # Preprocess
    is_preproc = "processed" in args.image.lower()
    tensor_img, pil_preprocessed = preprocess_image(args.image, img_size=384, is_already_preprocessed=is_preproc)
    tensor_img = tensor_img.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(tensor_img)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_grade = int(np.argmax(probs))
        confidence = probs[pred_grade] * 100.0

    sev_name, clinical_rec = SEVERITY_LEVELS[pred_grade]

    print("\n" + "=" * 75)
    print("             RETINAL FUNDUS CLINICAL DIAGNOSIS REPORT             ")
    print("=" * 75)
    print(f"  Input Image:         {os.path.basename(args.image)}")
    print(f"  Diagnostic Model:    {display_name}")
    print(f"  Predicted DR Grade:  Grade {pred_grade} ({sev_name})")
    print(f"  Confidence:          {confidence:.2f}%")
    print(f"  Clinical Action:     {clinical_rec}")
    print("-" * 75)
    print("  Class Probability Distribution:")
    for i, (name, p) in enumerate(zip(DR_CLASS_NAMES, probs)):
        bar = "#" * int(p * 30)
        marker = " <== PREDICTED" if i == pred_grade else ""
        print(f"    Grade {i} ({name:<16}): {p*100:>5.2f}% | {bar:<30}{marker}")
    print("=" * 75)

    if args.save_plot:
        out_dir = os.path.join(REPO_ROOT, "results", "plots")
        os.makedirs(out_dir, exist_ok=True)
        img_stem = os.path.splitext(os.path.basename(args.image))[0]
        out_path = os.path.join(out_dir, f"prediction_{img_stem}_{args.model}.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        ax1.imshow(pil_preprocessed)
        ax1.set_title(f"Preprocessed Fundus ({os.path.basename(args.image)})\nPred: Grade {pred_grade} - {sev_name} ({confidence:.1f}%)",
                      fontsize=10, fontweight="bold")
        ax1.axis("off")

        colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c", "#8e44ad"]
        y_pos = np.arange(len(DR_CLASS_NAMES))
        bars = ax2.barh(y_pos, probs * 100, color=colors, edgecolor="black", alpha=0.85)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(DR_CLASS_NAMES)
        ax2.set_xlabel("Predicted Probability (%)", fontweight="bold")
        ax2.set_xlim(0, 100)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.invert_yaxis()

        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 1.5, bar.get_y() + bar.get_height() / 2, f"{width:.1f}%",
                     va="center", ha="left", fontsize=9, fontweight="bold")

        plt.suptitle(f"Clinical AI Diagnosis | {display_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"\n  Saved visual diagnostic report to: {out_path}")


if __name__ == "__main__":
    main()
