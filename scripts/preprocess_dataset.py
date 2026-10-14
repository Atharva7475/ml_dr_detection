"""
High-Speed Dataset Preprocessing Script for APTOS 2019 Diabetic Retinopathy.

Pre-computes and caches:
1. Retinal margin circular cropping + square padding
2. High-quality antialiased resizing (default 384x384, configurable)
3. Ben Graham Gaussian illumination normalization

Using multi-threading on multi-core workstations to process all 3,662 images
in under 2 minutes, reducing training epoch time from ~6,900s to ~8s.

Usage:
    python scripts/preprocess_dataset.py --img-size 384
    python scripts/preprocess_dataset.py --img-size 224
"""

import os
import sys
import glob
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data.retinal_crop import crop_retina
from src.data.ben_graham import ben_graham_filter


def process_and_save_single_image(
    input_path: str,
    output_dir: str,
    img_size: int = 384,
    tolerance: int = 15,
    sigma_ratio: float = 1.0 / 30.0,
    overwrite: bool = False,
) -> bool:
    """
    Process one raw fundus image and save the preprocessed output to disk.
    """
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_dir, filename)

    if not overwrite and os.path.isfile(output_path):
        return True

    try:
        # 1. Load raw image
        raw_img = Image.open(input_path).convert("RGB")

        # 2. Retinal aperture crop + square padding
        cropped_img = crop_retina(
            raw_img,
            tolerance=tolerance,
            preserve_aspect_ratio=True,
            return_metadata=False,
        )

        # 3. High-quality resize BEFORE Gaussian blur (huge speedup!)
        resized_img = cropped_img.resize((img_size, img_size), Image.BICUBIC)

        # 4. Ben Graham illumination normalization on resized image
        norm_img = ben_graham_filter(
            resized_img,
            sigma_ratio=sigma_ratio,
            mask_background=True,
            return_pil=True,
        )

        # 5. Save preprocessed image
        norm_img.save(output_path, format="PNG")
        return True
    except Exception as e:
        print(f"Error processing {filename}: {e}", file=sys.stderr)
        return False


def run_preprocessing(
    raw_dir: str,
    output_dir: str,
    img_size: int = 384,
    num_workers: int = 12,
    overwrite: bool = False,
):
    """
    Preprocess all images in raw_dir in parallel and save to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(raw_dir, "*.*")))
    image_paths = [
        p for p in image_paths if p.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    total_images = len(image_paths)
    if total_images == 0:
        print(f"No images found in {raw_dir}!")
        return

    print("=" * 70)
    print("      APTOS 2019 HIGH-SPEED RETINAL DATASET PREPROCESSING      ")
    print("=" * 70)
    print(f"  Source Directory:     {raw_dir}")
    print(f"  Destination:          {output_dir}")
    print(f"  Total Images:         {total_images:,}")
    print(f"  Target Resolution:    {img_size}x{img_size}")
    print(f"  Worker Threads:       {num_workers}")
    print(f"  Pipeline:             Retina Crop -> Resize -> Ben Graham Norm")
    print("=" * 70)

    start_time = time.time()
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_and_save_single_image,
                p,
                output_dir,
                img_size,
                15,
                1.0 / 30.0,
                overwrite,
            ): p
            for p in image_paths
        }

        for future in as_completed(futures):
            res = future.result()
            if res:
                completed += 1
            else:
                errors += 1

            if completed % 500 == 0 or completed == total_images:
                elapsed = time.time() - start_time
                fps = completed / max(elapsed, 0.001)
                remaining = (total_images - completed) / max(fps, 0.001)
                print(
                    f"  Progress: {completed:>4}/{total_images} ({completed/total_images*100:5.1f}%) "
                    f"| Speed: {fps:5.1f} img/s | Elapsed: {elapsed:5.1f}s | ETA: {remaining:5.1f}s"
                )

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"  Preprocessing complete!")
    print(f"  Processed {completed:,} images in {total_time:.1f}s ({total_time/60:.2f} min).")
    print(f"  Average speed: {completed / total_time:.1f} images/second.")
    if errors > 0:
        print(f"  Warning: {errors} images failed.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess APTOS dataset images")
    parser.add_argument("--img-size", type=int, default=384, help="Target image size (default: 384)")
    parser.add_argument("--raw-dir", type=str, default="src/data/raw/train_images", help="Raw images directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--workers", type=int, default=12, help="Number of worker threads (default: 12)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed images")

    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join("src", "data", "processed", f"images_{args.img_size}")

    run_preprocessing(
        raw_dir=args.raw_dir,
        output_dir=args.out_dir,
        img_size=args.img_size,
        num_workers=args.workers,
        overwrite=args.overwrite,
    )
