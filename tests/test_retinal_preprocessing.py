"""
Comprehensive unit tests and validation checks for Retinal Cropping and Multi-Resolution Preprocessing (STEP 2).

Validates:
1. Circular retinal field detection and margin removal.
2. RGB channel ordering preservation.
3. Safe fallback on corrupt, pitch-black, uniform, or degenerate images.
4. Square aspect ratio padding.
5. Configurable resolutions (224x224 and 384x384).
6. Finite, normalized tensor values with zero NaNs or Infs.
7. Split consistency across train, val, and test pipelines.
8. Baseline equivalence when do_crop=False.
9. Forward-pass compatibility with EfficientNet-B0 at both 224 and 384 resolutions.
10. Non-destructive processing (original disk files remain untouched).
"""

import os
import sys
import unittest
import numpy as np
import torch
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.retinal_crop import (
    crop_retina,
    detect_retina_bbox,
    pad_to_square,
    RetinalCropTransform,
)
from src.data.transforms_v2 import (
    get_crop_train_transforms,
    get_crop_val_transforms,
    get_crop_test_transforms,
)
from configs.preprocessing_config import PreprocessingConfig, PRESETS
from src.models.efficientnet import build_efficientnet_b0


class TestRetinalCropAlgorithm(unittest.TestCase):
    """Test suite for retinal cropping logic and edge cases."""

    def _create_synthetic_fundus(self, h=1000, w=1500, radius=400, bg_val=5, fg_val=(180, 100, 50)):
        """Creates a synthetic fundus image: black rectangular background with an illuminated circle."""
        arr = np.ones((h, w, 3), dtype=np.uint8) * bg_val
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = dist_from_center <= radius
        arr[mask] = fg_val
        return arr, (cx - radius, cy - radius, cx + radius, cy + radius)

    def test_synthetic_circular_retina_detection(self):
        """Verify cropping removes black borders around a circular retina."""
        arr, expected_box = self._create_synthetic_fundus(h=1000, w=1600, radius=400)
        bbox, is_cropped, reason = detect_retina_bbox(arr, tolerance=15)

        self.assertTrue(is_cropped, f"Expected is_cropped=True, got reason: {reason}")
        xmin, ymin, xmax, ymax = bbox

        # Margin width should have removed ~400 pixels from left and right
        self.assertGreater(xmin, 300, "Should detect substantial left margin")
        self.assertLess(xmax, 1300, "Should detect substantial right margin")
        self.assertGreaterEqual(ymin, 0)
        self.assertLessEqual(ymax, 1000)

    def test_rgb_channel_preservation(self):
        """Verify that RGB channels are strictly preserved (no BGR reversal)."""
        # Distinguishable RGB color: Red=210, Green=120, Blue=40
        arr, _ = self._create_synthetic_fundus(h=600, w=800, radius=200, fg_val=(210, 120, 40))
        img_pil = Image.fromarray(arr)

        cropped_pil, meta = crop_retina(img_pil, return_metadata=True, preserve_aspect_ratio=False)
        cropped_np = np.array(cropped_pil)

        # Center pixel of cropped retina must match original RGB components
        center_color = cropped_np[cropped_np.shape[0] // 2, cropped_np.shape[1] // 2]
        self.assertEqual(int(center_color[0]), 210, f"R channel altered: {center_color[0]}")
        self.assertEqual(int(center_color[1]), 120, f"G channel altered: {center_color[1]}")
        self.assertEqual(int(center_color[2]), 40, f"B channel altered: {center_color[2]}")

    def test_pad_to_square(self):
        """Verify aspect-ratio preserving square padding."""
        rect_img = np.ones((400, 600, 3), dtype=np.uint8) * 100
        padded = pad_to_square(rect_img)

        self.assertEqual(padded.shape, (600, 600, 3))
        # Top and bottom padding should be black (0)
        self.assertEqual(int(padded[:100, :].sum()), 0)
        self.assertEqual(int(padded[-100:, :].sum()), 0)
        # Center should contain original content
        self.assertEqual(int(padded[100:500, :].mean()), 100)

    def test_failsafe_pitch_black_image(self):
        """Verify safe fallback on completely black images."""
        black_img = np.zeros((500, 500, 3), dtype=np.uint8)
        out, meta = crop_retina(black_img, return_metadata=True)

        self.assertFalse(meta["is_cropped"])
        self.assertEqual(out.shape, (500, 500, 3))
        self.assertIn("insufficient_foreground", meta["reason"])

    def test_failsafe_uniform_white_image(self):
        """Verify safe fallback on uniform / saturated images."""
        white_img = np.ones((500, 500, 3), dtype=np.uint8) * 250
        out, meta = crop_retina(white_img, return_metadata=True)

        self.assertFalse(meta["is_cropped"])
        self.assertEqual(out.shape, (500, 500, 3))

    def test_failsafe_degenerate_tiny_speck(self):
        """Verify safe fallback on degenerate tiny bright artifacts."""
        arr = np.zeros((800, 800, 3), dtype=np.uint8)
        arr[395:405, 395:405] = 200  # 10x10 speck
        out, meta = crop_retina(arr, return_metadata=True)

        self.assertFalse(meta["is_cropped"])
        self.assertEqual(out.shape, (800, 800, 3))

    def test_already_cropped_image_preserved(self):
        """Verify images already tightly framed are not over-cropped."""
        # Circle fills entire frame
        arr, _ = self._create_synthetic_fundus(h=600, w=600, radius=295)
        out, meta = crop_retina(arr, return_metadata=True)

        self.assertFalse(meta["is_cropped"])
        self.assertEqual(out.shape, (600, 600, 3))


class TestConfigurableTransformsPipeline(unittest.TestCase):
    """Test suite for configurable transforms and multi-resolution compatibility."""

    def setUp(self):
        self.test_img = Image.new("RGB", (800, 1200), color=(150, 80, 40))

    def test_resolution_224_output_tensor(self):
        """Verify 224x224 transforms produce [3, 224, 224] tensors."""
        t_train = get_crop_train_transforms(img_size=224, do_crop=True)
        t_val = get_crop_val_transforms(img_size=224, do_crop=True)

        out_train = t_train(self.test_img)
        out_val = t_val(self.test_img)

        self.assertEqual(out_train.shape, torch.Size([3, 224, 224]))
        self.assertEqual(out_val.shape, torch.Size([3, 224, 224]))
        self.assertEqual(out_train.dtype, torch.float32)

    def test_resolution_384_output_tensor(self):
        """Verify 384x384 transforms produce [3, 384, 384] tensors."""
        t_train = get_crop_train_transforms(img_size=384, do_crop=True)
        t_val = get_crop_val_transforms(img_size=384, do_crop=True)

        out_train = t_train(self.test_img)
        out_val = t_val(self.test_img)

        self.assertEqual(out_train.shape, torch.Size([3, 384, 384]))
        self.assertEqual(out_val.shape, torch.Size([3, 384, 384]))
        self.assertEqual(out_val.dtype, torch.float32)

    def test_pixel_values_normalized_and_finite(self):
        """Verify normalized tensors contain finite float values with zero NaNs or Infs."""
        t_val = get_crop_val_transforms(img_size=384, do_crop=True)
        tensor = t_val(self.test_img)

        self.assertTrue(torch.isfinite(tensor).all(), "Tensor contains NaN or Inf values")
        # Standard ImageNet normalized values typically fall within [-3.0, 3.0]
        self.assertGreater(tensor.min().item(), -5.0)
        self.assertLess(tensor.max().item(), 5.0)

    def test_validation_deterministic_consistency(self):
        """Verify validation and test pipelines are strictly deterministic."""
        t_val = get_crop_val_transforms(img_size=224, do_crop=True)
        t_test = get_crop_test_transforms(img_size=224, do_crop=True)

        out1 = t_val(self.test_img)
        out2 = t_val(self.test_img)
        out_test = t_test(self.test_img)

        self.assertTrue(torch.allclose(out1, out2), "Val transform is not deterministic")
        self.assertTrue(torch.allclose(out1, out_test), "Test transform does not match Val transform")

    def test_baseline_equivalence_when_uncropped(self):
        """Verify that setting do_crop=False reproduces standard resizing without cropping."""
        t_no_crop = get_crop_val_transforms(img_size=224, do_crop=False, interpolation="bilinear")
        from src.data.transforms import get_val_transforms as get_baseline_val_transforms
        t_baseline = get_baseline_val_transforms(img_size=224)

        out_no_crop = t_no_crop(self.test_img)
        out_baseline = t_baseline(self.test_img)

        self.assertTrue(
            torch.allclose(out_no_crop, out_baseline, atol=1e-5),
            "Uncropped v2 pipeline should match baseline v1 transforms"
        )


class TestEfficientNetCompatibility(unittest.TestCase):
    """Test suite verifying EfficientNet-B0 architectural compatibility with 224 and 384 inputs."""

    def setUp(self):
        self.model = build_efficientnet_b0(num_classes=5, pretrained=False)
        self.model.eval()

    def test_forward_pass_224(self):
        """Verify EfficientNet-B0 forward pass with 224x224 batch."""
        dummy_batch = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = self.model(dummy_batch)
        self.assertEqual(logits.shape, torch.Size([2, 5]))
        self.assertTrue(torch.isfinite(logits).all())

    def test_forward_pass_384(self):
        """Verify EfficientNet-B0 forward pass with 384x384 batch."""
        dummy_batch = torch.randn(2, 3, 384, 384)
        with torch.no_grad():
            logits = self.model(dummy_batch)
        self.assertEqual(logits.shape, torch.Size([2, 5]))
        self.assertTrue(torch.isfinite(logits).all())


class TestDiskNonDestructiveProperty(unittest.TestCase):
    """Verify that preprocessing operates purely in memory and does not alter disk files."""

    def test_raw_files_unmodified(self):
        sample_path = os.path.join(PROJECT_ROOT, "src", "data", "raw", "train_images", "bda8c973b09d.png")
        if not os.path.isfile(sample_path):
            self.skipTest(f"Sample image not found: {sample_path}")

        mtime_before = os.path.getmtime(sample_path)
        size_before = os.path.getsize(sample_path)

        # Execute full cropping pipeline
        img = Image.open(sample_path).convert("RGB")
        t = get_crop_train_transforms(img_size=384, do_crop=True)
        _ = t(img)

        mtime_after = os.path.getmtime(sample_path)
        size_after = os.path.getsize(sample_path)

        self.assertEqual(mtime_before, mtime_after, "File modification timestamp was altered!")
        self.assertEqual(size_before, size_after, "File size was altered on disk!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
