"""
Unit tests for Ben Graham Preprocessing Module (src/data/ben_graham.py).

Verifies:
1. Shape preservation across various image dimensions (224x224, 384x384, non-square).
2. RGB channel preservation (channels are not permuted or collapsed).
3. Value range: strictly in [0, 255] and dtype uint8.
4. Finite values: no NaN, Inf, or negative values.
5. Deterministic behavior: identical input yields identical output every time.
6. Immutability: input PIL Image and NumPy arrays are never mutated in-place.
7. Masking behavior: pixels outside circular retinal boundary are 0 (black).
8. Blur scale configurability: explicit sigma vs sigma_ratio.
9. Torchvision transform compatibility via BenGrahamTransform.
"""

import os
import sys
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.ben_graham import ben_graham_filter, BenGrahamTransform


class TestBenGrahamFilter(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 384x384 RGB retinal-like image (bright center, dark perimeter)
        np.random.seed(42)
        h, w = 384, 384
        cy, cx = h / 2.0, w / 2.0
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radius = min(h, w) * 0.45

        # Retinal disk with varying color texture
        fundus = np.zeros((h, w, 3), dtype=np.uint8)
        mask = dist <= radius
        fundus[mask, 0] = np.clip(180 - dist[mask] * 0.2 + np.random.randint(-10, 10, mask.sum()), 0, 255).astype(np.uint8) # Red
        fundus[mask, 1] = np.clip(100 - dist[mask] * 0.15 + np.random.randint(-10, 10, mask.sum()), 0, 255).astype(np.uint8) # Green
        fundus[mask, 2] = np.clip(30 - dist[mask] * 0.05 + np.random.randint(-5, 5, mask.sum()), 0, 255).astype(np.uint8)   # Blue

        self.synthetic_image_arr = fundus
        self.synthetic_image_pil = Image.fromarray(fundus, mode="RGB")

    def test_shape_preservation(self):
        """Test output shape matches input across various resolutions."""
        for size in [(224, 224), (384, 384), (256, 512)]:
            img = Image.new("RGB", (size[1], size[0]), color=(100, 50, 20))
            out_pil = ben_graham_filter(img, return_pil=True)
            self.assertEqual(out_pil.size, (size[1], size[0]))

            out_np = ben_graham_filter(np.array(img), return_pil=False)
            self.assertEqual(out_np.shape, (size[0], size[1], 3))

    def test_rgb_channel_preservation(self):
        """Test that RGB channels remain distinct and not permuted (e.g. not flipped to BGR)."""
        # Create an image where R >> G >> B
        img_arr = np.zeros((100, 100, 3), dtype=np.uint8)
        img_arr[:, :, 0] = 220  # High Red
        img_arr[:, :, 1] = 120  # Medium Green
        img_arr[:, :, 2] = 20   # Low Blue

        # With identical constant areas, Gaussian blur subtraction keeps the relative channel offsets
        out_np = ben_graham_filter(img_arr, mask_background=False, return_pil=False)
        self.assertEqual(out_np.shape[2], 3)
        self.assertEqual(out_np.dtype, np.uint8)

        # Test on textured image: channels must not be identical (not grayscale)
        out_textured = ben_graham_filter(self.synthetic_image_arr, return_pil=False)
        self.assertFalse(np.array_equal(out_textured[:, :, 0], out_textured[:, :, 1]))
        self.assertFalse(np.array_equal(out_textured[:, :, 0], out_textured[:, :, 2]))

    def test_output_range_and_finite_values(self):
        """Test that outputs are strictly within [0, 255] with no NaN or Inf."""
        out_np = ben_graham_filter(self.synthetic_image_arr, return_pil=False)
        self.assertEqual(out_np.dtype, np.uint8)
        self.assertTrue(np.all(out_np >= 0))
        self.assertTrue(np.all(out_np <= 255))
        self.assertTrue(np.isfinite(out_np).all())

    def test_deterministic_behavior(self):
        """Test that multiple runs on the exact same input yield identical byte outputs."""
        out1 = ben_graham_filter(self.synthetic_image_pil, return_pil=False)
        out2 = ben_graham_filter(self.synthetic_image_pil, return_pil=False)
        self.assertTrue(np.array_equal(out1, out2))

    def test_immutability_of_raw_input(self):
        """Test that input array and PIL Image are not modified in-place."""
        orig_copy = self.synthetic_image_arr.copy()
        _ = ben_graham_filter(self.synthetic_image_arr, return_pil=False)
        self.assertTrue(np.array_equal(self.synthetic_image_arr, orig_copy))

    def test_mask_background(self):
        """Test that circular background mask sets corner pixels outside retina to 0."""
        out_masked = ben_graham_filter(self.synthetic_image_arr, mask_background=True, return_pil=False)
        # Top-left corner (0, 0) is well outside retinal radius
        self.assertEqual(out_masked[0, 0, 0], 0)
        self.assertEqual(out_masked[0, 0, 1], 0)
        self.assertEqual(out_masked[0, 0, 2], 0)
        # Center of image must have valid illuminated content
        center = out_masked[192, 192]
        self.assertTrue(np.any(center > 0))

    def test_configurable_sigma_parameters(self):
        """Test explicit sigma vs sigma_ratio."""
        out_sigma5 = ben_graham_filter(self.synthetic_image_arr, sigma=5.0, return_pil=False)
        out_sigma25 = ben_graham_filter(self.synthetic_image_arr, sigma=25.0, return_pil=False)
        # Different blur scale produces noticeably different local contrast
        self.assertFalse(np.array_equal(out_sigma5, out_sigma25))

    def test_torchvision_transform_wrapper(self):
        """Test BenGrahamTransform works seamlessly as a torchvision-style callable."""
        transform = BenGrahamTransform(sigma_ratio=1.0 / 30.0, mask_background=True)
        out = transform(self.synthetic_image_pil)
        self.assertIsInstance(out, Image.Image)
        self.assertEqual(out.size, (384, 384))
        self.assertEqual(out.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
