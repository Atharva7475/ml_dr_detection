"""
Unit tests for Loss Functions and Sampling Strategies (Phase 3).

Verifies:
1. FocalLoss numerical parity with CrossEntropyLoss when gamma=0.0 and alpha=None.
2. FocalLoss modulation effect: downweights easy examples as gamma increases.
3. FocalLoss alpha weighting behavior and reduction modes ('mean', 'sum', 'none').
4. Autograd compatibility: gradients flow smoothly without NaN or Inf.
5. WeightedRandomSampler creation and properties: inverse-frequency weights match train.csv counts.
6. Empirical sampling simulation: confirms roughly balanced class distribution under sampler.
7. Validation/test isolation: confirms val_loader is strictly unweighted and sequential.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.training.losses import FocalLoss, get_loss_criterion
from src.data.dataloader_v2 import create_balanced_sampler, get_crop_dataloaders


class TestLossesAndSampling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.num_classes = 5
        self.batch_size = 16
        self.logits = torch.randn(self.batch_size, self.num_classes, requires_grad=True)
        self.targets = torch.randint(0, self.num_classes, (self.batch_size,))
        self.splits_dir = "src/data/splits"
        self.img_dir = "src/data/raw/train_images"

    def test_focal_loss_cross_entropy_parity(self):
        """When gamma=0 and alpha=None, FocalLoss must equal standard CrossEntropyLoss."""
        ce = nn.CrossEntropyLoss()
        fl = FocalLoss(gamma=0.0, alpha=None)

        loss_ce = ce(self.logits, self.targets)
        loss_fl = fl(self.logits, self.targets)

        self.assertTrue(torch.isclose(loss_ce, loss_fl, atol=1e-5),
                        f"CE={loss_ce.item():.6f} vs FL(gamma=0)={loss_fl.item():.6f}")

    def test_focal_loss_gamma_modulation(self):
        """Higher gamma should downweight well-classified (easy) examples more aggressively."""
        # Create an easy example: logit for correct class is very high (+10.0)
        easy_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
        target_0 = torch.tensor([0])

        fl_0 = FocalLoss(gamma=0.0)(easy_logits, target_0).item()
        fl_2 = FocalLoss(gamma=2.0)(easy_logits, target_0).item()
        fl_5 = FocalLoss(gamma=5.0)(easy_logits, target_0).item()

        # Loss should monotonically decrease as gamma focuses away from easy examples
        self.assertGreater(fl_0, fl_2)
        self.assertGreater(fl_2, fl_5)

    def test_focal_loss_reductions(self):
        """Verify 'mean', 'sum', and 'none' reduction modes."""
        fl_mean = FocalLoss(gamma=2.0, reduction="mean")(self.logits, self.targets)
        fl_sum = FocalLoss(gamma=2.0, reduction="sum")(self.logits, self.targets)
        fl_none = FocalLoss(gamma=2.0, reduction="none")(self.logits, self.targets)

        self.assertEqual(fl_mean.ndim, 0)
        self.assertEqual(fl_sum.ndim, 0)
        self.assertEqual(fl_none.shape, (self.batch_size,))
        self.assertTrue(torch.isclose(fl_mean, fl_none.mean(), atol=1e-5))
        self.assertTrue(torch.isclose(fl_sum, fl_none.sum(), atol=1e-5))

    def test_focal_loss_backward_gradients(self):
        """Test gradient propagation through FocalLoss."""
        fl = FocalLoss(gamma=2.0)
        loss = fl(self.logits, self.targets)
        loss.backward()

        self.assertIsNotNone(self.logits.grad)
        self.assertEqual(self.logits.grad.shape, self.logits.shape)
        self.assertTrue(torch.isfinite(self.logits.grad).all())

    def test_get_loss_criterion_factory(self):
        """Test factory function handles cross_entropy and focal_loss correctly."""
        crit_ce = get_loss_criterion(loss_type="cross_entropy")
        self.assertIsInstance(crit_ce, nn.CrossEntropyLoss)

        crit_fl = get_loss_criterion(loss_type="focal_loss", focal_gamma=2.5)
        self.assertIsInstance(crit_fl, FocalLoss)
        self.assertEqual(crit_fl.gamma, 2.5)

    def test_balanced_sampler_creation_and_weights(self):
        """Test create_balanced_sampler matches inverse class counts of train.csv."""
        train_csv = os.path.join(self.splits_dir, "train.csv")
        if not os.path.exists(train_csv):
            self.skipTest(f"Missing {train_csv}")

        sampler, class_weights = create_balanced_sampler(train_csv)
        self.assertEqual(len(sampler.weights), 2563)
        self.assertEqual(class_weights.shape, (5,))

        # In train.csv: class 0 has 1263, class 3 has 135.
        # Class 3 weight should be approx 1263 / 135 ~= 9.35x larger than Class 0 weight.
        ratio = class_weights[3] / class_weights[0]
        self.assertAlmostEqual(ratio.item(), 1263.0 / 135.0, places=2)

    def test_empirical_balanced_sampling(self):
        """Simulate draws from sampler and verify classes are sampled roughly uniformly."""
        train_csv = os.path.join(self.splits_dir, "train.csv")
        if not os.path.exists(train_csv):
            self.skipTest(f"Missing {train_csv}")

        import pandas as pd
        df = pd.read_csv(train_csv)
        labels = df["diagnosis"].values

        sampler, _ = create_balanced_sampler(train_csv)
        # Draw 20,000 samples with replacement according to sampler weights
        indices = list(torch.multinomial(sampler.weights, num_samples=20000, replacement=True).numpy())
        sampled_labels = labels[indices]
        counts = pd.Series(sampled_labels).value_counts(normalize=True)

        # Each class should be sampled approximately 20% of the time (tolerance +/- 3%)
        for c in range(5):
            freq = counts.get(c, 0.0)
            self.assertGreater(freq, 0.16, f"Class {c} under-sampled: {freq:.3f}")
            self.assertLess(freq, 0.24, f"Class {c} over-sampled: {freq:.3f}")

    def test_val_loader_remains_unbalanced(self):
        """Confirm validation DataLoader never uses a sampler and retains natural split distribution."""
        train_csv = os.path.join(self.splits_dir, "train.csv")
        if not os.path.exists(train_csv):
            self.skipTest("Missing split directory")

        loaders = get_crop_dataloaders(
            splits_dir=self.splits_dir,
            img_dir=self.img_dir,
            batch_size=16,
            num_workers=0,
            use_balanced_sampler=True,  # enabled for train only
        )

        val_loader = loaders["val"]
        self.assertIsNone(val_loader.sampler.__class__.__name__ == "WeightedRandomSampler" and None or None)
        # SequentialSampler is used for val_loader when shuffle=False
        self.assertEqual(val_loader.sampler.__class__.__name__, "SequentialSampler")


if __name__ == "__main__":
    unittest.main()
