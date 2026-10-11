"""
End-to-End Master Pipeline Verification & Sanity Test Suite.

Verifies:
  1. Forward and backward passes for all 5 experiment configurations (E0, E1, E2, E3, E4).
  2. Loss computation, gradient flow, and optimizer steps under both CrossEntropyLoss and FocalLoss.
  3. Preprocessing transform pipeline execution (cropping, resizing, Ben Graham normalization).
  4. Checkpoint saving and reloading: bit-for-bit parameter state restoration and metadata preservation.
  5. Short overfit sanity test: confirms that EfficientNet-B0 can rapidly overfit a tiny batch,
     verifying optimization dynamics and gradient descent mechanics.
"""

import os
import sys
import tempfile
import unittest
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.experiments import EXPERIMENTS, get_experiment_config
from src.models.efficientnet import build_efficientnet_b0
from src.data.transforms_v2 import get_crop_train_transforms
from src.training.losses import get_loss_criterion
from src.training.trainer import Trainer


class TestMasterPipeline(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)

    def test_all_experiment_forward_backward_smoke(self):
        """Verify 1-batch forward/backward pass across all 5 experiment configurations."""
        for exp_id in ["E0", "E1", "E2", "E3", "E4"]:
            cfg = get_experiment_config(exp_id)
            img_size = cfg.img_size
            batch_size = 4
            num_classes = 5

            # 1. Verify transform builds cleanly
            transform = get_crop_train_transforms(
                img_size=img_size,
                do_crop=cfg.do_crop,
                use_ben_graham=cfg.use_ben_graham,
                ben_graham_sigma_ratio=cfg.ben_graham_sigma_ratio,
                ben_graham_mask_background=cfg.ben_graham_mask_background,
            )
            # Apply transform to synthetic PIL Image
            dummy_pil = Image.fromarray(np.random.randint(0, 255, (400, 500, 3), dtype=np.uint8))
            transformed_tensor = transform(dummy_pil)
            self.assertEqual(transformed_tensor.shape, (3, img_size, img_size))

            # 2. Build model and criterion
            model = build_efficientnet_b0(num_classes=num_classes, pretrained=False, freeze_backbone=False)
            criterion = get_loss_criterion(
                loss_type=cfg.loss_type,
                focal_gamma=cfg.focal_gamma,
                focal_alpha=cfg.focal_alpha,
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            # 3. Dummy batch forward pass
            dummy_batch = torch.randn(batch_size, 3, img_size, img_size)
            dummy_labels = torch.randint(0, num_classes, (batch_size,))

            optimizer.zero_grad()
            outputs = model(dummy_batch)
            self.assertEqual(outputs.shape, (batch_size, num_classes))
            self.assertTrue(torch.isfinite(outputs).all())

            # 4. Backward pass
            loss = criterion(outputs, dummy_labels)
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(loss.item(), 0.0)

            loss.backward()

            # 5. Verify gradients exist on classification head
            head_grad = model.classifier[1].weight.grad
            self.assertIsNotNone(head_grad)
            self.assertTrue(torch.isfinite(head_grad).all())
            self.assertTrue((head_grad.abs() > 0).any())

            optimizer.step()

    def test_checkpoint_save_and_restore_integrity(self):
        """Verify checkpoint saving and exact weight restoration."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            model1 = build_efficientnet_b0(num_classes=5, pretrained=False)
            dummy_data = [(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))]
            optimizer = torch.optim.AdamW(model1.parameters(), lr=1e-3)

            trainer = Trainer(
                model=model1,
                train_loader=dummy_data,
                val_loader=dummy_data,
                optimizer=optimizer,
                checkpoint_dir=tmp_dir,
                checkpoint_filename="test_ckpt.pth",
                num_epochs=1,
                stage_name="smoke_test",
            )
            ckpt_path = trainer._save_checkpoint(epoch=0, val_loss=0.45, val_acc=0.80)
            self.assertTrue(os.path.isfile(ckpt_path))

            # Inspect checkpoint content
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.assertIn("model_state_dict", checkpoint)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertEqual(checkpoint["best_validation_loss"], 0.45)
            self.assertEqual(checkpoint["best_validation_accuracy"], 0.80)

            # Load into fresh model and verify bitwise parameter equality
            model2 = build_efficientnet_b0(num_classes=5, pretrained=False)
            model2.load_state_dict(checkpoint["model_state_dict"])

            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                self.assertTrue(torch.equal(p1, p2))

    def test_overfit_sanity_on_small_subset(self):
        """Sanity check: Model must rapidly decrease loss when overfitting a tiny batch of 4 samples."""
        model = build_efficientnet_b0(num_classes=5, pretrained=False, freeze_backbone=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        criterion = nn.CrossEntropyLoss()

        # Fixed tiny batch (using 128x128 for fast CPU execution)
        images = torch.randn(4, 3, 128, 128)
        targets = torch.tensor([0, 1, 2, 3])

        initial_loss = None
        final_loss = None

        model.train()
        outputs = None
        for step in range(25):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if step == 0:
                initial_loss = loss.item()
            if step == 24:
                final_loss = loss.item()

        # Loss must drop substantially during training (drops > 80% to near zero)
        self.assertLess(final_loss, initial_loss * 0.2,
                        f"Failed to overfit tiny batch: initial={initial_loss:.4f}, final={final_loss:.4f}")

        # Accuracy on this batch should reach 100% on the fitted outputs
        preds = outputs.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
        self.assertEqual(acc, 1.0, f"Expected 100% training accuracy on 4 samples, got {acc:.2%}")


if __name__ == "__main__":
    unittest.main()
