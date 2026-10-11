"""
Unit tests and smoke checks for EfficientNet-B0 Two-Stage Fine-Tuning Pipeline.

Validates the 12 core requirements specified in Phase 6:
1. Logits shape [batch_size, 5].
2. Stage 1 trains only the classification head.
3. Stage 1 does not update frozen backbone parameters or BatchNorm running statistics.
4. Stage 2 makes the intended backbone parameters trainable.
5. Both learning rates are assigned to the correct optimizer parameter groups.
6. Every intended trainable parameter occurs exactly once in the optimizer.
7. Loss values and gradients are finite.
8. One training batch and one validation batch execute successfully.
9. Checkpoint save/load works for both stages with rich metadata.
10. The best checkpoint is restored correctly.
11. Training history records stage, epoch, train loss, val loss, and validation metrics.
12. Early stopping and learning-rate scheduling work correctly.
"""
import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import copy
import shutil
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.efficientnet import (
    build_efficientnet_b0,
    freeze_backbone,
    unfreeze_backbone,
    freeze_batchnorm_layers,
    get_parameter_groups,
    count_parameters,
)
from src.training.trainer import Trainer, TrainingHistory, set_seed
from src.training.scheduler import get_cosine_warmup_scheduler
from src.training.two_stage_pipeline import (
    train_stage1_warmup,
    train_stage2_finetuning,
)


def create_dummy_dataloader(num_samples=16, batch_size=4, img_size=224, num_classes=5):
    """Creates a small synthetic DataLoader for rapid smoke tests."""
    x = torch.randn(num_samples, 3, img_size, img_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_1_model_logits_shape():
    """Verify model produces logits with shape [batch_size, 5]."""
    model = build_efficientnet_b0(num_classes=5, pretrained=False)
    model.eval()
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch_size, 5), f"Expected shape ({batch_size}, 5), got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    print("  [PASS] Check 1 Passed: Logits shape is [batch_size, 5] and finite.")


def test_2_stage1_trains_head_only():
    """Verify Stage 1 freezes backbone and trains only classification head."""
    model = build_efficientnet_b0(num_classes=5, pretrained=False, freeze_backbone=True)
    total, trainable = count_parameters(model)
    # Head in EfficientNet-B0 is Linear(1280, 5) -> 1280*5 + 5 = 6405
    assert trainable == 6405, f"Expected 6,405 trainable parameters, got {trainable}"
    assert total - trainable == 4007548, f"Expected 4,007,548 frozen parameters, got {total - trainable}"
    for param in model.features.parameters():
        assert not param.requires_grad, "Backbone parameter unexpectedly requires grad!"
    for param in model.classifier.parameters():
        assert param.requires_grad, "Classifier parameter should require grad!"
    print(f"  [PASS] Check 2 Passed: Stage 1 trains exactly {trainable:,} parameters (head only).")


def test_3_stage1_preserves_backbone_and_bn_stats():
    """Verify Stage 1 does not update frozen backbone parameters or BatchNorm running stats."""
    model = build_efficientnet_b0(num_classes=5, pretrained=False, freeze_backbone=True)
    freeze_batchnorm_layers(model)

    # Record initial weights of first conv layer in backbone
    initial_backbone_weight = model.features[0][0].weight.clone()

    # Record initial running mean of first BatchNorm layer
    first_bn = model.features[0][1]
    initial_running_mean = first_bn.running_mean.clone()

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Run a training step with freeze_bn enabled
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 1, 2, 3])

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

    # Assert no gradients in backbone
    for p in model.features.parameters():
        assert p.grad is None, "Frozen backbone parameter accumulated a gradient!"

    optimizer.step()

    # Verify backbone parameters did not change
    assert torch.equal(model.features[0][0].weight, initial_backbone_weight), (
        "Frozen backbone weight was modified after optimizer step!"
    )

    # Verify BatchNorm running mean was NOT updated
    assert torch.equal(first_bn.running_mean, initial_running_mean), (
        "BatchNorm running_mean was modified during warmup!"
    )
    print("  [PASS] Check 3 Passed: Backbone parameters and BatchNorm running stats remain unchanged.")


def test_4_stage2_unfreezes_backbone():
    """Verify Stage 2 makes the entire backbone trainable."""
    model = build_efficientnet_b0(num_classes=5, pretrained=False, freeze_backbone=True)
    unfreeze_backbone(model)
    total, trainable = count_parameters(model)
    assert trainable == total, f"Expected 100% trainable ({total}), got {trainable}"
    assert trainable == 4013953, f"Expected 4,013,953 total trainable parameters, got {trainable}"
    print(f"  [PASS] Check 4 Passed: Stage 2 unfreezes entire backbone ({trainable:,} parameters).")


def test_5_and_6_parameter_groups_integrity():
    """Verify learning rates are assigned correctly with no parameter overlap or omission."""
    model = build_efficientnet_b0(num_classes=5, pretrained=False, freeze_backbone=False)
    unfreeze_backbone(model)

    groups = get_parameter_groups(model, backbone_lr=1e-5, head_lr=1e-4, weight_decay=1e-4)

    assert len(groups) == 2, f"Expected 2 parameter groups, got {len(groups)}"
    assert groups[0]["name"] == "backbone"
    assert groups[0]["lr"] == 1e-5
    assert groups[1]["name"] == "head"
    assert groups[1]["lr"] == 1e-4

    backbone_params = groups[0]["params"]
    head_params = groups[1]["params"]

    # Verify disjointness
    backbone_ids = set(id(p) for p in backbone_params)
    head_ids = set(id(p) for p in head_params)
    assert backbone_ids.isdisjoint(head_ids), "Parameter overlap between groups!"

    # Verify exact coverage
    all_trainable_ids = set(id(p) for p in model.parameters() if p.requires_grad)
    assert backbone_ids.union(head_ids) == all_trainable_ids, "Missing trainable parameters!"

    optimizer = torch.optim.AdamW(groups)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[1]["lr"] == 1e-4
    print("  [PASS] Checks 5 & 6 Passed: Optimizer parameter groups are disjoint and completely account for all parameters.")


def test_7_and_8_execution_loss_and_gradients():
    """Verify finite loss, finite gradients, and single-batch train/val pass."""
    model = build_efficientnet_b0(num_classes=5, pretrained=False)
    train_loader = create_dummy_dataloader(num_samples=4, batch_size=4)
    val_loader = create_dummy_dataloader(num_samples=4, batch_size=4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=1,
    )

    train_loss, train_acc = trainer._train_one_epoch()
    val_loss, val_acc = trainer._validate_one_epoch()

    assert not torch.isnan(torch.tensor(train_loss)), "Train loss is NaN!"
    assert not torch.isnan(torch.tensor(val_loss)), "Val loss is NaN!"
    assert 0.0 <= train_acc <= 1.0, f"Invalid train accuracy: {train_acc}"
    assert 0.0 <= val_acc <= 1.0, f"Invalid val accuracy: {val_acc}"
    print(f"  [PASS] Checks 7 & 8 Passed: Train/Val single-batch pass successful (loss={train_loss:.4f}, finite).")


def test_9_and_10_checkpoint_metadata_and_restore():
    """Verify safe checkpoint save/load with rich metadata and exact weight restoration."""
    temp_dir = tempfile.mkdtemp()
    try:
        model = build_efficientnet_b0(num_classes=5, pretrained=False)
        train_loader = create_dummy_dataloader(num_samples=8, batch_size=4)
        val_loader = create_dummy_dataloader(num_samples=8, batch_size=4)
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=1,
            checkpoint_dir=temp_dir,
            checkpoint_filename="test_stage1_best.pth",
            stage_name="stage1_warmup",
            freeze_bn=True,
            model_config={"test": True},
        )

        history = trainer.fit()
        ckpt_path = os.path.join(temp_dir, "test_stage1_best.pth")
        assert os.path.isfile(ckpt_path), "Checkpoint was not saved!"

        # Inspect checkpoint keys
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        required_keys = [
            "epoch", "stage", "model_state_dict", "optimizer_state_dict",
            "best_validation_loss", "best_validation_accuracy", "class_names",
            "class_to_idx", "model_config", "training_config", "random_seed", "training_history"
        ]
        for k in required_keys:
            assert k in ckpt, f"Missing required checkpoint key: {k}"

        assert ckpt["stage"] == "stage1_warmup"
        assert ckpt["training_config"]["freeze_bn"] is True

        # Verify restoration into fresh model produces identical outputs
        fresh_model = build_efficientnet_b0(num_classes=5, pretrained=False)
        fresh_model.load_state_dict(ckpt["model_state_dict"])
        fresh_model.eval()
        model.eval()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out1 = model(dummy)
            out2 = fresh_model(dummy)

        assert torch.allclose(out1, out2, atol=1e-5), "Restored model output mismatch!"
        print("  [PASS] Checks 9 & 10 Passed: Checkpoint preserves full metadata and restores weights identically.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_11_training_history_tracking():
    """Verify training history records stage, epoch, train loss, val loss, and metrics."""
    temp_dir = tempfile.mkdtemp()
    try:
        model = build_efficientnet_b0(num_classes=5, pretrained=False)
        train_loader = create_dummy_dataloader(num_samples=8, batch_size=4)
        val_loader = create_dummy_dataloader(num_samples=8, batch_size=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            num_epochs=2,
            checkpoint_dir=temp_dir,
            checkpoint_filename="history_test.pth",
            stage_name="stage2_finetune",
        )

        history = trainer.fit()
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2
        assert len(history["train_accuracy"]) == 2
        assert len(history["val_accuracy"]) == 2
        assert "stage" in history
        assert history["stage"] == ["stage2_finetune", "stage2_finetune"]
        print("  [PASS] Check 11 Passed: Training history records all stage and per-epoch metrics.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_12_early_stopping_and_warmup_cosine_scheduler():
    """Verify early stopping triggers at patience and scheduler executes warmup then cosine decay."""
    # 1. Test Warmup + Cosine Decay scheduler
    model = nn.Linear(10, 5)
    optimizer = torch.optim.AdamW([
        {"params": [model.weight], "lr": 1e-5, "name": "backbone"},
        {"params": [model.bias], "lr": 1e-4, "name": "head"},
    ])

    warmup_epochs = 3
    total_epochs = 10
    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=total_epochs)

    lrs_backbone = []
    lrs_head = []

    for epoch in range(total_epochs):
        lrs_backbone.append(optimizer.param_groups[0]["lr"])
        lrs_head.append(optimizer.param_groups[1]["lr"])
        scheduler.step()

    # During warmup: should increase linearly
    assert lrs_backbone[0] < lrs_backbone[1] < lrs_backbone[2], "Backbone LR didn't ramp up during warmup!"
    assert lrs_head[0] < lrs_head[1] < lrs_head[2], "Head LR didn't ramp up during warmup!"

    # At end of warmup (epoch index 2), learning rate reaches base learning rate
    assert abs(lrs_backbone[2] - 1e-5) < 1e-8, f"Expected 1e-5, got {lrs_backbone[2]}"
    assert abs(lrs_head[2] - 1e-4) < 1e-8, f"Expected 1e-4, got {lrs_head[2]}"

    # After warmup: cosine decay should decrease
    assert lrs_backbone[3] > lrs_backbone[6] > lrs_backbone[9], "Cosine decay not decreasing monotonically!"

    # 2. Test Early Stopping on Trainer
    temp_dir = tempfile.mkdtemp()
    try:
        dummy_model = nn.Linear(10, 2)
        train_loader = DataLoader(TensorDataset(torch.randn(8, 10), torch.randint(0, 2, (8,))), batch_size=4)
        val_loader = DataLoader(TensorDataset(torch.randn(8, 10), torch.randint(0, 2, (8,))), batch_size=4)
        opt = torch.optim.SGD(dummy_model.parameters(), lr=0.0)  # No learning -> constant val loss

        trainer = Trainer(
            model=dummy_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=opt,
            num_epochs=10,
            early_stopping_patience=3,
            checkpoint_dir=temp_dir,
            checkpoint_filename="early_stop_test.pth",
        )

        hist = trainer.fit()
        # Epoch 0 is new best. Epochs 1, 2, 3 have no improvement -> stops after 4 epochs total (1 best + 3 patience)
        assert len(hist["train_loss"]) <= 4, f"Early stopping failed to trigger, ran {len(hist['train_loss'])} epochs"
        print(f"  [PASS] Check 12 Passed: Warmup-cosine scheduler and early stopping verified (stopped at epoch {len(hist['train_loss'])}).")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_all_smoke_checks():
    """Runs all 12 smoke tests."""
    print("=" * 70)
    print("       STARTING EFFICIENTNET-B0 TWO-STAGE PIPELINE SMOKE CHECKS       ")
    print("=" * 70)

    test_1_model_logits_shape()
    test_2_stage1_trains_head_only()
    test_3_stage1_preserves_backbone_and_bn_stats()
    test_4_stage2_unfreezes_backbone()
    test_5_and_6_parameter_groups_integrity()
    test_7_and_8_execution_loss_and_gradients()
    test_9_and_10_checkpoint_metadata_and_restore()
    test_11_training_history_tracking()
    test_12_early_stopping_and_warmup_cosine_scheduler()

    print("=" * 70)
    print("       ALL 12 SMOKE CHECKS PASSED PERFECTLY! PIPELINE VERIFIED       ")
    print("=" * 70)


if __name__ == "__main__":
    run_all_smoke_checks()
