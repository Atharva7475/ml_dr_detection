"""
Lightweight Model Construction Tests for Colab.

For each of the 4 models, verifies:
  - Model can be instantiated
  - Input [2, 3, 224, 224] → Output [2, 5]
  - No NaN in output
  - Parameter count works

Usage (Colab cell):
    import sys
    sys.path.insert(0, "/content/drive/MyDrive/DR_project")
    exec(open("/content/drive/MyDrive/DR_project/src/models/test_models.py").read())
"""

import torch
import torch.nn as nn


def test_model(name, build_fn, count_fn, summary_fn, **kwargs):
    """Test a single model builder."""
    print(f"\n{'=' * 60}")
    print(f"  TESTING: {name}")
    print(f"{'=' * 60}")

    # 1. Instantiate
    model = build_fn(**kwargs)
    assert isinstance(model, nn.Module), f"{name}: not an nn.Module"
    print(f"  [PASS] Model instantiated")

    # 2. Parameter counts
    total, trainable = count_fn(model)
    assert total > 0, f"{name}: total params = 0"
    assert trainable > 0, f"{name}: trainable params = 0"
    print(f"  [PASS] Total params:     {total:>12,}")
    print(f"  [PASS] Trainable params: {trainable:>12,}")

    # 3. Forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == torch.Size([2, 5]), \
        f"{name}: expected [2, 5], got {list(output.shape)}"
    print(f"  [PASS] Output shape: {list(output.shape)}")

    # 4. No NaN
    assert not torch.isnan(output).any(), f"{name}: NaN in output"
    print(f"  [PASS] No NaN in output")

    # 5. Summary
    summary_fn(model)

    print(f"\n  {name}: ALL TESTS PASSED ✓")
    return total, trainable


def run_all_tests():
    """Run construction tests for all 4 models."""
    import sys, os
    # Ensure project root is on path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.models.resnet50 import build_resnet50
    from src.models.resnet50 import count_parameters as resnet_count
    from src.models.resnet50 import print_model_summary as resnet_summary

    from src.models.efficientnet import build_efficientnet_b0
    from src.models.efficientnet import count_parameters as eff_count
    from src.models.efficientnet import print_model_summary as eff_summary

    from src.models.fusion import build_fusion_model
    from src.models.fusion import count_parameters as fusion_count
    from src.models.fusion import print_model_summary as fusion_summary

    from src.models.cnn_transformer import build_cnn_transformer
    from src.models.cnn_transformer import count_parameters as ct_count
    from src.models.cnn_transformer import print_model_summary as ct_summary

    print("=" * 60)
    print("     MODEL CONSTRUCTION TESTS (4 MODELS)")
    print("=" * 60)

    results = {}

    # Model 1: ResNet50
    t, tr = test_model(
        "ResNet50",
        build_resnet50, resnet_count, resnet_summary,
        num_classes=5, pretrained=True, freeze_backbone=True,
    )
    results["ResNet50"] = (t, tr)

    # Model 2: EfficientNet-B0
    t, tr = test_model(
        "EfficientNet-B0",
        build_efficientnet_b0, eff_count, eff_summary,
        num_classes=5, pretrained=True, freeze_backbone=True,
    )
    results["EfficientNet-B0"] = (t, tr)

    # Model 3: EfficientNet-B0 + DenseNet121 Fusion
    t, tr = test_model(
        "EfficientNet-B0 + DenseNet121 Fusion",
        build_fusion_model, fusion_count, fusion_summary,
        num_classes=5, pretrained=True, freeze_backbones=True,
    )
    results["Fusion"] = (t, tr)

    # Model 4: CNN + Transformer Hybrid
    t, tr = test_model(
        "CNN + Transformer Hybrid",
        build_cnn_transformer, ct_count, ct_summary,
        num_classes=5, pretrained=True, freeze_cnn_backbone=True,
    )
    results["CNN+Transformer"] = (t, tr)

    # Summary table
    print("\n\n" + "=" * 60)
    print("     MODEL PARAMETER COMPARISON")
    print("=" * 60)
    print(f"  {'Model':<35} {'Total':>12} {'Trainable':>12}")
    print("-" * 60)
    for name, (total, trainable) in results.items():
        print(f"  {name:<35} {total:>12,} {trainable:>12,}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("  ALL 4 MODEL CONSTRUCTION TESTS: PASS ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
