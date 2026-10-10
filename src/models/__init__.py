# src/models - Model zoo for Diabetic Retinopathy classification.
#
# Available models:
#   build_resnet50         - ResNet50 transfer learning  (src.models.resnet50)
#   build_efficientnet_b0  - EfficientNet-B0             (src.models.efficientnet)
#   build_fusion_model     - EfficientNet-B0 + DenseNet121 fusion (src.models.fusion)
#   build_cnn_transformer  - CNN + Transformer hybrid    (src.models.cnn_transformer)
#
# All models follow a common interface:
#   - Input:  [B, 3, 224, 224]
#   - Output: [B, num_classes]
#   - Builder function returns nn.Module
#   - count_parameters(model) → (total, trainable)
#   - print_model_summary(model) → formatted console output
