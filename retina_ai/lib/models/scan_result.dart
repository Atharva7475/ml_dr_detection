import 'package:uuid/uuid.dart';

class ScanResult {
  final String id;
  final DateTime scannedAt;
  final String imagePath;
  final int predictedGrade;
  final List<double> probabilities; // length 5
  final double quadraticWeightedKappa;
  final double sensitivity;
  final double specificity;
  final String modelUsed;
  final String patientId;
  final String eyeLaterality; // 'OD (Right Eye)' or 'OS (Left Eye)'
  final String imageQuality;

  ScanResult({
    required this.id,
    required this.scannedAt,
    required this.imagePath,
    required this.predictedGrade,
    required this.probabilities,
    required this.quadraticWeightedKappa,
    required this.sensitivity,
    required this.specificity,
    required this.modelUsed,
    this.patientId = 'PT-8942',
    this.eyeLaterality = 'OD (Right Eye)',
    this.imageQuality = 'Diagnostic Grade (Optimal)',
  });

  factory ScanResult.mock(
    String imagePath, {
    String model = 'Ensemble Fusion (EffNet-B0 + DenseNet121)',
    String patientId = 'PT-8942',
    String eyeLaterality = 'OD (Right Eye)',
  }) {
    const uuid = Uuid();
    
    // Realistic simulation based on image path if testing preloaded samples
    int grade = 2; // Default moderate
    List<double> probs = [0.03, 0.09, 0.64, 0.18, 0.06];

    if (imagePath.toLowerCase().contains('normal')) {
      grade = 0;
      probs = [0.94, 0.04, 0.01, 0.005, 0.005];
    } else if (imagePath.toLowerCase().contains('pathology')) {
      grade = 3;
      probs = [0.01, 0.02, 0.12, 0.76, 0.09];
    }

    return ScanResult(
      id: uuid.v4(),
      scannedAt: DateTime.now(),
      imagePath: imagePath,
      predictedGrade: grade,
      probabilities: probs,
      quadraticWeightedKappa: 0.912,
      sensitivity: 0.964,
      specificity: 0.981,
      modelUsed: model,
      patientId: patientId,
      eyeLaterality: eyeLaterality,
      imageQuality: 'Diagnostic Grade (Optimal 45°)',
    );
  }

  Map<String, dynamic> toMap() => {
    'id': id,
    'scannedAt': scannedAt.toIso8601String(),
    'imagePath': imagePath,
    'predictedGrade': predictedGrade,
    'probabilities': probabilities,
    'quadraticWeightedKappa': quadraticWeightedKappa,
    'sensitivity': sensitivity,
    'specificity': specificity,
    'modelUsed': modelUsed,
    'patientId': patientId,
    'eyeLaterality': eyeLaterality,
    'imageQuality': imageQuality,
  };

  factory ScanResult.fromMap(Map<String, dynamic> map) {
    return ScanResult(
      id: map['id'] ?? const Uuid().v4(),
      scannedAt: DateTime.tryParse(map['scannedAt'] ?? '') ?? DateTime.now(),
      imagePath: map['imagePath'] ?? '',
      predictedGrade: map['predictedGrade'] ?? 0,
      probabilities: (map['probabilities'] as List<dynamic>?)?.map((e) => (e as num).toDouble()).toList() ?? [0.2, 0.2, 0.2, 0.2, 0.2],
      quadraticWeightedKappa: (map['quadraticWeightedKappa'] as num?)?.toDouble() ?? 0.912,
      sensitivity: (map['sensitivity'] as num?)?.toDouble() ?? 0.964,
      specificity: (map['specificity'] as num?)?.toDouble() ?? 0.981,
      modelUsed: map['modelUsed'] ?? 'Ensemble Fusion',
      patientId: map['patientId'] ?? 'PT-8942',
      eyeLaterality: map['eyeLaterality'] ?? 'OD (Right Eye)',
      imageQuality: map['imageQuality'] ?? 'Diagnostic Grade',
    );
  }
}

class ModelInfo {
  final String name;
  final String description;
  final double accuracy;
  final double f1Score;
  final double aucScore;
  final double qwk;
  final String backboneDetails;

  const ModelInfo({
    required this.name,
    required this.description,
    required this.accuracy,
    required this.f1Score,
    required this.aucScore,
    required this.qwk,
    required this.backboneDetails,
  });
}

const List<ModelInfo> kModels = [
  ModelInfo(
    name: 'Dual-Backbone Fusion',
    description: 'Hierarchical Feature Concat of EfficientNet-B0 & DenseNet121 with Multi-Head Self-Attention.',
    accuracy: 0.958,
    f1Score: 0.942,
    aucScore: 0.984,
    qwk: 0.912,
    backboneDetails: 'EffNet-B0 (1280-dim) + DenseNet-121 (1024-dim) -> Projection MLP (512) -> Softmax (5 Classes)',
  ),
  ModelInfo(
    name: 'ResNet50 Clinical Baseline',
    description: 'Residual 50-layer deep convolutional network with ImageNet transfer weights fine-tuned on APTOS.',
    accuracy: 0.931,
    f1Score: 0.910,
    aucScore: 0.962,
    qwk: 0.874,
    backboneDetails: 'ResNet-50 (2048-dim penultimate) -> Dropout(0.3) -> Linear classifier (5 Classes)',
  ),
];
