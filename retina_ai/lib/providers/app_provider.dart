import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/scan_result.dart';

enum AnalysisState { idle, loading, done, error }

class AppProvider extends ChangeNotifier {
  AnalysisState _state = AnalysisState.idle;
  ScanResult? _result;
  String? _error;
  final List<ScanResult> _history = [];
  String _selectedModel = 'Dual-Backbone Fusion';
  int _totalScans = 0;

  AnalysisState get state => _state;
  ScanResult? get result => _result;
  String? get error => _error;
  List<ScanResult> get history => List.unmodifiable(_history);
  String get selectedModel => _selectedModel;
  int get totalScans => _totalScans;

  AppProvider() {
    _loadHistory();
  }

  void setModel(String model) {
    _selectedModel = model;
    notifyListeners();
  }

  Future<void> analyzeImage({
    required String imagePath,
    String patientId = 'PT-8942',
    String eyeLaterality = 'OD (Right Eye)',
  }) async {
    _state = AnalysisState.loading;
    _error = null;
    notifyListeners();

    try {
      // Simulate high-performance clinical inference pipeline
      await Future.delayed(const Duration(milliseconds: 2400));

      final result = ScanResult.mock(
        imagePath,
        model: _selectedModel,
        patientId: patientId,
        eyeLaterality: eyeLaterality,
      );

      _result = result;
      _history.insert(0, result);
      _totalScans = _history.length;
      await _saveHistory();

      _state = AnalysisState.done;
    } catch (e) {
      _state = AnalysisState.error;
      _error = e.toString();
    }

    notifyListeners();
  }

  void reset() {
    _state = AnalysisState.idle;
    _result = null;
    _error = null;
    notifyListeners();
  }

  Future<void> _saveHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final list = _history.map((r) => jsonEncode(r.toMap())).toList();
      await prefs.setStringList('retina_clinical_history', list);
    } catch (_) {}
  }

  Future<void> _loadHistory() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final list = prefs.getStringList('retina_clinical_history') ?? [];
      
      if (list.isNotEmpty) {
        _history.clear();
        for (final item in list) {
          try {
            final map = jsonDecode(item) as Map<String, dynamic>;
            _history.add(ScanResult.fromMap(map));
          } catch (_) {}
        }
      }

      // If empty, seed initial clinical reference records for immediate demonstration
      if (_history.isEmpty) {
        _history.addAll([
          ScanResult(
            id: 'REC-2026-9041',
            scannedAt: DateTime.now().subtract(const Duration(hours: 3)),
            imagePath: 'assets/images/retina_sample_pathology.jpg',
            predictedGrade: 3,
            probabilities: [0.01, 0.03, 0.14, 0.74, 0.08],
            quadraticWeightedKappa: 0.912,
            sensitivity: 0.964,
            specificity: 0.981,
            modelUsed: 'Dual-Backbone Fusion',
            patientId: 'PT-9041',
            eyeLaterality: 'OD (Right Eye)',
            imageQuality: 'Diagnostic Grade (Optimal 45°)',
          ),
          ScanResult(
            id: 'REC-2026-8832',
            scannedAt: DateTime.now().subtract(const Duration(days: 1, hours: 2)),
            imagePath: 'assets/images/retina_sample_normal.jpg',
            predictedGrade: 0,
            probabilities: [0.95, 0.03, 0.01, 0.005, 0.005],
            quadraticWeightedKappa: 0.912,
            sensitivity: 0.964,
            specificity: 0.981,
            modelUsed: 'Dual-Backbone Fusion',
            patientId: 'PT-8832',
            eyeLaterality: 'OS (Left Eye)',
            imageQuality: 'Diagnostic Grade (Optimal 45°)',
          ),
          ScanResult(
            id: 'REC-2026-7419',
            scannedAt: DateTime.now().subtract(const Duration(days: 2, hours: 5)),
            imagePath: 'assets/images/doctor_lab.jpg',
            predictedGrade: 2,
            probabilities: [0.04, 0.11, 0.65, 0.16, 0.04],
            quadraticWeightedKappa: 0.912,
            sensitivity: 0.964,
            specificity: 0.981,
            modelUsed: 'ResNet50 Clinical Baseline',
            patientId: 'PT-7419',
            eyeLaterality: 'OD (Right Eye)',
            imageQuality: 'Acceptable (Mild Flare)',
          ),
        ]);
      }

      _totalScans = _history.length;
      notifyListeners();
    } catch (_) {}
  }
}
