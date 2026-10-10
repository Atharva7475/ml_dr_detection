import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/theme.dart';
import '../providers/app_provider.dart';
import 'results_screen.dart';

class ScanScreen extends StatefulWidget {
  final String? initialSamplePath;
  const ScanScreen({super.key, this.initialSamplePath});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> with SingleTickerProviderStateMixin {
  File? _selectedFile;
  String? _selectedAssetPath;
  String _patientId = 'PT-8942';
  String _eyeLaterality = 'OD (Right Eye)';
  int _analysisStep = 0;
  late AnimationController _scanLineController;

  final List<String> _pipelineSteps = [
    'Illumination normalization & CLAHE contrast enhancement...',
    'Dual-backbone feature extraction (EffNet-B0 + DenseNet121)...',
    'Foveal & optic disc landmark verification...',
    'Evaluating microaneurysms, hemorrhages & exudates...',
    'Computing ICDR 5-tier classification & confidence index...',
  ];

  @override
  void initState() {
    super.initState();
    _scanLineController = AnimationController(vsync: this, duration: const Duration(milliseconds: 1800));
    if (widget.initialSamplePath != null) {
      _selectedAssetPath = widget.initialSamplePath;
    }
  }

  @override
  void dispose() {
    _scanLineController.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: source, imageQuality: 95);
    if (picked != null) {
      setState(() {
        _selectedFile = File(picked.path);
        _selectedAssetPath = null;
      });
    }
  }

  void _loadSample(String assetPath) {
    setState(() {
      _selectedAssetPath = assetPath;
      _selectedFile = null;
    });
  }

  Future<void> _startClinicalAnalysis() async {
    final provider = context.read<AppProvider>();
    final path = _selectedFile?.path ?? _selectedAssetPath;
    if (path == null) return;

    _scanLineController.repeat(reverse: true);

    // Simulate multi-stage telemetry
    for (int i = 0; i < _pipelineSteps.length; i++) {
      if (!mounted) {
        _scanLineController.stop();
        return;
      }
      setState(() => _analysisStep = i);
      await Future.delayed(const Duration(milliseconds: 400));
    }

    await provider.analyzeImage(
      imagePath: path,
      patientId: _patientId,
      eyeLaterality: _eyeLaterality,
    );

    _scanLineController.stop();

    if (mounted && provider.state == AnalysisState.done && provider.result != null) {
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => ResultsScreen(
          result: provider.result!,
          imageFile: _selectedFile,
          assetPath: _selectedAssetPath,
        ),
      ));
    }
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();
    final isAnalyzing = provider.state == AnalysisState.loading;

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text('Retinography Workbench', style: GoogleFonts.plusJakartaSans(fontSize: 16, fontWeight: FontWeight.w700)),
        centerTitle: true,
        backgroundColor: AppColors.surface,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 18),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions: [
          Container(
            margin: const EdgeInsets.only(right: 16),
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: AppColors.surfaceElevated,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: AppColors.cardBorder),
            ),
            child: Row(
              children: [
                const Icon(Icons.lens_blur_rounded, size: 14, color: AppColors.primaryLight),
                const SizedBox(width: 4),
                Text('45° MODALITY', style: GoogleFonts.plusJakartaSans(fontSize: 10, fontWeight: FontWeight.w700, color: AppColors.textSecondary)),
              ],
            ),
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildPatientMetadataBar(),
              const SizedBox(height: 18),
              _buildViewportArea(isAnalyzing),
              const SizedBox(height: 18),
              if (isAnalyzing) ...[
                _buildAnalysisProgress(),
              ] else ...[
                _buildSourceSelection(),
                const SizedBox(height: 16),
                _buildClinicalReferencePicker(),
                const SizedBox(height: 20),
                _buildActionButtons(),
              ],
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildPatientMetadataBar() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.cardBorder),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Expanded(
                child: Row(
                  children: [
                    const Icon(Icons.badge_outlined, size: 16, color: AppColors.primaryLight),
                    const SizedBox(width: 8),
                    Text('Patient ID: ', style: GoogleFonts.plusJakartaSans(fontSize: 12, color: AppColors.textMuted)),
                    Text(_patientId, style: GoogleFonts.plusJakartaSans(fontSize: 13, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
                  ],
                ),
              ),
              InkWell(
                onTap: () {
                  _showEditPatientDialog();
                },
                child: Text('Edit', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w600, color: AppColors.primaryLight)),
              ),
            ],
          ),
          const Divider(height: 16, color: AppColors.cardBorder),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Eye Examined:', style: GoogleFonts.plusJakartaSans(fontSize: 12, color: AppColors.textMuted)),
              Row(
                children: [
                  _LateralityChip(
                    label: 'OD (Right)',
                    selected: _eyeLaterality.contains('OD'),
                    onTap: () => setState(() => _eyeLaterality = 'OD (Right Eye)'),
                  ),
                  const SizedBox(width: 8),
                  _LateralityChip(
                    label: 'OS (Left)',
                    selected: _eyeLaterality.contains('OS'),
                    onTap: () => setState(() => _eyeLaterality = 'OS (Left Eye)'),
                  ),
                ],
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _showEditPatientDialog() {
    final controller = TextEditingController(text: _patientId);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        backgroundColor: AppColors.surface,
        title: Text('Edit Patient Record', style: GoogleFonts.plusJakartaSans(fontSize: 16, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(labelText: 'Patient Case Identifier', hintText: 'e.g. PT-9402'),
          style: GoogleFonts.plusJakartaSans(color: Colors.white),
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          ElevatedButton(
            onPressed: () {
              if (controller.text.trim().isNotEmpty) {
                setState(() => _patientId = controller.text.trim());
              }
              Navigator.pop(ctx);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  Widget _buildViewportArea(bool isAnalyzing) {
    final hasImage = _selectedFile != null || _selectedAssetPath != null;

    return Container(
      width: double.infinity,
      height: 290,
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: hasImage ? AppColors.primary.withValues(alpha: 0.6) : AppColors.cardBorder,
          width: hasImage ? 1.5 : 1,
        ),
        boxShadow: [
          if (hasImage) BoxShadow(color: AppColors.primary.withValues(alpha: 0.1), blurRadius: 20),
        ],
      ),
      clipBehavior: Clip.antiAlias,
      child: Stack(
        alignment: Alignment.center,
        children: [
          if (hasImage) ...[
            if (_selectedFile != null)
              Image.file(_selectedFile!, width: double.infinity, height: double.infinity, fit: BoxFit.cover)
            else if (_selectedAssetPath != null)
              Image.asset(_selectedAssetPath!, width: double.infinity, height: double.infinity, fit: BoxFit.cover),
          ] else ...[
            _buildOpticalTargetPlaceholder(),
          ],

          // Professional Reticle Overlay
          Positioned.fill(
            child: CustomPaint(
              painter: _OpticalCrosshairPainter(hasImage: hasImage),
            ),
          ),

          // Telemetry scanning line during analysis
          if (isAnalyzing)
            AnimatedBuilder(
              animation: _scanLineController,
              builder: (ctx, child) {
                return Positioned(
                  top: 290 * _scanLineController.value,
                  left: 0,
                  right: 0,
                  child: Container(
                    height: 2,
                    decoration: const BoxDecoration(
                      boxShadow: [
                        BoxShadow(color: AppColors.primary, blurRadius: 10, spreadRadius: 2),
                      ],
                      gradient: LinearGradient(
                        colors: [Colors.transparent, AppColors.primaryLight, Colors.transparent],
                      ),
                    ),
                  ),
                );
              },
            ),

          // Top Viewport Indicator
          Positioned(
            top: 12,
            left: 12,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.black.withValues(alpha: 0.65),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(color: AppColors.cardBorder),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Container(width: 6, height: 6, decoration: BoxDecoration(color: hasImage ? AppColors.gradeNormal : AppColors.textMuted, shape: BoxShape.circle)),
                  const SizedBox(width: 6),
                  Text(hasImage ? 'DIAGNOSTIC RESOLUTION 45°' : 'ALIGNMENT TARGET', style: GoogleFonts.plusJakartaSans(fontSize: 10, fontWeight: FontWeight.w700, color: Colors.white)),
                ],
              ),
            ),
          ),

          if (hasImage && !isAnalyzing)
            Positioned(
              bottom: 12,
              right: 12,
              child: InkWell(
                onTap: () => setState(() {
                  _selectedFile = null;
                  _selectedAssetPath = null;
                }),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.75),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: Colors.white24),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.refresh_rounded, size: 12, color: Colors.white),
                      const SizedBox(width: 4),
                      Text('Clear', style: GoogleFonts.plusJakartaSans(fontSize: 10, color: Colors.white, fontWeight: FontWeight.w600)),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildOpticalTargetPlaceholder() {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Container(
          width: 72,
          height: 72,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: AppColors.surfaceElevated,
            border: Border.all(color: AppColors.cardBorder, width: 1.5),
          ),
          child: const Icon(Icons.center_focus_strong_rounded, size: 36, color: AppColors.primaryLight),
        ),
        const SizedBox(height: 14),
        Text('Optical Posterior Pole Alignment', style: GoogleFonts.plusJakartaSans(fontSize: 14, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
        const SizedBox(height: 4),
        Text('Import digital color fundus photograph or choose reference sample', textAlign: TextAlign.center, style: GoogleFonts.plusJakartaSans(fontSize: 11, color: AppColors.textMuted)),
      ],
    );
  }

  Widget _buildAnalysisProgress() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.primary.withValues(alpha: 0.4)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2, valueColor: AlwaysStoppedAnimation(AppColors.primaryLight)),
              ),
              const SizedBox(width: 10),
              Text('CLINICAL INFERENCE ENGINE ACTIVE', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.primaryLight, letterSpacing: 0.8)),
            ],
          ),
          const SizedBox(height: 14),
          LinearProgressIndicator(
            value: (_analysisStep + 1) / _pipelineSteps.length,
            backgroundColor: AppColors.cardBorder,
            valueColor: const AlwaysStoppedAnimation(AppColors.primary),
            borderRadius: BorderRadius.circular(4),
            minHeight: 6,
          ),
          const SizedBox(height: 12),
          Text(
            _pipelineSteps[_analysisStep],
            style: GoogleFonts.plusJakartaSans(fontSize: 12, fontWeight: FontWeight.w600, color: AppColors.textPrimary),
          ),
          const SizedBox(height: 4),
          Text('Dual-Backbone Feature Concatenation • DenseNet121 + EfficientNet-B0', style: GoogleFonts.plusJakartaSans(fontSize: 10, color: AppColors.textMuted)),
        ],
      ),
    );
  }

  Widget _buildSourceSelection() {
    return Row(
      children: [
        Expanded(
          child: _SourceButton(
            icon: Icons.photo_library_outlined,
            label: 'Import Retinogram',
            subtitle: 'Device Gallery / PACS File',
            onTap: () => _pickImage(ImageSource.gallery),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: _SourceButton(
            icon: Icons.camera_alt_outlined,
            label: 'Fundus Camera',
            subtitle: 'Direct Ophthalmoscope',
            onTap: () => _pickImage(ImageSource.camera),
          ),
        ),
      ],
    );
  }

  Widget _buildClinicalReferencePicker() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.cardBorder),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.library_books_outlined, size: 16, color: AppColors.primaryLight),
              const SizedBox(width: 8),
              Text('CLINICAL REFERENCE TEST SAMPLES', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.textMuted, letterSpacing: 0.8)),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _QuickSamplePill(
                  title: 'Normal Retina',
                  subtitle: 'Stage 0 • Grade 0',
                  color: AppColors.gradeNormal,
                  isSelected: _selectedAssetPath == 'assets/images/retina_sample_normal.jpg',
                  onTap: () => _loadSample('assets/images/retina_sample_normal.jpg'),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _QuickSamplePill(
                  title: 'Severe NPDR',
                  subtitle: 'Stage 3 • Grade 3',
                  color: AppColors.gradeSevere,
                  isSelected: _selectedAssetPath == 'assets/images/retina_sample_pathology.jpg',
                  onTap: () => _loadSample('assets/images/retina_sample_pathology.jpg'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildActionButtons() {
    final hasImage = _selectedFile != null || _selectedAssetPath != null;

    return SizedBox(
      width: double.infinity,
      height: 52,
      child: ElevatedButton.icon(
        style: ElevatedButton.styleFrom(
          backgroundColor: hasImage ? AppColors.primary : AppColors.surfaceElevated,
          foregroundColor: hasImage ? Colors.white : AppColors.textMuted,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          elevation: hasImage ? 4 : 0,
        ),
        onPressed: hasImage ? _startClinicalAnalysis : null,
        icon: const Icon(Icons.analytics_rounded, size: 20),
        label: Text('Execute ICDR Retinal Evaluation', style: GoogleFonts.plusJakartaSans(fontSize: 14, fontWeight: FontWeight.w700)),
      ),
    );
  }
}

class _LateralityChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;

  const _LateralityChip({required this.label, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: selected ? AppColors.primary.withValues(alpha: 0.15) : AppColors.card,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: selected ? AppColors.primary : AppColors.cardBorder),
        ),
        child: Text(
          label,
          style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: selected ? FontWeight.w700 : FontWeight.w500, color: selected ? AppColors.primaryLight : AppColors.textSecondary),
        ),
      ),
    );
  }
}

class _SourceButton extends StatelessWidget {
  final IconData icon;
  final String label;
  final String subtitle;
  final VoidCallback onTap;

  const _SourceButton({required this.icon, required this.label, required this.subtitle, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(14),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 12),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.cardBorder),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Icon(icon, color: AppColors.primaryLight, size: 22),
            const SizedBox(height: 8),
            Text(label, style: GoogleFonts.plusJakartaSans(fontSize: 12, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
            const SizedBox(height: 2),
            Text(subtitle, style: GoogleFonts.plusJakartaSans(fontSize: 10, color: AppColors.textMuted)),
          ],
        ),
      ),
    );
  }
}

class _QuickSamplePill extends StatelessWidget {
  final String title;
  final String subtitle;
  final Color color;
  final bool isSelected;
  final VoidCallback onTap;

  const _QuickSamplePill({
    required this.title,
    required this.subtitle,
    required this.color,
    required this.isSelected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: isSelected ? color.withValues(alpha: 0.15) : AppColors.surfaceElevated,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: isSelected ? color : AppColors.cardBorder, width: isSelected ? 1.5 : 1),
        ),
        child: Row(
          children: [
            Container(width: 8, height: 8, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
            const SizedBox(width: 8),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(title, style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
                  Text(subtitle, style: GoogleFonts.plusJakartaSans(fontSize: 9, color: AppColors.textSecondary)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _OpticalCrosshairPainter extends CustomPainter {
  final bool hasImage;
  _OpticalCrosshairPainter({required this.hasImage});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = hasImage ? Colors.white.withValues(alpha: 0.25) : AppColors.cardBorder
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    final center = Offset(size.width / 2, size.height / 2);

    // Crosshair lines
    canvas.drawLine(Offset(center.dx - 30, center.dy), Offset(center.dx + 30, center.dy), paint);
    canvas.drawLine(Offset(center.dx, center.dy - 30), Offset(center.dx, center.dy + 30), paint);

    // Center Macula Target Ring
    canvas.drawCircle(center, 40, paint);

    // Optic disc offset indicator circle (Nasal side)
    final opticDiscCenter = Offset(center.dx + 65, center.dy - 5);
    final dashedPaint = Paint()
      ..color = hasImage ? AppColors.primaryLight.withValues(alpha: 0.4) : Colors.transparent
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;
    canvas.drawCircle(opticDiscCenter, 22, dashedPaint);
  }

  @override
  bool shouldRepaint(covariant _OpticalCrosshairPainter oldDelegate) => oldDelegate.hasImage != hasImage;
}
