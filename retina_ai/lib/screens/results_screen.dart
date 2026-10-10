import 'dart:io';
import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:share_plus/share_plus.dart';
import '../core/theme.dart';
import '../models/scan_result.dart';
import 'scan_screen.dart';

class ResultsScreen extends StatefulWidget {
  final ScanResult result;
  final File? imageFile;
  final String? assetPath;

  const ResultsScreen({
    super.key,
    required this.result,
    this.imageFile,
    this.assetPath,
  });

  @override
  State<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen> {
  int _overlayMode = 0; // 0: Standard, 1: Anatomical Landmarks, 2: Pathology Overlay

  void _shareReport() {
    final grade = widget.result.predictedGrade;
    final text = '''
ST. JUDE OPHTHALMIC INSTITUTE - CLINICAL REPORT
Patient ID: ${widget.result.patientId}
Examined Eye: ${widget.result.eyeLaterality}
ICDR Grade: Stage $grade (${DRGrade.nameForGrade(grade)})
Confidence: ${(widget.result.probabilities[grade] * 100).toStringAsFixed(1)}%
ICD-10 Code: ${DRGrade.icd10ForGrade(grade)}
Recommendation: ${DRGrade.recommendationForGrade(grade)}
Evaluation Timestamp: ${widget.result.scannedAt.toLocal()}
''';
    Share.share(text, subject: 'Retinal Evaluation Report - ${widget.result.patientId}');
  }

  @override
  Widget build(BuildContext context) {
    final grade = widget.result.predictedGrade;
    final gradeColor = DRGrade.colorForGrade(grade);

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text('Ophthalmic Evaluation Report', style: GoogleFonts.plusJakartaSans(fontSize: 16, fontWeight: FontWeight.w700)),
        centerTitle: true,
        backgroundColor: AppColors.surface,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 18),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.share_outlined, size: 20, color: AppColors.primaryLight),
            onPressed: _shareReport,
            tooltip: 'Share Report',
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          physics: const BouncingScrollPhysics(),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildReportHeader(),
              const SizedBox(height: 16),
              _buildPrimaryVerdictBanner(grade, gradeColor),
              const SizedBox(height: 18),
              _buildInteractiveRetinogramViewer(gradeColor),
              const SizedBox(height: 18),
              _buildProbabilitySpectrum(grade, gradeColor),
              const SizedBox(height: 18),
              _buildDiagnosticTelemetry(),
              const SizedBox(height: 18),
              _buildClinicalRecommendationCard(grade, gradeColor),
              const SizedBox(height: 22),
              _buildActionFooter(context),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildReportHeader() {
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
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(6),
                    decoration: BoxDecoration(color: AppColors.surfaceElevated, borderRadius: BorderRadius.circular(8)),
                    child: const Icon(Icons.description_outlined, size: 16, color: AppColors.primaryLight),
                  ),
                  const SizedBox(width: 8),
                  Text('CASE ${widget.result.patientId}', style: GoogleFonts.plusJakartaSans(fontSize: 13, fontWeight: FontWeight.w800, color: AppColors.textPrimary)),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: AppColors.surfaceElevated,
                  borderRadius: BorderRadius.circular(6),
                  border: Border.all(color: AppColors.cardBorder),
                ),
                child: Text(widget.result.eyeLaterality, style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.primaryLight)),
              ),
            ],
          ),
          const Divider(height: 16, color: AppColors.cardBorder),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Modality: Digital Color Fundus 45°', style: GoogleFonts.plusJakartaSans(fontSize: 11, color: AppColors.textMuted)),
              Text(
                '${widget.result.scannedAt.day}/${widget.result.scannedAt.month}/${widget.result.scannedAt.year} • ${widget.result.scannedAt.hour.toString().padLeft(2, "0")}:${widget.result.scannedAt.minute.toString().padLeft(2, "0")}',
                style: GoogleFonts.plusJakartaSans(fontSize: 11, color: AppColors.textSecondary),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildPrimaryVerdictBanner(int grade, Color gradeColor) {
    return FadeInDown(
      duration: const Duration(milliseconds: 500),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: gradeColor.withValues(alpha: 0.5), width: 1.5),
          boxShadow: [
            BoxShadow(color: gradeColor.withValues(alpha: 0.08), blurRadius: 20),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: gradeColor.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(color: gradeColor.withValues(alpha: 0.4)),
                  ),
                  child: Icon(DRGrade.iconForGrade(grade), color: gradeColor, size: 22),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('PRIMARY DIAGNOSTIC STAGING', style: GoogleFonts.plusJakartaSans(fontSize: 10, fontWeight: FontWeight.w700, color: AppColors.textMuted, letterSpacing: 0.8)),
                      Text(DRGrade.shortNameForGrade(grade), style: GoogleFonts.plusJakartaSans(fontSize: 18, fontWeight: FontWeight.w800, color: gradeColor)),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: gradeColor.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Column(
                    children: [
                      Text('${(widget.result.probabilities[grade] * 100).toStringAsFixed(1)}%', style: GoogleFonts.plusJakartaSans(fontSize: 15, fontWeight: FontWeight.w800, color: gradeColor)),
                      Text('Confidence', style: GoogleFonts.plusJakartaSans(fontSize: 9, color: AppColors.textMuted)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              DRGrade.descriptionForGrade(grade),
              style: GoogleFonts.plusJakartaSans(fontSize: 12, color: AppColors.textSecondary, height: 1.4),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: AppColors.surfaceElevated,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: AppColors.cardBorder),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.code_rounded, size: 14, color: AppColors.textMuted),
                  const SizedBox(width: 6),
                  Text('ICD-10 Code: ${DRGrade.icd10ForGrade(grade)}', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w600, color: AppColors.textPrimary)),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildInteractiveRetinogramViewer(Color gradeColor) {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.cardBorder),
      ),
      padding: const EdgeInsets.all(14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('RETINOGRAM VISUALIZATION', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.textMuted, letterSpacing: 0.8)),
              Text('Interactive Layers', style: GoogleFonts.plusJakartaSans(fontSize: 10, color: AppColors.primaryLight, fontWeight: FontWeight.w600)),
            ],
          ),
          const SizedBox(height: 10),
          Container(
            height: 220,
            width: double.infinity,
            decoration: BoxDecoration(
              color: Colors.black,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppColors.cardBorder),
            ),
            clipBehavior: Clip.antiAlias,
            child: Stack(
              alignment: Alignment.center,
              children: [
                _buildFundusImage(),
                if (_overlayMode == 1) _buildLandmarksOverlay(),
                if (_overlayMode == 2) _buildPathologyOverlay(gradeColor),
                Positioned(
                  top: 8,
                  left: 8,
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
                    decoration: BoxDecoration(color: Colors.black.withValues(alpha: 0.7), borderRadius: BorderRadius.circular(4)),
                    child: Text(
                      _overlayMode == 0 ? 'Mode: Standard' : _overlayMode == 1 ? 'Mode: Landmarks' : 'Mode: Pathology Map',
                      style: GoogleFonts.plusJakartaSans(fontSize: 9, fontWeight: FontWeight.w700, color: Colors.white),
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              _OverlayButton(label: 'Standard', isSelected: _overlayMode == 0, onTap: () => setState(() => _overlayMode = 0)),
              const SizedBox(width: 8),
              _OverlayButton(label: 'Landmarks', isSelected: _overlayMode == 1, onTap: () => setState(() => _overlayMode = 1)),
              const SizedBox(width: 8),
              _OverlayButton(label: 'Pathology Map', isSelected: _overlayMode == 2, onTap: () => setState(() => _overlayMode = 2)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildFundusImage() {
    if (widget.imageFile != null) {
      return Image.file(widget.imageFile!, width: double.infinity, height: double.infinity, fit: BoxFit.cover);
    }
    if (widget.assetPath != null) {
      return Image.asset(widget.assetPath!, width: double.infinity, height: double.infinity, fit: BoxFit.cover);
    }
    if (widget.result.imagePath.startsWith('assets/')) {
      return Image.asset(widget.result.imagePath, width: double.infinity, height: double.infinity, fit: BoxFit.cover);
    }
    try {
      final f = File(widget.result.imagePath);
      if (f.existsSync()) {
        return Image.file(f, width: double.infinity, height: double.infinity, fit: BoxFit.cover);
      }
    } catch (_) {}
    return Image.asset('assets/images/retina_sample_normal.jpg', width: double.infinity, height: double.infinity, fit: BoxFit.cover);
  }

  Widget _buildLandmarksOverlay() {
    return Positioned.fill(
      child: CustomPaint(
        painter: _LandmarksPainter(),
      ),
    );
  }

  Widget _buildPathologyOverlay(Color gradeColor) {
    return Positioned.fill(
      child: CustomPaint(
        painter: _PathologyPainter(color: gradeColor),
      ),
    );
  }

  Widget _buildProbabilitySpectrum(int grade, Color gradeColor) {
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
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('ICDR PROBABILITY DISTRIBUTION', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.textMuted, letterSpacing: 0.8)),
              Text('Softmax Output', style: GoogleFonts.plusJakartaSans(fontSize: 10, color: AppColors.textSecondary)),
            ],
          ),
          const SizedBox(height: 14),
          ...List.generate(5, (i) {
            final prob = widget.result.probabilities[i];
            final isTarget = i == grade;
            final stageColor = DRGrade.colorForGrade(i);

            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        DRGrade.shortNameForGrade(i),
                        style: GoogleFonts.plusJakartaSans(
                          fontSize: 12,
                          fontWeight: isTarget ? FontWeight.w700 : FontWeight.w500,
                          color: isTarget ? stageColor : AppColors.textSecondary,
                        ),
                      ),
                      Text(
                        '${(prob * 100).toStringAsFixed(1)}%',
                        style: GoogleFonts.plusJakartaSans(
                          fontSize: 12,
                          fontWeight: isTarget ? FontWeight.w800 : FontWeight.w600,
                          color: isTarget ? stageColor : AppColors.textMuted,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(3),
                    child: LinearProgressIndicator(
                      value: prob,
                      backgroundColor: AppColors.surfaceElevated,
                      valueColor: AlwaysStoppedAnimation(isTarget ? stageColor : AppColors.cardBorderLight),
                      minHeight: 5,
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildDiagnosticTelemetry() {
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
          Text('VALIDATION TELEMETRY & MODEL INDEX', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.textMuted, letterSpacing: 0.8)),
          const SizedBox(height: 12),
          Row(
            children: [
              _TelemetryPill(label: 'κ Agreement', value: '${widget.result.quadraticWeightedKappa}', color: AppColors.gradeNormal),
              const SizedBox(width: 8),
              _TelemetryPill(label: 'Sensitivity', value: '${(widget.result.sensitivity * 100).toStringAsFixed(1)}%', color: AppColors.primaryLight),
              const SizedBox(width: 8),
              _TelemetryPill(label: 'Specificity', value: '${(widget.result.specificity * 100).toStringAsFixed(1)}%', color: AppColors.accentTeal),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'Ensemble: ${widget.result.modelUsed}',
            style: GoogleFonts.plusJakartaSans(fontSize: 11, color: AppColors.textSecondary),
          ),
        ],
      ),
    );
  }

  Widget _buildClinicalRecommendationCard(int grade, Color gradeColor) {
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
              Icon(Icons.assignment_turned_in_outlined, size: 18, color: gradeColor),
              const SizedBox(width: 8),
              Text('CLINICAL DIRECTIVE & FOLLOW-UP PLAN', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.textMuted, letterSpacing: 0.8)),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            DRGrade.recommendationForGrade(grade),
            style: GoogleFonts.plusJakartaSans(fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.textPrimary, height: 1.5),
          ),
          const SizedBox(height: 10),
          Text(
            '• Patient counseling regarding HbA1c < 7.0% target\n• Blood pressure target < 130/80 mmHg\n• Check for macular edema symptoms (blurred central vision, metamorphopsia)',
            style: GoogleFonts.plusJakartaSans(fontSize: 11, color: AppColors.textSecondary, height: 1.5),
          ),
        ],
      ),
    );
  }

  Widget _buildActionFooter(BuildContext context) {
    return Column(
      children: [
        SizedBox(
          width: double.infinity,
          height: 50,
          child: ElevatedButton.icon(
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primary,
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
            ),
            onPressed: () {
              Navigator.of(context).pushReplacement(MaterialPageRoute(
                builder: (_) => const ScanScreen(),
              ));
            },
            icon: const Icon(Icons.add_photo_alternate_rounded, size: 18),
            label: Text('New Patient Retinogram', style: GoogleFonts.plusJakartaSans(fontSize: 14, fontWeight: FontWeight.w700)),
          ),
        ),
      ],
    );
  }
}

class _OverlayButton extends StatelessWidget {
  final String label;
  final bool isSelected;
  final VoidCallback onTap;

  const _OverlayButton({required this.label, required this.isSelected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: Container(
          padding: const EdgeInsets.symmetric(vertical: 8),
          decoration: BoxDecoration(
            color: isSelected ? AppColors.primary.withValues(alpha: 0.2) : AppColors.surfaceElevated,
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: isSelected ? AppColors.primary : AppColors.cardBorder),
          ),
          alignment: Alignment.center,
          child: Text(
            label,
            style: GoogleFonts.plusJakartaSans(
              fontSize: 11,
              fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
              color: isSelected ? AppColors.primaryLight : AppColors.textSecondary,
            ),
          ),
        ),
      ),
    );
  }
}

class _TelemetryPill extends StatelessWidget {
  final String label;
  final String value;
  final Color color;

  const _TelemetryPill({required this.label, required this.value, required this.color});

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(10),
        decoration: BoxDecoration(
          color: AppColors.surfaceElevated,
          borderRadius: BorderRadius.circular(10),
          border: Border.all(color: AppColors.cardBorder),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label, style: GoogleFonts.plusJakartaSans(fontSize: 10, color: AppColors.textMuted)),
            const SizedBox(height: 4),
            Text(value, style: GoogleFonts.plusJakartaSans(fontSize: 14, fontWeight: FontWeight.w800, color: color)),
          ],
        ),
      ),
    );
  }
}

class _LandmarksPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);

    final foveaPaint = Paint()
      ..color = AppColors.primaryLight
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    // Fovea macula target
    canvas.drawCircle(center, 28, foveaPaint);
    canvas.drawLine(Offset(center.dx - 15, center.dy), Offset(center.dx + 15, center.dy), foveaPaint);
    canvas.drawLine(Offset(center.dx, center.dy - 15), Offset(center.dx, center.dy + 15), foveaPaint);

    // Optic disc circle
    final discCenter = Offset(center.dx + 65, center.dy - 8);
    final discPaint = Paint()
      ..color = Colors.amberAccent
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;
    canvas.drawCircle(discCenter, 22, discPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _PathologyPainter extends CustomPainter {
  final Color color;
  _PathologyPainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);

    final highlightPaint = Paint()
      ..color = color.withValues(alpha: 0.6)
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    final fillPaint = Paint()
      ..color = color.withValues(alpha: 0.2)
      ..style = PaintingStyle.fill;

    // Simulated lesion callouts
    final r1 = RRect.fromRectAndRadius(Rect.fromLTWH(center.dx - 45, center.dy - 35, 24, 24), const Radius.circular(6));
    canvas.drawRRect(r1, fillPaint);
    canvas.drawRRect(r1, highlightPaint);

    final r2 = RRect.fromRectAndRadius(Rect.fromLTWH(center.dx + 20, center.dy + 25, 30, 20), const Radius.circular(6));
    canvas.drawRRect(r2, fillPaint);
    canvas.drawRRect(r2, highlightPaint);

    final r3 = RRect.fromRectAndRadius(Rect.fromLTWH(center.dx - 20, center.dy + 40, 22, 22), const Radius.circular(6));
    canvas.drawRRect(r3, fillPaint);
    canvas.drawRRect(r3, highlightPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
