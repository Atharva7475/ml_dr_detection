import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/theme.dart';
import '../models/scan_result.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text('Clinical Protocol & Standards', style: GoogleFonts.plusJakartaSans(fontSize: 16, fontWeight: FontWeight.w700)),
        centerTitle: true,
        backgroundColor: AppColors.surface,
        elevation: 0,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          physics: const BouncingScrollPhysics(),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildInstitutionalBanner(),
              const SizedBox(height: 20),
              _buildSectionTitle('ENSEMBLE ARCHITECTURES & BENCHMARKS'),
              const SizedBox(height: 12),
              ...kModels.map((m) => _buildModelValidationCard(m)),
              const SizedBox(height: 20),
              _buildSectionTitle('ICDR CLINICAL CLASSIFICATION CRITERIA'),
              const SizedBox(height: 12),
              _buildICDRProtocolTable(),
              const SizedBox(height: 20),
              _buildSectionTitle('VALIDATION DATASETS & CLINICAL AUDIT'),
              const SizedBox(height: 12),
              _buildValidationDatasetCard(),
              const SizedBox(height: 20),
              _buildRegulatoryNotice(),
              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInstitutionalBanner() {
    return FadeInDown(
      duration: const Duration(milliseconds: 500),
      child: Container(
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: AppColors.cardBorder),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: AppColors.primary.withValues(alpha: 0.12),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppColors.primary.withValues(alpha: 0.3)),
                  ),
                  child: const Icon(Icons.local_hospital_rounded, color: AppColors.primaryLight, size: 24),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('St. Jude Retinal Diagnostic Suite', style: GoogleFonts.plusJakartaSans(fontSize: 16, fontWeight: FontWeight.w800, color: AppColors.textPrimary)),
                      const SizedBox(height: 2),
                      Text('Clinical Decision Support System (CDSS) • v2.4.0', style: GoogleFonts.plusJakartaSans(fontSize: 11, color: AppColors.textSecondary)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),
            Text(
              'Engineered for comprehensive ophthalmologists, vitreoretinal specialists, and clinical screening programs. Adheres strictly to the International Clinical Diabetic Retinopathy (ICDR) Disease Severity Scale.',
              style: GoogleFonts.plusJakartaSans(fontSize: 12, color: AppColors.textSecondary, height: 1.5),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.textMuted, letterSpacing: 0.8),
    );
  }

  Widget _buildModelValidationCard(ModelInfo model) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
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
              Text(model.name, style: GoogleFonts.plusJakartaSans(fontSize: 14, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: AppColors.gradeNormal.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Text('QWK κ: ${model.qwk}', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.gradeNormal)),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(model.description, style: GoogleFonts.plusJakartaSans(fontSize: 12, color: AppColors.textSecondary, height: 1.4)),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(color: AppColors.surfaceElevated, borderRadius: BorderRadius.circular(10)),
            child: Text(
              model.backboneDetails,
              style: GoogleFonts.plusJakartaSans(fontSize: 10, color: AppColors.textMuted, height: 1.3),
            ),
          ),
          const SizedBox(height: 12),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _MetricItem(label: 'Accuracy', value: '${(model.accuracy * 100).toStringAsFixed(1)}%'),
              _MetricItem(label: 'Macro F1', value: '${(model.f1Score * 100).toStringAsFixed(1)}%'),
              _MetricItem(label: 'AUROC', value: '${(model.aucScore * 100).toStringAsFixed(1)}%'),
              _MetricItem(label: 'κ Score', value: '${model.qwk}'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildICDRProtocolTable() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.cardBorder),
      ),
      child: Column(
        children: List.generate(5, (i) {
          final color = DRGrade.colorForGrade(i);
          return Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 22,
                  height: 22,
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(6),
                    border: Border.all(color: color.withValues(alpha: 0.4)),
                  ),
                  alignment: Alignment.center,
                  child: Text('$i', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w800, color: color)),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(DRGrade.nameForGrade(i), style: GoogleFonts.plusJakartaSans(fontSize: 13, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
                      const SizedBox(height: 2),
                      Text(DRGrade.clinicalDescriptions[i], style: GoogleFonts.plusJakartaSans(fontSize: 11, color: AppColors.textSecondary, height: 1.3)),
                    ],
                  ),
                ),
              ],
            ),
          );
        }),
      ),
    );
  }

  Widget _buildValidationDatasetCard() {
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
              const Icon(Icons.storage_rounded, size: 16, color: AppColors.primaryLight),
              const SizedBox(width: 8),
              Text('APTOS 2019 Blindness Detection Cohort', style: GoogleFonts.plusJakartaSans(fontSize: 13, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            '3,662 validated fundus images evaluated under standardized mydriatic protocols across multiple clinical centers in rural India. Cross-validated against Messidor-2 benchmark cohort.',
            style: GoogleFonts.plusJakartaSans(fontSize: 11, color: AppColors.textSecondary, height: 1.4),
          ),
          const SizedBox(height: 10),
          const Row(
            children: [
              _DatasetStat(label: 'Training Split', value: '80% (2,929)'),
              SizedBox(width: 12),
              _DatasetStat(label: 'Validation Split', value: '10% (366)'),
              SizedBox(width: 12),
              _DatasetStat(label: 'Independent Test', value: '10% (367)'),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildRegulatoryNotice() {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppColors.surfaceElevated,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.cardBorder),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.health_and_safety_outlined, size: 18, color: AppColors.primaryLight),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'INVESTIGATIONAL CLINICAL WORKSTATION: Designed solely to assist licensed ophthalmologists and trained medical personnel. Final clinical diagnosis and patient management remains the sole responsibility of the attending practitioner.',
              style: GoogleFonts.plusJakartaSans(fontSize: 10, color: AppColors.textMuted, height: 1.4),
            ),
          ),
        ],
      ),
    );
  }
}

class _MetricItem extends StatelessWidget {
  final String label;
  final String value;
  const _MetricItem({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: GoogleFonts.plusJakartaSans(fontSize: 10, color: AppColors.textMuted)),
        const SizedBox(height: 2),
        Text(value, style: GoogleFonts.plusJakartaSans(fontSize: 13, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
      ],
    );
  }
}

class _DatasetStat extends StatelessWidget {
  final String label;
  final String value;
  const _DatasetStat({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: GoogleFonts.plusJakartaSans(fontSize: 9, color: AppColors.textMuted)),
        Text(value, style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
      ],
    );
  }
}
