import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:animate_do/animate_do.dart';
import 'package:google_fonts/google_fonts.dart';
import '../core/theme.dart';
import '../models/scan_result.dart';
import '../providers/app_provider.dart';
import 'results_screen.dart';

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});
  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  int _filterGrade = -1;
  String _searchQuery = '';
  final TextEditingController _searchController = TextEditingController();

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<AppProvider>();
    
    final filteredHistory = provider.history.where((r) {
      final matchesGrade = _filterGrade < 0 || r.predictedGrade == _filterGrade;
      final matchesQuery = _searchQuery.isEmpty || 
          r.patientId.toLowerCase().contains(_searchQuery.toLowerCase()) ||
          r.id.toLowerCase().contains(_searchQuery.toLowerCase());
      return matchesGrade && matchesQuery;
    }).toList();

    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text('Patient Diagnostic Registry', style: GoogleFonts.plusJakartaSans(fontSize: 16, fontWeight: FontWeight.w700)),
        centerTitle: true,
        backgroundColor: AppColors.surface,
        elevation: 0,
        actions: [
          Container(
            margin: const EdgeInsets.only(right: 16),
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: AppColors.surfaceElevated,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: AppColors.cardBorder),
            ),
            child: Text('${provider.history.length} Cases', style: GoogleFonts.plusJakartaSans(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.primaryLight)),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 10),
              child: _buildSearchBar(),
            ),
            _buildFilterChips(),
            const SizedBox(height: 10),
            Expanded(
              child: filteredHistory.isEmpty ? _buildEmptyState() : _buildList(filteredHistory),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSearchBar() {
    return Container(
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppColors.cardBorder),
      ),
      child: TextField(
        controller: _searchController,
        onChanged: (val) => setState(() => _searchQuery = val.trim()),
        style: GoogleFonts.plusJakartaSans(fontSize: 13, color: AppColors.textPrimary),
        decoration: InputDecoration(
          hintText: 'Search by Patient ID (e.g. PT-8942)...',
          hintStyle: GoogleFonts.plusJakartaSans(fontSize: 13, color: AppColors.textMuted),
          prefixIcon: const Icon(Icons.search_rounded, size: 20, color: AppColors.textMuted),
          suffixIcon: _searchQuery.isNotEmpty
              ? IconButton(
                  icon: const Icon(Icons.clear_rounded, size: 18, color: AppColors.textMuted),
                  onPressed: () {
                    _searchController.clear();
                    setState(() => _searchQuery = '');
                  },
                )
              : null,
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(vertical: 14),
        ),
      ),
    );
  }

  Widget _buildFilterChips() {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        children: [
          _RegistryChip(
            label: 'All Stages',
            selected: _filterGrade < 0,
            color: AppColors.primary,
            onTap: () => setState(() => _filterGrade = -1),
          ),
          const SizedBox(width: 8),
          ...List.generate(5, (i) => Padding(
            padding: const EdgeInsets.only(right: 8),
            child: _RegistryChip(
              label: 'Stage $i',
              selected: _filterGrade == i,
              color: DRGrade.colorForGrade(i),
              onTap: () => setState(() => _filterGrade = i),
            ),
          )),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: AppColors.surfaceElevated,
                shape: BoxShape.circle,
                border: Border.all(color: AppColors.cardBorder),
              ),
              child: const Icon(Icons.folder_open_rounded, size: 40, color: AppColors.textMuted),
            ),
            const SizedBox(height: 16),
            Text('No Matching Clinical Records', style: GoogleFonts.plusJakartaSans(fontSize: 15, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
            const SizedBox(height: 6),
            Text('No archived fundus examinations match the selected filter criteria.', textAlign: TextAlign.center, style: GoogleFonts.plusJakartaSans(fontSize: 12, color: AppColors.textMuted)),
          ],
        ),
      ),
    );
  }

  Widget _buildList(List<ScanResult> history) {
    return ListView.builder(
      physics: const BouncingScrollPhysics(),
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
      itemCount: history.length,
      itemBuilder: (ctx, i) {
        final scan = history[i];
        final grade = scan.predictedGrade;
        final gradeColor = DRGrade.colorForGrade(grade);

        return FadeInUp(
          delay: Duration(milliseconds: i * 40),
          child: Container(
            margin: const EdgeInsets.only(bottom: 10),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: AppColors.cardBorder),
            ),
            child: InkWell(
              onTap: () {
                Navigator.of(context).push(MaterialPageRoute(
                  builder: (_) => ResultsScreen(
                    result: scan,
                    imageFile: null,
                    assetPath: scan.imagePath,
                  ),
                ));
              },
              borderRadius: BorderRadius.circular(16),
              child: Padding(
                padding: const EdgeInsets.all(14),
                child: Row(
                  children: [
                    // Retinal fundus thumbnail
                    Container(
                      width: 52,
                      height: 52,
                      decoration: BoxDecoration(
                        borderRadius: BorderRadius.circular(12),
                        color: Colors.black,
                        border: Border.all(color: gradeColor.withValues(alpha: 0.5)),
                      ),
                      clipBehavior: Clip.antiAlias,
                      child: _buildThumbImage(scan.imagePath),
                    ),
                    const SizedBox(width: 14),
                    // Details
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Text(scan.patientId, style: GoogleFonts.plusJakartaSans(fontSize: 13, fontWeight: FontWeight.w800, color: AppColors.textPrimary)),
                              const SizedBox(width: 8),
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                decoration: BoxDecoration(
                                  color: AppColors.surfaceElevated,
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: Text(scan.eyeLaterality, style: GoogleFonts.plusJakartaSans(fontSize: 10, fontWeight: FontWeight.w600, color: AppColors.textSecondary)),
                              ),
                            ],
                          ),
                          const SizedBox(height: 6),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: gradeColor.withValues(alpha: 0.15),
                              borderRadius: BorderRadius.circular(4),
                              border: Border.all(color: gradeColor.withValues(alpha: 0.3)),
                            ),
                            child: Text(
                              DRGrade.shortNameForGrade(grade),
                              style: GoogleFonts.plusJakartaSans(fontSize: 10, fontWeight: FontWeight.w700, color: gradeColor),
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            '${scan.scannedAt.day}/${scan.scannedAt.month}/${scan.scannedAt.year} • ${scan.scannedAt.hour.toString().padLeft(2, "0")}:${scan.scannedAt.minute.toString().padLeft(2, "0")}',
                            style: GoogleFonts.plusJakartaSans(fontSize: 10, color: AppColors.textMuted),
                          ),
                        ],
                      ),
                    ),
                    // Confidence
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          '${(scan.probabilities[grade] * 100).toStringAsFixed(1)}%',
                          style: GoogleFonts.plusJakartaSans(fontSize: 14, fontWeight: FontWeight.w800, color: gradeColor),
                        ),
                        Text('Confidence', style: GoogleFonts.plusJakartaSans(fontSize: 9, color: AppColors.textMuted)),
                        const SizedBox(height: 6),
                        const Icon(Icons.chevron_right_rounded, size: 18, color: AppColors.textMuted),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildThumbImage(String path) {
    if (path.startsWith('assets/')) {
      return Image.asset(path, fit: BoxFit.cover, errorBuilder: (_, __, ___) => const Icon(Icons.remove_red_eye_outlined, color: AppColors.primaryLight));
    }
    try {
      final f = File(path);
      if (f.existsSync()) {
        return Image.file(f, fit: BoxFit.cover);
      }
    } catch (_) {}
    return const Icon(Icons.remove_red_eye_rounded, color: AppColors.primaryLight);
  }
}

class _RegistryChip extends StatelessWidget {
  final String label;
  final bool selected;
  final Color color;
  final VoidCallback onTap;

  const _RegistryChip({
    required this.label,
    required this.selected,
    required this.color,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: selected ? color.withValues(alpha: 0.2) : AppColors.surface,
          borderRadius: BorderRadius.circular(8),
          border: Border.all(color: selected ? color : AppColors.cardBorder),
        ),
        child: Text(
          label,
          style: GoogleFonts.plusJakartaSans(
            fontSize: 11,
            fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
            color: selected ? (color == AppColors.primary ? AppColors.primaryLight : color) : AppColors.textSecondary,
          ),
        ),
      ),
    );
  }
}
