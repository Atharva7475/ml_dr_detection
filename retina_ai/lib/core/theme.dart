import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Clinical Design System for Professional Retinal Diagnostic Workstation
/// Molded according to top-tier medical UI benchmark standards.
class AppColors {
  // Deep Obsidian & Midnight Navy Surfaces
  static const Color background      = Color(0xFF070C15);
  static const Color surface         = Color(0xFF0F1728);
  static const Color surfaceElevated = Color(0xFF162238);
  static const Color card            = Color(0xFF111927);
  static const Color cardBorder      = Color(0xFF1E2C44);
  static const Color cardBorderLight = Color(0xFF2B3D5E);

  // Professional Medical Brand Palette (Cerulean, Clinical Sapphire, Icy White)
  static const Color primary         = Color(0xFF38A3E5); // Vibrant Cerulean Button
  static const Color primaryDark     = Color(0xFF0284C7);
  static const Color primaryLight    = Color(0xFF60B5F6);
  static const Color primaryGlow     = Color(0x1F38A3E5);
  
  static const Color accentTeal      = Color(0xFF14B8A6);
  static const Color accentNavy      = Color(0xFF1E3A8A);

  // Clinical Status Badges (Direct from reference UI)
  static const Color badgeSessionBg   = Color(0xFF172C4C);
  static const Color badgeSessionText = Color(0xFF93C5FD);
  
  static const Color badgeHighRiskBg   = Color(0xFF36181E);
  static const Color badgeHighRiskText = Color(0xFFF87171);
  static const Color badgeHighRiskBorder = Color(0xFF57252C);

  static const Color badgeRoutineBg   = Color(0xFF0F2C23);
  static const Color badgeRoutineText = Color(0xFF34D399);
  static const Color badgeRoutineBorder = Color(0xFF194A3B);

  static const Color badgeModerateBg   = Color(0xFF332415);
  static const Color badgeModerateText = Color(0xFFFBBF24);
  static const Color badgeModerateBorder = Color(0xFF543B20);

  // ICDR Standard Clinical Severity Color Scale
  static const Color gradeNormal     = Color(0xFF10B981); // Stage 0: No DR (Emerald)
  static const Color gradeMild       = Color(0xFFF59E0B); // Stage 1: Mild NPDR (Amber)
  static const Color gradeModerate   = Color(0xFFF97316); // Stage 2: Moderate NPDR (Orange)
  static const Color gradeSevere     = Color(0xFFEF4444); // Stage 3: Severe NPDR (Crimson)
  static const Color gradeProlif     = Color(0xFFA855F7); // Stage 4: Proliferative DR (Purple)

  // Typography Palette
  static const Color textPrimary     = Color(0xFFF8FAFC);
  static const Color textSecondary   = Color(0xFF94A3B8);
  static const Color textMuted       = Color(0xFF64748B);
  static const Color textDim         = Color(0xFF475569);

  // Gradients
  static const LinearGradient primaryGradient = LinearGradient(
    colors: [Color(0xFF0284C7), Color(0xFF38A3E5)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );

  static const LinearGradient backgroundGradient = LinearGradient(
    colors: [Color(0xFF060A12), Color(0xFF0A111E)],
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
  );
}

class AppTheme {
  static ThemeData get dark {
    final base = ThemeData.dark();
    final textTheme = GoogleFonts.plusJakartaSansTextTheme(base.textTheme);

    return base.copyWith(
      scaffoldBackgroundColor: AppColors.background,
      colorScheme: const ColorScheme.dark(
        primary:   AppColors.primary,
        secondary: AppColors.accentTeal,
        surface:   AppColors.surface,
        error:     AppColors.gradeSevere,
      ),
      textTheme: textTheme.copyWith(
        displayLarge: GoogleFonts.plusJakartaSans(
          fontSize: 28, fontWeight: FontWeight.w800, color: AppColors.textPrimary, letterSpacing: -0.5,
        ),
        displayMedium: GoogleFonts.plusJakartaSans(
          fontSize: 22, fontWeight: FontWeight.w700, color: AppColors.textPrimary, letterSpacing: -0.3,
        ),
        titleLarge: GoogleFonts.plusJakartaSans(
          fontSize: 18, fontWeight: FontWeight.w700, color: AppColors.textPrimary,
        ),
        titleMedium: GoogleFonts.plusJakartaSans(
          fontSize: 15, fontWeight: FontWeight.w600, color: AppColors.textPrimary,
        ),
        bodyLarge: GoogleFonts.plusJakartaSans(
          fontSize: 14, fontWeight: FontWeight.w400, color: AppColors.textPrimary, height: 1.5,
        ),
        bodyMedium: GoogleFonts.plusJakartaSans(
          fontSize: 13, fontWeight: FontWeight.w400, color: AppColors.textSecondary, height: 1.4,
        ),
        bodySmall: GoogleFonts.plusJakartaSans(
          fontSize: 11, fontWeight: FontWeight.w500, color: AppColors.textMuted,
        ),
        labelLarge: GoogleFonts.plusJakartaSans(
          fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.textPrimary, letterSpacing: 0.3,
        ),
      ),
      cardTheme: CardThemeData(
        color: AppColors.card,
        elevation: 0,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(18),
          side: const BorderSide(color: AppColors.cardBorder, width: 1),
        ),
      ),
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: IconThemeData(color: AppColors.textPrimary),
        titleTextStyle: TextStyle(
          color: AppColors.textPrimary,
          fontSize: 17,
          fontWeight: FontWeight.w700,
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: AppColors.primary,
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 20),
          elevation: 0,
          textStyle: GoogleFonts.plusJakartaSans(fontWeight: FontWeight.w700, fontSize: 14),
        ),
      ),
    );
  }
}

/// International Clinical Diabetic Retinopathy (ICDR) Standard Helper
class DRGrade {
  static const List<String> names = [
    'No Apparent Retinopathy',
    'Mild NPDR',
    'Moderate NPDR',
    'Severe NPDR',
    'Proliferative DR',
  ];

  static const List<String> shortNames = [
    'Stage 0 • Normal',
    'Stage 1 • Mild',
    'Stage 2 • Moderate',
    'Stage 3 • Severe',
    'Stage 4 • PDR',
  ];

  static const List<String> clinicalDescriptions = [
    'No microaneurysms, hemorrhages, or retinal lesions detected. Posterior pole and vascular arcades unremarkable.',
    'Microaneurysms only. Early diabetic capillary changes without macular edema or retinal ischemia.',
    'More than microaneurysms but less than severe NPDR. Notable dot/blot hemorrhages and/or hard lipid exudates.',
    'Severe NPDR (4-2-1 rule): >20 intraretinal hemorrhages in all 4 quadrants, venous beading in ≥2 quadrants, or prominent IRMA in ≥1 quadrant.',
    'Proliferative Diabetic Retinopathy (PDR): Frank neovascularization of the disc/retina (NVD/NVE) or vitreous/preretinal hemorrhage.',
  ];

  static const List<String> clinicalRecommendations = [
    'Annual routine dilated ophthalmic screening. Maintain baseline HbA1c & glycemic controls.',
    'Follow-up dilated fundus exam in 6-12 months. Monitor blood pressure and serum lipids.',
    'Referral to comprehensive ophthalmologist in 3-6 months. Obtain macular OCT if DME suspected.',
    'Urgent referral to vitreoretinal subspecialist within 2-4 weeks. High risk of near-term neovascular conversion.',
    'Immediate retinal specialist consultation within 24-48 hours. Evaluate for panretinal photocoagulation (PRP) or anti-VEGF therapy.',
  ];

  static const List<String> icd10Codes = [
    'Z01.00 (Ophthalmic exam normal)',
    'E11.329 (Type 2 DM with mild NPDR)',
    'E11.339 (Type 2 DM with moderate NPDR)',
    'E11.349 (Type 2 DM with severe NPDR)',
    'E11.359 (Type 2 DM with proliferative DR)',
  ];

  static const List<Color> colors = [
    AppColors.gradeNormal,
    AppColors.gradeMild,
    AppColors.gradeModerate,
    AppColors.gradeSevere,
    AppColors.gradeProlif,
  ];

  static const List<IconData> icons = [
    Icons.verified_outlined,
    Icons.info_outline,
    Icons.warning_amber_rounded,
    Icons.error_outline_rounded,
    Icons.emergency_outlined,
  ];

  static Color colorForGrade(int g) => (g >= 0 && g < 5) ? colors[g] : AppColors.textMuted;
  static String nameForGrade(int g)  => (g >= 0 && g < 5) ? names[g]  : 'Indeterminate';
  static String shortNameForGrade(int g) => (g >= 0 && g < 5) ? shortNames[g] : 'Unknown';
  static IconData iconForGrade(int g) => (g >= 0 && g < 5) ? icons[g] : Icons.help_outline;
  static String descriptionForGrade(int g) => (g >= 0 && g < 5) ? clinicalDescriptions[g] : 'Clinical evaluation required.';
  static String recommendationForGrade(int g) => (g >= 0 && g < 5) ? clinicalRecommendations[g] : 'Consult specialist.';
  static String icd10ForGrade(int g) => (g >= 0 && g < 5) ? icd10Codes[g] : 'E11.319';
}
