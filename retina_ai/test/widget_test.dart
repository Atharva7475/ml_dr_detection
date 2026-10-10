import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:retina_ai/main.dart';
import 'package:retina_ai/providers/app_provider.dart';

void main() {
  testWidgets('Retinal Diagnostic Suite smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(
      ChangeNotifierProvider(
        create: (_) => AppProvider(),
        child: const RetinaAIApp(),
      ),
    );
    expect(find.byType(RetinaAIApp), findsOneWidget);
    await tester.pumpAndSettle();
  });
}
