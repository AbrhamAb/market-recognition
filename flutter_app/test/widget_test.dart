// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:hive/hive.dart';

import 'package:wagaLine/main.dart';

void main() {
  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    final dir = await Directory.systemTemp.createTemp('hive_test');
    Hive.init(dir.path);
  });

  testWidgets('MarketRecApp builds and shows primary actions',
      (WidgetTester tester) async {
    await tester.pumpWidget(const MarketRecApp());
    await tester.pump(const Duration(milliseconds: 1300));

    expect(find.text('wagaLine'), findsWidgets);
    expect(find.text('Scan Item'), findsOneWidget);
  });
}
