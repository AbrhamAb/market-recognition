import 'package:hive/hive.dart';

class HistoryStore {
  static const String _boxName = 'prediction_history';
  static const String _recordsKey = 'records';

  Box? _box;

  Future<void> init() async {
    if (_box != null && _box!.isOpen) return;
    _box = await Hive.openBox(_boxName);
  }

  Future<List<Map<String, dynamic>>> loadRecords() async {
    await init();
    final raw = (_box!.get(_recordsKey, defaultValue: const []) as List)
        .whereType<dynamic>()
        .toList();
    return raw
        .map((e) => Map<String, dynamic>.from(e as Map))
        .toList(growable: false);
  }

  Future<void> addRecord(Map<String, dynamic> record) async {
    await init();
    final existing = (_box!.get(_recordsKey, defaultValue: const []) as List)
        .whereType<dynamic>()
        .toList();
    _box!.put(_recordsKey, [record, ...existing]);
  }

  /// Delete records at the given list of indices (indices correspond to
  /// the order returned by `loadRecords()` — newest first).
  Future<void> deleteRecords(List<int> indices) async {
    await init();
    final existing = (_box!.get(_recordsKey, defaultValue: const []) as List)
        .toList(growable: true);
    if (existing.isEmpty) return;
    final unique = indices.toSet().where((i) => i >= 0).toList();
    unique.sort((a, b) => b.compareTo(a));
    for (final idx in unique) {
      if (idx < existing.length) existing.removeAt(idx);
    }
    await _box!.put(_recordsKey, existing);
  }

  /// Clear all stored history records.
  Future<void> clearAll() async {
    await init();
    await _box!.put(_recordsKey, <dynamic>[]);
  }
}
