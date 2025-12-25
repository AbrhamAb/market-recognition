import 'dart:io';

import 'package:flutter/foundation.dart';

import 'api_client.dart';
import 'history_store.dart';

class PredictionRecord {
  PredictionRecord({
    required this.item,
    required this.unit,
    required this.qty,
    required this.pricePerUnit,
    required this.total,
    required this.confidence,
    required this.timestamp,
    required this.suggestions,
    this.imagePath,
  });

  factory PredictionRecord.fromResult(
    Map<String, dynamic> result, {
    String? imagePath,
    DateTime? timestamp,
  }) {
    final qty = (result['qty'] as num?)?.toDouble() ?? 1.0;
    final pricePerUnit = (result['price_per_unit'] as num?)?.toDouble() ?? 0.0;
    final total = (result['total'] as num?)?.toDouble() ?? pricePerUnit * qty;
    final confidence = (result['confidence'] as num?)?.toDouble() ?? 0.0;
    final unit = (result['unit'] ?? '').toString();
    final item = (result['item'] ?? 'Unknown item').toString();
    final suggestions = ((result['top_k'] as List?) ?? [])
        .whereType<Map>()
        .map((e) => (e['label'] ?? '').toString())
        .where((e) => e.isNotEmpty)
        .toList();

    return PredictionRecord(
      item: item,
      unit: unit,
      qty: qty,
      pricePerUnit: pricePerUnit,
      total: total,
      confidence: confidence,
      timestamp: timestamp ?? DateTime.now(),
      imagePath: imagePath,
      suggestions: suggestions,
    );
  }

  final String item;
  final String unit;
  final double qty;
  final double pricePerUnit;
  final double total;
  final double confidence;
  final DateTime timestamp;
  final String? imagePath;
  final List<String> suggestions;

  Map<String, dynamic> toMap() {
    return {
      'item': item,
      'unit': unit,
      'qty': qty,
      'price_per_unit': pricePerUnit,
      'total': total,
      'confidence': confidence,
      'timestamp': timestamp.toIso8601String(),
      'image_path': imagePath,
      'suggestions': suggestions,
    };
  }

  static PredictionRecord fromMap(Map<String, dynamic> map) {
    final qty = (map['qty'] as num?)?.toDouble() ?? 1.0;
    final pricePerUnit = (map['price_per_unit'] as num?)?.toDouble() ?? 0.0;
    final total = (map['total'] as num?)?.toDouble() ?? pricePerUnit * qty;
    final confidence = (map['confidence'] as num?)?.toDouble() ?? 0.0;
    final unit = (map['unit'] ?? '').toString();
    final item = (map['item'] ?? 'Unknown item').toString();
    final suggestions =
        ((map['suggestions'] as List?) ?? []).whereType<String>().toList();
    final tsString = (map['timestamp'] ?? '').toString();
    final timestamp = DateTime.tryParse(tsString) ?? DateTime.now();
    final imagePath = (map['image_path'] ?? '') as String?;

    return PredictionRecord(
      item: item,
      unit: unit,
      qty: qty,
      pricePerUnit: pricePerUnit,
      total: total,
      confidence: confidence,
      timestamp: timestamp,
      imagePath: (imagePath?.isEmpty ?? true) ? null : imagePath,
      suggestions: suggestions,
    );
  }
}

class PredictState extends ChangeNotifier {
  PredictState(this._apiClient, this._historyStore) {
    _init();
  }

  final ApiClient _apiClient;
  final HistoryStore _historyStore;

  File? image;
  bool busy = false;
  bool saving = false;
  bool loadingHistory = true;
  Map<String, dynamic>? result;
  String? error;
  final List<PredictionRecord> history = [];

  Future<void> _init() async {
    await _historyStore.init();
    final stored = await _historyStore.loadRecords();
    history
      ..clear()
      ..addAll(
        stored.map((e) => PredictionRecord.fromMap(e)).toList(growable: false),
      );
    loadingHistory = false;
    notifyListeners();
  }

  void setImage(File? file) {
    image = file;
    notifyListeners();
  }

  void clearResult() {
    result = null;
    notifyListeners();
  }

  Future<bool> acceptCurrentResult() async {
    if (result == null) return false;
    final record = PredictionRecord.fromResult(
      result!,
      imagePath: image?.path,
      timestamp: DateTime.now(),
    );

    saving = true;
    notifyListeners();
    try {
      await _historyStore.addRecord(record.toMap());
      history.insert(0, record);
      return true;
    } finally {
      saving = false;
      notifyListeners();
    }
  }

  /// Delete stored history records at the provided indices (newest-first order).
  Future<void> deleteHistory(List<int> indices) async {
    if (indices.isEmpty) return;
    // Update persistent store first
    await _historyStore.deleteRecords(indices);

    // Remove from in-memory list (ensure descending order to avoid shifting)
    final unique = indices.toSet().where((i) => i >= 0).toList();
    unique.sort((a, b) => b.compareTo(a));
    for (final idx in unique) {
      if (idx < history.length) history.removeAt(idx);
    }
    notifyListeners();
  }

  Future<void> predict({
    required String vendorId,
    required double qty,
    double? buyPricePerUnit,
  }) async {
    if (image == null) {
      error = 'Please pick an image first';
      notifyListeners();
      return;
    }
    busy = true;
    error = null;
    result = null;
    notifyListeners();
    try {
      result = await _apiClient.predict(
        image: image!,
        vendorId: vendorId.isEmpty ? 'unknown' : vendorId,
        qty: qty,
        buyPricePerUnit: buyPricePerUnit,
      );
    } catch (e) {
      error = e.toString();
    } finally {
      busy = false;
      notifyListeners();
    }
  }
}
