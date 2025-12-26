import 'dart:io';

import 'package:dio/dio.dart';

class ApiClient {
  ApiClient({
    Dio? dio,
    this.baseUrl = const String.fromEnvironment(
      'BACKEND_URL',
      defaultValue: 'http://10.0.2.2:8000',
    ),
  }) : _dio = dio ??
            Dio(BaseOptions(
                connectTimeout: const Duration(seconds: 15),
                receiveTimeout: const Duration(seconds: 30)));

  final Dio _dio;
  final String baseUrl;

  Future<Map<String, dynamic>> predict({
    required File image,
    required String vendorId,
    required double qty,
    double? buyPricePerUnit,
  }) async {
    final formData = FormData.fromMap({
      'file': await MultipartFile.fromFile(image.path,
          filename: image.uri.pathSegments.isNotEmpty
              ? image.uri.pathSegments.last
              : 'upload.jpg'),
      'vendor_id': vendorId,
      'qty': qty,
      'buy_price_per_unit': buyPricePerUnit ?? '',
    });

    final resp = await _dio.post('$baseUrl/predict', data: formData);
    if (resp.statusCode != null &&
        resp.statusCode! >= 200 &&
        resp.statusCode! < 300) {
      return Map<String, dynamic>.from(resp.data as Map);
    }
    throw Exception('Predict failed with status ${resp.statusCode}');
  }

  Future<Map<String, dynamic>?> getPrice(String itemKey) async {
    try {
      final resp = await _dio.get('$baseUrl/prices/$itemKey');
      if (resp.statusCode != null &&
          resp.statusCode! >= 200 &&
          resp.statusCode! < 300) {
        return Map<String, dynamic>.from(resp.data as Map);
      }
      return null;
    } catch (_) {
      return null;
    }
  }
}
