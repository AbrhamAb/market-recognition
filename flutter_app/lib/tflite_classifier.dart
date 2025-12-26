import 'dart:io';
import 'package:flutter/services.dart';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class TFLiteClassifier {
  Interpreter? _interpreter;
  List<String> _labels = [];
  final int inputSize;
  List<int>? _inputShape;

  TFLiteClassifier({this.inputSize = 224});

  Future<void> load(
      {String modelAsset = 'assets/models/product_classifier.tflite',
      String labelsAsset = 'assets/models/labels.txt'}) async {
    _interpreter = await Interpreter.fromAsset(modelAsset);
    final labelsData = await rootBundle.loadString(labelsAsset);
    _labels = labelsData.split('\n').where((s) => s.trim().isNotEmpty).toList();

    // Log input tensor shape and run a quick warmup inference to detect
    // shape/type mismatches early. Output will appear in `flutter run` logs.
    try {
      final inShape =
          _interpreter!.getInputTensor(0).shape; // e.g. [1,224,224,3]
      _inputShape = inShape;
      debugPrint('TFLite model input shape: $inShape');

      // Warmup: build nested zeros matching the interpreter input shape
      final List<int> warmupShape = _inputShape ?? [1, inputSize, inputSize, 3];
      List<dynamic> buildZeros(List<int> shape) {
        if (shape.length == 1) return List.filled(shape[0], 0.0);
        return List.generate(shape[0], (_) => buildZeros(shape.sublist(1)));
      }

      final inputTensor = buildZeros(warmupShape);
      final output = List.generate(1, (_) => List.filled(_labels.length, 0.0));
      _interpreter!.run(inputTensor, output);
      debugPrint('TFLite warmup inference succeeded');
    } catch (e, st) {
      debugPrint('TFLite load/warmup failed: $e');
      debugPrint(st.toString());
      // Keep interpreter assigned but rethrow so caller can handle.
      rethrow;
    }
  }

  bool get isLoaded => _interpreter != null && _labels.isNotEmpty;

  /// Predict top-k labels for the provided image file.
  /// Returns a list of maps: { 'label': String, 'confidence': double }
  List<Map<String, dynamic>> predict(File imageFile, {int topK = 3}) {
    if (!isLoaded) throw StateError('TFLite model not loaded');

    // Read and decode image using `package:image` to control resize and
    // pixel layout. Build a 4D input shaped [1, H, W, C] as nested Lists
    // of doubles which the Interpreter accepts.
    final bytes = imageFile.readAsBytesSync();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) throw StateError('Could not decode image');
    final img.Image resized =
        img.copyResize(decoded, width: inputSize, height: inputSize);

    // Build a flat Float32List input; support NHWC ([1,H,W,3]) and
    // NCHW ([1,3,H,W]) model layouts by ordering the float buffer accordingly.
    final int h = resized.height;
    final int w = resized.width;
    // Bytes per pixel may be 3 (RGB) or 4 (RGBA) depending on decoder.
    final Uint8List pixels = resized.getBytes();
    final int expectedPixels = w * h;
    final int bytesPerPixel = (pixels.length ~/ expectedPixels);
    debugPrint(
        'Resized: ${w}x$h, pixel bytes: ${pixels.length}, expected pixels: $expectedPixels, bpp: $bytesPerPixel');
    if (bytesPerPixel < 3) {
      throw StateError('Unexpected bytes-per-pixel: $bytesPerPixel');
    }

    // Build nested input matching model rank (NHWC or NCHW) and run inference
    _inputShape ??= _interpreter!.getInputTensor(0).shape;
    final List<int> runtimeShape = _inputShape!;
    final bool runtimeIsNCHW = runtimeShape.length == 4 && runtimeShape[1] == 3;

    List<List<double>> output =
        List.generate(1, (_) => List.filled(_labels.length, 0.0));

    if (runtimeIsNCHW) {
      // [1,3,h,w]
      final inputTensor = List.generate(
        1,
        (_) => List.generate(
          3,
          (_) => List.generate(h, (_) => List.filled(w, 0.0)),
        ),
      );
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final int base = (y * w + x) * bytesPerPixel;
          inputTensor[0][0][y][x] = pixels[base] / 255.0;
          inputTensor[0][1][y][x] = pixels[base + 1] / 255.0;
          inputTensor[0][2][y][x] = pixels[base + 2] / 255.0;
        }
      }
      debugPrint('Using runtime NCHW input tensor');
      _interpreter!.run(inputTensor, output);
    } else {
      // [1,h,w,3]
      final inputTensor = List.generate(
        1,
        (_) => List.generate(
          h,
          (_) => List.generate(w, (_) => List.filled(3, 0.0)),
        ),
      );
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final int base = (y * w + x) * bytesPerPixel;
          inputTensor[0][y][x][0] = pixels[base] / 255.0;
          inputTensor[0][y][x][1] = pixels[base + 1] / 255.0;
          inputTensor[0][y][x][2] = pixels[base + 2] / 255.0;
        }
      }
      debugPrint('Using runtime NHWC input tensor');
      _interpreter!.run(inputTensor, output);
    }

    final List<double> scores = output[0].cast<double>();
    debugPrint('TFLite raw scores (first 10): ${scores.take(10).toList()}');
    final List<int> indices = List<int>.generate(scores.length, (i) => i);
    indices.sort((a, b) => scores[b].compareTo(scores[a]));

    final top = indices
        .take(topK)
        .map((i) => {
              'label': _labels[i],
              'confidence': scores[i],
            })
        .toList(growable: false);

    debugPrint('TFLite top: $top');

    return top;
  }
}
