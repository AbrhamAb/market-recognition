**TFLite On-Device Integration**

- **Purpose:** run image classification locally in the Flutter app using a TFLite model.

- **Files (expected):**
  - `flutter_app/assets/models/product_classifier.tflite`
  - `flutter_app/assets/models/labels.txt` (one label per line)

- **Add plugins** (in `pubspec.yaml` dependencies):
  - `tflite_flutter` and `tflite_flutter_helper`.

Example `pubspec.yaml` dependency snippet:

```yaml
dependencies:
  tflite_flutter: ^0.10.0
  tflite_flutter_helper: ^0.3.0
```

- **Basic usage (Dart)** — load model, preprocess image, run inference, map top-k labels:

```dart
import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

class TFLiteClassifier {
  late Interpreter _interpreter;
  late List<String> _labels;
  final int inputSize = 224;

  Future<void> load() async {
    _interpreter = await Interpreter.fromAsset('assets/models/product_classifier.tflite');
    final labelsData = await File('assets/models/labels.txt').readAsString();
    _labels = labelsData.split('\n').where((s) => s.trim().isNotEmpty).toList();
  }

  TensorImage _preprocess(File imageFile) {
    final imageProcessor = ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(inputSize, inputSize))
        .add(ResizeOp(inputSize, inputSize, ResizeMethod.NEAREST_NEIGHBOUR))
        .add(NormalizeOp(0, 255))
        .build();

    final inputImage = TensorImage.fromFile(imageFile);
    return imageProcessor.process(inputImage);
  }

  List<Map<String, dynamic>> predict(File imageFile, {int topK = 3}) {
    final input = _preprocess(imageFile);
    final outputShape = [1, _labels.length];
    final outputBuffer = TensorBuffer.createFixedSize(outputShape, TfLiteType.float32);

    _interpreter.run(input.buffer, outputBuffer.buffer);

    final scores = outputBuffer.getDoubleList();
    final sorted = List<int>.generate(scores.length, (i) => i)
      ..sort((a, b) => scores[b].compareTo(scores[a]));

    final top = sorted.take(topK).map((i) => {
      'label': _labels[i],
      'confidence': scores[i],
    }).toList();
    return top;
  }
}
```

- **Notes & tips:**
  - Use `NormalizeOp(0.0, 255.0)` if your model expects inputs in [0,1]. Adjust as needed.
  - For smaller models, use quantized TFLite or float16 to reduce size and speed up inference.
  - Test on real devices — performance varies widely between phones.
  - Remove or fallback `ApiClient.predict()` usage in `flutter_app/lib/api_client.dart` when using on-device inference, or add a toggle in app settings to switch between server and local inference.
