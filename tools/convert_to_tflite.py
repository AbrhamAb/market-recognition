#!/usr/bin/env python3
"""
Convert a Keras model to TFLite with optional quantization.

Usage:
  python tools/convert_to_tflite.py \
      --input model/run_10epochs/product_classifier.keras \
      --output model/run_10epochs/product_classifier.tflite \
      [--quantize none|dynamic|float16]

This script supports three quantization modes:
  - none: full float32 (default)
  - dynamic: dynamic range quantization (smaller, no calibration needed)
  - float16: float16 quantization (good size/accuracy tradeoff)

Note: full integer quantization requires a representative dataset and
is not implemented here.
"""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to Keras model (.keras, .h5 or SavedModel dir)")
    parser.add_argument("--output", required=True, help="Path to write TFLite model")
    parser.add_argument("--quantize", choices=["none", "dynamic", "float16"], default="none")
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except Exception as e:
        print("TensorFlow is not installed in this Python environment.")
        print("Install via: pip install tensorflow")
        raise

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input model not found: {input_path}")
        sys.exit(2)

    print(f"Loading model from {input_path} ...")
    # tf.keras can load .keras, .h5, or a SavedModel directory
    try:
        model = tf.keras.models.load_model(str(input_path))
    except Exception as e:
        print("Failed to load model with tf.keras.load_model:", e)
        print("If your model is a SavedModel dir and loading fails, ensure the directory is correct.")
        raise

    print("Converting model to TFLite (quantize=%s) ..." % args.quantize)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if args.quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif args.quantize == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print("TFLite conversion failed:", e)
        raise

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote TFLite model to {output_path} ({size_mb:.2f} MB)")

    # Optionally copy into Flutter app assets if directory exists
    flutter_assets = Path("flutter_app/assets/models")
    if flutter_assets.exists():
        dest = flutter_assets / output_path.name
        from shutil import copy2
        copy2(output_path, dest)
        print(f"Also copied TFLite model to Flutter assets: {dest}")

if __name__ == "__main__":
    main()
