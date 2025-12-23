"""Train a lightweight image classifier (MobileNetV2) for market items.

Usage example (from repo root):

python model/train.py --data-dir ./dataset --output-dir ./model \
  --epochs 10 --batch-size 32 --img-size 224 224

Expected dataset layout:
dataset/
  train/
    banana/
      img1.jpg
      ...
    garlic/
      img1.jpg
      ...
"""

import argparse
import json
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
# Use the keras namespace from the tensorflow module
# Pyright/Pylance sometimes can't resolve `tf.keras`; ignore that static warning here
keras = tf.keras  # type: ignore[attr-defined]


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="./dataset",
                   help="Dataset root containing train/ subfolder")
    p.add_argument("--output-dir", type=str, default="./model",
                   help="Where to save model artifacts")
    p.add_argument("--img-size", type=int, nargs=2, default=(224, 224))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--fine-tune", action="store_true",
                   help="Unfreeze base model for fine-tuning")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    return p


def save_labels(class_indices, output_dir):
    labels = [cls for cls, _ in sorted(
        class_indices.items(), key=lambda kv: kv[1])]
    os.makedirs(output_dir, exist_ok=True)
    labels_path = os.path.join(output_dir, "labels.txt")
    with open(labels_path, "w", encoding="utf-8") as f:
        f.write("\n".join(labels))
    # also save json for convenience
    with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    return labels_path


def build_model(num_classes, img_size, learning_rate=1e-3, fine_tune=False):
    base = MobileNetV2(weights="imagenet", include_top=False,
                       input_shape=(*img_size, 3))
    base.trainable = bool(fine_tune)

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(base.input, out)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = make_parser().parse_args()
    img_size = tuple(args.img_size)
    batch = args.batch_size

    train_dir = os.path.join(args.data_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Expected training images under {train_dir}. Create folders like {train_dir}\\banana, {train_dir}\\garlic, etc.")

    # Training data: apply augmentation (from dataset/train)
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        brightness_range=(0.8, 1.2),
    )

    # Validation data: only rescale (from dataset/validation)
    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(args.data_dir, "train"),
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        shuffle=True,
    )

    val_dir = os.path.join(args.data_dir, "validation")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected validation images under {val_dir}. Create folders like {val_dir}\\banana, {val_dir}\\garlic, etc.")

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        shuffle=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    labels_path = save_labels(train_gen.class_indices, args.output_dir)

    model = build_model(
        num_classes=train_gen.num_classes,
        img_size=img_size,
        learning_rate=args.learning_rate,
        fine_tune=args.fine_tune,
    )

    ckpt_path = os.path.join(args.output_dir, "product_classifier.h5")
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max"
        ),
        keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True),
    ]

    # Compute simple class weights to help with imbalance
    try:
        counts = np.bincount(train_gen.classes)
        total = counts.sum()
        num_classes = len(counts)
        class_weight = {
            i: float(total / (num_classes * counts[i])) for i in range(num_classes)}
    except Exception:
        class_weight = None

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    # Export SavedModel for deployment (Keras 3 API)
    export_dir = os.path.join(args.output_dir, "product_classifier")
    model.export(export_dir)

    # Optionally save native Keras format for offline use
    keras_path = os.path.join(args.output_dir, "product_classifier.keras")
    model.save(keras_path)

    print(
        "Training complete.\n"
        f"Best checkpoint: {ckpt_path}\n"
        f"SavedModel: {export_dir}\n"
        f"Keras: {keras_path}\n"
        f"Labels: {labels_path}"
    )


if __name__ == "__main__":
    main()
