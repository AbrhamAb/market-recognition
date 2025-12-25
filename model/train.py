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
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import math
import random
from tensorflow.keras.utils import Sequence
# use explicit keras namespace


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
    p.add_argument("--patience", type=int, default=3,
                   help="EarlyStopping patience in epochs (restore best weights)")
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--target-percentile", type=float, default=0.8,
                   help="Percentile of class counts to use as target per-class (0-1)."
                   )
    p.add_argument("--max-repeats-per-image", type=int, default=10,
                   help="Cap how many times a single source image may be repeated per epoch.")
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

    # Training data: augmentation parameters (applied on-the-fly)
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

    # Build class list and mapping from train folder so labels/order are deterministic
    train_root = os.path.join(args.data_dir, "train")
    classes = [d for d in sorted(os.listdir(train_root)) if os.path.isdir(
        os.path.join(train_root, d))]
    if not classes:
        raise FileNotFoundError(
            f"No class subfolders found under {train_root}")
    class_indices = {cls: i for i, cls in enumerate(classes)}

    # Inspect class counts and pick a sensible target to avoid excessive repetition
    counts = [len([f for f in os.listdir(os.path.join(train_root, c)) if f.lower(
    ).endswith((".jpg", ".jpeg", ".png"))]) for c in classes]
    max_count = max(counts)
    min_count = min(counts)
    # percentile-based target (e.g., 0.8 => 80th percentile)
    pct = float(args.target_percentile)
    pct = min(max(pct, 0.0), 1.0)
    pct_val = int(np.percentile(counts, pct * 100.0))
    # cap target at max_count and ensure at least min_count
    target_default = max(min(pct_val, max_count), min_count)
    # apply a soft cap to avoid enormous repetition; allow CLI override via max_repeats_per_image
    max_repeats = int(args.max_repeats_per_image)
    # compute global allowed target considering per-image repeat cap
    # for each class, allowed_target = min(target_default, len(files)*max_repeats)

    class BalancedSequence(Sequence):
        def __init__(self, root, classes, class_indices, datagen, img_size, batch_size, target=None, seed=42):
            self.root = root
            self.classes = classes
            self.class_indices = class_indices
            self.datagen = datagen
            self.img_size = tuple(img_size)
            self.batch_size = batch_size
            self.seed = seed

            # gather filepaths per class
            self.files_by_class = {}
            for cls in classes:
                p = os.path.join(root, cls)
                files = [os.path.join(p, f) for f in os.listdir(
                    p) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                self.files_by_class[cls] = files

            counts = [len(self.files_by_class[c]) for c in classes]
            if any(c == 0 for c in counts):
                raise FileNotFoundError(
                    "One or more classes contain no images; please check dataset/train/ structure")

            self.max_count = max(counts)
            # initial target candidate
            self.target = target or self.max_count
            # enforce per-image repeat cap: do not request more than len(files)*max_repeats for each class
            self.max_repeats = max_repeats
            self.num_classes = len(classes)
            self.samples = self.target * self.num_classes
            self._build_epoch_list()

        def _build_epoch_list(self):
            # Build a list of (filepath, class_index) of length self.samples
            rnd = random.Random(self.seed)
            epoch_list = []
            for cls in self.classes:
                files = self.files_by_class[cls]
                # enforce per-class allowed target
                allowed = min(self.target, len(files) * self.max_repeats)
                need = int(allowed)
                # sample with replacement if need > available
                chosen = [rnd.choice(files) for _ in range(need)]
                epoch_list.extend([(p, self.class_indices[cls])
                                  for p in chosen])
            rnd.shuffle(epoch_list)
            self.epoch_list = epoch_list
            # update samples to actual epoch list length (handles caps)
            self.samples = len(self.epoch_list)

        def __len__(self):
            # number of batches for the current epoch list
            return int(math.ceil(len(self.epoch_list) / float(self.batch_size)))

        def __getitem__(self, idx):
            start = idx * self.batch_size
            end = min(start + self.batch_size, self.samples)
            batch_items = self.epoch_list[start:end]
            batch_x = np.zeros(
                (len(batch_items), self.img_size[0], self.img_size[1], 3), dtype=np.float32)
            batch_y = np.zeros(
                (len(batch_items), self.num_classes), dtype=np.float32)
            for i, (path, cls_idx) in enumerate(batch_items):
                img = load_img(path, target_size=self.img_size)
                arr = img_to_array(img)
                # apply random transform + standardize
                arr = self.datagen.random_transform(arr)
                arr = self.datagen.standardize(arr)
                batch_x[i] = arr
                batch_y[i, cls_idx] = 1.0
            return batch_x, batch_y

        def on_epoch_end(self):
            # regenerate and reshuffle epoch list to provide fresh sampling
            self._build_epoch_list()

    train_seq = BalancedSequence(
        train_root, classes, class_indices, train_datagen, img_size, batch)

    val_dir = os.path.join(args.data_dir, "validation")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Expected validation images under {val_dir}. Create folders like {val_dir}\\banana, {val_dir}\\garlic, etc.")

    # Ensure validation generator uses the same class ordering
    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch,
        class_mode="categorical",
        shuffle=False,
        classes=classes,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    labels_path = save_labels(class_indices, args.output_dir)

    model = build_model(
        num_classes=len(classes),
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
            patience=args.patience, restore_best_weights=True),
    ]

    # When using a balanced on-the-fly generator, class_weight is not necessary
    model.fit(
        train_seq,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
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
