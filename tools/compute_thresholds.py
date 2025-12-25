"""
Compute per-class confidence thresholds using the validation set.

Outputs a JSON with per-class thresholds and summary metrics.

Usage:
python tools/compute_thresholds.py \
  --model-dir ./model/run_10epochs \
  --val-dir ./dataset/validation \
  --out ./model/run_10epochs/thresholds.json

The script will try to use scikit-learn if available for reporting; otherwise it uses
basic precision/recall/F1 computations.
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Any, cast

# Prefer the TF-provided keras namespace; if the static analyzer can't resolve
# attributes on `tf.keras`, fall back at runtime to standalone `keras` if
# available. At runtime we always use `tf.keras` as primary loader to avoid
# ambiguity between Keras versions.
keras = getattr(tf, "keras", None)  # type: ignore[attr-defined]
try:
    if keras is None:
        import keras as keras  # type: ignore
        keras = keras  # type: ignore
except Exception:
    # leave keras as whatever we got from tf (possibly None)
    pass

try:
    from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
    SKLEARN = True
except Exception:
    SKLEARN = False


def load_labels(model_dir):
    p = os.path.join(model_dir, "labels.txt")
    if not os.path.exists(p):
        raise FileNotFoundError(f"labels.txt not found in {model_dir}")
    with open(p, "r", encoding="utf-8") as f:
        labels = [l.strip() for l in f.readlines() if l.strip()]
    return labels


def load_model(model_dir) -> Any:
    # Prefer Keras saved file or SavedModel dir under model_dir/product_classifier
    candidates = [
        os.path.join(model_dir, "product_classifier.keras"),
        os.path.join(model_dir, "product_classifier.h5"),
        os.path.join(model_dir, "product_classifier"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                # Prefer tf.keras.models at runtime; if not available, try the
                # `keras` alias. This avoids analyzer false-positives while
                # remaining robust at runtime.
                if hasattr(tf, "keras") and hasattr(tf.keras, "models"):
                    m = tf.keras.models.load_model(c)
                elif keras is not None and hasattr(keras, "models"):
                    m = keras.models.load_model(c)
                else:
                    raise RuntimeError(
                        "No keras.models available to load model")
                print("Loaded model from", c)
                return m
            except Exception as e:
                print("load_model failed for", c, "error:", e)
    raise FileNotFoundError("No usable model found under " + model_dir)


def predict_on_generator(model, gen):
    steps = int(np.ceil(gen.samples / float(gen.batch_size)))
    # model may be a Keras model or a TFSMLayer wrapper; cast to Any to
    # satisfy static analyzers that `predict` exists.
    model_obj = cast(Any, model)
    preds = model_obj.predict(gen, steps=steps, verbose=1)
    if isinstance(preds, dict):
        preds = list(preds.values())[0]
    return np.array(preds)


def choose_best_thresholds(preds, true_idxs, labels):
    # preds: N x C
    N, C = preds.shape
    thresholds = np.linspace(0.0, 1.0, 101)
    per_class = {}
    # predicted index for each sample
    pred_idxs = np.argmax(preds, axis=1)

    for c in range(C):
        best_t = 0.0
        best_f1 = -1.0
        best_metrics = (0, 0, 0)
        # evaluate thresholds by treating predictions as class-c when pred==c and prob_c >= t
        for t in thresholds:
            pred_is_c = (pred_idxs == c) & (preds[:, c] >= t)
            tp = int(np.sum(pred_is_c & (true_idxs == c)))
            fp = int(np.sum(pred_is_c & (true_idxs != c)))
            fn = int(np.sum((true_idxs == c) & ~pred_is_c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
                best_metrics = (prec, rec, f1)
        per_class[labels[c]] = {
            "threshold": best_t,
            "precision": best_metrics[0],
            "recall": best_metrics[1],
            "f1": best_metrics[2],
            "support": int(np.sum(true_idxs == c))
        }
    return per_class


def apply_thresholds(preds, labels_map, thresholds, unknown_label="unknown"):
    # preds: N x C, labels_map: index->label name, thresholds: label->threshold
    pred_idxs = np.argmax(preds, axis=1)
    pred_probs = preds[np.arange(preds.shape[0]), pred_idxs]
    accepted = []
    final_preds = []
    for i, pidx in enumerate(pred_idxs):
        label = labels_map[pidx]
        t = thresholds.get(label, 0.0)
        if pred_probs[i] >= t:
            final_preds.append(pidx)
            accepted.append(True)
        else:
            final_preds.append(-1)  # unknown
            accepted.append(False)
    return np.array(final_preds), np.array(accepted)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                   help="Model run directory (contains labels.txt and product_classifier)")
    p.add_argument("--val-dir", required=True,
                   help="Validation dataset directory (same layout as train/)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, nargs=2, default=(224, 224))
    p.add_argument("--out", type=str, default=None,
                   help="Output JSON path for thresholds")
    args = p.parse_args()

    labels = load_labels(args.model_dir)
    model = load_model(args.model_dir)

    # build generator (do not shuffle so we can map true labels)
    val_datagen = ImageDataGenerator(rescale=1.0/255)
    val_gen = val_datagen.flow_from_directory(
        args.val_dir,
        target_size=tuple(args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False,
        classes=labels,
    )

    preds = predict_on_generator(model, val_gen)
    true_idxs = np.array(val_gen.classes)

    per_class = choose_best_thresholds(preds, true_idxs, labels)

    # apply per-class thresholds to get final predictions (unknown when below per-class threshold)
    thresholds_map = {k: v['threshold'] for k, v in per_class.items()}
    final_preds, accepted = apply_thresholds(preds, labels, thresholds_map)

    # compute summary metrics
    report = {}
    # Map unknown to index -1; compute overall accuracy ignoring unknowns or counting unknown as separate
    known_mask = final_preds != -1
    known_count = int(np.sum(known_mask))
    total = preds.shape[0]
    correct_known = int(np.sum((final_preds == true_idxs) & known_mask))
    accuracy_known = correct_known / known_count if known_count > 0 else 0.0
    accuracy_overall = int(np.sum(final_preds == true_idxs)) / total

    report['total_samples'] = int(total)
    report['known_count'] = known_count
    report['accuracy_known'] = float(accuracy_known)
    report['accuracy_overall'] = float(accuracy_overall)
    report['per_class'] = per_class

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print('Wrote thresholds report to', args.out)
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
