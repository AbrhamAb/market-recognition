"""Evaluate a trained model on the `dataset/test/` folder.

Produces a JSON report and prints a concise summary.

Usage:
python tools/evaluate_test.py \
  --model-dir ./model/run_10epochs \
  --test-dir ./dataset/test \
  --thresholds ./model/run_10epochs/thresholds.json \
  --out ./model/run_10epochs/test_eval.json
"""

import argparse
import json
import os
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN = True
except Exception:
    SKLEARN = False


def load_labels(model_dir):
    p = os.path.join(model_dir, "labels.txt")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def load_model(model_dir):
    candidates = [
        os.path.join(model_dir, "product_classifier.keras"),
        os.path.join(model_dir, "product_classifier.h5"),
        os.path.join(model_dir, "product_classifier"),
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                return tf.keras.models.load_model(c)
            except Exception:
                # try next
                pass
    raise FileNotFoundError("No model found under " + model_dir)


def predict_on_dir(model: Any, test_dir: str, labels, img_size=(224, 224), batch_size=32):
    gen = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        test_dir,
        target_size=tuple(img_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        classes=labels,
    )
    steps = int(np.ceil(gen.samples / float(gen.batch_size)))
    preds = model.predict(gen, steps=steps, verbose=1)
    if isinstance(preds, dict):
        preds = list(preds.values())[0]
    return np.array(preds), np.array(gen.classes), list(gen.filenames)


def apply_thresholds(preds: np.ndarray, labels: list, thresholds: Dict[str, float]):
    pred_idxs = np.argmax(preds, axis=1)
    pred_probs = preds[np.arange(preds.shape[0]), pred_idxs]
    final = []
    for i, pidx in enumerate(pred_idxs):
        label = labels[pidx]
        t = thresholds.get(label, 0.0)
        if pred_probs[i] >= t:
            final.append(pidx)
        else:
            final.append(-1)
    return np.array(final)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--test-dir", required=True)
    p.add_argument("--thresholds", default=None,
                   help="Path to thresholds.json (optional)")
    p.add_argument("--out", default=None, help="Output JSON report path")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, nargs=2, default=(224, 224))
    args = p.parse_args()

    labels = load_labels(args.model_dir)
    model = load_model(args.model_dir)

    preds, true_idxs, filenames = predict_on_dir(
        model, args.test_dir, labels, img_size=tuple(args.img_size), batch_size=args.batch_size)

    # load thresholds
    thresholds = {}
    if args.thresholds and os.path.exists(args.thresholds):
        with open(args.thresholds, 'r', encoding='utf-8') as f:
            tdata = json.load(f)
            # tolerate nested format
            per = tdata.get('per_class') if isinstance(tdata, dict) and 'per_class' in tdata else tdata.get(
                'per_class') if isinstance(tdata, dict) else None
            if per is None:
                # try top-level per_class
                per = tdata.get('per_class') if isinstance(
                    tdata, dict) else None
            if per is None and isinstance(tdata, dict):
                # maybe compute_thresholds output directly stores per_class
                per = tdata.get('per_class', {})
            # if per still empty, assume tdata maps class->metrics
            per = per or tdata
            for k, v in per.items():
                thresholds[k] = v['threshold'] if isinstance(
                    v, dict) and 'threshold' in v else float(v)

    if thresholds:
        final = apply_thresholds(preds, labels, thresholds)
    else:
        # no thresholds: accept top prediction
        final = np.argmax(preds, axis=1)

    # compute metrics
    total = len(true_idxs)
    unknown_mask = final == -1
    unknown_count = int(np.sum(unknown_mask))
    known_mask = ~unknown_mask
    known_count = int(np.sum(known_mask))

    correct_known = int(np.sum((final == true_idxs) & known_mask))
    accuracy_known = correct_known / known_count if known_count > 0 else 0.0
    accuracy_overall = int(np.sum(final == true_idxs)) / total

    report = {
        'total': int(total),
        'known_count': known_count,
        'unknown_count': unknown_count,
        'accuracy_known': float(accuracy_known),
        'accuracy_overall': float(accuracy_overall),
    }

    # per-class breakdown using sklearn if available
    if SKLEARN:
        # map unknown (-1) to a label string for reporting
        y_true = list(true_idxs)
        y_pred = [int(x) for x in final]
        # classification_report expects labels as ints; replace unknown (-1) with a value not in range
        import numpy as _np
        known_idx_mask = _np.array(y_pred) != -1
        if known_count > 0:
            from sklearn.metrics import classification_report, confusion_matrix
            rep = classification_report(_np.array(y_true)[known_idx_mask], _np.array(
                y_pred)[known_idx_mask], target_names=labels, zero_division=0)
            cm = confusion_matrix(
                _np.array(y_true)[known_idx_mask], _np.array(y_pred)[known_idx_mask])
            report['classification_report_known'] = rep
            report['confusion_matrix_known'] = cm.tolist()

    # Save per-sample results
    samples = []
    for fn, t, p, f in zip(filenames, true_idxs, preds.tolist(), final.tolist()):
        samples.append({
            'file': fn,
            'true_idx': int(t),
            'pred_idx': int(f),
            'pred_top_prob': float(np.max(p)),
        })
    report['samples'] = samples

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print('Wrote evaluation report to', args.out)
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
