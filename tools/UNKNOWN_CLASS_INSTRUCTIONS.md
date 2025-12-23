## Adding an `unknown` class (outlier exposure) — quick guide

Goal
----
Teach the classifier to recognize inputs that are not any of your labeled product classes (faces, cars, scenery, etc.).

Steps
-----
1. Gather negative images (out-of-distribution examples)
   - Collect a few hundred to a few thousand images of things that are NOT your product classes: people/faces, streets, buildings, animals, household objects, etc.
   - Use public datasets (COCO, OpenImages) or your own photos.

2. Copy negatives into project
   - Use the helper script to copy into `dataset/raw/unknown`:

```powershell
python tools/prepare_unknown.py --src C:\path\to\negatives --dest dataset/raw
```

3. Run the splitter (or manually place)

```powershell
# splits dataset/raw into train/ validation/ test/ (70/15/15)
python tools/split_dataset.py --seed 42
```

4. Verify the `unknown` class was created under `dataset/train/unknown`, `dataset/validation/unknown`, `dataset/test/unknown`.

5. Retrain your model

```powershell
python model/train.py --data-dir ./dataset --output-dir ./model --epochs 10 --batch-size 32 --img-size 224 224
```

6. Use thresholding at inference (already implemented in backend)
   - Backend will return `unknown` when top softmax prob < `CONF_THRESHOLD`.
   - Tune `CONF_THRESHOLD` by evaluating on validation set and held-out OOD samples (choose value like 0.5–0.7).

7. (Optional) Improve robustness
   - Calibrate probabilities (temperature scaling) using validation logits.
   - Build an embedding-based detector (centroids or k-NN) on penultimate-layer features.
   - Use ensembles or Mahalanobis/energy-based OOD detectors for stronger guarantees.

Notes
-----
- The `unknown` class must be diverse to generalize.
- Thresholding + unknown class together tend to work well in practice.
- After retraining, test with face images and other unrelated photos to confirm `unknown` is returned.

If you want, I can:
- (A) run `prepare_unknown.py` for you if you give the negatives path, and then run the splitter and a short training run;
- (B) add a small calibration helper script to compute and apply temperature scaling;
- (C) implement embedding centroid-based OOD detection in `backend/main.py`.
