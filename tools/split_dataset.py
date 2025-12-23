#!/usr/bin/env python3
"""Split images from dataset/raw into train/ validation/ and test/ folders.

Usage (from project root):

python tools/split_dataset.py

Optional args: --src, --dest, --train, --val, --test, --seed
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import imghdr
from typing import List, Tuple


IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"}


def is_image_file(path: str) -> bool:
    # Quick check by extension first
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if ext in IMAGE_EXTS:
        return True
    # Fallback to imghdr
    try:
        return imghdr.what(path) is not None
    except Exception:
        return False


def split_counts(n: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val
    return n_train, n_val, n_test


def process_class(src_class_dir: str, dest_root: str, class_name: str, ratios: Tuple[float, float, float], seed: int | None) -> Tuple[int, int, int]:
    files = [f for f in os.listdir(src_class_dir) if os.path.isfile(
        os.path.join(src_class_dir, f))]
    files = [f for f in files if is_image_file(os.path.join(src_class_dir, f))]
    if not files:
        return 0, 0, 0

    if seed is not None:
        random.Random(seed).shuffle(files)
    else:
        random.shuffle(files)

    n = len(files)
    n_train, n_val, n_test = split_counts(n, ratios[0], ratios[1])

    splits = (
        ("train", files[:n_train]),
        ("validation", files[n_train: n_train + n_val]),
        ("test", files[n_train + n_val:]),
    )

    for split_name, split_files in splits:
        dest_dir = os.path.join(dest_root, split_name, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        for fname in split_files:
            src_path = os.path.join(src_class_dir, fname)
            dest_path = os.path.join(dest_dir, fname)
            # Copy file metadata as well
            shutil.copy2(src_path, dest_path)

    return n_train, n_val, n_test


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Split dataset into train/validation/test by class")
    p.add_argument("--src", default="dataset/raw",
                   help="Source raw dataset folder (class subfolders)")
    p.add_argument("--dest", default="dataset",
                   help="Destination root (will contain train/ validation/ test/)")
    p.add_argument("--train", type=float, default=0.7, help="Train ratio")
    p.add_argument("--val", type=float, default=0.15, help="Validation ratio")
    p.add_argument("--test", type=float, default=0.15, help="Test ratio")
    p.add_argument("--seed", type=int, default=None,
                   help="Optional random seed for reproducible splits")
    args = p.parse_args(argv)

    total_ratios = args.train + args.val + args.test
    if abs(total_ratios - 1.0) > 1e-6:
        p.error("Ratios must sum to 1.0")

    if not os.path.isdir(args.src):
        p.error(f"Source folder not found: {args.src}")

    class_dirs = [d for d in sorted(os.listdir(
        args.src)) if os.path.isdir(os.path.join(args.src, d))]
    if not class_dirs:
        print(f"No class subfolders found in {args.src}")
        return

    summary = {}
    for cls in class_dirs:
        src_class_dir = os.path.join(args.src, cls)
        n_train, n_val, n_test = process_class(
            src_class_dir, args.dest, cls, (args.train, args.val, args.test), args.seed)
        summary[cls] = (n_train, n_val, n_test)
        print(f"{cls}: train={n_train}, validation={n_val}, test={n_test}")

    totals = [sum(v[i] for v in summary.values()) for i in range(3)]
    print("---")
    print(
        f"Totals: train={totals[0]}, validation={totals[1]}, test={totals[2]}")


if __name__ == "__main__":
    main()
