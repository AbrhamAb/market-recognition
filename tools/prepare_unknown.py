#!/usr/bin/env python3
"""Copy a folder of negative/out-of-distribution images into dataset/raw/unknown.

Usage (from project root):

python tools/prepare_unknown.py --src C:/path/to/negatives --dest dataset/raw

This will create `dataset/raw/unknown/` and copy image files there.
"""
from __future__ import annotations

import argparse
import os
import shutil
import imghdr

IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "gif", "tiff", "webp"}


def is_image(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower().lstrip('.')
    if ext in IMAGE_EXTS:
        return True
    try:
        return imghdr.what(path) is not None
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(
        description="Prepare unknown/outlier images into dataset/raw/unknown")
    p.add_argument("--src", required=True,
                   help="Source folder containing negative images")
    p.add_argument("--dest", default="dataset/raw",
                   help="Destination raw folder (default: dataset/raw)")
    args = p.parse_args()

    if not os.path.isdir(args.src):
        raise SystemExit(f"Source folder not found: {args.src}")

    unknown_dir = os.path.join(args.dest, "unknown")
    os.makedirs(unknown_dir, exist_ok=True)

    count = 0
    for root, _, files in os.walk(args.src):
        for f in files:
            src_path = os.path.join(root, f)
            if not is_image(src_path):
                continue
            dst_path = os.path.join(unknown_dir, f)
            # avoid overwriting duplicates
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(f)
                i = 1
                while os.path.exists(os.path.join(unknown_dir, f)):
                    f = f"{base}_{i}{ext}"
                    i += 1
                dst_path = os.path.join(unknown_dir, f)
            shutil.copy2(src_path, dst_path)
            count += 1

    print(f"Copied {count} images to {unknown_dir}")


if __name__ == "__main__":
    main()
