"""Reorganize a downloaded Kaggle dataset into dataset/train/<class>/image.jpg.

Example (after using kagglehub.dataset_download):
    python tools/reorg_kaggle.py --source "C:/Users/mommy/.cache/kagglehub/datasets/kritikseth/fruit-and-vegetable-image-recognition/versions/latest" --subdir train

If the dataset already has class folders under the source (or under a subdir like 'train' or 'Training'), this script copies images into dataset/train/<class> with lowercase snake_case names.
"""
import argparse
import os
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sanitize(name: str) -> str:
    name = name.strip().lower().replace(" ", "_").replace("-", "_")
    return name


def copy_class_dir(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for root, _, files in os.walk(src_dir):
        for f in files:
            if Path(f).suffix.lower() not in IMG_EXTS:
                continue
            count += 1
            src_path = Path(root) / f
            dst_path = dst_dir / f"img_{count:05d}{src_path.suffix.lower()}"
            shutil.copy2(src_path, dst_path)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Root of the downloaded dataset")
    parser.add_argument("--target", default="./dataset/train",
                        help="Destination root (default: ./dataset/train)")
    parser.add_argument("--subdir", default=None,
                        help="Optional subdirectory within source (e.g., train, Training, 'Vegetable Images')")
    args = parser.parse_args()

    src_root = Path(args.source)
    if args.subdir:
        src_root = src_root / args.subdir
    if not src_root.exists():
        raise FileNotFoundError(f"Source not found: {src_root}")

    dst_root = Path(args.target)
    dst_root.mkdir(parents=True, exist_ok=True)

    class_dirs = [p for p in src_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {src_root}")

    total = 0
    for cdir in sorted(class_dirs):
        cls_name = sanitize(cdir.name)
        dst_cls = dst_root / cls_name
        n = copy_class_dir(cdir, dst_cls)
        total += n
        print(f"Copied {n} images for class '{cls_name}'")
    print(f"Done. Total images: {total}. Output in {dst_root.resolve()}")


if __name__ == "__main__":
    main()
