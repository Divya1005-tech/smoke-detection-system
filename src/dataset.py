"""
Dataset loader for the D-Fire dataset layout:

    data/DFireDataset/train/images/*.jpg
    data/DFireDataset/train/labels/*.txt   (same basename as the image)
    data/DFireDataset/test/images/*.jpg
    data/DFireDataset/test/labels/*.txt

Each label .txt is YOLO format (see utils.py for parsing).
"""
from __future__ import annotations
import os
from dataclasses import dataclass

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Sample:
    image_path: str
    label_path: str
    name: str


def list_samples(images_dir: str, labels_dir: str) -> list[Sample]:
    """Pair every image in `images_dir` with its label file in `labels_dir`
    (same filename, .txt extension). Images with no label file are still
    included (label_path may not exist -> treated as zero ground-truth boxes,
    e.g. a non-smoke image)."""
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Images folder not found: {images_dir}\n"
            f"Put your D-Fire images here before running detection/evaluation."
        )

    samples = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(IMG_EXTS):
            continue
        name = os.path.splitext(fname)[0]
        image_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, name + ".txt")
        samples.append(Sample(image_path=image_path, label_path=label_path, name=name))

    if not samples:
        raise RuntimeError(
            f"No images found in {images_dir}. "
            f"Supported extensions: {IMG_EXTS}"
        )
    return samples
