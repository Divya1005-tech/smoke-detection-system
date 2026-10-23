"""
Train a real, trained AI/ML model (a Random Forest classifier) to replace
the unsupervised FCM clustering step with a supervised model that has
actually learned from D-Fire's labeled data.

This is the "AI model" piece of the project: FCM (used elsewhere in this
project, matching the paper) is an *unsupervised clustering* algorithm — it
never learns from labeled data or saves a model. This script instead:

1. Runs the existing candidate-region proposal on every TRAIN image
2. For every candidate blob, extracts the same 7-dim feature vector already
   used elsewhere (features.py)
3. Labels each blob 1 (smoke) or 0 (not smoke) using ground-truth boxes
4. Trains a RandomForestClassifier (scikit-learn) on these (features, label) pairs
5. Saves the trained model to models/smoke_classifier.joblib

Usage
-----
    python main.py train --split train --limit 500
"""
from __future__ import annotations
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from background_subtraction import single_image_candidate_regions
from features import extract_features_for_blobs
from dataset import list_samples
from utils import load_yolo_labels, iou_xywh, CLASS_SMOKE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "smoke_classifier.joblib")


def load_image(path: str):
    """Local copy to avoid a circular import with detect_image.py."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _box_contains_point(box, px: float, py: float) -> bool:
    x, y, w, h = box
    return x <= px <= x + w and y <= py <= y + h


def build_training_data(images_dir: str, labels_dir: str, min_area: int = 600,
                         label_iou_thresh: float = 0.05, limit=None):
    """Run candidate detection on every training image and label each blob
    against the real ground-truth boxes. Returns (X, y) arrays ready for
    scikit-learn.

    A blob is labeled "smoke" if EITHER:
    - it overlaps a ground-truth smoke box by at least label_iou_thresh IoU, OR
    - the ground-truth box's center point falls inside the candidate blob.
    """
    samples = list_samples(images_dir, labels_dir)
    if limit:
        samples = samples[:limit]

    all_X, all_y = [], []
    for i, s in enumerate(samples):
        try:
            img = load_image(s.image_path)
        except FileNotFoundError as e:
            print(f"  Skipping unreadable image: {s.image_path} ({e})")
            continue

        h, w = img.shape[:2]
        gt_boxes = load_yolo_labels(s.label_path, w, h, keep_class=CLASS_SMOKE)

        mask, boxes, labeled_mask, label_ids = single_image_candidate_regions(
            img, min_area=min_area
        )
        if len(boxes) == 0:
            continue

        feats = extract_features_for_blobs(img, labeled_mask, boxes, label_ids)

        for box, feat in zip(boxes, feats):
            is_smoke = False
            for gt in gt_boxes:
                if iou_xywh(box, gt) >= label_iou_thresh:
                    is_smoke = True
                    break
                gx, gy, gw, gh = gt
                center_x, center_y = gx + gw / 2, gy + gh / 2
                if _box_contains_point(box, center_x, center_y):
                    is_smoke = True
                    break
            all_X.append(feat)
            all_y.append(1 if is_smoke else 0)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(samples)} images, "
                  f"{len(all_X)} blobs labeled so far...")

    return np.array(all_X), np.array(all_y)


def train_and_save(images_dir: str, labels_dir: str, min_area: int = 600, limit=None) -> None:
    print("Building training data from candidate blobs + ground-truth labels...")
    X, y = build_training_data(images_dir, labels_dir, min_area=min_area, limit=limit)

    print(f"\nTotal blobs: {len(y)}  |  Smoke: {int(y.sum())}  |  Not smoke: {int((y == 0).sum())}")
    if len(y) < 20 or y.sum() == 0:
        print("Not enough labeled data to train a meaningful model. "
              "Try a larger --limit or check your label files.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.sum() > 1 else None
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                  class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    print("\n=== Held-out test performance (20% of blobs, not used in training) ===")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["not_smoke", "smoke"]))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    print(f"Saved trained model -> {MODEL_PATH}")


def load_trained_model():
    """Load the trained model if it exists, else return None."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None