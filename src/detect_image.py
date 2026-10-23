"""
Full single-image smoke detection pipeline, mirroring the paper's
Section II flow (background subtraction -> smoke features -> FCM clustering)
adapted for still images (see background_subtraction.py docstring).

Pipeline
--------
1. single_image_candidate_regions()  -> candidate blobs (foreground proposal)
2. extract_features_for_blobs()      -> 7-dim feature vector per blob
3. Classification of each blob as smoke / not smoke, using ONE of:
   a) a trained Random Forest model (src/train_classifier.py), if a saved
      model exists at models/smoke_classifier.joblib — a real, supervised
      AI model trained on D-Fire's labels. Used automatically when present.
   b) FuzzyCMeans(c=2) clustering (fcm.py), matching the paper's method —
      used as a fallback if no trained model has been saved yet.
"""
from __future__ import annotations
import cv2
import numpy as np

from background_subtraction import single_image_candidate_regions
from features import extract_features_for_blobs, normalize_features
from fcm import FuzzyCMeans
from train_classifier import load_trained_model


def detect_smoke_in_image(bgr: np.ndarray, min_area: int = 600,
                           m: float = 2.0, epsilon: float = 1e-3):
    """Run the full pipeline on one BGR image.

    Returns
    -------
    dict with keys:
        mask         : candidate foreground mask (uint8)
        all_boxes    : every candidate blob's (x,y,w,h)
        smoke_boxes  : subset of all_boxes classified as smoke
        labels       : hard classification label per blob (0/1)
        method       : "trained_model" or "fcm", showing which was used
    """
    mask, boxes, labeled_mask, label_ids = single_image_candidate_regions(
        bgr, min_area=min_area
    )

    if len(boxes) == 0:
        return {"mask": mask, "all_boxes": [], "smoke_boxes": [], "labels": np.array([]),
                "method": "none"}

    X = extract_features_for_blobs(bgr, labeled_mask, boxes, label_ids)

    trained_model = load_trained_model()
    if trained_model is not None:
        # (a) use the trained AI model — features go in raw (the model was
        # trained on the same raw feature scale, no normalisation needed)
        labels = trained_model.predict(X)
        smoke_boxes = [b for b, lbl in zip(boxes, labels) if lbl == 1]
        return {"mask": mask, "all_boxes": boxes, "smoke_boxes": smoke_boxes,
                "labels": labels, "method": "trained_model"}

    # (b) fallback: paper's FCM clustering method
    Xn = normalize_features(X)
    if Xn.shape[0] < 2:
        return {"mask": mask, "all_boxes": boxes, "smoke_boxes": boxes,
                "labels": np.zeros(len(boxes), dtype=int), "method": "fcm"}

    fcm = FuzzyCMeans(c=2, m=m, epsilon=epsilon)
    labels = fcm.fit_predict(Xn)

    # decide which cluster (0 or 1) represents "smoke":
    # feature order = [mean_h, mean_s, mean_v, edge_density, solidity, extent, blur_variance]
    # smoke -> lower saturation (idx 1), lower edge_density (idx 3),
    #          and lower blur_variance (idx 6, i.e. soft/diffuse texture,
    #          not sharp hard-edged objects like cars or road markings)
    centers = fcm.centers_
    smoke_score = centers[:, 1] + centers[:, 3] + centers[:, 6]  # lower = more smoke-like
    smoke_cluster = int(np.argmin(smoke_score))

    smoke_boxes = [b for b, lbl in zip(boxes, labels) if lbl == smoke_cluster]

    return {"mask": mask, "all_boxes": boxes, "smoke_boxes": smoke_boxes,
            "labels": labels, "method": "fcm"}


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img
