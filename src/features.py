"""
Feature extraction per candidate blob, used as input to FCM clustering
(fcm.py). These features operationalise the smoke description in Section
2.1 of the reference paper: colour (bluish-white/white to greyish-black),
diffuse/rough boundaries, and irregular, varying shape.

Each blob -> one feature vector:
    [mean_hue, mean_saturation, mean_value, edge_density, solidity, extent, blur_variance]

- mean_saturation: smoke is desaturated (low colour saturation)
- mean_value: smoke ranges from whitish (high V) to blackish (low V)
- edge_density: smoke has soft/diffuse edges -> low local edge density
- solidity: area / convex-hull area -> smoke blobs are irregular (lower
  solidity) compared to solid man-made objects
- extent: area / bounding-box area -> another irregularity/shape cue
- blur_variance: variance of the Laplacian inside the blob -> a standard
  "sharpness" measure. Smoke is soft and diffuse (LOW variance). Cars,
  painted road markings, and building edges stay comparatively sharp even
  after blurring, so they score HIGH here. This is the main feature that
  separates real smoke from man-made clutter that happens to be pale-coloured.
"""
from __future__ import annotations
import cv2
import numpy as np


def extract_blob_features(bgr: np.ndarray, mask: np.ndarray,
                           box: tuple[int, int, int, int]) -> np.ndarray:
    """Compute the 7-dim feature vector for one candidate blob.

    Parameters
    ----------
    bgr : full original image (BGR)
    mask : full-size binary mask (255 = candidate foreground) for this blob only
    box : (x, y, w, h) bounding box of the blob
    """
    x, y, w, h = box
    roi_bgr = bgr[y:y + h, x:x + w]
    roi_mask = mask[y:y + h, x:x + w]

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    m = roi_mask > 0
    if m.sum() == 0:
        m = np.ones(roi_mask.shape, dtype=bool)  # fallback, shouldn't normally happen

    mean_h = float(hsv[:, :, 0][m].mean())
    mean_s = float(hsv[:, :, 1][m].mean())
    mean_v = float(hsv[:, :, 2][m].mean())

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_density = float((edges[m] > 0).mean()) if m.sum() > 0 else 0.0

    # texture sharpness: variance of Laplacian, restricted to the blob pixels
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    blur_variance = float(laplacian[m].var()) if m.sum() > 1 else 0.0

    # shape irregularity via contour of the blob mask
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area / hull_area) if hull_area > 0 else 1.0
        extent = float(area / (w * h)) if (w * h) > 0 else 0.0
    else:
        solidity, extent = 1.0, 0.0

    return np.array([mean_h, mean_s, mean_v, edge_density, solidity, extent,
                      blur_variance], dtype=np.float64)


def extract_features_for_blobs(bgr: np.ndarray, labeled_mask: np.ndarray,
                                boxes: list[tuple[int, int, int, int]],
                                label_ids: list[int]) -> np.ndarray:
    """Vectorised helper: build the (n_blobs, 7) feature matrix used by fcm.py.

    labeled_mask : output of cv2.connectedComponentsWithStats (each blob has
                   its own integer label)
    label_ids    : the connected-component label id corresponding to each box
    """
    feats = []
    for box, lbl in zip(boxes, label_ids):
        single_mask = np.where(labeled_mask == lbl, 255, 0).astype(np.uint8)
        feats.append(extract_blob_features(bgr, single_mask, box))
    return np.array(feats) if feats else np.zeros((0, 7))


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Min-max normalise each feature column to [0, 1] so no single feature
    (e.g. hue 0-179 vs solidity 0-1) dominates the FCM distance calculation."""
    if X.shape[0] == 0:
        return X
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return (X - mins) / ranges
