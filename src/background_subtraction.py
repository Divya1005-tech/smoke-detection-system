"""
Background subtraction, following Section II.1 of the reference paper
("Background Subtraction Using Frame Differencing"):

    P[f(t)] = P[I(t)] - P[B]                                   (Eq. 1)
    | P[f(t+1)] - P[f(t)] | > Threshold                        (Eq. 2)

Two entry points are provided:

1. `VideoFrameDifferencer` — the paper's literal method, for a sequence of
   VIDEO frames (needs a genuine background frame + moving foreground).

2. `single_image_candidate_regions` — the D-Fire dataset used for this
   project is made of INDIVIDUAL still images (no paired background frame
   exists for most of them), so true frame-differencing background
   subtraction cannot be applied image-by-image. Section 2.1 of the paper
   ("Features of smoke") is used instead as the basis for a still-image
   candidate-region proposal step that plays the same structural role as
   background subtraction: it produces a foreground/candidate mask of
   "moving-region-like" blobs to hand to the FCM clustering stage. This is
   documented in the project README.
"""
from __future__ import annotations
import cv2
import numpy as np


class VideoFrameDifferencer:
    """Implements Eq. 1 & 2 of the paper for real video sequences."""

    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self.background: np.ndarray | None = None

    def set_background(self, frame_gray: np.ndarray) -> None:
        self.background = frame_gray.astype(np.float32)

    def apply(self, frame_gray: np.ndarray) -> np.ndarray:
        """Return a binary foreground mask for one frame (Eq. 1 + threshold in Eq. 2)."""
        if self.background is None:
            self.set_background(frame_gray)
            return np.zeros_like(frame_gray, dtype=np.uint8)

        diff = cv2.absdiff(frame_gray.astype(np.float32), self.background)  # Eq. 1
        mask = (diff > self.threshold).astype(np.uint8) * 255              # Eq. 2
        # slowly adapt background so slow-moving smoke is still eventually captured
        self.background = 0.98 * self.background + 0.02 * frame_gray.astype(np.float32)
        return mask


def single_image_candidate_regions(bgr: np.ndarray,
                                    canny_lo: int = 40,
                                    canny_hi: int = 120,
                                    min_area: int = 600):
    """
    Produce a candidate foreground/smoke-region mask for a SINGLE still image,
    using the smoke colour/texture description in Section 2.1 of the paper:

      - "colour of smoke ranges from bluish-white to white and from
         greyish-black to black" -> low colour saturation, and value NOT at
         the extreme brightness of open sky glare nor the extreme dark of
         shadow (both common false-positive sources).
      - "surface and boundary of the smoke regions are usually rough and
         coarse" -> low-to-moderate edge density relative to sharp, hard
         edges of normal scene objects (cars, road markings, buildings).

    Returns
    -------
    mask : uint8 binary mask, candidate regions = 255
    boxes : list of (x, y, w, h) bounding boxes of connected components
            with area >= min_area
    labeled_mask : int32 array, same size as image, each kept blob carries
                   its connected-component label id (0 = background)
    label_ids : list of label ids, same order/length as `boxes`, so
                boxes[k] corresponds to labeled_mask == label_ids[k]
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 1) low-saturation, greyish/whitish colour condition (smoke colour cue).
    #    Tighter than a naive threshold: excludes near-pure-white sky glare
    #    (v > 235) and very dark shadow (v < 90), both of which share
    #    smoke's low-saturation signature and cause false positives.
    color_mask = ((s < 50) & (v > 90) & (v < 235)).astype(np.uint8) * 255

    # 2) edges: smoke has soft, diffuse boundaries -> low Canny edge density
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    edge_density = cv2.blur(edges.astype(np.float32) / 255.0, (15, 15))
    soft_edge_mask = (edge_density < 0.12).astype(np.uint8) * 255

    # 3) combine colour + soft-edge cues to approximate a "foreground-like"
    #    candidate region, analogous to the thresholded difference image
    #    P[f(t)] in Eq. 1/2 of the paper.
    combined = cv2.bitwise_and(color_mask, soft_edge_mask)

    # clean up: remove tiny speckle, close small gaps (paper: "rough and coarse"
    # boundaries -> a mild morphological close helps merge diffuse smoke blobs)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    boxes = []
    label_ids = []
    clean_mask = np.zeros_like(combined)
    kept_labels = np.zeros_like(labels)
    for i in range(1, num):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x, y, w, hh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                           stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            boxes.append((x, y, w, hh))
            label_ids.append(i)
            clean_mask[labels == i] = 255
            kept_labels[labels == i] = i

    return clean_mask, boxes, kept_labels, label_ids
