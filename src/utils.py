"""
Small shared helpers: YOLO-format box conversion, drawing, and IoU.

D-Fire label format (one .txt per image, YOLO style):
    <class_id> <x_center> <y_center> <width> <height>
all four numbers normalised to [0, 1] relative to image width/height.
D-Fire classes: 0 = smoke, 1 = fire  (check your copy's data.yaml/README —
some releases order them the other way; adjust CLASS_SMOKE below if so).
"""
from __future__ import annotations
import cv2
import numpy as np

CLASS_SMOKE = 0
CLASS_FIRE = 1


def yolo_to_xywh(yolo_box: tuple[float, float, float, float],
                  img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert normalised (xc, yc, w, h) -> pixel (x, y, w, h) top-left format."""
    xc, yc, w, h = yolo_box
    bw, bh = w * img_w, h * img_h
    x = int(round(xc * img_w - bw / 2))
    y = int(round(yc * img_h - bh / 2))
    return x, y, int(round(bw)), int(round(bh))


def load_yolo_labels(label_path: str, img_w: int, img_h: int,
                      keep_class: int | None = CLASS_SMOKE) -> list[tuple[int, int, int, int]]:
    """Read a YOLO .txt label file and return pixel-space (x, y, w, h) boxes.

    If keep_class is not None, only boxes of that class are returned
    (default: only smoke boxes, since this project targets smoke detection).
    If the file doesn't exist (image has no annotated objects), returns [].
    """
    boxes = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                if keep_class is not None and cls != keep_class:
                    continue
                xc, yc, w, h = map(float, parts[1:])
                boxes.append(yolo_to_xywh((xc, yc, w, h), img_w, img_h))
    except FileNotFoundError:
        pass
    return boxes


def iou_xywh(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """Intersection-over-Union between two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def draw_boxes(bgr: np.ndarray, boxes: list[tuple[int, int, int, int]],
                color: tuple[int, int, int] = (0, 0, 255), label: str | None = None,
                thickness: int = 2) -> np.ndarray:
    """Draw (x, y, w, h) boxes on a copy of the image."""
    out = bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if label:
            cv2.putText(out, label, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def side_by_side(*images: np.ndarray, height: int = 300) -> np.ndarray:
    """Resize a list of BGR images to the same height and stack horizontally,
    for building the kind of "original | mask | detected" figure the paper uses."""
    resized = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        scale = height / h
        resized.append(cv2.resize(img, (int(w * scale), height)))
    return np.hstack(resized)
