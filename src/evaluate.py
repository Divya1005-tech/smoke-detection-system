"""
Evaluation against D-Fire ground-truth smoke boxes, producing the same
kind of True-Positive / False-Positive percentage tables as Table I and
Table II in the reference paper.

Matching rule: a predicted smoke box counts as a True Positive if it has
IoU >= iou_thresh with ANY ground-truth smoke box in that image (each
ground-truth box can only be matched once). Unmatched predictions are
False Positives; unmatched ground-truth boxes are False Negatives.
"""
from __future__ import annotations
import csv
import os

import cv2

from dataset import list_samples
from detect_image import detect_smoke_in_image, load_image
from utils import load_yolo_labels, iou_xywh, CLASS_SMOKE


def evaluate_split(images_dir: str, labels_dir: str, iou_thresh: float = 0.3,
                    min_area: int = 600, limit: int | None = None,
                    skip: int = 0) -> dict:
    samples = list_samples(images_dir, labels_dir)
    if skip:
        samples = samples[skip:]
    if limit:
        samples = samples[:limit]

    total_tp = total_fp = total_fn = 0
    total_gt_frames = 0        # frames that actually contain smoke (ground truth)
    total_gt_boxes = 0
    per_image_rows = []

    for s in samples:
        try:
            img = load_image(s.image_path)
        except FileNotFoundError as e:
            print(f"  Skipping unreadable image: {s.image_path} ({e})")
            continue

        h, w = img.shape[:2]
        gt_boxes = load_yolo_labels(s.label_path, w, h, keep_class=CLASS_SMOKE)

        result = detect_smoke_in_image(img, min_area=min_area)
        pred_boxes = result["smoke_boxes"]

        matched_gt = set()
        tp = 0
        for pb in pred_boxes:
            best_iou, best_j = 0.0, -1
            for j, gb in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = iou_xywh(pb, gb)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_j)

        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        if gt_boxes:
            total_gt_frames += 1
        total_gt_boxes += len(gt_boxes)

        per_image_rows.append({
            "image": s.name,
            "gt_boxes": len(gt_boxes),
            "pred_boxes": len(pred_boxes),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
        })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    summary = {
        "num_images": len(samples),
        "num_smoke_frames": total_gt_frames,
        "total_gt_boxes": total_gt_boxes,
        "true_positive": total_tp,
        "false_positive": total_fp,
        "false_negative": total_fn,
        "precision": round(precision * 100, 2),
        "recall_true_positive_rate": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
    }
    return {"summary": summary, "per_image": per_image_rows}


def save_report(results: dict, out_dir: str, split_name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # summary CSV (paper-style Table I/II combined)
    summary_path = os.path.join(out_dir, f"{split_name}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in results["summary"].items():
            writer.writerow([k, v])

    # per-image detail CSV
    detail_path = os.path.join(out_dir, f"{split_name}_per_image.csv")
    with open(detail_path, "w", newline="") as f:
        fieldnames = ["image", "gt_boxes", "pred_boxes", "true_positive",
                      "false_positive", "false_negative"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results["per_image"])

    print(f"Saved: {summary_path}")
    print(f"Saved: {detail_path}")