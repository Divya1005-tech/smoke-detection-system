# """
# Command-line entry point for the smoke detection project.

# Examples
# --------
# py main.py image --input input5.jpg
# py main.py evaluate --split test --skip 400
# py main.py train --split test --limit 400
# py main.py video --input test_smoke_video.mp4
# """
# from __future__ import annotations
# import argparse
# import os
# import sys

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# import cv2

# from detect_image import detect_smoke_in_image, load_image
# from utils import draw_boxes, side_by_side
# from background_subtraction import VideoFrameDifferencer
# from evaluate import evaluate_split, save_report
# from alert import raise_alert
# from phone_alert import send_phone_alert
# from train_classifier import train_and_save

# OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# def run_on_image(input_path: str, min_area: int) -> None:
#     img = load_image(input_path)
#     result = detect_smoke_in_image(img, min_area=min_area)

#     name = os.path.basename(input_path)
#     n_smoke = len(result["smoke_boxes"])

#     if n_smoke > 0:
#         raise_alert(name, n_smoke)          # terminal banner + beep
#         send_phone_alert(name, n_smoke)     # push alert to your phone
#     else:
#         print("No smoke detected - no alert sent.")

#     mask_bgr = cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR)
#     boxed = draw_boxes(img, result["smoke_boxes"], color=(0, 0, 255), label="smoke")
#     combo = side_by_side(img, mask_bgr, boxed)

#     os.makedirs(os.path.join(OUT_DIR, "boxes"), exist_ok=True)
#     os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)
#     base = os.path.splitext(name)[0]

#     mask_path = os.path.join(OUT_DIR, "masks", f"{base}_mask.png")
#     boxed_path = os.path.join(OUT_DIR, "boxes", f"{base}_detected.png")
#     combo_path = os.path.join(OUT_DIR, "boxes", f"{base}_sidebyside.png")

#     cv2.imwrite(mask_path, result["mask"])
#     cv2.imwrite(boxed_path, boxed)
#     cv2.imwrite(combo_path, combo)

#     print(f"Candidate blobs found : {len(result['all_boxes'])}")
#     print(f"Classified as smoke   : {n_smoke}")
#     print(f"Saved side-by-side    -> {combo_path}")


# def run_evaluate(split, iou_thresh, min_area, limit, skip=0) -> None:
#     base = os.path.join(os.path.dirname(__file__), "data", "DFireDataset", split)
#     results = evaluate_split(os.path.join(base, "images"), os.path.join(base, "labels"),
#                               iou_thresh=iou_thresh, min_area=min_area,
#                               limit=limit, skip=skip)
#     print("\n=== Summary ===")
#     for k, v in results["summary"].items():
#         print(f"{k:28s}: {v}")
#     save_report(results, os.path.join(OUT_DIR, "reports"), split)


# def run_on_video(input_path: str, threshold: int) -> None:
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"Could not open video: {input_path}")

#     differ = VideoFrameDifferencer(threshold=threshold)
#     out_dir = os.path.join(OUT_DIR, "masks")
#     os.makedirs(out_dir, exist_ok=True)

#     frame_idx = 0
#     while True:
#         ok, frame = cap.read()
#         if not ok:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         mask = differ.apply(gray)
#         if frame_idx % 15 == 0:
#             cv2.imwrite(os.path.join(out_dir, f"frame_{frame_idx:05d}_mask.png"), mask)
#         frame_idx += 1

#     cap.release()
#     print(f"Processed {frame_idx} frames. Sample masks saved in {out_dir}")


# def main():
#     parser = argparse.ArgumentParser(description="Smoke detection: background subtraction + FCM / Random Forest")
#     sub = parser.add_subparsers(dest="command", required=True)

#     p_img = sub.add_parser("image", help="Run detection on a single image")
#     p_img.add_argument("--input", required=True, help="Path to an image file")
#     p_img.add_argument("--min-area", type=int, default=600)

#     p_eval = sub.add_parser("evaluate", help="Evaluate a dataset split against ground truth")
#     p_eval.add_argument("--split", choices=["train", "test"], default="test")
#     p_eval.add_argument("--iou-thresh", type=float, default=0.3)
#     p_eval.add_argument("--min-area", type=int, default=600)
#     p_eval.add_argument("--limit", type=int, default=None)
#     p_eval.add_argument("--skip", type=int, default=0)

#     p_vid = sub.add_parser("video", help="Frame-differencing on a video")
#     p_vid.add_argument("--input", required=True)
#     p_vid.add_argument("--threshold", type=int, default=5)

#     p_train = sub.add_parser("train", help="Train the Random Forest classifier")
#     p_train.add_argument("--split", choices=["train", "test"], default="train")
#     p_train.add_argument("--min-area", type=int, default=600)
#     p_train.add_argument("--limit", type=int, default=None)

#     args = parser.parse_args()

#     if args.command == "image":
#         run_on_image(args.input, args.min_area)
#     elif args.command == "evaluate":
#         run_evaluate(args.split, args.iou_thresh, args.min_area, args.limit, args.skip)
#     elif args.command == "video":
#         run_on_video(args.input, args.threshold)
#     elif args.command == "train":
#         base = os.path.join(os.path.dirname(__file__), "data", "DFireDataset", args.split)
#         train_and_save(os.path.join(base, "images"), os.path.join(base, "labels"),
#                         min_area=args.min_area, limit=args.limit)


# if __name__ == "__main__":
#     main()

"""
Command-line entry point for the smoke detection project.

Examples
--------
py main.py image --input input5.jpg
py main.py evaluate --split test --skip 400
py main.py train --split test --limit 400
py main.py video --input INPUT.mp4
py main.py video --input INPUT.mp4 --every 15 --confirm 2 --cooldown 60
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import cv2

from detect_image import detect_smoke_in_image, load_image
from utils import draw_boxes, side_by_side
from background_subtraction import VideoFrameDifferencer
from evaluate import evaluate_split, save_report
from alert import raise_alert
from phone_alert import send_phone_alert
from train_classifier import train_and_save

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def run_on_image(input_path: str, min_area: int) -> None:
    img = load_image(input_path)
    result = detect_smoke_in_image(img, min_area=min_area)

    name = os.path.basename(input_path)
    n_smoke = len(result["smoke_boxes"])

    if n_smoke > 0:
        raise_alert(name, n_smoke)          # terminal banner + beep
        send_phone_alert(name, n_smoke)     # push alert to your phone
    else:
        print("No smoke detected - no alert sent.")

    mask_bgr = cv2.cvtColor(result["mask"], cv2.COLOR_GRAY2BGR)
    boxed = draw_boxes(img, result["smoke_boxes"], color=(0, 0, 255), label="smoke")
    combo = side_by_side(img, mask_bgr, boxed)

    os.makedirs(os.path.join(OUT_DIR, "boxes"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "masks"), exist_ok=True)
    base = os.path.splitext(name)[0]

    mask_path = os.path.join(OUT_DIR, "masks", f"{base}_mask.png")
    boxed_path = os.path.join(OUT_DIR, "boxes", f"{base}_detected.png")
    combo_path = os.path.join(OUT_DIR, "boxes", f"{base}_sidebyside.png")

    cv2.imwrite(mask_path, result["mask"])
    cv2.imwrite(boxed_path, boxed)
    cv2.imwrite(combo_path, combo)

    print(f"Candidate blobs found : {len(result['all_boxes'])}")
    print(f"Classified as smoke   : {n_smoke}")
    print(f"Saved side-by-side    -> {combo_path}")


def run_evaluate(split, iou_thresh, min_area, limit, skip=0) -> None:
    base = os.path.join(os.path.dirname(__file__), "data", "DFireDataset", split)
    results = evaluate_split(os.path.join(base, "images"), os.path.join(base, "labels"),
                              iou_thresh=iou_thresh, min_area=min_area,
                              limit=limit, skip=skip)
    print("\n=== Summary ===")
    for k, v in results["summary"].items():
        print(f"{k:28s}: {v}")
    save_report(results, os.path.join(OUT_DIR, "reports"), split)


def run_on_video(input_path: str, threshold: int, min_area: int,
                 every: int, confirm: int, cooldown: float, no_alert: bool) -> None:
    """Frame differencing (as in the reference paper) + smoke classification
    on every `every`-th frame. An alert is sent when smoke is found in
    `confirm` checked frames in a row, at most once per `cooldown` video-seconds."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    differ = VideoFrameDifferencer(threshold=threshold)
    mask_dir = os.path.join(OUT_DIR, "masks")
    det_dir = os.path.join(OUT_DIR, "video_detections")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(det_dir, exist_ok=True)

    name = os.path.basename(input_path)
    frame_idx = 0
    streak = 0
    last_alert_time = -1e9
    alerts_sent = 0
    checks = 0
    smoke_checks = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = differ.apply(gray)

        if frame_idx % every == 0:
            cv2.imwrite(os.path.join(mask_dir, f"frame_{frame_idx:05d}_mask.png"), mask)

            result = detect_smoke_in_image(frame, min_area=min_area)
            n_smoke = len(result["smoke_boxes"])
            checks += 1
            t = frame_idx / fps

            if n_smoke > 0:
                smoke_checks += 1
                streak += 1
                boxed = draw_boxes(frame, result["smoke_boxes"], color=(0, 0, 255), label="smoke")
                cv2.imwrite(os.path.join(det_dir, f"frame_{frame_idx:05d}_detected.png"), boxed)
                print(f"[{t:7.1f}s] frame {frame_idx}: smoke in {n_smoke} region(s) "
                      f"(streak {streak}/{confirm})")

                if streak >= confirm and (t - last_alert_time) >= cooldown:
                    label = f"{name} at {t:.0f}s"
                    raise_alert(label, n_smoke)
                    if not no_alert:
                        send_phone_alert(label, n_smoke)
                    last_alert_time = t
                    alerts_sent += 1
            else:
                streak = 0

        frame_idx += 1

    cap.release()
    print(f"\nProcessed {frame_idx} frames ({frame_idx / fps:.1f} s of video).")
    print(f"Frames checked by classifier : {checks}")
    print(f"Checked frames with smoke    : {smoke_checks}")
    print(f"Alerts sent                  : {alerts_sent}")
    print(f"Masks -> {mask_dir}")
    print(f"Detected frames -> {det_dir}")


def main():
    parser = argparse.ArgumentParser(description="Smoke detection: background subtraction + FCM / Random Forest")
    sub = parser.add_subparsers(dest="command", required=True)

    p_img = sub.add_parser("image", help="Run detection on a single image")
    p_img.add_argument("--input", required=True, help="Path to an image file")
    p_img.add_argument("--min-area", type=int, default=600)

    p_eval = sub.add_parser("evaluate", help="Evaluate a dataset split against ground truth")
    p_eval.add_argument("--split", choices=["train", "test"], default="test")
    p_eval.add_argument("--iou-thresh", type=float, default=0.3)
    p_eval.add_argument("--min-area", type=int, default=600)
    p_eval.add_argument("--limit", type=int, default=None)
    p_eval.add_argument("--skip", type=int, default=0)

    p_vid = sub.add_parser("video", help="Smoke detection + alerts on a video")
    p_vid.add_argument("--input", required=True)
    p_vid.add_argument("--threshold", type=int, default=5)
    p_vid.add_argument("--min-area", type=int, default=600)
    p_vid.add_argument("--every", type=int, default=15,
                       help="Run the classifier on every Nth frame")
    p_vid.add_argument("--confirm", type=int, default=2,
                       help="Consecutive smoke frames needed before alerting")
    p_vid.add_argument("--cooldown", type=float, default=60.0,
                       help="Minimum video-seconds between phone alerts")
    p_vid.add_argument("--no-alert", action="store_true",
                       help="Do not send the phone alert (terminal alert only)")

    p_train = sub.add_parser("train", help="Train the Random Forest classifier")
    p_train.add_argument("--split", choices=["train", "test"], default="train")
    p_train.add_argument("--min-area", type=int, default=600)
    p_train.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.command == "image":
        run_on_image(args.input, args.min_area)
    elif args.command == "evaluate":
        run_evaluate(args.split, args.iou_thresh, args.min_area, args.limit, args.skip)
    elif args.command == "video":
        run_on_video(args.input, args.threshold, args.min_area,
                     args.every, args.confirm, args.cooldown, args.no_alert)
    elif args.command == "train":
        base = os.path.join(os.path.dirname(__file__), "data", "DFireDataset", args.split)
        train_and_save(os.path.join(base, "images"), os.path.join(base, "labels"),
                        min_area=args.min_area, limit=args.limit)


if __name__ == "__main__":
    main()