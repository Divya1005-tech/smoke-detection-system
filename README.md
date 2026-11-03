# Smoke Detection using Background Subtraction + Fuzzy C-Means

A Python implementation inspired by:

> S.R. Vijayalakshmi, S. Muruganand, *"Smoke detection in video images using
> background subtraction method for early fire alarm system"*, ICCES 2017.

Adapted to run on the **D-Fire dataset** (still images with YOLO-format
smoke/fire bounding boxes) instead of raw video, plus an optional true
video mode that follows the paper's method exactly, and a working
notification system (local alert + real SMS via Twilio) that the paper's
own title promises but never actually implements.

---

## 1. How this maps to the paper

| Paper section | This project |
|---|---|
| Sec II.1 — Background Subtraction (frame differencing, Eq. 1–2) | `src/background_subtraction.py` — `VideoFrameDifferencer` implements this literally for **video**. For **still images** (no paired background frame exists), `single_image_candidate_regions()` uses the smoke colour/edge description from **Sec 2.1** (low saturation, whitish-to-blackish colour, soft/diffuse edges) to produce an equivalent foreground candidate mask. |
| Sec 2.1 — Features of smoke | `src/features.py` — per-blob features: hue, saturation, value, edge density, solidity, extent, and **blur variance** (texture sharpness — an addition beyond the paper, added to reject pale hard-edged objects like cars). |
| Sec II — FCM clustering (Eq. 3–6) | `src/fcm.py` — Fuzzy C-Means implemented directly from the paper's equations (numpy only, no external clustering library required). |
| Sec III — Table I / II (True Positive %, False Positive %) | `src/evaluate.py` + `main.py evaluate` — computes the same style of metrics against D-Fire's ground-truth boxes on the full 3,600+ image test set, saved as CSV in `outputs/reports/`. |
| Fig. 1–3 (original / mask / detected) | `main.py image` saves exactly this 3-panel comparison to `outputs/boxes/*_sidebyside.png`. |
| Title says "early fire **alarm** system" but paper never builds one | `src/alert.py` (local terminal alert + beep) and `src/sms_alert.py` (real SMS via Twilio) — this project actually implements the alarm/notification step. |

**Why still images instead of true background subtraction:** the paper's
core method needs a static background frame and a moving foreground across
a video sequence. D-Fire is a dataset of individual photos, so there's no
"previous frame" to subtract. This project keeps the same overall pipeline
shape (foreground/candidate proposal → smoke features → FCM classification)
but swaps the foreground-proposal step for one based on the smoke
color/edge description the paper itself uses in Section 2.1.

---

## 2. Project structure

```
smoke_detection_project/
├── data/DFireDataset/
│   ├── train/{images,labels}/   <- put your D-Fire train data here
│   └── test/{images,labels}/    <- put your D-Fire test data here
├── src/
│   ├── background_subtraction.py
│   ├── fcm.py
│   ├── features.py
│   ├── dataset.py
│   ├── detect_image.py
│   ├── evaluate.py
│   ├── alert.py         <- local terminal alert + beep
│   ├── sms_alert.py      <- real SMS notification via Twilio
│   └── utils.py
├── outputs/
│   ├── masks/     <- saved candidate/segmentation masks
│   ├── boxes/     <- saved detection images (boxes drawn) + side-by-side figures
│   └── reports/   <- CSV summary + per-image evaluation tables
├── main.py
├── requirements.txt
└── README.md
```

---

## 3. Setup

```bash
pip install -r requirements.txt
```

## 4. Add your dataset

Copy your downloaded D-Fire files so they look like:

```
data/DFireDataset/train/images/xxxx.jpg
data/DFireDataset/train/labels/xxxx.txt
data/DFireDataset/test/images/xxxx.jpg
data/DFireDataset/test/labels/xxxx.txt
```

> D-Fire label format per line: `class_id x_center y_center width height`
> (all normalized 0–1). Default assumes `0 = smoke`, `1 = fire` — adjust
> `CLASS_SMOKE` at the top of `src/utils.py` if your copy differs.

## 5. Run it

**Single image (produces original / mask / detected figure):**
```bash
python main.py image --input data/DFireDataset/test/images/some_image.jpg
```
If smoke is detected, this also triggers a local terminal alert (`src/alert.py`)
and attempts to send an SMS (`src/sms_alert.py`, no-op until Twilio is configured).
Outputs go to `outputs/masks/` and `outputs/boxes/`.

**Evaluate a whole split against ground truth (paper-style Table I/II):**
```bash
python main.py evaluate --split test
```
Outputs go to `outputs/reports/test_summary.csv` and `test_per_image.csv`.

Useful flags: `--iou-thresh 0.3` (how much overlap counts as a correct
detection), `--min-area 600` (ignore tiny noise blobs), `--limit 50` (only
evaluate the first 50 images while testing).

**(Optional) literal paper method on a real video file:**
```bash
python main.py video --input path/to/clip.mp4 --threshold 5
```

## 6. Setting up the SMS alert (optional)

1. Create a free account at https://twilio.com/try-twilio
2. Copy your **Account SID**, **Auth Token**, and free **Twilio phone number**
   from the console (console.twilio.com)
3. Verify each recipient's phone number under
   *Phone Numbers > Manage > Verified Caller IDs* (required on trial accounts)
4. Open `src/sms_alert.py` and fill in `TWILIO_ACCOUNT_SID`,
   `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`, and the `RECIPIENTS` list
   with your verified numbers.

**Important:** this SMS feature is a working technical prototype for a
student project demo — not a production emergency system. Given this
detector's measured false-positive rate (see Section 7), it should never be
connected to real residents or a real fire department without a much more
accurate detector, human review before any alert is sent, and proper
authorization from the relevant authorities.

---

## 7. Results (measured on the full D-Fire test set, 3,622 images)

| Metric | IoU ≥ 0.3 (strict) | IoU ≥ 0.1 (loose) |
|---|---|---|
| Precision | 6.27% | 9.47% |
| Recall (True Positive Rate) | 20.43% | 30.86% |
| F1 Score | 9.6% | 14.49% |

The original paper reports ~85–90% accuracy on a small set of controlled
videos using true frame-differencing with a fixed camera background. This
project's much lower numbers on a large, diverse, real-world still-image
dataset reflect two main, diagnosed failure modes:

1. **Coarse localization** — the detector often finds the right general
   area of smoke but draws an oversized region (e.g. an entire hazy sky)
   instead of a tight box, hurting IoU-based matching against precise
   human-drawn ground truth.
2. **Color ambiguity** — clouds and pale objects (cars, road markings)
   share smoke's low-saturation "greyish/whitish" signature, causing false
   positives; conversely, fire-lit smoke at night appears orange/red (high
   saturation) and is missed entirely by the same color rule, causing false
   negatives.

