# Smoke Detection in Still Images

Smoke detection in still photographs using colour and texture features, Fuzzy C-Means (FCM) or Random Forest classification, and smartphone push alerts.

This project is based on the background-subtraction and FCM method of Vijayalakshmi and Muruganand [1], which works on video. Because a single photo has no background frame, this version replaces frame differencing with colour and edge rules that propose candidate smoke regions. It then classifies each region and sends an alert to a phone when smoke is found.

> **Disclaimer:** this is a student prototype. It produces many false alarms and misses smoke, so it must **not** be used as a life-safety fire alarm.

## Features

- **Candidate regions from a single image:** HSV colour thresholds, local edge density, morphological clean-up and connected components.
- **Seven features per region:** mean hue, saturation and value, edge density, solidity, extent and Laplacian variance (a sharpness measure).
- **Two classifiers:**
  - a trained **Random Forest** (used automatically if `models/smoke_classifier.joblib` exists);
  - **FCM** clustering from the reference paper (used as a fallback).
- **Evaluation** against D-Fire ground-truth boxes (IoU matching, precision, recall, F1).
- **Alerts:** terminal banner with beeps (Windows) and a high-priority push notification through [ntfy](https://ntfy.sh).
- **Video mode:** frame differencing, with periodic smoke classification of frames.

## How it works

```
image -> HSV colour mask + soft-edge mask -> clean-up -> candidate regions
      -> 7 features per region -> Random Forest (or FCM) -> smoke / not smoke
      -> terminal alarm + phone notification
```

## Project structure

```
.
├── main.py                       # command-line entry point
├── requirements.txt
├── src/
│   ├── background_subtraction.py # candidate regions; frame differencing for video
│   ├── features.py               # 7-feature vector per region
│   ├── fcm.py                    # Fuzzy C-Means (NumPy only)
│   ├── train_classifier.py       # trains and saves the Random Forest
│   ├── detect_image.py           # single-image pipeline
│   ├── evaluate.py               # IoU matching, precision / recall / F1
│   ├── dataset.py, utils.py      # label parsing, IoU, drawing helpers
│   ├── alert.py                  # terminal alarm
│   └── phone_alert.py            # ntfy push notification
├── data/DFireDataset/            # NOT included - download it yourself (see below)
├── models/                       # saved model (created by the train command)
└── outputs/                      # masks, detections, CSV reports
```

## Installation

Requires Python 3.10 or newer (tested with Python 3.12 and 3.14).

```bash
git clone https://github.com/[YOUR-USERNAME]/[YOUR-REPO].git
cd [YOUR-REPO]
pip install -r requirements.txt
```

## Dataset

The D-Fire dataset is not included in this repository. Download it from
https://github.com/gaia-solutions-on-demand/DFireDataset and arrange it like this:

```
data/DFireDataset/
├── train/
│   ├── images/   *.jpg
│   └── labels/   *.txt
└── test/
    ├── images/   *.jpg
    └── labels/   *.txt
```

Labels are in YOLO format: `class x_centre y_centre width height`, normalised to 0-1. The code treats **class 0 as smoke** and class 1 as fire, so check this against the version of the dataset you use. If your copy differs, change `CLASS_SMOKE` in `src/utils.py`. Only smoke boxes are used.

## Usage

Use `python` instead of `py` if you are not on Windows.

**Detect smoke in one image**

```bash
py main.py image --input path/to/photo.jpg
```

Saves the candidate mask, the detections and a side-by-side figure to `outputs/`, and sends the alert if smoke is found.

**Train the Random Forest**

```bash
py main.py train --split test --limit 400
```

Trains on the first 400 images of the chosen split and saves `models/smoke_classifier.joblib`. After that, the image and evaluate commands use the Random Forest automatically. Delete the model file to go back to FCM.

**Evaluate against ground truth**

```bash
py main.py evaluate --split test
```

Options: `--limit N` (only the first N images), `--skip N` (ignore the first N images), `--iou-thresh 0.3`, `--min-area 600`. Results are saved to `outputs/reports/`.

To avoid testing on training images, skip the images you trained on:

```bash
py main.py evaluate --split test --skip 400
```

**Video**

```bash
py main.py video --input path/to/video.mp4
```

Run `py main.py video --help` to see all video options.

## Phone alerts (ntfy)

1. Install the **ntfy** app on your phone and subscribe to a topic with a name that is hard to guess.
2. Set the same name in `src/phone_alert.py`:

   ```python
   TOPIC = "your-private-topic-name"
   ```

3. On Android, open the ntfy notification settings and allow the highest-priority channel to make sound and vibrate (and to override Do Not Disturb if you want).

A successful request only means ntfy received the message. Whether the phone makes a sound depends on its notification settings, volume and network connection. ntfy topics are public unless you set up access control, so never send sensitive information, and do not commit your real topic name to a public repository.

## Results

Evaluated on D-Fire images at an IoU threshold of 0.3:

| Run | Images | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Random Forest, all working-copy images | 1,332 | 472 | 921 | 928 | 33.88 % | 33.71 % | 33.80 % |
| Random Forest, held-out images only | 932 | 369 | 729 | 649 | 33.61 % | 36.25 % | 34.88 % |
| FCM only (earlier run, different image set) | 3,622 | 435 | 6,502 | 1,694 | 6.27 % | 20.43 % | 9.60 % |


## Limitations

- Colour and edge thresholds are fixed, and candidate regions are often too large or badly placed, which lowers IoU-based scores.
- Clouds, haze, pale objects and light-coloured vehicles can look like smoke to the colour rules, and smoke lit orange by flames at night can be missed.
- Only smoke is detected; flames are not a separate class.
- The classifier was trained on only 400 images (874 candidate regions).
- The video mode was only checked on a synthetic clip.

## Future work

- Replace or add a trained detector for smoke and fire, such as a YOLO model.
- Add hard negative examples (clouds, haze) to the training data.
- Compare FCM and the Random Forest on exactly the same held-out set, and report average precision and inference time.
- Combine the video motion masks directly with the classifier and test live camera feeds.
- Add delivery checks for alerts and require a human to confirm detections.

## References

1. S. R. Vijayalakshmi and S. Muruganand, "Smoke detection in video images using background subtraction method for early fire alarm system," in *Proc. 2nd Int. Conf. Communication and Electronics Systems (ICCES)*, 2017, pp. 167-171.
2. Gaia Solutions on Demand, "D-Fire: An image data set for fire and smoke detection," GitHub repository. https://github.com/gaia-solutions-on-demand/DFireDataset

