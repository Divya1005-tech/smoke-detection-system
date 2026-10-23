# Project Q&A Prep Sheet — Smoke Detection Project

Simple answers to questions your professor might ask. Read through once before
your meeting.

---

## Basic / "What is this?" questions

**Q: What is this project about?**
A: It detects smoke in photos and raises an alert when it does. It's based
on a research paper that used video and MATLAB — I rebuilt the same idea
in Python, adapted for a still-image dataset called D-Fire, and added a
real alert/notification system the paper never built.

**Q: Why didn't you use MATLAB like the paper?**
A: Python is free, doesn't need a license, and has powerful open-source
libraries (OpenCV, NumPy) that do the same image processing MATLAB does.
The underlying math and method are identical to the paper — only the
programming language changed.

**Q: What dataset did you use?**
A: D-Fire — a public dataset of ~3,600+ test images (plus a training set)
photographed by real surveillance cameras, each labeled with bounding boxes
showing where smoke or fire appears in the photo.

**Q: Why images instead of video, if the paper used video?**
A: The paper's core method (background subtraction) needs a fixed
background frame and a moving object across many video frames. D-Fire is
individual photos, not video clips, so there's no "previous frame" to
compare against. I adapted the method to work on single images instead,
using the same smoke color/texture description from the paper.

---

## Method / "How does it work?" questions

**Q: What is preprocessing in your project, and where does it happen?**
A: Preprocessing is the set of steps that clean up and prepare each image
before the smoke classifier looks at it. It happens automatically in
`src/background_subtraction.py`, and includes:
1. Converting the image to HSV color space (separates color from brightness)
2. Color thresholding (looking for greyish/whitish, low-saturation regions)
3. Edge detection (Canny) to find soft vs. sharp boundaries
4. Cleaning up the result — removing tiny noise spots, filling small gaps

**Q: What is "background subtraction" in your project?**
A: In the original paper, it means comparing a video frame to a background
image to find what moved. Since I don't have paired background/foreground
video frames, I built an equivalent step for still images: I flag pixels
that match smoke's known color and texture pattern (low saturation,
soft/diffuse edges) as "candidate regions" — this plays the same role as
background subtraction did in the paper.

**Q: What is Fuzzy C-Means (FCM), and why did you use it?**
A: It's a clustering algorithm — it groups data into categories, but instead
of forcing each item into exactly one group, it calculates a "degree of
membership" (how strongly something belongs to each group) before making a
final decision. The paper uses it to sort candidate blobs into "smoke" vs.
"not smoke." I implemented it directly from the paper's mathematical
equations (Section II, Equations 3–6) using NumPy.

**Q: What features do you extract from each candidate region?**
A: For every candidate blob, I calculate:
- Average hue, saturation, and brightness (color)
- Edge density (how many hard edges it has)
- Solidity and extent (shape irregularity — smoke has rough, irregular
  edges, unlike solid objects)
- Blur variance (texture sharpness — smoke is soft/diffuse; cars and road
  markings stay sharp) — this is my own addition, not in the paper.

**Q: Why did you add "blur variance" as a feature?**
A: Early testing showed the program was mistaking pale objects like parked
cars for smoke, because they share smoke's low-saturation color. Cars have
sharp edges even when slightly blurred, while real smoke is soft and
diffuse everywhere. Adding this texture-sharpness feature helped the
classifier tell them apart.

---

## Alert system questions

**Q: Does your project actually send an alert, or just detect smoke?**
A: Both. When smoke is detected, two things happen: (1) a visible red
ALERT banner and a beep sound in the terminal (`src/alert.py`), and (2) an
attempt to send a real SMS text message via Twilio (`src/sms_alert.py`) to
a configured list of recipients.

**Q: Why did you add an alert system? The paper doesn't have one.**
A: The paper's own title is "...for early fire **alarm** system," but the
paper itself only ever does detection — it never builds an actual alarm or
notification step. I noticed that gap and built the missing piece myself.

**Q: Are you really sending SMS to real people's phones and the fire brigade?**
A: For the demo, only to verified test numbers I control (like my own
phone) — Twilio's trial account only allows sending to verified numbers,
which conveniently keeps this safe. I'm not sending to real residents or a
real fire department, because this detector's accuracy isn't high enough
to be trusted for real emergency alerts without a human checking first —
that's a deliberate, explainable design decision, not a limitation I missed.

**Q: What would it take to make this a real deployment?**
A: A much more accurate detector (likely a trained deep-learning model,
not hand-picked color rules), a human reviewing each detection before any
real alert goes out, properly consented contact lists instead of assumed
resident numbers, and official agreement with the local fire department to
receive automated alerts at all.

---

## Results / "How accurate is it?" questions

**Q: What accuracy did you get?**
A: On the full 3,622-image test set:
- Precision: ~6–9%
- Recall (true positive rate): ~20–31%
- (Range depends on how strict the "correct detection" rule is)

**Q: What do precision and recall mean, simply?**
A: Precision = out of everything I flagged as smoke, how much was actually
smoke. Recall = out of all the real smoke in the images, how much did I
actually catch. Low precision means too many false alarms; low recall means
too many misses.

**Q: Why is your accuracy so much lower than the paper's 85–90%?**
A: Two main reasons:
1. The paper tested on a small set of clean, controlled videos with a fixed
   camera and true background subtraction — a much easier setup.
2. I tested on 3,622 varied real-world photos with no background reference,
   different lighting, weather, and scenes — a much harder, more realistic
   challenge.

**Q: What is IoU, and why does it matter for your results?**
A: IoU (Intersection over Union) measures how much a predicted box overlaps
with the real, human-labeled box — used to decide if a detection "counts."
At a loose 10% overlap requirement, my recall nearly doubled (20% → 31%)
compared to a stricter 30% requirement — showing my detector often finds
the right general area, but draws boxes too large/coarse to precisely
match the tight human-drawn labels.

**Q: What caused most of your false positives?**
A: Two things with a similar color signature to smoke: hazy/cloudy sky, and
pale objects like parked cars and painted road lines.

**Q: Can you give a specific example of a failure?**
A: Yes — I tested an image with real fire and a dramatic smoke plume at
night. My detector missed it completely, because the smoke was lit
orange/red by the fire below it — the opposite of the grey/white color my
rule expects. I also tested a plain cloudy-sky photo with no smoke at all,
and my detector falsely flagged the clouds as smoke, because clouds share
smoke's low-saturation grey/white look. These are two opposite, specific,
diagnosed failure modes — not just "it doesn't work well."

---

## Limitations / "What would you improve?" questions

**Q: What's the biggest limitation of your method?**
A: It relies on hand-picked color and texture rules instead of learning
from data. A trained model can learn subtle patterns that distinguish
smoke from sky/haze/cars automatically; my rule-based thresholds can't
fully do that.

**Q: What would you do differently next time / in future work?**
A: Use a trained deep-learning object detector (like YOLO) instead of
hand-crafted color rules — D-Fire's labels are actually designed for this.
This mirrors what the original paper's own authors suggested as their
future work too.

**Q: Did you validate your code, or just assume it works?**
A: I tested it step by step — first on a synthetic test image I made
myself to confirm the pipeline runs correctly, then on real D-Fire images,
checked individual results against the dataset's ground-truth labels by
hand (visualizing the actual labeled smoke box on the photo), and finally
ran a full evaluation across all 3,622 test images to get real numbers.

---

## Code / "Show me the code" questions

**Q: Can you show me where each step happens?**
A:
- Preprocessing (color + edge candidate detection) → `src/background_subtraction.py`
- Feature extraction → `src/features.py`
- FCM clustering → `src/fcm.py`
- Full pipeline that connects everything → `src/detect_image.py`
- Ground-truth comparison / accuracy scoring → `src/evaluate.py`
- Local alert (terminal + beep) → `src/alert.py`
- SMS alert (Twilio) → `src/sms_alert.py`
- Command-line entry point → `main.py`

**Q: Did you write the FCM code yourself, or use a ready-made library?**
A: I implemented it myself from the paper's equations, using only NumPy —
so it matches the math in the paper directly, rather than being a
black-box library call.

**Q: How do you run it?**
A: `python main.py image --input <path>` for one photo (also triggers an
alert if smoke is found), or `python main.py evaluate --split test` to test
against the whole dataset and get accuracy numbers.
