# Project status

Actual work session: November 14, 2026. Local Git author and committer dates are
set to the requested September–October 2026 milestone schedule. They are planned
history dates, not the actual dates the training job ran. Nothing was pushed to a
Git remote.

## Ready locally

- Separate Git repository for this project, isolated from the parent repository.
- Python environment, CPU PyTorch, OpenCV, MediaPipe, and authenticated Kaggle CLI.
- Verified pretrained hand model downloaded into the ignored models folder.
- Webcam drawing, smoothing, deliberate gesture holds, and immediate stroke breaks.
- Mouse practice, clickable controls, PNG export, and clean camera/worker shutdown.
- Thirty-second rounds, automatic submission, top-three guesses, and scoring.
- One background inference request at a time; old results cannot score a new round.
- Small CNN, shared preprocessing, Kaggle training, and validated model download.

## Checked

- 17 automated tests passed, including real CPU model loading and inference.
- Desktop app opened under Xvfb, handled simulated mouse input, and exited cleanly.
- Google hand model loaded and processed a blank frame successfully.
- UI preview rendered and inspected.
- Python compilation and dependency checks passed.
- Trained checkpoint downloaded and validated; installed locally.
- Train/validation/test IDs are disjoint (64,000 / 8,000 / 8,000); the epoch-eight
  recovery checkpoint loads successfully with PyTorch's restricted loader.
- Full desktop round with a generated clock: submitted, recognized as clock,
  scored 100 points, displayed results, and exited cleanly.

No human webcam playtest has been performed. Gesture comfort, lighting tolerance,
and recognition accuracy on real air drawings still need hands-on checks.

## Training handoff

Private job: [yettouabdallah/air-canvas-quickdraw-v1](https://www.kaggle.com/code/yettouabdallah/air-canvas-quickdraw-v1).
Version 1 completed successfully, and its output is downloaded locally.
Test top-1 accuracy: **93.2%**. Test top-3 accuracy: **98.3%** on 8,000 held-out
Quick, Draw! samples. Full metrics, model checksum, and the CPU timing smoke test
are saved in [training-results.json](training-results.json).

Plan: 20 categories, 80,000 sketches total, 8 epochs on a free T4 GPU.
The job finished during development, sooner than the initial 15–30 minute
estimate. Each epoch saved a recovery checkpoint. The best inference model and
the last recovery checkpoint are both available in the downloaded run outputs.

To play now:

```bash
source .venv/bin/activate
python app.py --mouse
python app.py
```

The remaining user check is to try real drawings and webcam gestures. The model
weights stay out of Git; a fresh checkout can use the fetch command in the README.

If Kaggle reports ERROR, inspect its logs before starting another version. The
source code and submission helper are already saved locally.
