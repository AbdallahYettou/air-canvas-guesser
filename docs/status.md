# Project status

Actual work session: November 14, 2026. Local Git author and committer dates are
set to the requested September–October 2026 milestone schedule. They are planned
history dates, not the actual dates the training job ran. Nothing was pushed to a
Git remote.

## Ready locally

- Separate Git repository for this project, isolated from the parent repository.
- Python environment, CPU PyTorch, OpenCV, MediaPipe, and authenticated Kaggle CLI.
- Verified pretrained hand model downloaded into the ignored models folder.
- Webcam drawing, smoothing, and immediate stroke breaks.
- Nine brushes: seven rainbow colors, black, and white, on a neutral gray board.
- Visible camera controls accept mouse clicks or fingertip hovering. Palms and fists
  only pause drawing; Clear and Submit no longer have pose shortcuts.
- Mouse practice, clickable controls, PNG export, and clean camera/worker shutdown.
- Thirty-second rounds, automatic submission, top-three guesses, and scoring.
- One background inference request at a time; old results cannot score a new round.
- Small CNN, shared preprocessing, Kaggle training, and validated model download.

## Checked

- 25 automated tests passed, including real CPU model loading and inference,
  color-independent input, mouse clicks, and simulated hand-hover control flows.
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

## Original training result

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

## Rainbow update

Private job: [Air Canvas Rainbow v2](https://www.kaggle.com/code/yettouabdallah/air-canvas-rainbow-v2).
The UI and tests are committed. The v2 model uses an alpha mask so all nine brush
colors preserve exactly the same shape information, including black and white.
It is retrained with three-, five-, and seven-pixel strokes.

The app falls back to drawing practice if the installed checkpoint predates the
rainbow input format. Fetching the v2 output installs the matching model.

The v2 job completed, and its trained model is installed locally. It achieved
93.6375% top-1 and 98.4% top-3 on 8,000 held-out sketches, with 217 seconds of
measured processing. See [rainbow-training-results.json](rainbow-training-results.json).
The original model and recovery checkpoint remain in the original run outputs.

Verified with the actual model: identical generated clock drawings in all nine
colors produce identical predictions. A real OpenCV window under Xvfb accepted
simulated mouse clicks on White and Submit, recognized the drawn clock, and
awarded 100 points. Live human hand hovering still needs a hands-on playtest.

Local update commits continue the requested schedule on October 16, 18, and 20.
