# Current status

Actual work session: November 14, 2026. Local commit dates follow the user's requested
September 29–October 21 milestone schedule; they are not the actual dates of Kaggle
execution. No Git remote push has been made.

## Ready locally

- Simple black ink on a white board; the color palette is removed.
- Fixed brush size, as requested.
- Larger buttons, a separately smoothed control cursor, sticky hit regions,
  a hover progress ring, and short-dropout tolerance.
- Optional Pinch click mode (P) with release-to-rearm behavior.
- More tolerant two-finger Move detection, using 3D hand landmarks when available.
- A visible pen-up state and a short transition guard against stray strokes.
- Undo for the last 30 strokes/actions, including Clear; history resets for a new round.
- Category browser showing only installed model labels; opening it pauses the timer.
- Explicit uncertainty feedback for weak or closely tied predictions.
- Existing 20-class checkpoint still installed and working on the CPU.

## Verified

- 34 automated tests pass, including both 100-class model architectures loading
  and running a CPU forward pass, movement, dropouts, pinch edges, Undo, and UI flows.
- A real OpenCV window under Xvfb accepted simulated clicks, browsed categories,
  drew a black clock, undid a stray stroke, submitted, and scored 100 points with
  the existing model.
- Python compilation and whitespace checks pass.
- The new UI preview was rendered and inspected.

Live human hand comfort and real air-drawing recognition have not been measured.
These fixes address code-level causes and have simulated-input coverage; the user's
webcam playtest is still needed.

## Running on Kaggle

[Air Canvas 100 Classes v3](https://www.kaggle.com/code/yettouabdallah/air-canvas-100-classes-v3)
was submitted successfully. It uses Quick, Draw! only: 100 classes, 400,000 sketches,
and eight epochs each for a small CNN and a compact residual network. The final
checkpoint is selected using validation accuracy and calibrated on validation data.

Initial estimate: 20–40 minutes including preparation and startup. Check the log for
measured epoch timing. Every epoch writes recovery checkpoints. No 100-class
accuracy or completed-model claim has been made yet.

Kaggle normalized the title to `air-canvas-100-classes-v3`; the helper now uses that
actual slug. This is the same submitted run, not a second job.

When complete:

```bash
source .venv/bin/activate
python scripts/kaggle_job.py status --username yettouabdallah
python scripts/kaggle_job.py fetch --username yettouabdallah
python app.py
```

Then check the selected architecture, accuracy, CPU latency, and 100-class UI,
record the results, and commit the result summary. Until then, the installed
20-class model keeps the game usable. Training setup and UI changes are saved locally.
