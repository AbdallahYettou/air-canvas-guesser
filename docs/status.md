# Current status

Actual work session: November 14, 2026. Local commit dates follow the user's requested
September 29–October 21 milestone schedule; they are not the actual dates of Kaggle
execution. No Git remote push has been made.

## Ready locally

- Simple black ink on a white board; the color palette is removed.
- Fixed brush size, as requested.
- Larger buttons, a separately smoothed control cursor, sticky hit regions,
  a hover progress ring, and short-dropout tolerance.
- Comfortable reach mapping: raw fingertips in the upper-middle camera area
  control the full canvas coordinate space. Lower buttons moved upward, so an
  upright palm has room below the fingertip. Cursor and ink share this mapping.
- Optional Pinch click mode (P) with release-to-rearm behavior.
- More tolerant two-finger Move detection, using 3D hand landmarks when available.
- A visible pen-up state and a short transition guard against stray strokes.
- Undo for the last 30 strokes/actions, including Clear; history resets for a new round.
- Category browser showing only installed model labels; opening it pauses the timer.
- Explicit uncertainty feedback for weak or closely tied predictions.
- Completed 100-class residual checkpoint installed and working on the CPU.
- Always-visible category count and explicit ranges/page numbers in the browser.

## Verified

- 37 automated tests pass, including both 100-class model architectures loading
  and running a CPU forward pass, movement, dropouts, pinch edges, Undo, and UI flows.
- A real OpenCV window under Xvfb accepted simulated clicks, browsed categories,
  drew a black clock, undid a stray stroke, submitted, and scored 100 points with
  the installed model. A further real-window check visited all five category pages
  and scored a round using the actual 100-class checkpoint.
- Every camera button is reachable in geometry tests with a fingertip above row
  250 and a representative upright wrist 180 pixels below it, still inside the frame.
- Python compilation and whitespace checks pass.
- The new UI preview was rendered and inspected.

Live human hand comfort and real air-drawing recognition have not been measured.
These fixes address code-level causes and have simulated-input coverage; the user's
webcam playtest is still needed.

## Completed on Kaggle

[Air Canvas 100 Classes v3](https://www.kaggle.com/code/yettouabdallah/air-canvas-100-classes-v3)
completed successfully. It used Quick, Draw! only: 100 classes, 400,000 sketches,
and eight epochs each for a small CNN and a compact residual network. The residual
checkpoint won on validation accuracy and was calibrated on validation data.

The residual model scored 90.4425% top-1 and 97.65% top-3 on 40,000 held-out samples,
compared with 78.795% / 91.8575% for the small CNN. The downloaded checkpoint contains
100 unique labels matching the configured list. Local CPU prediction measured
2.85 ms median / 3.63 ms p95 including preprocessing. See
[100-class-results.json](100-class-results.json) for the comparison and checksum.

Kaggle normalized the title to `air-canvas-100-classes-v3`; the helper now uses that
actual slug. This is the same submitted run, not a second job.

To download again on a fresh checkout:

```bash
source .venv/bin/activate
python scripts/kaggle_job.py status --username yettouabdallah
python scripts/kaggle_job.py fetch --username yettouabdallah
python app.py
```

The model and UI are ready locally. An app process opened before installation still
holds the old 20-class model in memory: close and reopen it to load all 100 classes.
The browser has five pages of up to 24 categories each. Live human hand comfort
and real air-drawing accuracy remain user playtest items.
