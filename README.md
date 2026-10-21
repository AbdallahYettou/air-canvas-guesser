# Air-Canvas Guesser

Draw with one finger, move with two fingers, and click Submit. A local desktop
Pictionary game with black ink on a white board. No color palette or brush-size
settings. Your webcam stays local, and guessing runs on the CPU.

## Run

The environment and working **100-class model** are installed in this workspace:

```bash
.venv/bin/python app.py
```

Use `--mouse` to draw with a mouse, `--practice` to remove the timer, or
`--camera 1` to try another webcam. Restart after downloading a new model.

For a fresh setup with Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-cpu.txt
python scripts/download_hand_model.py
```

The [dependency lock](requirements-lock.txt) records the tested Linux/Python 3.12
versions. Add `-c requirements-lock.txt` to both install commands to reproduce them.
An absent or incompatible model starts drawing practice without AI guesses.

## Controls

| Action | Hand / onscreen control | Shortcut |
| --- | --- | --- |
| Draw | Index finger extended; other fingers curled | Mouse drag in `--mouse` mode |
| Move without ink | Index and middle fingers extended; lower fingers can relax | Release the mouse |
| Submit | Click or hold over Submit | Enter |
| Clear | Click or hold over Clear | C |
| Undo | Undo the last stroke, or restore a cleared board | U |
| New round | Click or hold over New round | N |
| Categories | Browse what the installed model recognizes; timer pauses | B |
| Click mode | Switch between Hover and Pinch | P |
| Save | Save the visible black-and-white drawing as PNG | S |
| Quit | Close the window | Q / Escape |

The buttons are larger, with a little tolerance
at their edges. Hover shows a progress ring. Brief tracking dropouts pause the
hold instead of clicking; longer losses reset it. Move away after a click to rearm.

The cursor sits **directly on your detected fingertip**, with no coordinate
stretching or cursor smoothing delay. Both button rows sit near the top of the
camera view so you can reach them with your palm upright and still in frame.
Draw below the controls. Stroke smoothing remains enabled for cleaner lines.

For **Pinch** mode, point at a button with your thumb and index finger separated,
then pinch them together. Release before another click. Pinching only clicks UI
controls; it never clears or submits based on a standalone hand pose. Mouse clicks
work in either mode. Palms and fists only pause drawing.

Move means **pen up**, not moving the existing picture. The status line explicitly
shows `Move - pen up`. The pose rules use 3D hand landmarks when available and
avoid drawing during brief transitions back from Move.

Undo keeps the last 30 strokes/actions and does not cross into an earlier round.
A round lasts 30 seconds; the timer submits automatically. Top-1 earns 100 points,
and the target in second or third place earns 30. Each round scores once.

Weak or nearly tied predictions show **Not sure**. This is a conservative display
hint, not a guarantee that every unknown object will be detected. Scores are model
probabilities, not certainty; the new model's temperature is fitted on validation data.

## 100-class model: installed

The installed model knows **100 classes**. The category browser reads the model's
actual labels and shows **24 per page across five pages**. Use Next / Previous;
the footer shows the range and total (for example, `1-24 of 100`). The total is
also visible above the drawing board. Restart an already-open app to load the new model.

The [100-class list](training/classes.json) uses only the Google Quick, Draw!
dataset. A private [Kaggle run](https://www.kaggle.com/code/yettouabdallah/air-canvas-100-classes-v3)
compared the small CNN and a compact residual model on the same data and epoch
budget. The residual model won on validation and achieved **90.4% top-1 / 97.7%
top-3** on 40,000 held-out sketches; the small CNN reached 78.8% top-1. The model
was selected using validation accuracy, not the test set. Local CPU prediction
measured **2.9 ms median**, including preprocessing. See [recorded results](docs/100-class-results.json).

To install the completed model on a fresh checkout:

```bash
source .venv/bin/activate
python scripts/kaggle_job.py status --username yettouabdallah
python scripts/kaggle_job.py fetch --username yettouabdallah
python app.py
```

Fetching validates the checkpoint before replacing the local model. Data, model
weights, credentials, and generated outputs stay out of Git. See
[training notes](docs/training.md) and [current status](docs/status.md).

## Check and troubleshoot

```bash
.venv/bin/python -m pytest -q
.venv/bin/python app.py --preview captures/preview.png
```

- Close other camera apps if the camera cannot open; try `--camera 1` or `--mouse`.
- Show one hand in good light. Start with your palm facing the camera.
- If hovering feels difficult, try Pinch mode with P; keep the pointer within the button.
- Run from a desktop terminal. A remote notebook cannot open this local GUI or webcam.
- Dataset accuracy is not a measurement of real air-drawing accuracy. Live hand
  comfort and recognition still need a human playtest.

## Credits

- [Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset), CC BY 4.0.
- [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).
- [Kaggle CLI](https://github.com/Kaggle/kaggle-cli).
