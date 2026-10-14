# Air-Canvas Guesser

Draw with your index finger. Hold up your palm. See what the AI thinks it is.

A local desktop game using OpenCV, MediaPipe hand tracking, and a small PyTorch
sketch classifier. No paid API. Webcam images stay on your computer. Only the
sketch classifier is trained, on Kaggle's free GPU tier.

## Start here

Python 3.12 is the development version. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-cpu.txt
python scripts/download_hand_model.py
python app.py
```

The tested Linux/Python 3.12 dependency versions are recorded in
[requirements-lock.txt](requirements-lock.txt). Add `-c requirements-lock.txt` to
both install commands to reproduce that environment; the CPU PyTorch index is
still needed for the second command.

The environment, hand model, and trained sketch model are already installed in
the current workspace. The first Kaggle run completed successfully: **93.2% top-1**
and **98.3% top-3** on 8,000 held-out Quick, Draw! sketches. See
[the recorded results](docs/training-results.json) for metrics and limitations.
For a quick try, run `.venv/bin/python app.py`.

On a fresh checkout, until the trained model is downloaded, the app opens in **practice mode**.
Drawing works, but guesses and scoring are unavailable. It never invents guesses.

No camera handy? Run `python app.py --mouse` and draw on the white board.
Use `--camera 1` to select another webcam, or `--practice` to remove the timer.

## Play

Click **New round** or press **N**. You have 30 seconds to draw the displayed word.
Hold your hand comfortably in view, with your palm facing the camera.

| Action | Hand gesture | Keyboard |
| --- | --- | --- |
| Draw | Index finger extended, other fingers curled | Mouse drag with `--mouse` |
| Move without drawing | Index and middle fingers extended | Release the mouse |
| Guess | Hold open palm for 0.65 seconds | Enter |
| Clear | Hold fist for 1.2 seconds | C |
| New round | Use the button | N |
| Save drawing | Use the button | S |
| Quit | Close the window | Q or Escape |

The timer also submits automatically. A correct first guess earns 100 points;
the target appearing second or third earns 30. Each round scores once. Model
percentages are softmax scores, not calibrated probabilities of being right.
Clear works during drawing; after a guess, start a new round.

## Get the trained model

The private training job is
[Air Canvas QuickDraw v1](https://www.kaggle.com/code/yettouabdallah/air-canvas-quickdraw-v1).
With Kaggle authenticated locally:

```bash
python scripts/kaggle_job.py status --username yettouabdallah
python scripts/kaggle_job.py fetch --username yettouabdallah
python app.py
```

Fetch validates the model before installing it. Weights go into the ignored
[models folder](models/); training metrics and logs go into `runs/kaggle/`.
For another account or a new training run, see [the training notes](docs/training.md).

## Checks and troubleshooting

```bash
python -m pytest -q
python app.py --preview captures/preview.png
```

- **Camera won't open:** close other camera apps, check permissions, try `--camera 1`.
- **Tracking feels unreliable:** improve lighting, show one hand, and keep it in frame.
- **No desktop display:** run in a local desktop terminal. A remote notebook cannot
  open your webcam or an OpenCV window. `--preview` works without a display.
- **The AI guesses poorly:** draw a large, simple outline. It only knows the
  [20 trained categories](training/classes.json). Air drawings differ from the dataset;
  the held-out test score does not measure real webcam drawing accuracy.
- **Model missing:** practice mode is expected until the Kaggle run finishes and
  its output is downloaded. Do not rename the hand model into the sketch model.

See [project status](docs/status.md) for completed checks and remaining work.

## Credits

- [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
- [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset),
  licensed under CC BY 4.0. The training recipe uses recognized vector sketches.
- [Kaggle CLI](https://github.com/Kaggle/kaggle-cli)

Large model files, datasets, credentials, and generated outputs stay out of Git.
