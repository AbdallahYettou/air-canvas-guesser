# Air-Canvas Guesser

Draw in the air with your index finger and let a small neural network guess
what you made. A local desktop game built with Python, OpenCV, MediaPipe, and
PyTorch. Your webcam frames stay on your computer.

## The plan

1. Get drawing and gesture controls feeling comfortable.
2. Add a timer, prompts, and asynchronous guesses.
3. Train a small sketch classifier on Kaggle's free GPU tier.
4. Download the weights and play locally.

We use an existing hand landmark model. Only the sketch classifier needs
training. No paid API or large language model is required.

## Environment

Python 3.12 is the development version. Create an environment inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install -r requirements-cpu.txt
```

Large model files, datasets, credentials, and generated outputs stay out of Git.

## Credits

- [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
- [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset),
  licensed under CC BY 4.0. The training recipe will use recognized vector sketches.
- [Kaggle CLI](https://github.com/Kaggle/kaggle-cli)
