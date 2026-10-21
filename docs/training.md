# Quick, Draw! training for CPU inference

Training uses only Google's recognized Quick, Draw! vector sketches. The browser
camera is not used as training data. Training runs on Kaggle's free T4 GPU; the
desktop app uses CPU PyTorch with two threads.

## Current comparison

- 100 verified dataset categories, keeping the previous 20 at the start of the list.
- 4,000 unique, nonempty, recognized drawings per category: 400,000 total.
- Seeded per-class split: 320,000 train, 40,000 validation, 40,000 test.
- Eight epochs each for `SketchCNN-v1` and `SketchResNet-v1`.
- Same data, optimizer settings, augmentations, epoch budget, and validation selection.
- Mixed precision on the GPU to shorten training; normal float32 CPU inference.
- Choose the candidate with the best validation top-1. Evaluate the test set only
  after that choice. Fit a scalar temperature on validation logits for display scores.

The prefix sampling is bounded, not a random sample of the entire Quick, Draw!
dataset. Exact drawing IDs are saved. Training uses three-, five-, and seven-pixel
strokes plus mild rotation, scaling, translation, and thickening; the app brush
remains fixed. Crop/pad/resize is shared between training and inference.

Before launch, on this machine with two CPU threads and a 64x64 input:

| Architecture | Parameters (100 classes) | Median forward pass | p95 |
| --- | ---: | ---: | ---: |
| SketchCNN-v1 | 261,860 | 1.15 ms | 1.80 ms |
| SketchResNet-v1 | 944,324 | 1.82 ms | 2.88 ms |

These are 100 warm forward passes with untrained weights, excluding preprocessing
and camera work. Verify actual inference again when the selected checkpoint arrives.
They are not hardware-independent performance guarantees.

## Run and fetch

The active private job is
[Air Canvas 100 Classes v3](https://www.kaggle.com/code/yettouabdallah/air-canvas-100-classes-v3).
The CLI uses normal local Kaggle credentials; never put tokens in Git.

```bash
source .venv/bin/activate
python scripts/kaggle_job.py prepare --username YOUR_USERNAME
python scripts/kaggle_job.py push --username YOUR_USERNAME
python scripts/kaggle_job.py status --username YOUR_USERNAME
kaggle kernels logs YOUR_USERNAME/air-canvas-100-classes-v3
```

The helper uploads a self-contained script generated from the shared model,
preprocessing, and training sources. It requests a free T4 with a two-hour limit.
Do not push again just to check progress: that starts another version.

When COMPLETE:

```bash
python scripts/kaggle_job.py fetch --username YOUR_USERNAME
python app.py
```

Files download to ignored `runs/air-canvas-100-classes-v3/`. The selected model is
validated and installed into the ignored [models folder](../models/).

Outputs include:

- `quickdraw_cnn.pth`: selected model, 100 labels, architecture, and temperature.
- `metrics.json` and `comparison.json`: held-out metrics and both candidates' results.
- `split_ids.json`: the exact IDs used in each split.
- One subfolder per architecture with `best.pth`, `last_checkpoint.pth`,
  `history.json`, and `summary.json`.

## Recovery

Each epoch atomically saves model/optimizer/scheduler/scaler state, the best weights,
random states, and progress. Kaggle must still preserve/publish the output for you
to retrieve it; checkpoints cannot guarantee recovery after a platform termination.

To recover one candidate, attach its checkpoint as a private Kaggle input and run
the generated training script in a GPU session, keeping the sample count, seed,
class list, and total epoch count unchanged:

```bash
python train.py --architectures SketchResNet-v1 --epochs 8 --resume /kaggle/input/YOUR_INPUT/last_checkpoint.pth
```

Recovery with a single architecture reports that candidate only. Do not interpret
it as a new two-model comparison. The dataset is downloaded again deterministically.

The historical 20-class results remain in [baseline results](training-results.json)
and [the second run](rainbow-training-results.json). Their accuracies do not directly
compare with the harder 100-class task.
