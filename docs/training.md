# Training on Kaggle

The model uses 20 categories, 4,000 recognized sketches per category,
and eight epochs. It has 251,540 parameters. Expect roughly 15–30 minutes on
a T4, plus queue/startup time. This is an estimate, not a measured guarantee.
The log prints an updated estimate after every epoch.

The original monochrome run completed in 211 seconds of measured processing on a Tesla T4,
excluding queue/startup and final output publishing. It reached 93.2% held-out
top-1 and 98.3% top-3 accuracy. See [the full results](training-results.json).

The rainbow update trains a separate v2 checkpoint using colored strokes with
an alpha channel and varying brush widths. All nine colors become the same
shape mask before classification, so changing color cannot change a guess for
otherwise identical strokes. This is still a 20-object classifier, not a color classifier.

The v2 run completed in 217 seconds of measured processing on a Tesla T4:
**93.6% top-1 / 98.4% top-3** on 8,000 held-out sketches. These sketches use the
new varied-width rendering, so the baseline and v2 scores are not a controlled
comparison. [The v2 results](rainbow-training-results.json) include a nine-color
consistency check with the installed model.

## Run it

Kaggle needs an authenticated account with GPU quota, Internet enabled, and any
account verification Kaggle requires. Use its free GPU allowance; no paid service
is configured. The CLI reads credentials from its normal local locations. Never
put a token in the repository or in a commit.

```bash
source .venv/bin/activate
python scripts/kaggle_job.py prepare --username YOUR_USERNAME
python scripts/kaggle_job.py push --username YOUR_USERNAME
python scripts/kaggle_job.py status --username YOUR_USERNAME
kaggle kernels logs YOUR_USERNAME/air-canvas-rainbow-v2
```

The push creates or updates a **private** Kaggle script with a two-hour time limit.
It uploads a generated single-file bundle of [preprocessing](../core/preprocess.py),
[the model](../core/model.py), and [the training loop](../training/train.py).
Kaggle does not automatically receive neighboring source files. The generated
bundle and account-specific metadata live in the ignored `.kaggle-run/` folder.

Check status before pushing again: pushing an existing slug starts another version.
An empty CLI log does not necessarily mean the run has stalled; check the Kaggle page.

## Download when complete

```bash
python scripts/kaggle_job.py fetch --username YOUR_USERNAME
python app.py
```

Outputs:

- `quickdraw_cnn.pth`: best validation model, labels, and preprocessing version.
- `last_checkpoint.pth`: model, optimizer, best weights, epoch, and random states.
- `history.json`: losses, validation scores, and epoch durations.
- `metrics.json`: final held-out top-1/top-3, confusion matrix, device, and versions.
- `split_ids.json`: exact drawing IDs in each split.

The fetch helper installs only a validated inference checkpoint into the local
[models folder](../models/). The rest remains in ignored `runs/air-canvas-rainbow-v2/`.

## How the data is handled

For each category, download only the first 4,000 unique recognized nonempty
vector sketches, shuffle with seed 42 plus the category index, and split 80/10/10
into train/validation/test. This is a bounded prefix sample, not a random sample
of the entire dataset. Validation chooses the checkpoint; the held-out test set
is evaluated only after training. Augmentation is applied only to training data.

Both live drawings and training examples use the same bounding-box crop,
square padding, 10% margin, and 64x64 grayscale representation. Training renders
vectors with three-, five-, or seven-pixel brushes and random per-stroke palette
colors, then extracts the alpha mask; mild rotation, scaling, translation, and stroke
thickening help with variation. There is still a domain gap to air drawings.

## If a run is interrupted

Each epoch saves a recovery checkpoint and the best inference model atomically.
Kaggle must still preserve/publish the run's output for you to retrieve it; local
checkpoint writes cannot guarantee recovery after a platform termination.

If the last checkpoint is available, attach it as a private Kaggle input, use
the generated training script there, and run with:

```bash
python train.py --resume /kaggle/input/YOUR_CHECKPOINT/last_checkpoint.pth --epochs 8
```

Keep the seed, sample count, class list, and preprocessing unchanged. `--epochs`
means the total epoch count, not extra epochs. Resumption downloads the same
bounded data again and restores training state. Training deliberately refuses to
run outside a Kaggle GPU session.
