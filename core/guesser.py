"""CPU recognition with a single background request at a time."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from core.preprocess import IMAGE_SIZE, PREPROCESS_VERSION, prepare_sketch


@dataclass(frozen=True)
class Guess:
    label: str
    score: float


class SketchGuesser:
    def __init__(self, checkpoint_path):
        import torch
        from core.model import create_model

        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"No sketch model at {path}. Download the Kaggle output first.")
        torch.set_num_threads(2)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("image_size") != IMAGE_SIZE or checkpoint.get("preprocess_version") != PREPROCESS_VERSION:
            raise ValueError("The sketch model uses a different preprocessing version. Retrain it with this project.")
        self.classes = checkpoint["classes"]
        if not self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError("The sketch model has an invalid class list")
        self.architecture = checkpoint.get("architecture", "SketchCNN-v1")
        self.temperature = float(checkpoint.get("temperature", 1.0))
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid model temperature")
        self.model = create_model(self.architecture, len(self.classes))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def predict(self, canvas) -> list[Guess]:
        import torch

        image = prepare_sketch(canvas)
        if image is None:
            return []
        tensor = torch.from_numpy(image.astype(np.float32) / 255)[None, None]
        with torch.inference_mode():
            scores, indices = (self.model(tensor) / self.temperature).softmax(dim=1)[0].topk(min(3, len(self.classes)))
        return [Guess(self.classes[int(i)], float(p)) for p, i in zip(scores, indices)]


def is_uncertain(guesses):
    """A conservative display hint, not an out-of-distribution detector."""
    if not guesses:
        return True
    margin = guesses[0].score - guesses[1].score if len(guesses) > 1 else guesses[0].score
    return guesses[0].score < 0.5 or margin < 0.15


class AsyncGuesser:
    def __init__(self, predictor):
        self.predictor = predictor
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sketch")
        self.future = None
        self.request_id = None

    @property
    def busy(self):
        return self.future is not None

    def submit(self, canvas, request_id) -> bool:
        if self.busy:
            return False
        self.request_id = request_id
        self.future = self.executor.submit(self.predictor.predict, canvas.copy())
        return True

    def poll(self, current_id):
        """Return (guesses, error), or None; silently discard obsolete results."""
        if self.future is None or not self.future.done():
            return None
        future, request_id = self.future, self.request_id
        self.future = None
        if request_id != current_id:
            return None
        try:
            return future.result(), None
        except Exception as exc:
            return [], f"Could not guess this drawing: {exc}"

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
