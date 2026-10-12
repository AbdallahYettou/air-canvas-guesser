import numpy as np
import pytest
import torch
from core.guesser import SketchGuesser
from core.model import SketchCNN
from core.preprocess import IMAGE_SIZE, PREPROCESS_VERSION


def checkpoint(path, version=PREPROCESS_VERSION):
    torch.manual_seed(42)
    torch.save({"state_dict": SketchCNN(4).state_dict(),
                "classes": ["cat", "fish", "house", "tree"],
                "image_size": IMAGE_SIZE, "preprocess_version": version}, path)


def test_checkpoint_roundtrip_and_real_forward_pass(tmp_path):
    path = tmp_path / "model.pth"
    checkpoint(path)
    guesser = SketchGuesser(path)
    image = np.zeros((480, 640), np.uint8)
    assert guesser.predict(image) == []
    image[100:150, 200:400] = 255
    guesses = guesser.predict(image)
    assert len(guesses) == 3
    assert all(0 <= guess.score <= 1 for guess in guesses)
    assert guesses == sorted(guesses, key=lambda guess: guess.score, reverse=True)


def test_incompatible_preprocessing_is_rejected(tmp_path):
    path = tmp_path / "model.pth"
    checkpoint(path, version=-1)
    with pytest.raises(ValueError, match="preprocessing"):
        SketchGuesser(path)
