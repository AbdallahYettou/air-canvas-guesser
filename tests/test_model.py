import numpy as np
import pytest
import torch
from core.guesser import SketchGuesser
from core.model import SketchCNN, create_model
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


@pytest.mark.parametrize("architecture", ["SketchCNN-v1", "SketchResNet-v1"])
def test_100_class_architectures_load_and_predict(tmp_path, architecture):
    classes = [f"class-{i}" for i in range(100)]
    model = create_model(architecture, len(classes))
    path = tmp_path / "model.pth"
    torch.save({"architecture": architecture, "classes": classes, "temperature": 1.3,
                "state_dict": model.state_dict(), "image_size": IMAGE_SIZE,
                "preprocess_version": PREPROCESS_VERSION}, path)
    guesser = SketchGuesser(path)
    image = np.zeros((480, 640), np.uint8)
    image[100:200, 200:210] = 255
    assert len(guesser.predict(image)) == 3
    assert guesser.classes == classes
