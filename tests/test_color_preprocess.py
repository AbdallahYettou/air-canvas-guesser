import numpy as np
import pytest
from core.colors import PALETTE
from core.preprocess import prepare_sketch, rasterize_strokes


def test_every_color_including_black_and_white_has_the_same_shape():
    strokes = [[[10, 100, 180], [20, 180, 20]], [[50, 140], [100, 100]]]
    expected = prepare_sketch(rasterize_strokes(strokes))
    for color in PALETTE.values():
        drawing = rasterize_strokes(strokes, colors=[color])
        np.testing.assert_array_equal(prepare_sketch(drawing), expected)
    mixed = rasterize_strokes(strokes, colors=list(PALETTE.values()))
    np.testing.assert_array_equal(prepare_sketch(mixed), expected)


def test_background_rgb_does_not_count_as_ink_and_opaque_rgb_is_rejected():
    transparent = np.full((20, 20, 4), 255, np.uint8)
    transparent[:, :, 3] = 0
    assert prepare_sketch(transparent) is None
    with pytest.raises(ValueError, match="alpha"):
        prepare_sketch(np.full((20, 20, 3), 255, np.uint8))
