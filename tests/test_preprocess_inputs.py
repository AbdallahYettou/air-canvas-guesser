import numpy as np
import pytest
COLORS = [(0, 0, 0), (255, 255, 255), (0, 0, 255)]
from core.preprocess import prepare_sketch, rasterize_strokes


def test_legacy_bgra_inputs_preserve_the_shape_mask():
    strokes = [[[10, 100, 180], [20, 180, 20]], [[50, 140], [100, 100]]]
    expected = prepare_sketch(rasterize_strokes(strokes))
    for color in COLORS:
        drawing = rasterize_strokes(strokes, colors=[color])
        np.testing.assert_array_equal(prepare_sketch(drawing), expected)
    mixed = rasterize_strokes(strokes, colors=list(COLORS))
    np.testing.assert_array_equal(prepare_sketch(mixed), expected)


def test_background_rgb_does_not_count_as_ink_and_opaque_rgb_is_rejected():
    transparent = np.full((20, 20, 4), 255, np.uint8)
    transparent[:, :, 3] = 0
    assert prepare_sketch(transparent) is None
    with pytest.raises(ValueError, match="alpha"):
        prepare_sketch(np.full((20, 20, 3), 255, np.uint8))
