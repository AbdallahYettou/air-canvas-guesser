"""The same crop/pad/resize path is used for training and live sketches."""

import cv2
import numpy as np

IMAGE_SIZE = 64
PREPROCESS_VERSION = 1


def prepare_sketch(canvas: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray | None:
    """Return white ink on black, centered in a uint8 square; None for no ink."""
    if canvas.ndim == 3:
        canvas = np.max(canvas[:, :, :3], axis=2)
    if canvas.ndim != 2 or not canvas.size:
        raise ValueError("Expected a nonempty grayscale or color image")
    ys, xs = np.nonzero(canvas > 8)
    if len(xs) == 0:
        return None
    crop = canvas[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = crop.shape
    edge = max(height, width)
    margin = max(2, round(edge * 0.1))
    square = np.zeros((edge + 2 * margin, edge + 2 * margin), np.uint8)
    top = (square.shape[0] - height) // 2
    left = (square.shape[1] - width) // 2
    square[top:top + height, left:left + width] = crop
    return cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)


def rasterize_strokes(strokes, thickness: int = 5) -> np.ndarray:
    """Render Quick, Draw! simplified vectors without connecting separate strokes."""
    canvas = np.zeros((280, 280), np.uint8)
    for xs, ys, *_ in strokes:
        points = np.array(list(zip(xs, ys)), dtype=np.int32) + 12
        if len(points) == 1:
            cv2.circle(canvas, tuple(points[0]), thickness // 2, 255, -1)
        elif len(points) > 1:
            cv2.polylines(canvas, [points], False, 255, thickness, cv2.LINE_AA)
    return canvas
