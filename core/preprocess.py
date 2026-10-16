"""The same crop/pad/resize path is used for training and live sketches."""

import cv2
import numpy as np

IMAGE_SIZE = 64
PREPROCESS_VERSION = 2


def prepare_sketch(canvas: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray | None:
    """Normalize a grayscale ink mask or BGRA strokes, ignoring brush color."""
    if canvas.ndim == 3:
        if canvas.shape[2] != 4:
            raise ValueError("Color sketches need an alpha channel to distinguish black/white ink from the background")
        canvas = canvas[:, :, 3]
    if canvas.ndim != 2 or not canvas.size:
        raise ValueError("Expected a nonempty ink mask or BGRA sketch")
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


def rasterize_strokes(strokes, thickness: int = 5, colors=None) -> np.ndarray:
    """Render Quick, Draw! simplified vectors without connecting separate strokes."""
    canvas = np.zeros((280, 280, 4) if colors else (280, 280), np.uint8)
    for index, (xs, ys, *_) in enumerate(strokes):
        color = (*colors[index % len(colors)], 255) if colors else 255
        points = np.array(list(zip(xs, ys)), dtype=np.int32) + 12
        if len(points) == 1:
            cv2.circle(canvas, tuple(points[0]), thickness // 2, color, -1)
        elif len(points) > 1:
            cv2.polylines(canvas, [points], False, color, thickness, cv2.LINE_AA)
    return canvas
