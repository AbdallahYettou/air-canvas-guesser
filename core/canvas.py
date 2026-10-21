"""Black ink, a white board, and a bounded stroke history for Undo."""

from collections import deque
import cv2
import numpy as np


class Canvas:
    def __init__(self, width=640, height=480, thickness=6):
        self.image = np.zeros((height, width), np.uint8)
        self.thickness = thickness
        self.points = deque(maxlen=5)
        self.previous = None
        self.history = deque(maxlen=30)

    def break_stroke(self):
        self.previous = None
        self.points.clear()

    def clear(self):
        if self.image.any():
            self.history.append(self.image.copy())
        self.image.fill(0)
        self.break_stroke()

    def new_round(self):
        self.clear()
        self.history.clear()

    def undo(self):
        self.break_stroke()
        if not self.history:
            return False
        self.image = self.history.pop()
        return True

    def snapshot(self):
        return self.image.copy()

    def display(self):
        return cv2.cvtColor(255 - self.image, cv2.COLOR_GRAY2BGR)

    def update(self, point, drawing):
        if not drawing or point is None:
            self.break_stroke()
            return
        point = np.asarray(point, dtype=float)
        if not np.isfinite(point).all():
            self.break_stroke()
            return
        height, width = self.image.shape
        point = np.clip(point, [0, 0], [width - 1, height - 1])
        if self.points and np.linalg.norm(point - self.points[-1]) > width * 0.2:
            self.break_stroke()
        self.points.append(point)
        weights = np.arange(1, len(self.points) + 1)
        current = tuple(np.rint(np.average(self.points, axis=0, weights=weights)).astype(int))
        if self.previous is None:
            self.history.append(self.image.copy())
            cv2.circle(self.image, current, self.thickness // 2, 255, -1)
        else:
            cv2.line(self.image, self.previous, current, 255, self.thickness, cv2.LINE_AA)
        self.previous = current
