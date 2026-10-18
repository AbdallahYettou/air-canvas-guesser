"""A clean canvas; the webcam image is never part of the drawing."""

from collections import deque
import cv2
import numpy as np
from core.colors import BOARD_COLOR, PALETTE


class Canvas:
    def __init__(self, width=640, height=480, thickness=6):
        self.image = np.zeros((height, width), np.uint8)
        self.ink = np.zeros((height, width, 4), np.uint8)
        self.color_name = "Black"
        self.thickness = thickness
        self.points = deque(maxlen=5)
        self.previous = None

    def break_stroke(self):
        self.previous = None
        self.points.clear()

    def clear(self):
        self.image.fill(0)
        self.ink.fill(0)
        self.break_stroke()

    def select_color(self, name):
        if name not in PALETTE:
            raise ValueError(f"Unknown brush color: {name}")
        self.color_name = name
        self.break_stroke()

    def snapshot(self):
        """Alpha carries the shape, even when the visible ink is black or white."""
        return self.ink.copy()

    def display(self):
        # Ink channels are already premultiplied by OpenCV's antialiasing.
        alpha = self.ink[:, :, 3:4].astype(np.float32) / 255
        return np.clip(self.ink[:, :, :3] + np.array(BOARD_COLOR) * (1 - alpha), 0, 255).astype(np.uint8)

    def update(self, point, drawing: bool):
        if not drawing or point is None:
            self.break_stroke()
            return
        point = np.asarray(point, dtype=float)
        if not np.isfinite(point).all():
            self.break_stroke()
            return
        height, width = self.image.shape
        point = np.clip(point, [0, 0], [width - 1, height - 1])
        # A tracking jump starts a new stroke instead of slashing across the board.
        if self.points and np.linalg.norm(point - self.points[-1]) > width * 0.2:
            self.break_stroke()
        self.points.append(point)
        weights = np.arange(1, len(self.points) + 1)
        current = tuple(np.rint(np.average(self.points, axis=0, weights=weights)).astype(int))
        if self.previous is None:
            cv2.circle(self.image, current, self.thickness // 2, 255, -1)
            cv2.circle(self.ink, current, self.thickness // 2, (*PALETTE[self.color_name], 255), -1)
        else:
            cv2.line(self.image, self.previous, current, 255, self.thickness, cv2.LINE_AA)
            cv2.line(self.ink, self.previous, current, (*PALETTE[self.color_name], 255), self.thickness, cv2.LINE_AA)
        self.previous = current
