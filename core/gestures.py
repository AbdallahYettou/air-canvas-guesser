"""Small gesture rules and time-based holds, independent of the camera."""

from dataclasses import dataclass
from enum import Enum
import numpy as np


class Gesture(str, Enum):
    IDLE = "idle"
    DRAW = "draw"
    MOVE = "move"


def classify_landmarks(points) -> Gesture:
    if points is None or len(points) != 21:
        return Gesture.IDLE
    p = np.asarray(points, dtype=float)
    if not np.isfinite(p).all():
        return Gesture.IDLE
    # Joint angles work for either hand and tolerate a tilted wrist.
    extended = []
    folded = []
    for base, joint, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        a, b = p[base] - p[joint], p[tip] - p[joint]
        cosine = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8)
        straight = cosine < -0.45
        outward = np.linalg.norm(p[tip] - p[0]) > np.linalg.norm(p[joint] - p[0]) * 1.03
        extended.append(straight and outward)
        folded.append(np.linalg.norm(p[tip] - p[0]) < np.linalg.norm(p[joint] - p[0]) * 1.05)
    # Open palms and fists simply pause. Only onscreen controls trigger actions.
    if extended[0] and extended[1] and not (extended[2] and extended[3]):
        return Gesture.MOVE
    if extended[0] and all(folded[1:]):
        return Gesture.DRAW
    return Gesture.IDLE


@dataclass
class GestureEvent:
    drawing: bool = False


class GestureController:
    def __init__(self):
        self.candidate = Gesture.IDLE
        self.since = 0.0
        self.move_until = 0.0

    def update(self, gesture: Gesture, now: float) -> GestureEvent:
        if gesture == Gesture.MOVE:
            self.move_until = now + 0.18
        if gesture != self.candidate:
            self.candidate, self.since = gesture, now
        return GestureEvent(drawing=gesture == Gesture.DRAW and now - self.since >= 0.12 and now >= self.move_until)


def pinch_ratio(points):
    """Thumb/index distance relative to palm width, independent of camera scale."""
    if points is None or len(points) != 21:
        return None
    p = np.asarray(points, dtype=float)
    width = np.linalg.norm(p[5] - p[17])
    if width < 1e-6 or not np.isfinite(p).all():
        return None
    return float(np.linalg.norm(p[4] - p[8]) / width)
