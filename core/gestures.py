"""Small gesture rules and time-based holds, independent of the camera."""

from dataclasses import dataclass
from enum import Enum
import numpy as np


class Gesture(str, Enum):
    IDLE = "idle"
    DRAW = "draw"
    MOVE = "move"
    SUBMIT = "submit"
    CLEAR = "clear"


def classify_landmarks(points) -> Gesture:
    if points is None or len(points) != 21:
        return Gesture.IDLE
    p = np.asarray(points, dtype=float)[:, :2]
    # Joint angles work for either hand and tolerate a tilted wrist.
    extended = []
    folded = []
    for base, joint, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        a, b = p[base] - p[joint], p[tip] - p[joint]
        cosine = np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8)
        straight = cosine < -0.65
        outward = np.linalg.norm(p[tip] - p[0]) > np.linalg.norm(p[joint] - p[0]) * 1.1
        extended.append(straight and outward)
        folded.append(np.linalg.norm(p[tip] - p[0]) < np.linalg.norm(p[joint] - p[0]) * 1.05)
    if all(extended):
        return Gesture.SUBMIT
    if all(folded):
        return Gesture.CLEAR
    if extended[0] and extended[1] and folded[2] and folded[3]:
        return Gesture.MOVE
    if extended[0] and all(folded[1:]):
        return Gesture.DRAW
    return Gesture.IDLE


@dataclass
class GestureEvent:
    drawing: bool = False
    action: Gesture | None = None
    progress: float = 0.0


class GestureController:
    def __init__(self):
        self.candidate = Gesture.IDLE
        self.since = 0.0
        self.fired = False
        self.holds = {Gesture.DRAW: 0.08, Gesture.SUBMIT: 0.65, Gesture.CLEAR: 1.2}

    def update(self, gesture: Gesture, now: float) -> GestureEvent:
        if gesture != self.candidate:
            self.candidate, self.since, self.fired = gesture, now, False
        delay = self.holds.get(gesture)
        if delay is None:
            return GestureEvent()
        progress = min(1.0, max(0.0, (now - self.since) / delay))
        if gesture == Gesture.DRAW:
            return GestureEvent(drawing=progress >= 1)
        action = None
        if progress >= 1 and not self.fired:
            action, self.fired = gesture, True
        return GestureEvent(action=action, progress=progress)
