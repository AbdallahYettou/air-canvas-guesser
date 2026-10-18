"""Visible camera controls shared by mouse clicks and fingertip hovering."""

import math
import cv2
import numpy as np
from core.colors import PALETTE
from core.detector import draw_landmarks

WIDTH, HEIGHT = 1320, 810
LEFT = (20, 150, 640, 480)
RIGHT = (680, 150, 620, 480)
BUTTONS = {
    "new": (30, 567, 145, 48), "clear": (185, 567, 145, 48),
    "submit": (340, 567, 155, 48), "save": (505, 567, 145, 48),
}
SWATCHES = {f"color:{name}": (30 + i * 69, 162, 60, 62)
            for i, name in enumerate(PALETTE)}
CONTROLS = {**SWATCHES, **BUTTONS}
BG, INK, MUTED, ACCENT = (27, 22, 20), (240, 242, 245), (170, 161, 151), (160, 224, 100)


def text(image, value, xy, scale=0.6, color=INK, thickness=1):
    cv2.putText(image, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def inside(x, y, rect):
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


def control_at(x, y):
    return next((action for action, rect in CONTROLS.items() if inside(x, y, rect)), None)


def control_enabled(action, game, model_ready):
    if action == "submit":
        return model_ready and game.state == "drawing"
    if action == "clear":
        return game.state != "submitting"
    return True


def board_point(x, y):
    if not inside(x, y, RIGHT):
        return None
    left, top, width, height = RIGHT
    return (x - left) * 640 / width, (y - top) * 480 / height


def camera_draw_area(point):
    # Reserve both toolbars, including the gaps between buttons, for controls.
    return point is not None and 0 <= point[0] < 640 and 90 <= point[1] < 400


class DwellPointer:
    """One activation per visit; moving away or losing tracking resets the hold."""
    def __init__(self):
        self.target = None
        self.since = 0.0
        self.fired = False
        self.progress = 0.0

    def update(self, target, now):
        if target != self.target:
            self.target, self.since, self.fired = target, now, False
        self.progress = 0.0
        if target is None:
            return None
        duration = 1.0 if target == "clear" else 0.75
        self.progress = min(1.0, max(0.0, (now - self.since) / duration))
        if self.progress >= 1 and not self.fired:
            self.fired = True
            return target
        return None


def render(frame, canvas, game, now, gesture="idle", points=None,
           cursor=None, fps=0, mouse=False, model_ready=False, hover=None, progress=0):
    screen = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
    text(screen, "AIR CANVAS", (20, 42), 1.0, ACCENT, 2)
    text(screen, "Pick a color. Draw something. Click Submit.", (20, 75), 0.59, MUTED)
    target = "Free drawing" if game.practice else f"Draw: {game.target}" if game.target else "Ready to draw?"
    text(screen, target, (680, 48), 1.0, INK, 2)
    remaining = "No timer" if game.practice else f"{math.ceil(game.remaining(now))}s left" if game.state == "drawing" else game.state.title()
    text(screen, f"{remaining}   |   Score {game.score}   |   Rounds {game.rounds}", (680, 82), 0.6, MUTED)
    text(screen, "CLICK A COLOR OR BUTTON" if mouse else f"CAMERA / {fps:.0f} fps / HOVER TO CLICK", (20, 132), 0.51, MUTED)
    text(screen, f"YOUR SKETCH  /  {canvas.color_name.upper()} INK", (680, 132), 0.55, MUTED)
    if frame is None:
        camera = np.full((480, 640, 3), (41, 34, 30), np.uint8)
        text(camera, "Pick a color above", (115, 205), 0.8)
        text(camera, "Then click and drag on the drawing board", (35, 250), 0.6, MUTED)
        text(camera, "Clear and Submit are right below", (65, 290), 0.6, MUTED)
    else:
        camera = frame.copy()
        draw_landmarks(camera, points)
        alpha = canvas.ink[:, :, 3:4].astype(np.float32) / 255
        camera = np.clip(camera * (1 - alpha) + canvas.ink[:, :, :3], 0, 255).astype(np.uint8)
    screen[150:630, 20:660] = camera
    screen[150:630, 680:1300] = cv2.resize(canvas.display(), (620, 480))
    # Solid toolbar backgrounds keep the controls legible over any webcam scene.
    cv2.rectangle(screen, (20, 150), (659, 237), (39, 33, 29), -1)
    cv2.rectangle(screen, (20, 552), (659, 629), (39, 33, 29), -1)
    for action, (x, y, width, height) in SWATCHES.items():
        name = action.split(":", 1)[1]
        selected = name == canvas.color_name
        if selected:
            cv2.rectangle(screen, (x, y), (x + width, y + height), ACCENT, 2)
        cv2.circle(screen, (x + width // 2, y + 23), 17, PALETTE[name], -1, cv2.LINE_AA)
        cv2.circle(screen, (x + width // 2, y + 23), 17, (150, 150, 150), 1, cv2.LINE_AA)
        label_width = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0][0]
        text(screen, name, (x + (width - label_width) // 2, y + 53), 0.34, ACCENT if selected else INK)
    labels = {"new": "New round", "submit": "Submit", "clear": "Clear", "save": "Save image"}
    for action, (x, y, width, height) in BUTTONS.items():
        enabled = control_enabled(action, game, model_ready)
        fill = (65, 116, 48) if action == "submit" and enabled else (64, 53, 44) if enabled else (44, 42, 39)
        cv2.rectangle(screen, (x, y), (x + width, y + height), fill, -1)
        label_width = cv2.getTextSize(labels[action], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        text(screen, labels[action], (x + (width - label_width) // 2, y + 31), 0.6, INK if enabled else MUTED)
    if hover in CONTROLS:
        x, y, width, height = CONTROLS[hover]
        cv2.rectangle(screen, (x, y), (x + width, y + height), ACCENT, 2)
        if progress:
            cv2.rectangle(screen, (x, y + height - 4), (x + int(width * progress), y + height), ACCENT, -1)
    if cursor is not None and not mouse:
        center = (int(LEFT[0] + cursor[0]), int(LEFT[1] + cursor[1]))
        if inside(*center, LEFT):
            cv2.circle(screen, center, 11, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(screen, center, 11, ACCENT, 2, cv2.LINE_AA)
    status = "Mouse drawing" if mouse else "Drawing" if gesture == "draw" and hover is None else "Move / choose a control"
    text(screen, status, (20, 668), 0.65, ACCENT)
    message = game.message if model_ready else "Practice mode: fetch the rainbow model to enable guessing."
    lines, line = [], ""
    for word in message.split():
        candidate = f"{line} {word}".strip()
        if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] > 630:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    for index, line in enumerate(lines[:2]):
        text(screen, line, (20, 705 + index * 23), 0.5)
    if game.guesses:
        for index, guess in enumerate(game.guesses):
            text(screen, f"{index + 1}. {guess.label}  {guess.score:.1%}", (700, 674 + index * 34), 0.65,
                 ACCENT if index == 0 else MUTED)
    else:
        text(screen, "Thinking..." if game.state == "submitting" else "Click Submit to see my top 3 guesses.", (700, 699), 0.57, MUTED)
    text(screen, "1 finger: draw / point   |   2 fingers: move   |   Hold over a control to click", (20, 771), 0.51, MUTED)
    text(screen, "Mouse clicks work too.   N: new round   C: clear   Enter: submit   S: save   Q: quit", (20, 799), 0.47, MUTED)
    return screen
