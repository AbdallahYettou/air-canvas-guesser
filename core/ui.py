"""A small OpenCV dashboard, with real buttons and a clean drawing board."""

import math
import cv2
import numpy as np
from core.detector import draw_landmarks

WIDTH, HEIGHT = 1320, 810
LEFT = (20, 150, 640, 480)
RIGHT = (680, 150, 620, 480)
BUTTONS = {
    "new": (20, 725, 180, 48), "submit": (215, 725, 180, 48),
    "clear": (410, 725, 140, 48), "save": (565, 725, 140, 48),
}
BG = (27, 22, 20)
INK = (240, 242, 245)
MUTED = (170, 161, 151)
ACCENT = (160, 224, 100)


def text(image, value, xy, scale=0.6, color=INK, thickness=1):
    cv2.putText(image, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def inside(x, y, rect):
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


def board_point(x, y):
    if not inside(x, y, RIGHT):
        return None
    left, top, width, height = RIGHT
    return (x - left) * 640 / width, (y - top) * 480 / height


def render(frame, canvas, game, now, gesture="idle", progress=0, points=None,
           cursor=None, fps=0, mouse=False, model_ready=False):
    screen = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
    text(screen, "AIR CANVAS", (20, 42), 1.0, ACCENT, 2)
    text(screen, "Draw in the air. See what the AI sees.", (20, 73), 0.57, MUTED)
    target = "Free drawing" if game.practice else f"Draw: {game.target}" if game.target else "Ready to draw?"
    text(screen, target, (680, 48), 1.0, INK, 2)
    remaining = "No timer" if game.practice else f"{math.ceil(game.remaining(now))}s left" if game.state == "drawing" else game.state.title()
    text(screen, f"{remaining}   |   Score {game.score}   |   Rounds {game.rounds}", (680, 82), 0.6, MUTED)
    text(screen, "MOUSE MODE" if mouse else f"CAMERA  /  {fps:.0f} fps", (20, 132), 0.55, MUTED)
    text(screen, "YOUR SKETCH", (680, 132), 0.55, MUTED)
    if frame is None:
        camera = np.full((480, 640, 3), (41, 34, 30), np.uint8)
        text(camera, "Click and drag on the white board", (35, 225), 0.7)
        text(camera, "Your webcam is off in mouse mode.", (35, 265), 0.55, MUTED)
    else:
        camera = frame.copy()
        draw_landmarks(camera, points)
        # Keep the ink vivid rather than dimming it with a global blend.
        mask = canvas.image > 8
        camera[mask] = (camera[mask] * 0.25 + np.array(ACCENT) * 0.75).astype(np.uint8)
        if cursor is not None:
            cv2.circle(camera, tuple(np.rint(cursor).astype(int)), 9, ACCENT, 2, cv2.LINE_AA)
    screen[150:630, 20:660] = camera
    board = cv2.cvtColor(255 - canvas.image, cv2.COLOR_GRAY2BGR)
    screen[150:630, 680:1300] = cv2.resize(board, (620, 480))
    status = "Mouse" if mouse else gesture.replace("_", " ").title()
    text(screen, status, (20, 661), 0.6, ACCENT)
    if progress:
        cv2.rectangle(screen, (160, 646), (360, 660), (65, 59, 54), -1)
        cv2.rectangle(screen, (160, 646), (160 + int(progress * 200), 660), ACCENT, -1)
    message = game.message
    if not model_ready:
        message = "Practice mode: download the Kaggle model to unlock AI guesses."
    text(screen, message[:100], (20, 699), 0.57, INK)
    labels = {"new": "New round [N]", "submit": "Guess [Enter]", "clear": "Clear [C]", "save": "Save [S]"}
    for action, (x, y, width, height) in BUTTONS.items():
        cv2.rectangle(screen, (x, y), (x + width, y + height), (64, 53, 44), -1)
        text(screen, labels[action], (x + 12, y + 30), 0.52)
    if game.guesses:
        for index, guess in enumerate(game.guesses):
            text(screen, f"{index + 1}. {guess.label}  {guess.score:.0%}", (730, 678 + index * 32), 0.59,
                 ACCENT if index == 0 else MUTED)
    else:
        text(screen, "Thinking..." if game.state == "submitting" else "Your top 3 guesses will appear here.", (730, 699), 0.57, MUTED)
    text(screen, "1 finger: draw   |   2 fingers: move   |   Palm: guess   |   Hold fist: clear   |   Q: quit", (20, 800), 0.47, MUTED)
    return screen
