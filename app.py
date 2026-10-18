"""Run the desktop game: python app.py (or python app.py --mouse)."""

import argparse
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from core.canvas import Canvas
from core.detector import HandTracker
from core.game import Game
from core.gestures import Gesture, GestureController
from core.guesser import AsyncGuesser, SketchGuesser
from core.ui import LEFT, DwellPointer, board_point, camera_draw_area, control_at, control_enabled, render
from core.colors import PALETTE

ROOT = Path(__file__).resolve().parent
WINDOW = "Air-Canvas Guesser"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mouse", action="store_true", help="Draw with the mouse; no webcam needed")
    parser.add_argument("--practice", action="store_true", help="Draw freely without a timer or score")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=ROOT / "models/quickdraw_cnn.pth")
    parser.add_argument("--preview", type=Path, help="Save a sample UI image without opening a window")
    return parser.parse_args()


def main():
    args = parse_args()
    classes = json.loads((ROOT / "training/classes.json").read_text())
    canvas = Canvas()
    if args.preview:
        game = Game(classes, practice=True)
        game.start(0)
        for index, name in enumerate(PALETTE):
            canvas.select_color(name)
            for x in range(90, 540, 6):
                canvas.update((x, 75 + index * 40 + 10 * np.sin(x / 30)), True)
        canvas.break_stroke()
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview), render(None, canvas, game, 0, mouse=True)):
            raise RuntimeError("Could not write preview image")
        print(f"Preview saved to {args.preview}")
        return
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("No desktop display found. Run this in a desktop terminal, or use --preview captures/preview.png.")
    try:
        predictor = SketchGuesser(args.model) if args.model.is_file() else None
    except ValueError as exc:
        print(f"{exc} Starting practice mode until the new model is downloaded.")
        predictor = None
    if predictor:
        classes = predictor.classes
    else:
        print("Sketch model is not installed yet. Starting in practice mode (no AI guesses).")
    game = Game(classes, practice=args.practice or predictor is None)
    controller = GestureController()
    pointer = DwellPointer()
    events = deque()
    mouse = {"point": None, "down": False, "hover": None}
    camera = tracker = worker = None
    window_open = False

    def mouse_event(event, x, y, flags, _):
        point = board_point(x, y)
        mouse["hover"] = control_at(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            action = control_at(x, y)
            if action:
                mouse["down"] = False
                mouse["point"] = None
                if control_enabled(action, game, predictor is not None):
                    events.append(action)
                return
            mouse["down"] = point is not None
        elif event == cv2.EVENT_LBUTTONUP:
            mouse["down"] = False
        # A lost release outside the window must not keep the pen down.
        elif event == cv2.EVENT_MOUSEMOVE and not (flags & cv2.EVENT_FLAG_LBUTTON):
            mouse["down"] = False
        mouse["point"] = point

    try:
        if not args.mouse:
            tracker = HandTracker(ROOT / "models/hand_landmarker.task")
            camera = cv2.VideoCapture(args.camera)
            if not camera.isOpened():
                raise RuntimeError(f"Cannot open camera {args.camera}. Try --camera 1 or --mouse.")
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if predictor:
            worker = AsyncGuesser(predictor)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW, 1188, 729)
        window_open = True
        cv2.setMouseCallback(WINDOW, mouse_event)
        if game.practice:
            game.start(time.monotonic())
        previous_time = time.monotonic()
        fps = 0.0
        while True:
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - previous_time, 0.001)
            previous_time = now
            frame, points, point = None, None, mouse["point"]
            gesture, progress, drawing = Gesture.IDLE, 0, mouse["down"]
            hover = mouse["hover"]
            if camera is not None:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("The camera stopped delivering frames. Reconnect it and restart.")
                frame = cv2.flip(cv2.resize(frame, (640, 480)), 1)
                gesture, point, points = tracker.detect(frame, now)
                event = controller.update(gesture, now)
                hover = control_at(LEFT[0] + point[0], LEFT[1] + point[1]) if point is not None else None
                eligible = gesture in [Gesture.DRAW, Gesture.MOVE] and control_enabled(hover, game, predictor is not None)
                action = pointer.update(hover if eligible else None, now)
                progress = pointer.progress
                if action:
                    events.append(action)
                drawing = event.drawing and camera_draw_area(point)
            if worker:
                result = worker.poll(game.request_id)
                if result is not None:
                    game.finish(*result)
            timed_out = not game.practice and game.state == "drawing" and game.remaining(now) <= 0
            if timed_out:
                events.append("timeout")
            while events:
                action = events.popleft()
                if action == "new":
                    game.start(now)
                    canvas.clear()
                elif action.startswith("color:"):
                    canvas.select_color(action.split(":", 1)[1])
                elif action == "clear" and game.state != "submitting":
                    if game.state in ["ready", "result"]:
                        game.start(now)
                    if game.clear():
                        canvas.clear()
                elif action == "save":
                    destination = ROOT / "captures" / f"sketch-{datetime.now():%Y%m%d-%H%M%S-%f}.png"
                    destination.parent.mkdir(exist_ok=True)
                    if cv2.imwrite(str(destination), canvas.display()):
                        print(f"Saved {destination}")
                        game.message = "Drawing saved in captures."
                elif action in ["submit", "timeout"] and game.state == "drawing":
                    # A queued timeout from the previous round must not submit a new one.
                    if action == "timeout" and (game.practice or game.remaining(now) > 0):
                        continue
                    if worker is None:
                        game.message = "Install the trained model first; you can keep drawing for now."
                    elif not canvas.image.any():
                        if action == "timeout":
                            game.state = "submitting"
                            game.finish([])
                        else:
                            game.message = "Draw something first, then ask me to guess."
                    elif worker.submit(canvas.snapshot(), game.request_id):
                        game.state = "submitting"
                        game.message = "Let me take a look..."
                        canvas.break_stroke()
                    else:
                        game.message = "Finishing the previous guess. Try again in a moment."
            timed_out = not game.practice and game.remaining(now) <= 0
            canvas.update(point, drawing and game.state == "drawing" and not timed_out)
            screen = render(frame, canvas, game, now, gesture=gesture.value, points=points, cursor=point,
                            fps=fps, mouse=args.mouse, model_ready=predictor is not None, hover=hover, progress=progress)
            cv2.imshow(WINDOW, screen)
            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27] or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            action = {ord("n"): "new", ord("c"): "clear", ord("s"): "save", 13: "submit", 10: "submit"}.get(key)
            if action:
                events.append(action)
    finally:
        if worker:
            worker.close()
        if camera:
            camera.release()
        if tracker:
            tracker.close()
        if window_open:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError, cv2.error) as exc:
        print(f"Air Canvas: {exc}", file=sys.stderr)
        raise SystemExit(1)
