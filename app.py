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
from core.controls import CategoryBrowser, DwellPointer, PinchClick, PointerFilter
from core.detector import HandTracker
from core.game import Game
from core.gestures import Gesture, GestureController, pinch_ratio
from core.guesser import AsyncGuesser, SketchGuesser
from core.ui import LEFT, board_point, camera_draw_area, control_at, control_enabled, render

ROOT = Path(__file__).resolve().parent
WINDOW = 'Air-Canvas Guesser'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mouse', action='store_true', help='Draw with the mouse; no webcam needed')
    parser.add_argument('--practice', action='store_true', help='Draw freely without a timer or score')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--click-mode', choices=['hover', 'pinch'], default='hover')
    parser.add_argument('--model', type=Path, default=ROOT / 'models/quickdraw_cnn.pth')
    parser.add_argument('--preview', type=Path, help='Save a sample UI without opening a window')
    return parser.parse_args()


def main():
    args = parse_args()
    classes = json.loads((ROOT / 'training/classes.json').read_text())
    canvas = Canvas()
    if args.preview:
        game = Game(classes, practice=True)
        game.start(0)
        for t in np.linspace(0, 2 * np.pi, 120):
            canvas.update((320 + 95 * np.cos(t), 230 + 95 * np.sin(t)), True)
        canvas.break_stroke()
        for y in np.linspace(230, 165, 15):
            canvas.update((320, y), True)
        canvas.break_stroke()
        for t in np.linspace(0, 1, 15):
            canvas.update((320 + 50 * t, 230 + 30 * t), True)
        canvas.break_stroke()
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.preview), render(None, canvas, game, 0, mouse=True, model_ready=args.model.is_file())):
            raise RuntimeError('Could not write preview image')
        print(f'Preview saved to {args.preview}')
        return
    if sys.platform.startswith('linux') and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        raise RuntimeError('No desktop display found. Run in a desktop terminal or use --preview captures/preview.png.')
    try:
        predictor = SketchGuesser(args.model) if args.model.is_file() else None
    except ValueError as exc:
        print(f'{exc} Starting practice mode until a compatible model is installed.')
        predictor = None
    if predictor:
        classes = predictor.classes
        print(f'Loaded {len(classes)} categories for CPU inference.')
    else:
        print('No sketch model installed. Practice mode has no AI guesses.')
    game = Game(classes, practice=args.practice or predictor is None)
    browser = CategoryBrowser(classes)
    controller, pointer, pinch, cursor_filter = GestureController(), DwellPointer(), PinchClick(), PointerFilter()
    click_mode = args.click_mode
    pointer_locked = False
    events = deque()
    mouse = {'point': None, 'down': False, 'hover': None}
    camera = tracker = worker = None
    window_open = False

    def enabled(action):
        return control_enabled(action, game, predictor is not None, canvas, browser)

    def mouse_event(event, x, y, flags, _):
        point = None if browser.open else board_point(x, y)
        action = control_at(x, y, browser)
        mouse['hover'] = action
        if event == cv2.EVENT_LBUTTONDOWN:
            if action:
                mouse['down'], mouse['point'] = False, None
                if enabled(action):
                    events.append(action)
                return
            mouse['down'] = point is not None
        elif event == cv2.EVENT_LBUTTONUP or (event == cv2.EVENT_MOUSEMOVE and not flags & cv2.EVENT_FLAG_LBUTTON):
            mouse['down'] = False
        mouse['point'] = point

    try:
        if not args.mouse:
            tracker = HandTracker(ROOT / 'models/hand_landmarker.task')
            camera = cv2.VideoCapture(args.camera)
            if not camera.isOpened():
                raise RuntimeError(f'Cannot open camera {args.camera}. Try --camera 1 or --mouse.')
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
        previous_time, fps = time.monotonic(), 0.0
        while True:
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - previous_time, 0.001)
            previous_time = now
            frame, points, point, cursor = None, None, mouse['point'], None
            gesture, progress, drawing = Gesture.IDLE, 0, mouse['down']
            hover = mouse['hover']
            if camera is not None:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError('The camera stopped delivering frames. Reconnect it and restart.')
                frame = cv2.flip(cv2.resize(frame, (640, 480)), 1)
                gesture, point, points = tracker.detect(frame, now)
                event = controller.update(gesture, now)
                cursor = cursor_filter.update(point, now)
                sticky = pointer.target if click_mode == 'hover' else pinch.target
                hover = control_at(LEFT[0] + cursor[0], LEFT[1] + cursor[1], browser, sticky) if cursor else None
                ratio = pinch_ratio(points)
                if hover is None:
                    pointer_locked = False
                if pointer_locked:
                    action = None
                elif click_mode == 'hover':
                    eligible = gesture in [Gesture.DRAW, Gesture.MOVE] and enabled(hover)
                    action = pointer.update(hover if eligible else None, now)
                    progress = pointer.progress if eligible else 0
                else:
                    action = pinch.update(hover if enabled(hover) else None, ratio, now)
                if action:
                    events.append(action)
                drawing = event.drawing and camera_draw_area(point) and camera_draw_area(cursor) and hover is None
                if click_mode == 'pinch' and (ratio is None or ratio < 0.5):
                    drawing = False
            if worker:
                result = worker.poll(game.request_id)
                if result is not None:
                    game.finish(*result)
            if not game.practice and game.state == 'drawing' and not browser.open and game.remaining(now) <= 0:
                events.append('timeout')
            while events:
                action = events.popleft()
                if action in ['categories', 'close']:
                    browser.open = not browser.open if action == 'categories' else False
                    game.pause(now) if browser.open else game.resume(now)
                    mouse['down'], drawing = False, False
                    canvas.break_stroke()
                    pointer.reset()
                    pinch.reset()
                    # The page changes the controls under the fingertip. Require
                    # leaving them before another hand click can be accepted.
                    pointer_locked = True
                elif action in ['previous', 'next'] and browser.open:
                    browser.turn(-1 if action == 'previous' else 1)
                elif action == 'mode':
                    click_mode = 'pinch' if click_mode == 'hover' else 'hover'
                    pointer.reset()
                    pinch.reset()
                    canvas.break_stroke()
                    drawing = False
                elif action == 'new':
                    browser.open = False
                    game.start(now)
                    canvas.new_round()
                    drawing = False
                elif action == 'undo' and game.state == 'drawing':
                    if canvas.undo():
                        game.revision += 1
                        game.guesses = []
                        game.message = 'Last stroke undone.'
                    drawing, mouse['down'] = False, False
                elif action == 'clear' and game.state != 'submitting':
                    if game.state in ['ready', 'result']:
                        game.start(now)
                        canvas.new_round()
                    elif game.clear():
                        canvas.clear()
                    drawing, mouse['down'] = False, False
                elif action == 'save':
                    destination = ROOT / 'captures' / f'sketch-{datetime.now():%Y%m%d-%H%M%S-%f}.png'
                    destination.parent.mkdir(exist_ok=True)
                    if cv2.imwrite(str(destination), canvas.display()):
                        print(f'Saved {destination}')
                        game.message = 'Drawing saved in captures.'
                elif action in ['submit', 'timeout'] and game.state == 'drawing' and not browser.open:
                    if action == 'timeout' and (game.practice or game.remaining(now) > 0):
                        continue
                    if worker is None:
                        game.message = 'Install the trained model first; you can keep drawing for now.'
                    elif not canvas.image.any():
                        if action == 'timeout':
                            game.state = 'submitting'
                            game.finish([])
                        else:
                            game.message = 'Draw something first, then ask me to guess.'
                    elif worker.submit(canvas.snapshot(), game.request_id):
                        game.state = 'submitting'
                        game.message = 'Let me take a look...'
                        canvas.break_stroke()
                    else:
                        game.message = 'Finishing the previous guess. Try again in a moment.'
            timed_out = not game.practice and game.remaining(now) <= 0
            drawing = drawing and game.state == 'drawing' and not timed_out and not browser.open
            canvas.update(point, drawing)
            screen = render(frame, canvas, game, now, gesture=gesture.value, points=points, cursor=cursor,
                            fps=fps, mouse=args.mouse, model_ready=predictor is not None, hover=hover, progress=progress,
                            browser=browser, click_mode=click_mode, drawing=drawing)
            cv2.imshow(WINDOW, screen)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 and browser.open:
                events.append('close')
            elif key in [ord('q'), 27] or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            action = {ord('n'): 'new', ord('c'): 'clear', ord('u'): 'undo', ord('b'): 'categories',
                      ord('p'): 'mode', ord('s'): 'save', 13: 'submit', 10: 'submit'}.get(key)
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


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, FileNotFoundError, ValueError, cv2.error) as exc:
        print(f'Air Canvas: {exc}', file=sys.stderr)
        raise SystemExit(1)
