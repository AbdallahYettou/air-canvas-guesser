"""Exercise the actual mouse callback, event loop, canvas, and background worker."""

import sys
import time
import cv2
import app
from core.guesser import Guess
from core.ui import BUTTONS, SWATCHES


def test_mouse_can_select_white_clear_select_black_and_submit(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    monkeypatch.setattr(sys, "argv", ["app.py", "--mouse", "--model", str(checkpoint)])
    monkeypatch.setenv("DISPLAY", ":test")
    received = []

    class Predictor:
        classes = ["clock"]

        def __init__(self, _):
            pass

        def predict(self, sketch):
            received.append(sketch)
            return [Guess("clock", 1.0)]

    monkeypatch.setattr(app, "SketchGuesser", Predictor)
    state = {"frame": 0, "callback": None, "done": False}
    for name in ["namedWindow", "resizeWindow", "destroyAllWindows", "imshow"]:
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *args: 1)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda _, callback: state.update(callback=callback))
    original_render = app.render

    def render(frame, canvas, game, now, **kwargs):
        state.update(canvas=canvas, game=game)
        return original_render(frame, canvas, game, now, **kwargs)

    monkeypatch.setattr(app, "render", render)

    def click(action):
        x, y, width, height = {**BUTTONS, **SWATCHES}[action]
        state["callback"](cv2.EVENT_LBUTTONDOWN, x + width // 2, y + height // 2, 0, None)
        state["callback"](cv2.EVENT_LBUTTONUP, x + width // 2, y + height // 2, 0, None)

    def wait(_):
        time.sleep(0.002)
        state["frame"] += 1
        frame = state["frame"]
        if frame == 1:
            click("new")
        elif frame == 2:
            click("color:White")
        elif frame in [3, 9]:
            state["callback"](cv2.EVENT_LBUTTONDOWN, 800, 350, cv2.EVENT_FLAG_LBUTTON, None)
        elif frame in [4, 10]:
            state["callback"](cv2.EVENT_MOUSEMOVE, 840, 350, cv2.EVENT_FLAG_LBUTTON, None)
        elif frame in [5, 11]:
            state["callback"](cv2.EVENT_LBUTTONUP, 840, 350, 0, None)
        elif frame == 6:
            assert state["canvas"].image.any()
            assert state["canvas"].color_name == "White"
            click("clear")
        elif frame == 7:
            assert not state["canvas"].image.any()
            click("color:Black")
        elif frame == 12:
            click("submit")
        elif frame > 12 and state["game"].state == "result":
            assert state["game"].score == 100
            state["done"] = True
            return ord("q")
        assert frame < 200, "App did not finish the submitted round"
        return -1

    monkeypatch.setattr(cv2, "waitKey", wait)
    app.main()
    assert state["done"]
    assert len(received) == 1
    assert received[0].shape == (480, 640, 4)
    assert received[0][:, :, 3].any()
    assert not received[0][:, :, :3].any()


def test_camera_hover_controls_do_not_draw_or_repeat_actions(tmp_path, monkeypatch):
    from core.gestures import Gesture
    from core.ui import LEFT
    import numpy as np

    checkpoint = tmp_path / 'model.pth'
    checkpoint.touch()
    monkeypatch.setattr(sys, 'argv', ['app.py', '--model', str(checkpoint)])
    monkeypatch.setenv('DISPLAY', ':test')
    state = {'frame': 0, 'now': 0.0, 'done': False}
    received = []

    class Predictor:
        classes = ['clock']

        def __init__(self, _):
            pass

        def predict(self, sketch):
            received.append(sketch)
            return [Guess('clock', 1.0)]

    def center(action):
        x, y, width, height = {**BUTTONS, **SWATCHES}[action]
        return x + width / 2 - LEFT[0], y + height / 2 - LEFT[1]

    class Tracker:
        def __init__(self, _):
            pass

        def detect(self, frame, now):
            number = state['frame']
            if number < 6:
                return Gesture.MOVE, center('new'), None
            if number < 12:
                return Gesture.MOVE, center('color:White'), None
            if number < 17:
                return Gesture.DRAW, (200 + number, 230), None
            if number < 24:
                return Gesture.MOVE, center('clear'), None
            if number < 29:
                return Gesture.DRAW, (250 + number, 240), None
            if number < 35:
                # A palm/fist is classified IDLE, even when sitting over Clear.
                return Gesture.IDLE, center('clear'), None
            return Gesture.MOVE, center('submit'), None

        def close(self):
            pass

    class Camera:
        def __init__(self, _):
            pass

        def isOpened(self):
            return True

        def set(self, *args):
            pass

        def read(self):
            return True, np.zeros((480, 640, 3), np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(app, 'SketchGuesser', Predictor)
    monkeypatch.setattr(app, 'HandTracker', Tracker)
    monkeypatch.setattr(cv2, 'VideoCapture', Camera)
    monkeypatch.setattr(app.time, 'monotonic', lambda: state['now'])
    for name in ['namedWindow', 'resizeWindow', 'destroyAllWindows', 'imshow', 'setMouseCallback']:
        monkeypatch.setattr(cv2, name, lambda *args: None)
    monkeypatch.setattr(cv2, 'getWindowProperty', lambda *args: 1)
    original_render = app.render

    def render(frame, canvas, game, now, **kwargs):
        number = state['frame']
        if number == 11:
            assert canvas.color_name == 'White'
            assert not canvas.image.any(), 'Hovering over a toolbar drew ink'
            assert game.round_id == 1, 'Holding New round fired more than once'
        if number == 23:
            assert not canvas.image.any(), 'Clear did not empty the canvas'
            assert game.revision == 1, 'Holding Clear fired more than once'
        if number == 34:
            assert canvas.image.any(), 'An idle pose over Clear erased the drawing'
        if game.state == 'result':
            assert game.score == 100
            state['done'] = True
        return original_render(frame, canvas, game, now, **kwargs)

    monkeypatch.setattr(app, 'render', render)

    def wait(_):
        time.sleep(0.002)
        state['frame'] += 1
        state['now'] += 0.2
        assert state['frame'] < 100
        return ord('q') if state['done'] else -1

    monkeypatch.setattr(cv2, 'waitKey', wait)
    app.main()
    assert state['done']
    assert len(received) == 1
    assert received[0][:, :, 3].any()
