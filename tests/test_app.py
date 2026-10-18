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
