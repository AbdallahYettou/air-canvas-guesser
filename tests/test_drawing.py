import numpy as np
from core.canvas import Canvas
from core.gestures import Gesture, GestureController
from core.preprocess import prepare_sketch, rasterize_strokes


def test_empty_canvas_does_not_produce_a_guess():
    assert prepare_sketch(np.zeros((480, 640), np.uint8)) is None


def test_crop_centers_a_wide_shape_without_squashing_it():
    image = np.zeros((480, 640), np.uint8)
    image[350:370, 20:220] = 255
    result = prepare_sketch(image)
    ys, xs = np.nonzero(result)
    assert result.shape == (64, 64)
    assert np.ptp(xs) > 8 * np.ptp(ys)
    assert abs(xs.mean() - 31.5) < 1
    assert abs(ys.mean() - 31.5) < 1


def test_missing_hand_breaks_stroke_and_smoothing_history():
    canvas = Canvas()
    canvas.update((20, 20), True)
    canvas.update(None, True)
    canvas.update((80, 20), True)
    assert canvas.image[20, 50] == 0
    assert canvas.image[20, 80] == 255


def test_reposition_and_clear_do_not_connect_strokes():
    canvas = Canvas()
    canvas.update((20, 20), True)
    canvas.update((40, 20), False)
    canvas.update((80, 20), True)
    assert canvas.image[20, 50] == 0
    canvas.clear()
    assert not canvas.image.any()
    assert canvas.previous is None


def test_submit_fires_once_per_hold_and_rearms_after_release():
    controller = GestureController()
    assert controller.update(Gesture.SUBMIT, 0).action is None
    assert controller.update(Gesture.SUBMIT, 0.7).action == Gesture.SUBMIT
    assert controller.update(Gesture.SUBMIT, 2).action is None
    controller.update(Gesture.IDLE, 3)
    controller.update(Gesture.SUBMIT, 4)
    assert controller.update(Gesture.SUBMIT, 4.7).action == Gesture.SUBMIT


def test_brief_fist_does_not_clear_and_tracking_loss_stops_ink():
    controller = GestureController()
    controller.update(Gesture.CLEAR, 0)
    assert controller.update(Gesture.CLEAR, 0.5).action is None
    controller.update(Gesture.DRAW, 0.6)
    assert controller.update(Gesture.DRAW, 0.8).drawing
    assert not controller.update(Gesture.IDLE, 0.81).drawing


def test_vector_strokes_stay_separate():
    image = rasterize_strokes([[[0, 10], [0, 0]], [[70, 80], [0, 0]]])
    assert image[12, 50] == 0
    assert prepare_sketch(image) is not None
