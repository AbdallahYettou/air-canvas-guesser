import numpy as np
from core.canvas import Canvas
from core.gestures import Gesture, GestureController, classify_landmarks
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


def test_drawing_is_debounced_and_tracking_loss_stops_ink():
    controller = GestureController()
    assert not controller.update(Gesture.DRAW, 0.6).drawing
    assert controller.update(Gesture.DRAW, 0.8).drawing
    assert not controller.update(Gesture.IDLE, 0.81).drawing
    assert not controller.update(Gesture.IDLE, 5).drawing


def test_vector_strokes_stay_separate():
    image = rasterize_strokes([[[0, 10], [0, 0]], [[70, 80], [0, 0]]])
    assert image[12, 50] == 0
    assert prepare_sketch(image) is not None


def test_landmark_rules_work_for_rotated_and_mirrored_hands():
    def hand(extended):
        points = np.zeros((21, 2))
        for index, base in enumerate([5, 9, 13, 17]):
            x = (index - 1.5) * 15
            points[base] = (x, -40)
            points[base + 1] = (x, -70)
            points[base + 2] = (x, -85 if extended[index] else -40)
            points[base + 3] = (x, -110 if extended[index] else -25)
        return points
    for flags, gesture in [([1, 0, 0, 0], Gesture.DRAW), ([1, 1, 0, 0], Gesture.MOVE),
                           ([1, 1, 1, 1], Gesture.IDLE), ([0, 0, 0, 0], Gesture.IDLE)]:
        points = hand(flags)
        assert classify_landmarks(points) == gesture
        assert classify_landmarks(points * [-1, 1]) == gesture
        assert classify_landmarks(points @ np.array([[0, -1], [1, 0]])) == gesture
    assert classify_landmarks(None) == Gesture.IDLE


def test_move_accepts_relaxed_lower_fingers_and_tilted_3d_pose():
    points = np.zeros((21, 3))
    for i, base in enumerate([5, 9, 13, 17]):
        x = (i - 1.5) * 15
        points[base] = (x, -40, 0)
        points[base + 1] = (x, -70, 0)
        points[base + 2] = (x, -85, 0)
        # Lower fingers are relaxed, not tightly folded toward the wrist.
        points[base + 3] = (x, -110 if i < 2 else -72, 0)
    assert classify_landmarks(points) == Gesture.MOVE
    rotation = np.array([[1, 0, 0], [0, 0.2, -np.sqrt(0.96)], [0, np.sqrt(0.96), 0.2]])
    assert classify_landmarks(points @ rotation) == Gesture.MOVE


def test_brief_draw_detection_during_move_never_connects_strokes():
    controller = GestureController()
    controller.update(Gesture.MOVE, 0)
    assert not controller.update(Gesture.DRAW, 0.02).drawing
    assert not controller.update(Gesture.DRAW, 0.15).drawing
    assert controller.update(Gesture.DRAW, 0.25).drawing
