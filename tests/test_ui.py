import numpy as np
from core.canvas import Canvas
from core.colors import PALETTE, BOARD_COLOR
from core.game import Game
from core.preprocess import prepare_sketch
from core.ui import CONTROLS, LEFT, DwellPointer, control_at, camera_draw_area, control_enabled


def test_palette_drawings_preserve_color_and_identical_model_input():
    expected = None
    for name, color in PALETTE.items():
        canvas = Canvas()
        canvas.select_color(name)
        for x in range(40, 100, 5):
            canvas.update((x, 100), True)
        assert tuple(canvas.display()[100, 50]) == color
        prepared = prepare_sketch(canvas.snapshot())
        if expected is None:
            expected = prepared
        np.testing.assert_array_equal(prepared, expected)
        canvas.clear()
        assert not canvas.image.any()
        assert not canvas.snapshot().any()
        assert tuple(canvas.display()[100, 50]) == BOARD_COLOR


def test_switching_colors_breaks_the_stroke_and_snapshots_are_independent():
    canvas = Canvas()
    canvas.update((30, 100), True)
    snapshot = canvas.snapshot()
    canvas.select_color("White")
    canvas.update((100, 100), True)
    assert canvas.image[100, 60] == 0
    canvas.clear()
    assert snapshot[:, :, 3].any()


def test_visible_controls_match_hit_regions_and_are_outside_drawing_area():
    assert len(CONTROLS) == 13
    for action, (x, y, width, height) in CONTROLS.items():
        center = (x + width / 2, y + height / 2)
        assert control_at(*center) == action
        assert not camera_draw_area((center[0] - LEFT[0], center[1] - LEFT[1]))
    assert camera_draw_area((320, 200))
    assert not camera_draw_area(None)


def test_hover_click_fires_once_then_requires_leaving_button():
    pointer = DwellPointer()
    assert pointer.update("submit", 0) is None
    assert pointer.update("submit", 0.5) is None
    assert pointer.update("submit", 0.8) == "submit"
    assert pointer.update("submit", 2) is None
    pointer.update(None, 3)
    pointer.update("clear", 4)
    assert pointer.update("clear", 4.8) is None
    assert pointer.update("clear", 5.1) == "clear"


def test_tracking_loss_cancels_hover_and_disabled_buttons_cannot_act():
    pointer = DwellPointer()
    pointer.update("clear", 0)
    pointer.update(None, 0.9)
    assert pointer.progress == 0
    assert pointer.update("clear", 2) is None
    assert pointer.update("clear", 2.2) is None
    game = Game(["clock"])
    assert not control_enabled("submit", game, True)
    game.start(0)
    assert control_enabled("submit", game, True)
    assert not control_enabled("submit", game, False)
    game.state = "submitting"
    assert not control_enabled("clear", game, True)
