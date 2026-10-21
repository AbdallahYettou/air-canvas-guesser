import numpy as np
from core.canvas import Canvas
from core.controls import CategoryBrowser, DwellPointer, PinchClick, PointerFilter, hit_test, map_hand_point
from core.game import Game
from core.ui import CONTROLS, LEFT, control_at, camera_draw_area, control_enabled


def test_undo_restores_strokes_and_clear_but_does_not_cross_rounds():
    canvas = Canvas()
    canvas.update((30, 100), True)
    first = canvas.snapshot()
    canvas.break_stroke()
    canvas.update((90, 100), True)
    assert canvas.undo()
    np.testing.assert_array_equal(canvas.image, first)
    canvas.clear()
    assert not canvas.image.any()
    assert canvas.undo()
    np.testing.assert_array_equal(canvas.image, first)
    canvas.new_round()
    assert not canvas.undo()
    assert not canvas.image.any()
    assert tuple(canvas.display()[100, 30]) == (255, 255, 255)


def test_black_ink_snapshot_is_independent_and_strokes_remain_separate():
    canvas = Canvas()
    canvas.update((30, 100), True)
    assert tuple(canvas.display()[100, 30]) == (0, 0, 0)
    snapshot = canvas.snapshot()
    canvas.break_stroke()
    canvas.update((100, 100), True)
    assert canvas.image[100, 60] == 0
    canvas.clear()
    assert snapshot.any()


def test_visible_controls_match_hit_regions_and_are_outside_drawing_area():
    assert len(CONTROLS) == 6
    for action, (x, y, width, height) in CONTROLS.items():
        center = x + width / 2, y + height / 2
        assert control_at(*center) == action
        assert not camera_draw_area((center[0] - LEFT[0], center[1] - LEFT[1]))
    assert camera_draw_area((320, 200))
    assert not camera_draw_area(None)


def test_hover_tolerates_short_dropout_but_does_not_click_when_tracking_lost():
    pointer = DwellPointer()
    for index in range(6):
        assert pointer.update('clear', index * 0.1) is None
    elapsed = pointer.elapsed
    assert pointer.update(None, 0.6) is None
    assert pointer.update('clear', 0.65) is None
    assert pointer.elapsed == elapsed
    fired = [pointer.update('clear', t / 10) for t in range(7, 20)]
    assert fired.count('clear') == 1
    assert pointer.update('clear', 2) is None
    pointer.update(None, 2.3)
    assert pointer.target is None
    assert pointer.progress == 0


def test_sticky_edges_do_not_switch_to_neighbor_until_entered():
    controls = {'first': (0, 0, 100, 60), 'second': (120, 0, 100, 60)}
    assert hit_test((108, 30), controls, 'first') == 'first'
    assert hit_test((132, 30), controls, 'first') == 'second'
    assert hit_test((108, 30), controls) is None


def test_pointer_smoothing_reduces_jitter_and_never_hides_tracking_loss():
    pointer = PointerFilter()
    pointer.update((100, 100), 0)
    smoothed = pointer.update((110, 100), 0.016)
    assert 100 < smoothed[0] < 105
    assert pointer.update(None, 0.03) is None
    assert pointer.update((500, 300), 1) == (500, 300)


def test_pinch_requires_open_then_close_over_a_stable_button():
    click = PinchClick()
    assert click.update('submit', 0.2, 0) is None
    assert click.update('submit', 0.7, 0.1) is None
    assert click.update('submit', 0.2, 0.3) == 'submit'
    assert click.update('submit', 0.2, 0.4) is None
    click.update(None, None, 0.5)
    assert click.update('clear', 0.2, 1) is None


def test_categories_page_only_through_installed_classes_and_buttons_disable():
    browser = CategoryBrowser([f'class-{i:03}' for i in range(100)])
    assert len(browser.visible) == 24
    browser.turn(99)
    assert len(browser.visible) == 4
    assert browser.page == 4
    game = Game(['clock'])
    assert not control_enabled('submit', game, True)
    game.start(0)
    assert control_enabled('submit', game, True)
    assert not control_enabled('submit', game, False)
    assert not control_enabled('undo', game, True, Canvas())
    game.state = 'submitting'
    assert not control_enabled('clear', game, True)


def test_all_camera_buttons_are_reachable_without_moving_palm_off_frame():
    for action, (x, y, width, height) in CONTROLS.items():
        target = np.array([x + width / 2 - LEFT[0], y + height / 2 - LEFT[1]])
        raw_tip = np.array([96, 60]) + target / [639, 479] * [448, 220]
        # Representative upright hand: wrist 180 pixels below the fingertip.
        assert raw_tip[1] + 180 < 480
        assert 96 <= raw_tip[0] <= 544
        assert raw_tip[1] < 250
        cursor = map_hand_point(raw_tip)
        assert control_at(LEFT[0] + cursor[0], LEFT[1] + cursor[1]) == action


def test_hand_mapping_handles_tracking_loss_and_clamps_edges():
    assert map_hand_point(None) is None
    assert map_hand_point((np.nan, 100)) is None
    assert map_hand_point((0, 0)) == (0, 0)
    assert map_hand_point((640, 480)) == (639, 479)
    np.testing.assert_allclose(map_hand_point((320, 170)), (319.5, 239.5))


def test_category_browser_exposes_all_100_unique_labels_across_five_pages():
    labels = [f'class-{i:03}' for i in range(100)]
    browser = CategoryBrowser(labels)
    found = []
    for page in range(browser.pages):
        assert browser.page == page
        found.extend(browser.visible)
        browser.turn(1)
    assert found == labels
    assert browser.pages == 5
