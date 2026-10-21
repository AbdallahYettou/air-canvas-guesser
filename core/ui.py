"""A simple monochrome board with large, visible hand controls."""

import math
import cv2
import numpy as np
from core.controls import inside, hit_test, DwellPointer
from core.detector import draw_landmarks

WIDTH, HEIGHT = 1480, 910
LEFT, RIGHT = (20, 150, 800, 600), (840, 150, 620, 465)
BUTTONS = {
    'new': (30, 160, 250, 60), 'undo': (295, 160, 250, 60), 'categories': (560, 160, 250, 60),
    'clear': (30, 232, 250, 60), 'submit': (295, 232, 250, 60), 'save': (560, 232, 250, 60),
}
BROWSER_BUTTONS = {
    'previous': BUTTONS['clear'], 'next': BUTTONS['submit'], 'close': BUTTONS['save'],
}
MODE_BUTTON = (20, 846, 220, 42)
CONTROLS = BUTTONS
BG, INK, MUTED, ACCENT = (27, 22, 20), (240, 242, 245), (170, 161, 151), (160, 224, 100)


def text(image, value, xy, scale=0.6, color=INK, thickness=1):
    cv2.putText(image, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def controls_for(browser=None):
    if browser is not None and browser.open:
        return {'categories': BUTTONS['categories'], **BROWSER_BUTTONS}
    return BUTTONS


def control_at(x, y, browser=None, sticky=None):
    if inside(x, y, MODE_BUTTON):
        return 'mode'
    return hit_test((x, y), controls_for(browser), sticky)


def control_enabled(action, game, model_ready, canvas=None, browser=None):
    if action == 'submit':
        return model_ready and game.state == 'drawing'
    if action == 'clear':
        return game.state != 'submitting'
    if action == 'undo':
        return game.state == 'drawing' and (canvas is None or bool(canvas.history))
    if action == 'previous':
        return browser is not None and browser.page > 0
    if action == 'next':
        return browser is not None and browser.page + 1 < browser.pages
    return action is not None


def board_point(x, y):
    if not inside(x, y, RIGHT):
        return None
    left, top, width, height = RIGHT
    return (x - left) * 640 / width, (y - top) * 480 / height


def camera_to_screen(point):
    """Apply only the camera panel's display scale, keeping the pointer on the finger."""
    return LEFT[0] + point[0] * LEFT[2] / 640, LEFT[1] + point[1] * LEFT[3] / 480


def screen_to_camera(point):
    return (point[0] - LEFT[0]) * 640 / LEFT[2], (point[1] - LEFT[1]) * 480 / LEFT[3]


def camera_draw_area(point):
    return point is not None and 0 <= point[0] < 640 and 136 <= point[1] < 452


def render(frame, canvas, game, now, gesture='idle', points=None, cursor=None, fps=0,
           mouse=False, model_ready=False, hover=None, progress=0, browser=None,
           click_mode='hover', drawing=False):
    screen = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
    text(screen, 'AIR CANVAS', (20, 42), 1.0, ACCENT, 2)
    text(screen, 'Draw. Move without ink. Click Submit.', (20, 75), 0.59, MUTED)
    target = 'Free drawing' if game.practice else f'Draw: {game.target}' if game.target else 'Ready to draw?'
    text(screen, target, (840, 48), 0.9, INK, 2)
    browsing = browser is not None and browser.open
    remaining = 'Paused' if browsing else 'No timer' if game.practice else f'{math.ceil(game.remaining(now))}s left' if game.state == 'drawing' else game.state.title()
    text(screen, f'{remaining}   |   Score {game.score}   |   Rounds {game.rounds}', (840, 82), 0.6, MUTED)
    text(screen, 'MOUSE CONTROLS' if mouse else f'CAMERA / {fps:.0f} fps / {click_mode.upper()} TO CLICK', (20, 132), 0.51, MUTED)
    text(screen, f'{len(game.classes)} CATEGORIES  /  {"BROWSING" if browsing else "YOUR SKETCH"}', (840, 132), 0.55, MUTED)
    if frame is None:
        camera = np.full((480, 640, 3), (41, 34, 30), np.uint8)
        text(camera, 'Draw on the white board', (65, 220), 0.8)
        text(camera, 'Undo keeps a small mistake from ruining a round.', (25, 265), 0.52, MUTED)
    else:
        camera = frame.copy()
        draw_landmarks(camera, points)
        alpha = canvas.image[:, :, None].astype(np.float32) / 255
        camera = np.clip(camera * (1 - alpha) + np.array(ACCENT) * alpha, 0, 255).astype(np.uint8)
    screen[150:750, 20:820] = cv2.resize(camera, (800, 600))
    screen[150:615, 840:1460] = cv2.resize(canvas.display(), (620, 465))
    cv2.rectangle(screen, (20, 150), (819, 226), (39, 33, 29), -1)
    cv2.rectangle(screen, (20, 226), (819, 306), (39, 33, 29), -1)
    if not mouse:
        cv2.rectangle(screen, (20, 720), (819, 749), (39, 33, 29), -1)
        text(screen, 'Cursor follows your fingertip. Controls are above.', (35, 741), 0.47, MUTED)
    labels = {'new': 'New round', 'undo': 'Undo stroke', 'categories': 'Categories',
              'clear': 'Clear', 'submit': 'Submit', 'save': 'Save image',
              'previous': '< Previous', 'next': 'Next >', 'close': 'Back to drawing'}
    controls = controls_for(browser)
    for action, (x, y, width, height) in controls.items():
        enabled = control_enabled(action, game, model_ready, canvas, browser)
        fill = (65, 116, 48) if action == 'submit' and enabled else (64, 53, 44) if enabled else (44, 42, 39)
        cv2.rectangle(screen, (x, y), (x + width, y + height), fill, -1)
        label_width = cv2.getTextSize(labels[action], cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
        text(screen, labels[action], (x + (width - label_width) // 2, y + 38), 0.6, INK if enabled else MUTED)
    if browsing:
        screen[150:615, 840:1460] = (42, 36, 31)
        text(screen, 'What I can recognize', (860, 188), 0.7, ACCENT)
        for index, label in enumerate(browser.visible):
            col, row = index // 8, index % 8
            text(screen, label, (860 + col * 196, 230 + row * 40), 0.46)
        first = browser.page * browser.page_size + 1
        last = min((browser.page + 1) * browser.page_size, len(browser.classes))
        text(screen, f'{first}-{last} of {len(browser.classes)}  |  Page {browser.page + 1}/{browser.pages}  |  paused', (860, 595), 0.48, MUTED)
    if hover in controls:
        x, y, width, height = controls[hover]
        cv2.rectangle(screen, (x, y), (x + width, y + height), ACCENT, 3)
        if progress:
            cv2.rectangle(screen, (x, y + height - 5), (x + int(width * progress), y + height), ACCENT, -1)
    if cursor is not None and not mouse:
        center = tuple(int(round(value)) for value in camera_to_screen(cursor))
        if inside(*center, LEFT):
            cv2.circle(screen, center, 13, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.circle(screen, center, 13, ACCENT, 2, cv2.LINE_AA)
            if progress:
                cv2.ellipse(screen, center, (18, 18), -90, 0, int(progress * 360), ACCENT, 3, cv2.LINE_AA)
    status = 'Drawing' if drawing else 'Move - pen up' if gesture == 'move' else 'Mouse ready' if mouse else 'Pen up / point at a control'
    text(screen, status, (20, 778), 0.65, ACCENT)
    message = game.message if model_ready else 'Practice mode: install a trained model to enable guessing.'
    lines, line = [], ''
    for word in message.split():
        candidate = f'{line} {word}'.strip()
        if cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] > 790:
            lines.append(line)
            line = word
        else:
            line = candidate
    lines.append(line)
    for index, line in enumerate(lines[:2]):
        text(screen, line, (20, 811 + index * 23), 0.5)
    if game.guesses:
        title = 'Not sure - closest guesses:' if game.uncertain else 'My guesses:'
        text(screen, title, (860, 657), 0.5, MUTED)
        for index, guess in enumerate(game.guesses):
            text(screen, f'{index + 1}. {guess.label}  {guess.score:.1%}', (860, 692 + index * 34), 0.61,
                 ACCENT if index == 0 and not game.uncertain else MUTED)
    else:
        text(screen, 'Thinking...' if game.state == 'submitting' else 'Click Submit to see my top 3 guesses.', (860, 699), 0.55, MUTED)
    x, y, width, height = MODE_BUTTON
    cv2.rectangle(screen, (x, y), (x + width, y + height), (64, 53, 44), -1)
    text(screen, f'Click: {click_mode.title()} [P]', (x + 12, y + 27), 0.54)
    text(screen, '1 finger: draw  |  2 fingers: move without ink  |  U: undo', (260, 873), 0.51, MUTED)
    text(screen, 'N: new round   C: clear   Enter: submit   B: categories   S: save   F: full screen   Q: quit', (20, 901), 0.46, MUTED)
    return screen
