"""Stable pointing and deliberate clicks; drawing uses its own raw fingertip."""

import math

def inside(x, y, rect, padding=0):
    left, top, width, height = rect
    return left - padding <= x < left + width + padding and top - padding <= y < top + height + padding


def hit_test(point, controls, sticky=None):
    if point is None:
        return None
    direct = next((name for name, rect in controls.items() if inside(*point, rect)), None)
    if sticky in controls and inside(*point, controls[sticky], padding=14):
        if direct is None or direct == sticky or not inside(*point, controls[direct], padding=-8):
            return sticky
    return direct


class DwellPointer:
    """Pause for brief dropouts; require leaving a control before it can fire again."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.target = None
        self.last_time = None
        self.last_seen = None
        self.elapsed = 0.0
        self.fired = False
        self.progress = 0.0
        self.present = False

    def update(self, target, now):
        dt = 0 if self.last_time is None else max(0, now - self.last_time)
        self.last_time = now
        if target is None:
            self.present = False
            if self.last_seen is None or now - self.last_seen > 0.18:
                self.reset()
            return None
        if target != self.target or self.last_seen is None or now - self.last_seen > 0.25:
            self.target, self.elapsed, self.fired = target, 0.0, False
        elif self.present:
            self.elapsed += dt
        self.last_seen, self.present = now, True
        duration = 1.0 if target in ['clear', 'new'] else 0.7
        self.progress = min(1.0, self.elapsed / duration)
        if self.progress >= 1 and not self.fired:
            self.fired = True
            return target
        return None


class PinchClick:
    """Arm with an open pinch, click once when closing over a stable target."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.armed = False
        self.closed = False
        self.target = None
        self.since = 0.0

    def update(self, target, ratio, now):
        if ratio is None:
            self.reset()
            return None
        if target != self.target:
            self.target, self.since = target, now
        if ratio > 0.5:
            self.armed, self.closed = True, False
            return None
        if ratio < 0.3:
            edge = not self.closed
            self.closed = True
            if edge:
                armed, self.armed = self.armed, False
                if armed and target is not None and now - self.since >= 0.12:
                    return target
        return None


class CategoryBrowser:
    page_size = 24

    def __init__(self, classes):
        self.classes = sorted(classes)
        self.open = False
        self.page = 0

    @property
    def pages(self):
        return max(1, math.ceil(len(self.classes) / self.page_size))

    @property
    def visible(self):
        start = self.page * self.page_size
        return self.classes[start:start + self.page_size]

    def turn(self, delta):
        self.page = max(0, min(self.pages - 1, self.page + delta))
