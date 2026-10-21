"""Round timing and scoring, kept separate from camera and model code."""

import random
from core.guesser import is_uncertain


class Game:
    def __init__(self, classes, seconds=30, practice=False):
        self.classes = list(classes)
        self.seconds = seconds
        self.practice = practice
        self.state = "ready"
        self.round_id = 0
        self.revision = 0
        self.started = 0.0
        self.target = ""
        self.score = 0
        self.rounds = 0
        self.guesses = []
        self.paused_at = None
        self.uncertain = False
        self.message = "Press N or click New round when you're ready."

    @property
    def request_id(self):
        return self.round_id, self.revision

    def start(self, now):
        choices = [word for word in self.classes if word != self.target] or self.classes
        self.target = "" if self.practice else random.choice(choices)
        self.round_id += 1
        self.revision = 0
        self.started = now
        self.paused_at = None
        self.state = "drawing"
        self.guesses = []
        self.uncertain = False
        self.message = "Draw something!" if self.practice else f"You have {self.seconds} seconds. Make it recognizable!"

    def remaining(self, now):
        effective_now = now if self.paused_at is None else self.paused_at
        return max(0.0, self.seconds - (effective_now - self.started))

    def pause(self, now):
        if self.paused_at is None:
            self.paused_at = now

    def resume(self, now):
        if self.paused_at is not None:
            self.started += now - self.paused_at
            self.paused_at = None

    def clear(self):
        if self.state != "drawing":
            return False
        self.revision += 1
        self.guesses = []
        self.message = "Fresh canvas." if self.practice else "Fresh canvas. The timer keeps running."
        return True

    def finish(self, guesses, error=None):
        if self.state != "submitting":
            return
        self.state, self.guesses = "result", guesses
        self.uncertain = is_uncertain(guesses)
        if error:
            self.message = error
            return
        if self.practice:
            self.message = "Not sure yet. These are my closest guesses." if guesses and self.uncertain else "Here are my guesses." if guesses else "No drawing submitted."
            return
        self.rounds += 1
        labels = [guess.label for guess in guesses]
        points = 100 if labels and labels[0] == self.target else 30 if self.target in labels else 0
        self.score += points
        self.message = f"Got it! +{points} points." if points else f"That was {self.target}. Try another round!"
        if guesses and self.uncertain:
            self.message = f"Not sure yet. Closest guesses below. +{points} points."
