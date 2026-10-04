from threading import Event
import numpy as np
from core.guesser import AsyncGuesser, Guess


class BlockingPredictor:
    def __init__(self):
        self.ready = Event()

    def predict(self, image):
        assert self.ready.wait(2)
        return [Guess("cat", float(image.sum()))]


def test_worker_uses_snapshot_and_does_not_queue_more_requests():
    predictor = BlockingPredictor()
    worker = AsyncGuesser(predictor)
    try:
        canvas = np.ones((2, 2), np.uint8)
        assert worker.submit(canvas, 1)
        canvas.fill(0)
        assert not worker.submit(canvas, 1)
        predictor.ready.set()
        worker.future.result(timeout=2)
        assert worker.poll(1) == ([Guess("cat", 4)], None)
    finally:
        predictor.ready.set()
        worker.close()


def test_result_from_a_cleared_canvas_is_discarded():
    predictor = BlockingPredictor()
    worker = AsyncGuesser(predictor)
    try:
        worker.submit(np.zeros((2, 2)), 1)
        predictor.ready.set()
        worker.future.result(timeout=2)
        assert worker.poll(2) is None
        assert not worker.busy
    finally:
        predictor.ready.set()
        worker.close()
