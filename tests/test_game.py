from core.game import Game
from core.guesser import Guess


def test_timer_and_clear_do_not_restart_the_round():
    game = Game(["cat"])
    game.start(100)
    request = game.request_id
    assert game.remaining(120) == 10
    assert game.clear()
    assert game.request_id != request
    assert game.remaining(131) == 0


def test_score_cannot_be_awarded_twice():
    game = Game(["cat"])
    game.start(0)
    game.state = "submitting"
    game.finish([Guess("cat", 0.8)])
    game.finish([Guess("cat", 0.8)])
    assert game.score == 100
    assert game.rounds == 1


def test_top_three_gets_partial_credit_and_practice_has_no_points():
    game = Game(["cat"])
    game.start(0)
    game.state = "submitting"
    game.finish([Guess("dog", 0.6), Guess("cat", 0.3)])
    assert game.score == 30
    practice = Game(["cat"], practice=True)
    practice.start(0)
    practice.state = "submitting"
    practice.finish([Guess("cat", 1)])
    assert practice.score == 0


def test_new_round_invalidates_an_old_request_and_error_does_not_score():
    game = Game(["cat", "fish"])
    game.start(0)
    request, target = game.request_id, game.target
    game.start(5)
    assert game.request_id != request
    assert game.target != target
    game.state = "submitting"
    game.finish([], "Model failed")
    assert game.score == 0
    assert game.message == "Model failed"
