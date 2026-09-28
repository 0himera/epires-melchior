"""Tests for Jev-powered decision points."""

from melchior.brain.jev_client import JevClient
from melchior.brain.decisions import (
    decide_strategy,
    assess_technique,
    rank_pilot,
    check_early_stop,
    classify_error,
    assess_budget,
)


def test_decide_strategy_initial():
    jev = JevClient(mode="mock")
    res = decide_strategy(jev, "Initial state: 0 trials executed.")
    assert res.strategy in ["explore", "exploit", "ablate", "ensemble"]
    assert res.confidence > 0.5


def test_assess_technique():
    jev = JevClient(mode="mock")
    res_boost = assess_technique(jev, "gradient boosting", "tabular", "none")
    assert res_boost.viable is True
    assert res_boost.probability > 0.5

    res_dropout = assess_technique(jev, "dropout", "tabular", "none")
    assert res_dropout.viable is False
    assert res_dropout.probability < 0.5


def test_rank_pilot():
    jev = JevClient(mode="mock")
    score_good = rank_pilot(jev, "Candidate: RF, metric: 0.95, accuracy: 0.95")
    score_bad = rank_pilot(jev, "Candidate: bad, error in traceback")
    assert score_good > score_bad


def test_check_early_stop():
    jev = JevClient(mode="mock")
    # Improving: should not stop
    dec_improving = check_early_stop(jev, [0.70, 0.80, 0.88, 0.92], 0.92)
    assert dec_improving.should_stop is False

    # Plateau / declining: should stop
    dec_plateau = check_early_stop(jev, [0.85, 0.85, 0.84, 0.84], 0.85)
    assert dec_plateau.should_stop is True


def test_classify_error():
    jev = JevClient(mode="mock")
    oom_res = classify_error(jev, "torch.cuda.OutOfMemoryError: CUDA out of memory", "")
    assert oom_res.error_type == "oom"
    assert oom_res.auto_fixable is True

    shape_res = classify_error(jev, "ValueError: Dimension mismatch size [10, 4] vs [10, 5]", "")
    assert shape_res.error_type == "shape"
    assert shape_res.auto_fixable is True


def test_assess_budget():
    jev = JevClient(mode="mock")
    res = assess_budget(jev, "Trials: 3, improvement ongoing")
    assert res.should_continue is True
