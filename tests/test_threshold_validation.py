"""Invalid threshold feedback must not corrupt or evict valid observations."""
import copy
import math

import pytest

from stream_fraud.threshold import CostAwareThreshold

BAD_NUMBERS = [math.nan, math.inf, -math.inf, True, False, "0.5", None, 10**1000]


@pytest.mark.parametrize("field", ["default_threshold", "tp_cost", "fp_cost"])
@pytest.mark.parametrize("value", BAD_NUMBERS)
def test_nonfinite_or_nonnumeric_options_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        CostAwareThreshold(**{field: value})


@pytest.mark.parametrize("field", ["tp_cost", "fp_cost"])
def test_negative_review_cost_rejected(field):
    with pytest.raises(ValueError, match=field):
        CostAwareThreshold(**{field: -0.01})


@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_invalid_buffer_capacity_rejected(value):
    with pytest.raises(ValueError, match="max_buffer"):
        CostAwareThreshold(max_buffer=value)


@pytest.mark.parametrize("field,value", [(field, value)
    for field, values in [("score", BAD_NUMBERS + [-0.01, 1.01]),
                          ("amount", BAD_NUMBERS + [-0.01]),
                          ("is_fraud", [-1, 2, 0.5, math.nan, math.inf, -math.inf, "false", "0", "1", None, 10**1000])]
    for value in values])
def test_rejected_feedback_preserves_full_buffer_and_future_tuning(field, value):
    tuner = CostAwareThreshold(max_buffer=2)
    tuner.observe(0.9, True, 100.0)
    tuner.observe(0.1, False, 10.0)
    control = copy.deepcopy(tuner)
    feedback = dict(score=0.5, is_fraud=True, amount=500.0)
    feedback[field] = value
    with pytest.raises(ValueError, match=field):
        tuner.observe(**feedback)
    assert tuner.__dict__ == control.__dict__
    assert tuner.tune() == control.tune()
    tuner.observe(0.8, True, 30.0)
    control.observe(0.8, True, 30.0)
    assert tuner.tune() == control.tune()


@pytest.mark.parametrize("method", ["total_cost", "confusion"])
@pytest.mark.parametrize("value", BAD_NUMBERS)
def test_bad_query_threshold_rejected_even_with_empty_buffer(method, value):
    with pytest.raises(ValueError, match="threshold"):
        getattr(CostAwareThreshold(), method)(value)


def test_valid_boundaries_zero_costs_and_flag_nothing_threshold():
    tuner = CostAwareThreshold(default_threshold=1.0001, tp_cost=0, fp_cost=0, max_buffer=2)
    tuner.observe(0, False, 0)
    tuner.observe(1, True, 100)
    assert tuner.total_cost(-1) == 0
    assert tuner.confusion(-1) == (1, 1, 0, 0)
    assert tuner.total_cost(1.0001) == 100
    assert tuner.confusion(1.0001) == (0, 0, 1, 1)
    assert tuner.tune().threshold == 1


@pytest.mark.parametrize("label", [False, True, 0, 1, 0.0, 1.0])
def test_binary_labels_preserve_decisions_and_are_normalized(label):
    tuner = CostAwareThreshold()
    control = CostAwareThreshold()
    tuner.observe(0.7, label, 100)
    control.observe(0.7, bool(label), 100)
    assert type(tuner._obs[0].is_fraud) is bool
    assert tuner.tune() == control.tune()
    for threshold in [0, 0.7, 1.0001]:
        assert tuner.confusion(threshold) == control.confusion(threshold)
        assert tuner.total_cost(threshold) == control.total_cost(threshold)
