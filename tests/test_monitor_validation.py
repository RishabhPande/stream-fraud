"""Invalid experimental-monitor inputs fail before changing streaming state."""
import copy
import math

import pytest

from stream_fraud.ks import KSMonitor, ks_statistic
from stream_fraud.page_hinkley import PageHinkleyMonitor

BAD_NUMBERS = [math.nan, math.inf, -math.inf, True, False, "0.5", None, 10**1000]


@pytest.mark.parametrize("field", ["reference_window", "current_window", "check_every"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_ks_rejects_invalid_window_options(field, value):
    with pytest.raises(ValueError, match=field):
        KSMonitor(**{field: value})


@pytest.mark.parametrize("value", BAD_NUMBERS + [0, 1, -0.1, 1.1])
def test_ks_rejects_invalid_alpha(value):
    with pytest.raises(ValueError, match="alpha"):
        KSMonitor(alpha=value)


@pytest.mark.parametrize("field", ["delta", "threshold"])
@pytest.mark.parametrize("value", BAD_NUMBERS + [0, -1])
def test_ph_rejects_invalid_numeric_options(field, value):
    with pytest.raises(ValueError, match=field):
        PageHinkleyMonitor(**{field: value})


@pytest.mark.parametrize("value", [-1, 1.5, True, None])
def test_ph_rejects_invalid_warmup(value):
    with pytest.raises(ValueError, match="warmup"):
        PageHinkleyMonitor(warmup=value)


@pytest.mark.parametrize("value", BAD_NUMBERS)
@pytest.mark.parametrize("kind", ["ks_reference", "ks_current", "ph"])
def test_bad_sample_leaves_state_and_future_results_unchanged(value, kind):
    m = PageHinkleyMonitor(warmup=0) if kind == "ph" else KSMonitor(reference_window=2, current_window=2)
    m.update(0.1)
    if kind == "ks_current":
        m.update(0.2)
    control = copy.deepcopy(m)
    with pytest.raises(ValueError, match="score"):
        m.update(value)
    assert m.__dict__ == control.__dict__
    for score in [0.2, 0.3, 0.9, 0.8] * 10:
        m.update(score)
        control.update(score)
        assert m.check() == control.check()


@pytest.mark.parametrize("a,b", [([], []), ([], [1]), ([1], [])])
def test_ks_statistic_rejects_empty_samples(a, b):
    with pytest.raises(ValueError, match="non-empty"):
        ks_statistic(a, b)


@pytest.mark.parametrize("value", BAD_NUMBERS)
@pytest.mark.parametrize("side", [0, 1])
def test_ks_statistic_rejects_bad_values_in_either_sample(value, side):
    samples = [[0.1, 0.2], [0.2, 0.3]]
    samples[side].append(value)
    with pytest.raises(ValueError, match="sample value"):
        ks_statistic(*samples)


def test_valid_boundary_options_and_unbounded_finite_statistics():
    assert not KSMonitor(reference_window=1, current_window=1, check_every=1).check().drifted
    assert not PageHinkleyMonitor(warmup=0).check().drifted
    assert ks_statistic([-3, -2], [4, 5]) == 1.0


def test_smallest_positive_alpha_does_not_underflow_log():
    assert math.isfinite(KSMonitor(alpha=math.ulp(0.0)).check().threshold)
