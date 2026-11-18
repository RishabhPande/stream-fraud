"""PSI inputs must fail before changing reference or current windows."""
from copy import deepcopy
import math

import pytest

from stream_fraud.drift import DriftMonitor


@pytest.mark.parametrize("field", ["n_bins", "reference_window", "current_window"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, None])
def test_invalid_histogram_and_window_sizes_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        DriftMonitor(**{field: value})


@pytest.mark.parametrize("value", [-1, math.nan, math.inf, -math.inf, True, False, "0.25", None, 10**1000])
def test_invalid_psi_threshold_rejected(value):
    with pytest.raises(ValueError, match="psi_threshold"):
        DriftMonitor(psi_threshold=value)


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf, -math.inf,
                                  True, False, "0.5", None, 10**1000])
@pytest.mark.parametrize("phase", ["empty", "reference", "current", "full"])
def test_bad_score_cannot_poison_reference_or_evict_current(value, phase):
    monitor = DriftMonitor(reference_window=2, current_window=2)
    samples = {"empty": [], "reference": [0.1], "current": [0.1, 0.2, 0.3],
               "full": [0.1, 0.2, 0.3, 0.4]}[phase]
    for score in samples:
        monitor.update(score)
    control = deepcopy(monitor)
    with pytest.raises(ValueError, match="score"):
        monitor.update(value)
    assert monitor.__dict__ == control.__dict__
    for score in [0, 1, 0.1, 0.9] * 3:
        monitor.update(score)
        control.update(score)
        assert monitor.check() == control.check()


def test_single_bin_windows_and_zero_threshold_are_supported():
    monitor = DriftMonitor(n_bins=1, reference_window=1, current_window=1, psi_threshold=0)
    monitor.update(0)
    assert not monitor.check().drifted
    monitor.update(1)
    assert monitor.check().psi == 0
    assert monitor.check().drifted
    monitor.reset()
    assert not monitor.reference_ready
    assert not monitor.check().drifted


def test_probability_endpoints_land_in_first_and_last_bins():
    monitor = DriftMonitor(n_bins=2, reference_window=2, current_window=2)
    assert monitor._bin([0, 1]) == [0.5, 0.5]
    for score in [0, 1, 0, 1]:
        monitor.update(score)
    assert monitor.check().psi == 0
    assert not monitor.check().drifted
