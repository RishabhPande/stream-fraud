import random

import pytest

from stream_fraud.page_hinkley import PageHinkleyMonitor
from stream_fraud.replay import compare, replay


def test_hand_computed_margin():
    # scores: five 0.0s then five 1.0s. Running mean and U are computable by hand:
    # after the five 0.0s, mean = 0 and U = -5 * delta.
    # each 1.0 then adds (1 - mean - delta); check the margin crosses only after enough mass.
    m = PageHinkleyMonitor(delta=0.1, threshold=1.0, warmup=1)
    for _ in range(5):
        m.update(0.0)
        assert not m.check().drifted
    fired_at = None
    for i in range(1, 6):
        m.update(1.0)
        if m.check().drifted and fired_at is None:
            fired_at = i
    # with delta 0.1 the step after the shift is ~0.9, 0.8, ... so the margin
    # needs at least two 1.0s to pass 1.0; require it fires within the five.
    assert fired_at is not None


def test_rejects_nonpositive_parameters():
    with pytest.raises(ValueError):
        PageHinkleyMonitor(delta=0)
    with pytest.raises(ValueError):
        PageHinkleyMonitor(threshold=0)
    with pytest.raises(ValueError):
        PageHinkleyMonitor(warmup=-1)


def test_warmup_blocks_early_alarms():
    m = PageHinkleyMonitor(delta=0.01, threshold=0.05, warmup=100)
    for _ in range(99):
        m.update(1.0)
        assert not m.check().drifted


def test_quiet_on_stationary_noise():
    rng = random.Random(0)
    m = PageHinkleyMonitor(delta=0.02, threshold=20.0)
    for _ in range(5000):
        m.update(rng.random() * 0.3)
        assert not m.check().drifted


def test_fires_on_a_clear_shift_and_recalibrates():
    rng = random.Random(0)
    m = PageHinkleyMonitor(delta=0.02, threshold=2.0, warmup=50)
    for _ in range(400):
        m.update(rng.random() * 0.2)
    fired = False
    for _ in range(800):
        m.update(0.6 + rng.random() * 0.4)
        fired = fired or m.check().drifted
    assert fired
    # the alarm resets the accumulator, so the monitor re-arms instead of latching
    assert not m.check().drifted
    m.reset()
    assert m.check().statistic == 0.0 and not m.check().drifted


def test_replay_records_ph_series_and_alarm():
    r = replay(n=1500, drift_at=900, seed=3)
    assert len(r.ph) == len(r.scores) == 1500
    assert r.ph_alarm_at is not None and r.ph_alarm_at >= 900


def test_replay_ph_is_deterministic():
    a, b = replay(n=1500, seed=3), replay(n=1500, seed=3)
    assert a.ph == b.ph and a.ph_alarm_at == b.ph_alarm_at


def test_compare_reports_ph_for_every_seed():
    out = compare(seeds=range(1, 4), n=1800, drift_at=900)
    for row in out["rows"]:
        assert set(row["ph"]) == {"delay", "early", "never", "false_alarm_no_drift"}


def test_compare_default_settings_detect_every_drift_without_false_alarms():
    # the documented defaults: all 20 seeds detected, no early alarms, no
    # alarms on the matching no-drift replays (README quotes these numbers)
    out = compare()
    rows = out["rows"]
    assert len(rows) == 20
    assert all(row["ph"]["delay"] is not None for row in rows)
    assert not any(row["ph"]["early"] for row in rows)
    assert not any(row["ph"]["never"] for row in rows)
    assert not any(row["ph"]["false_alarm_no_drift"] for row in rows)
