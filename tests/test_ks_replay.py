import random

from stream_fraud.ks import KSMonitor, ks_statistic
from stream_fraud.replay import compare, replay


def test_ks_statistic_known_values():
    assert ks_statistic([1, 2, 3], [1, 2, 3]) == 0.0
    assert ks_statistic([0, 0, 0], [1, 1, 1]) == 1.0
    # ties: half the mass moves from 0 to 1
    assert abs(ks_statistic([0, 0, 1, 1], [0, 1, 1, 1]) - 0.25) < 1e-12


def test_ks_monitor_quiet_on_stationary_noise():
    rng = random.Random(0)
    m = KSMonitor(reference_window=400, current_window=400)
    for _ in range(3000):
        m.update(rng.random())
        assert not m.check().drifted


def test_ks_monitor_fires_on_a_clear_shift_and_resets():
    rng = random.Random(0)
    m = KSMonitor(reference_window=400, current_window=400)
    for _ in range(400):
        m.update(rng.random() * 0.3)
    fired = False
    for _ in range(800):
        m.update(0.5 + rng.random() * 0.5)
        fired = fired or m.check().drifted
    assert fired
    m.reset()
    assert not m.reference_ready and not m.check().drifted


def test_replay_is_deterministic_and_records_alarms():
    a, b = replay(n=1500, seed=3), replay(n=1500, seed=3)
    assert a.scores == b.scores and a.psi_alarm_at == b.psi_alarm_at
    assert len(a.scores) == len(a.psi) == len(a.ks) == 1500


def test_replay_without_drift_has_no_drift_point():
    r = replay(n=900, drift_at=None, seed=1)
    assert r.drift_at is None


def test_compare_reports_every_seed_including_misses():
    out = compare(seeds=range(1, 4), n=1800, drift_at=900)
    assert [row["seed"] for row in out["rows"]] == [1, 2, 3]
    for row in out["rows"]:
        for k in ("psi", "ks"):
            assert set(row[k]) == {"delay", "early", "never", "false_alarm_no_drift"}
