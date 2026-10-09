from datetime import datetime, timedelta, timezone

from stream_fraud.drift import DriftMonitor
from stream_fraud.features import VelocityFeatures
from stream_fraud.scorer import RiskScorer
from stream_fraud.stream import Transaction, TransactionStream


def _txn(card, ts, amount=20.0, merchant="Target", mcc="5411", fraud=False):
    return Transaction(
        tx_id="t", card_id=card, amount=amount,
        merchant=merchant, mcc=mcc, ts=ts, is_fraud=fraud,
    )


BASE = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_velocity_counts_across_windows():
    vf = VelocityFeatures()
    # 3 txns within 5m, 1 more 30m ago, 1 more 12h ago — same card
    times = [BASE - timedelta(hours=12), BASE - timedelta(minutes=30),
             BASE - timedelta(minutes=4), BASE - timedelta(minutes=2), BASE]
    for t in times:
        f = vf.update(_txn("c1", t))
    assert f.count_5m == 3
    assert f.count_1h == 4
    assert f.distinct_merchants_1h == 1
    assert f.time_since_last_txn == 120.0


def test_amount_windows():
    vf = VelocityFeatures()
    vf.update(_txn("c2", BASE - timedelta(hours=2), amount=100.0))
    f = vf.update(_txn("c2", BASE, amount=50.0))
    assert f.amount_1h == 50.0
    assert f.amount_24h == 150.0


def test_fraud_burst_scores_high():
    stream = TransactionStream(n=3000, fraud_rate=0.05, seed=7)
    vf, scorer = VelocityFeatures(), RiskScorer()
    burst_scores, legit_scores = [], []
    for txn in stream:
        feats = vf.update(txn)
        res = scorer.score(txn, feats)
        if txn.is_fraud:
            burst_scores.append(res.score)
        else:
            legit_scores.append(res.score)
    assert burst_scores, "expected fraud txns in stream"
    # later txns in each burst should fire hard
    assert max(burst_scores) >= 0.9
    avg_legit = sum(legit_scores) / len(legit_scores)
    assert avg_legit < 0.2, f"legit avg score too high: {avg_legit}"


def test_benign_txn_low_score():
    vf, scorer = VelocityFeatures(), RiskScorer()
    f = vf.update(_txn("c3", BASE, amount=42.50))
    res = scorer.score(_txn("c3", BASE), f)
    assert res.score < 0.3
    assert not res.flagged


def test_stream_labels_sane():
    stream = TransactionStream(n=5000, fraud_rate=0.02, seed=1)
    frauds = sum(1 for t in stream if t.is_fraud)
    rate = frauds / 5000
    # bursts emit 5-8 txns per fraud event, so effective fraud txn rate > event rate
    assert 0.05 < rate < 0.20, f"fraud txn rate {rate:.3f} looks off"


def _run_pipeline(n, seed, inject_at=None, fraud_rate=0.02):
    stream = TransactionStream(n=n, fraud_rate=fraud_rate, seed=seed)
    vf, scorer = VelocityFeatures(), RiskScorer()
    monitor = DriftMonitor(reference_window=400, current_window=400)
    emitted = 0
    fired = None
    for txn in stream:
        if inject_at is not None and emitted == inject_at:
            stream.inject_drift()
        feats = vf.update(txn)
        score = scorer.score(txn, feats).score
        monitor.update(score)
        emitted += 1
        if monitor.check().drifted and fired is None:
            fired = emitted
    return fired, monitor.check().psi


def test_drift_monitor_flags_injected_drift():
    fired_at, psi = _run_pipeline(n=1600, seed=3, inject_at=800)
    assert fired_at is not None, "drift monitor never fired after inject_drift()"
    assert psi >= 0.25, f"PSI {psi} below threshold after drift"


def test_drift_monitor_stays_quiet_without_drift():
    fired_at, psi = _run_pipeline(n=1600, seed=3, inject_at=None)
    assert fired_at is None, f"drift monitor fired at {fired_at} with no drift"
    assert psi < 0.25, f"PSI {psi} unexpectedly high without drift"


def test_drift_result_before_reference_ready():
    m = DriftMonitor(reference_window=400, current_window=400)
    m.update(0.1)
    r = m.check()
    assert not r.drifted and r.psi == 0.0
