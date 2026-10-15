"""Tests for CostAwareThreshold."""

from stream_fraud.features import VelocityFeatures
from stream_fraud.scorer import RiskScorer
from stream_fraud.stream import TransactionStream
from stream_fraud.threshold import CostAwareThreshold


def test_total_cost_math():
    t = CostAwareThreshold(tp_cost=2.0, fp_cost=8.0)
    t.observe(0.9, True, 100.0)    # fraud, flagged at 0.6 -> analyst cost
    t.observe(0.3, False, 50.0)   # legit, not flagged
    assert t.total_cost(0.6) == 2.0
    t.observe(0.95, False, 100.0)  # legit, flagged -> review + friction
    assert t.total_cost(0.6) == 10.0
    # miss-all threshold: the fraud is missed -> lose the money
    assert t.total_cost(1.0001) == 100.0


def test_tune_picks_cheapest():
    t = CostAwareThreshold()
    t.observe(0.9, True, 100.0)
    t.observe(0.3, False, 50.0)
    res = t.tune()
    assert res.tuned
    assert res.threshold == 0.9
    assert res.expected_cost == 2.0
    assert res.n == 2


def test_high_value_fraud_pushes_threshold_below_f1_optimal():
    """$5k chargebacks are worth catching even at heavy review cost,
    while F1-optimal prefers the precision of a higher threshold."""
    t = CostAwareThreshold(tp_cost=2.0, fp_cost=8.0)
    t.observe(0.7, True, 5000.0)
    t.observe(0.4, True, 5000.0)
    for _ in range(8):
        t.observe(0.5, False, 50.0)
    res = t.tune()
    # F1 is maximized at 0.7 (tp=1, fp=0, fn=1 -> F1=0.667),
    # but cost-optimal is 0.4: $68 of reviews beats a $5,000 miss
    assert res.threshold == 0.4
    assert res.expected_cost == 68.0
    # F1-optimal threshold 0.7 costs $5,002 here (2 + 5,000)
    assert t.total_cost(0.7) == 5002.0


def test_tune_empty_buffer_returns_untuned():
    res = CostAwareThreshold().tune()
    assert not res.tuned
    assert res.threshold is None
    assert res.n == 0


def test_confusion_counts():
    t = CostAwareThreshold()
    t.observe(0.9, True, 100.0)
    t.observe(0.7, False, 50.0)
    t.observe(0.2, True, 80.0)
    t.observe(0.1, False, 10.0)
    tp, fp, fn, tn = t.confusion(0.6)
    assert (tp, fp, fn, tn) == (1, 1, 1, 1)
    tp, fp, fn, tn = t.confusion(0.15)
    assert (tp, fp, fn, tn) == (2, 1, 0, 1)


def test_buffer_evicts_oldest():
    t = CostAwareThreshold(max_buffer=3)
    for i in range(5):
        t.observe(0.1 * i, i == 3, 100.0)  # i=3 is a $100 fraud
    assert len(t) == 3
    # retained: 0.2 (legit), 0.3 (fraud), 0.4 (legit)
    # threshold 0.3 flags 2 -> $10 reviews, misses nothing -> cheapest
    res = t.tune()
    assert res.n == 3
    assert res.threshold == 0.3
    assert res.expected_cost == 10.0


def test_tune_beats_or_ties_baseline_on_real_stream():
    stream = TransactionStream(n=1500, fraud_rate=0.05, seed=9)
    vf, scorer = VelocityFeatures(), RiskScorer()
    tuner = CostAwareThreshold(default_threshold=scorer.threshold)
    for txn in stream:
        res = scorer.score(txn, vf.update(txn))
        tuner.observe(res.score, txn.is_fraud, txn.amount)
    result = tuner.tune()
    assert result.tuned and result.n == 1500
    assert result.expected_cost <= result.baseline_cost
