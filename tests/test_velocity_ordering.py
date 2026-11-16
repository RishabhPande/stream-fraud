"""Event-time ordering and boundary contracts for per-card velocity windows."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from stream_fraud.features import VelocityFeatures
from stream_fraud.stream import Transaction

BASE = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def txn(ts, card="card", amount=10.0, merchant="shop"):
    return Transaction("t", card, amount, merchant, "5411", ts)


@pytest.mark.parametrize("late_by", [timedelta(microseconds=1), timedelta(minutes=5),
                                    timedelta(hours=1), timedelta(days=2)])
def test_late_transaction_rejected_without_changing_card_state(late_by):
    features = VelocityFeatures()
    features.update(txn(BASE))
    before = deepcopy(features._cards)
    with pytest.raises(ValueError, match="nondecreasing.*per card"):
        features.update(txn(BASE - late_by, amount=999.0, merchant="late"))
    assert features._cards == before
    # A subsequent valid event must see neither the late amount nor its merchant.
    result = features.update(txn(BASE + timedelta(seconds=1)))
    assert (result.count_5m, result.count_1h, result.amount_1h,
            result.amount_24h, result.distinct_merchants_1h,
            result.time_since_last_txn) == (2, 2, 20.0, 20.0, 1, 1.0)


def test_late_event_cannot_reintroduce_expired_history():
    features = VelocityFeatures()
    features.update(txn(BASE, amount=100.0))
    features.update(txn(BASE + timedelta(days=2), amount=20.0))
    with pytest.raises(ValueError, match="nondecreasing.*per card"):
        features.update(txn(BASE + timedelta(hours=1), amount=500.0))
    result = features.update(txn(BASE + timedelta(days=2, seconds=1), amount=30.0))
    assert result.amount_24h == 50.0
    assert result.count_1h == 2
    assert result.time_since_last_txn == 1.0


def test_equal_timestamps_are_allowed_and_current_event_is_included():
    features = VelocityFeatures()
    features.update(txn(BASE))
    result = features.update(txn(BASE, amount=20.0, merchant="other"))
    assert result.count_5m == result.count_1h == 2
    assert result.amount_1h == result.amount_24h == 30.0
    assert result.distinct_merchants_1h == 2
    assert result.time_since_last_txn == 0.0


def test_different_cards_do_not_share_an_ordering_watermark():
    features = VelocityFeatures()
    features.update(txn(BASE + timedelta(days=2), card="first"))
    result = features.update(txn(BASE, card="second"))
    assert result.count_1h == 1
    assert result.amount_24h == 10.0
    assert result.time_since_last_txn == -1.0
    assert features.card_count() == 2


def test_aware_datetimes_compare_instants_not_clock_labels():
    features = VelocityFeatures()
    features.update(txn(BASE))
    eastern = timezone(timedelta(hours=-5))
    result = features.update(txn(BASE.astimezone(eastern)))
    assert result.time_since_last_txn == 0.0
    with pytest.raises(ValueError, match="nondecreasing.*per card"):
        features.update(txn((BASE - timedelta(seconds=1)).astimezone(eastern)))


@pytest.mark.parametrize("window,field", [(timedelta(minutes=5), "count_5m"),
                                         (timedelta(hours=1), "count_1h"),
                                         (timedelta(days=1), "amount_24h")])
def test_window_lower_bound_is_inclusive(window, field):
    features = VelocityFeatures()
    features.update(txn(BASE))
    at_boundary = features.update(txn(BASE + window))
    assert getattr(at_boundary, field) == (20.0 if field == "amount_24h" else 2)
    after_boundary = features.update(txn(BASE + window + timedelta(microseconds=1)))
    assert getattr(after_boundary, field) == (20.0 if field == "amount_24h" else 2)
