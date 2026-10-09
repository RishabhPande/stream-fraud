"""Sliding-window velocity features, pure stdlib.

For every transaction we compute, per card:
- count_5m: txns in the last 5 minutes
- count_1h: txns in the last 1 hour
- amount_1h: total spend in the last 1 hour
- amount_24h: total spend in the last 24 hours
- distinct_merchants_1h: distinct merchants in the last hour
- time_since_last_txn: seconds since the card's previous txn

State is kept in bounded deques per card and pruned on each update so
memory stays flat no matter how long the stream runs.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import timedelta

from stream_fraud.stream import Transaction

W5M = timedelta(minutes=5)
W1H = timedelta(hours=1)
W24H = timedelta(hours=24)


@dataclass
class FeatureVector:
    count_5m: int
    count_1h: int
    amount_1h: float
    amount_24h: float
    distinct_merchants_1h: int
    time_since_last_txn: float  # seconds; -1 if no prior txn


@dataclass
class _CardState:
    txns: deque = None          # (ts, amount, merchant) within 24h
    last_ts: object = None

    def __init__(self):
        self.txns = deque()


class VelocityFeatures:
    def __init__(self):
        self._cards: dict[str, _CardState] = defaultdict(_CardState)

    def _prune(self, state: _CardState, now) -> None:
        cutoff = now - W24H
        while state.txns and state.txns[0][0] < cutoff:
            state.txns.popleft()

    def update(self, txn: Transaction) -> FeatureVector:
        """Record txn and return the feature vector (current txn included)."""
        state = self._cards[txn.card_id]
        self._prune(state, txn.ts)

        time_since = (
            (txn.ts - state.last_ts).total_seconds()
            if state.last_ts is not None else -1.0
        )
        state.txns.append((txn.ts, txn.amount, txn.merchant))
        state.last_ts = txn.ts

        cutoff_5m = txn.ts - W5M
        cutoff_1h = txn.ts - W1H
        in_1h = [t for t in state.txns if t[0] >= cutoff_1h]

        return FeatureVector(
            count_5m=sum(1 for t in in_1h if t[0] >= cutoff_5m),
            count_1h=len(in_1h),
            amount_1h=round(sum(t[1] for t in in_1h), 2),
            amount_24h=round(sum(t[1] for t in state.txns), 2),
            distinct_merchants_1h=len({t[2] for t in in_1h}),
            time_since_last_txn=round(time_since, 2),
        )

    def card_count(self) -> int:
        return len(self._cards)
