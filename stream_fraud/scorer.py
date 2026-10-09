"""Rules + z-score risk scorer on velocity features.

Design: a tiny interpretable baseline, not a production model.
- Rules catch textbook patterns (velocity burst, huge single txn, fast repeat).
- A z-score layer compares this txn's 1h amount against the card's own
  24h baseline, catching "out of character" spend.
Scores are 0-1 with human-readable reasons — the skeleton a real streaming
scoring service (feature store -> model -> explain) would flesh out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stream_fraud.features import FeatureVector
from stream_fraud.stream import Transaction


@dataclass
class ScoreResult:
    score: float                # 0-1
    reasons: list[str] = field(default_factory=list)
    flagged: bool = False       # score >= threshold


class RiskScorer:
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        # per-card running stats for the z-score layer (amount_24h context)
        self._means: dict[str, float] = {}
        self._m2: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def _zscore(self, card_id: str, amount_1h: float) -> float:
        """Welford z-score of this hour's spend vs the card's history."""
        c = self._counts.get(card_id, 0)
        if c < 5:
            self._update_stats(card_id, amount_1h)
            return 0.0
        mean = self._means[card_id]
        var = self._m2[card_id] / (c - 1) if c > 1 else 0.0
        z = (amount_1h - mean) / (var ** 0.5) if var > 0 else 0.0
        self._update_stats(card_id, amount_1h)
        return z

    def _update_stats(self, card_id: str, x: float) -> None:
        c = self._counts.get(card_id, 0) + 1
        self._counts[card_id] = c
        if c == 1:
            self._means[card_id] = x
            self._m2[card_id] = 0.0
        else:
            delta = x - self._means[card_id]
            self._means[card_id] += delta / c
            self._m2[card_id] += delta * (x - self._means[card_id])

    def score(self, txn: Transaction, feats: FeatureVector) -> ScoreResult:
        s = 0.0
        reasons: list[str] = []

        # Rule 1: velocity burst — many txns in 5 minutes (card testing)
        if feats.count_5m >= 4:
            s += 0.55
            reasons.append(f"velocity burst: {feats.count_5m} txns in 5m")
        elif feats.count_5m >= 2 and feats.time_since_last_txn >= 0 and feats.time_since_last_txn < 60:
            s += 0.25
            reasons.append(f"rapid repeat: {feats.count_5m} txns in 5m")

        # Rule 2: huge single txn relative to card norms
        z = self._zscore(txn.card_id, feats.amount_1h)
        if z > 4:
            s += 0.5
            reasons.append(f"amount anomaly: z={z:.1f} vs card baseline")
        elif z > 2.5:
            s += 0.25
            reasons.append(f"elevated spend: z={z:.1f} vs card baseline")

        # Rule 3: large cross-border-style txn (high amount, odd MCC)
        if txn.amount >= 900 and txn.mcc == "6012":
            s += 0.60
            reasons.append(f"large cross-border txn: ${txn.amount:,.0f}")

        # Rule 4: single merchant hammering in the last hour
        if feats.count_1h >= 6 and feats.distinct_merchants_1h == 1:
            s += 0.35
            reasons.append(f"single-merchant hammering: {feats.count_1h} txns/1h")

        s = min(s, 1.0)
        if not reasons:
            reasons.append("no risk signals")
        return ScoreResult(score=round(s, 3), reasons=reasons, flagged=s >= self.threshold)
