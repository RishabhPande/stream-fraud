from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import random


@dataclass
class Transaction:
    tx_id: str
    card_id: str
    amount: float            # USD
    merchant: str
    mcc: str                 # merchant category code
    ts: datetime             # event time (UTC)
    is_fraud: bool = False
    fraud_mode: str = ""     # which fraud pattern produced this txn ("" for legit)


MERCHANTS = [
    ("Whole Foods", "5411"), ("Shell", "5541"), ("Amazon", "5999"),
    ("Netflix", "4899"), ("Delta", "4511"), ("Starbucks", "5812"),
    ("Target", "5411"), ("Uber", "4121"), ("Best Buy", "5732"),
    ("CVS", "5912"),
]

# Fraud pattern A: card-testing bursts — 5-8 rapid small txns within minutes.
# Fraud pattern B: large cross-border txn.
BURST_MERCHANT = ("TestMerchant", "5999")


class TransactionStream:
    """Simulated card-transaction stream.

    Legit traffic: each card transacts a few times a day, $5-$250 amounts.
    Fraud comes in two modes; call inject_drift() mid-stream to switch modes.
    """

    def __init__(
        self,
        n: int = 10_000,
        fraud_rate: float = 0.01,
        seed: int = 42,
        cards: int = 200,
    ):
        self.n = n
        self.fraud_rate = fraud_rate
        self.rng = random.Random(seed)
        self.card_ids = [f"card_{i:04d}" for i in range(cards)]
        self._drifted = False
        self._mode = "card_testing"  # current fraud mode
        self._emitted = 0
        self._tx_seq = 0
        self._now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def inject_drift(self, fraud_rate_multiplier: float = 6.0) -> None:
        """Switch fraud pattern: card-testing bursts -> large cross-border txns.

        The new campaign also runs hotter (higher fraud rate) — typical when
        a fraud ring switches tactics — which is what makes the score
        distribution shift detectable.
        """
        self._drifted = True
        self._mode = "cross_border"
        self.fraud_rate = min(self.fraud_rate * fraud_rate_multiplier, 0.5)

    @property
    def drifted(self) -> bool:
        return self._drifted

    def _next_ts(self) -> datetime:
        # avg ~10 txns/min across the population
        self._now += timedelta(seconds=self.rng.expovariate(1 / 6))
        return self._now

    def _legit(self) -> Transaction:
        card = self.rng.choice(self.card_ids)
        merchant, mcc = self.rng.choice(MERCHANTS)
        amount = round(self.rng.lognormvariate(3.2, 0.8), 2)  # median ~$25
        self._tx_seq += 1
        return Transaction(
            tx_id=f"tx_{self._tx_seq:07d}", card_id=card,
            amount=amount, merchant=merchant, mcc=mcc, ts=self._next_ts(),
        )

    def _burst(self) -> list[Transaction]:
        """5-8 rapid small-amount txns on one card (card testing)."""
        card = self.rng.choice(self.card_ids)
        txns = []
        for _ in range(self.rng.randint(5, 8)):
            self._tx_seq += 1
            txns.append(Transaction(
                tx_id=f"tx_{self._tx_seq:07d}", card_id=card,
                amount=round(self.rng.uniform(1, 9), 2),
                merchant=BURST_MERCHANT[0], mcc=BURST_MERCHANT[1],
                ts=self._next_ts(), is_fraud=True, fraud_mode="card_testing",
            ))
        return txns

    def _cross_border(self) -> Transaction:
        card = self.rng.choice(self.card_ids)
        self._tx_seq += 1
        return Transaction(
            tx_id=f"tx_{self._tx_seq:07d}", card_id=card,
            amount=round(self.rng.uniform(900, 4000), 2),
            merchant="ForeignExchange", mcc="6012", ts=self._next_ts(),
            is_fraud=True, fraud_mode="cross_border",
        )

    def _fraud(self) -> list[Transaction]:
        if self._mode == "card_testing":
            return self._burst()
        return [self._cross_border()]

    def __iter__(self):
        return self

    def __next__(self) -> Transaction:
        if self._emitted >= self.n:
            raise StopIteration
        # buffer fraud bursts so a single __next__ returns one txn
        if not hasattr(self, "_pending") or not self._pending:
            if self.rng.random() < self.fraud_rate:
                self._pending = self._fraud()
            else:
                self._pending = [self._legit()]
        txn = self._pending.pop(0)
        self._emitted += 1
        return txn
