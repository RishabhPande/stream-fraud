"""Cost-aware threshold tuning: flag by $ loss, not F1.

A fixed score threshold (e.g. 0.6) is a blunt instrument: it treats a $5
card-test and a $4,000 cross-border chargeback as equally worth catching.
In reality each decision outcome has a dollar price:

- TP (fraud caught): analyst confirmation cost — the flag still needs a human look.
- FP (legit flagged): review cost + customer friction (step-up auth, decline,
  support ticket) — more expensive than a clean catch.
- FN (fraud missed): the transaction amount — that money is gone.
- TN: nothing.

CostAwareThreshold collects labeled observations (score, label, amount) —
simulating the delayed label feedback a streaming system gets — and picks
the threshold that minimizes total expected cost over a candidate sweep.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LabeledObservation:
    score: float       # 0-1 scorer output
    is_fraud: bool     # ground-truth label (arrives late in production)
    amount: float      # txn amount in USD — the cost of missing it


@dataclass
class TuningResult:
    threshold: float | None  # argmin over candidates; None if no observations
    expected_cost: float      # total $ cost of the observations at threshold
    baseline_cost: float      # total $ cost at the fixed default threshold
    n: int                    # observations used
    tuned: bool               # False when the buffer was empty


class CostAwareThreshold:
    """Online buffer + argmin threshold tuner."""

    def __init__(
        self,
        default_threshold: float = 0.6,
        tp_cost: float = 2.0,   # analyst confirmation per caught fraud
        fp_cost: float = 8.0,   # review + customer friction per false alarm
        max_buffer: int = 50_000,
    ):
        self.default_threshold = default_threshold
        self.tp_cost = tp_cost
        self.fp_cost = fp_cost
        self._obs: list[LabeledObservation] = []
        self.max_buffer = max_buffer

    def observe(self, score: float, is_fraud: bool, amount: float) -> None:
        """Record one labeled observation; oldest evicted past max_buffer."""
        self._obs.append(LabeledObservation(score, is_fraud, amount))
        if len(self._obs) > self.max_buffer:
            del self._obs[: len(self._obs) - self.max_buffer]

    def total_cost(self, threshold: float) -> float:
        """Total $ cost if every observation were decided at `threshold`."""
        total = 0.0
        for o in self._obs:
            if o.score >= threshold:
                total += self.tp_cost if o.is_fraud else self.fp_cost
            elif o.is_fraud:
                total += o.amount              # missed fraud: lose the money
        return total

    def confusion(self, threshold: float) -> tuple[int, int, int, int]:
        """(tp, fp, fn, tn) if every observation were decided at `threshold`."""
        tp = fp = fn = tn = 0
        for o in self._obs:
            if o.score >= threshold:
                if o.is_fraud:
                    tp += 1
                else:
                    fp += 1
            elif o.is_fraud:
                fn += 1
            else:
                tn += 1
        return tp, fp, fn, tn

    def tune(self) -> TuningResult:
        """Sweep candidate thresholds; return the cheapest.

        Candidates are the unique observed scores (plus a hair above 1.0 so
        "flag nothing" is considered). Ties break toward the HIGHER threshold
        — fewer reviews for the same dollar outcome.
        """
        n = len(self._obs)
        if n == 0:
            return TuningResult(None, 0.0, 0.0, 0, False)
        candidates = sorted({o.score for o in self._obs} | {1.0001})
        best_t, best_cost = candidates[0], float("inf")
        for t in candidates:
            cost = self.total_cost(t)
            if cost < best_cost:            # strict < keeps the higher threshold
                best_cost, best_t = cost, t
        return TuningResult(
            threshold=round(best_t, 4),
            expected_cost=round(best_cost, 2),
            baseline_cost=round(self.total_cost(self.default_threshold), 2),
            n=n,
            tuned=True,
        )

    def clear(self) -> None:
        self._obs.clear()

    def __len__(self) -> int:
        return len(self._obs)
