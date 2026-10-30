"""Two-sample Kolmogorov-Smirnov drift detector on the score distribution.

Second detector next to PSI (drift.py), same update()/check() interface. PSI
bins scores into 10 buckets and needs a threshold chosen by rule of thumb.
KS compares the full empirical distributions of a frozen reference window and
a rolling current window, and its threshold comes from the sample sizes:

    D = max |F_ref(x) - F_cur(x)|        alarm when D > c(alpha) * sqrt((n + m) / (n * m))

with c(alpha) = sqrt(-ln(alpha / 2) / 2).

Caveat: alpha is the level of ONE two-sample test on independent samples. Here
the test is repeated every `check_every` transactions on overlapping windows,
and the scores are bursty and autocorrelated, so alpha is NOT the false-alarm
rate of the monitor. Treat it as a sensitivity knob and measure false alarms
on no-drift replays (stream_fraud.replay.compare). Experimental option, not a
replacement for DriftMonitor.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass


@dataclass
class KSResult:
    drifted: bool
    statistic: float
    threshold: float


def ks_statistic(a: list[float], b: list[float]) -> float:
    """Two-sample KS statistic, exact, ties handled by evaluating at every observed value."""
    sa, sb = sorted(a), sorted(b)
    n, m = len(sa), len(sb)
    d = 0.0
    for x in sorted(set(sa) | set(sb)):
        d = max(d, abs(bisect_right(sa, x) / n - bisect_right(sb, x) / m))
    return d


class KSMonitor:
    def __init__(
        self,
        reference_window: int = 500,
        current_window: int = 500,
        alpha: float = 0.001,
        check_every: int = 10,
    ):
        self.reference_window = reference_window
        self.current_window = current_window
        self.alpha = alpha
        self.check_every = check_every
        self._ref: list[float] = []
        self._cur: deque[float] = deque(maxlen=current_window)
        self.reference_ready = False
        self._since = 0
        self._last = KSResult(False, 0.0, self._crit(reference_window, current_window))

    def _crit(self, n: int, m: int) -> float:
        c = math.sqrt(-math.log(self.alpha / 2) / 2)
        return c * math.sqrt((n + m) / (n * m))

    def update(self, score: float) -> None:
        if not self.reference_ready:
            self._ref.append(score)
            if len(self._ref) >= self.reference_window:
                self.reference_ready = True
            return
        self._cur.append(score)
        self._since += 1
        if len(self._cur) >= self.current_window and self._since >= self.check_every:
            self._since = 0
            d = ks_statistic(self._ref, list(self._cur))
            crit = self._crit(len(self._ref), len(self._cur))
            self._last = KSResult(d > crit, round(d, 4), round(crit, 4))

    def check(self) -> KSResult:
        return self._last

    def reset(self) -> None:
        self._ref = []
        self._cur.clear()
        self.reference_ready = False
        self._since = 0
        self._last = KSResult(False, 0.0, self._crit(self.reference_window, self.current_window))
