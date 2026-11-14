"""Page-Hinkley streaming change-point detector on the score series.

PSI (drift.py) and KS (ks.py) are window-vs-reference detectors: the current
window has to fill before they can say anything, so their first alarm always
lags the drift by roughly a full window. Page-Hinkley is the classic
streaming alternative — one O(1)-memory cumulative sum that alarms as soon
as the running mean has shifted by more than `delta` for long enough
(defaults tuned on the synthetic stream below — see stream_fraud.replay):

    mean_t = running mean of every score since the last reset
    U_t    = U_{t-1} + (x_t - mean_t - delta),        U_0 = 0
    alarm when  U_t - min(U_0..U_t) > threshold

`delta` is the smallest mean shift worth alarming on, in score units.
`threshold` trades detection delay against false alarms. Neither has a
probabilistic meaning here: the scores are bursty and autocorrelated, so
measure the monitor's false-alarm rate on no-drift replays
(stream_fraud.replay.compare) — the same caveat as the KS monitor. An alarm
resets the accumulator and running mean, so the detector re-calibrates on
the post-drift stream instead of latching. Experimental option, not a
replacement for DriftMonitor.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PHResult:
    drifted: bool
    statistic: float    # current margin U_t - min(U_0..U_t)
    threshold: float


class PageHinkleyMonitor:
    def __init__(
        self,
        delta: float = 0.02,
        threshold: float = 20.0,
        warmup: int = 50,
    ):
        if delta <= 0:
            raise ValueError("delta must be positive (score units)")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        self.delta = delta
        self.threshold = threshold
        self.warmup = warmup
        self._n = 0
        self._mean = 0.0
        self._u = 0.0
        self._min_u = 0.0
        self._last = PHResult(False, 0.0, threshold)

    def update(self, score: float) -> None:
        self._n += 1
        self._mean += (score - self._mean) / self._n
        self._u += score - self._mean - self.delta
        self._min_u = min(self._min_u, self._u)
        margin = self._u - self._min_u
        if self._n >= self.warmup and margin > self.threshold:
            self._last = PHResult(True, round(margin, 4), self.threshold)
            # Re-calibrate on the post-drift stream instead of latching.
            self._n = 0
            self._mean = 0.0
            self._u = 0.0
            self._min_u = 0.0
        else:
            self._last = PHResult(False, round(margin, 4), self.threshold)

    def check(self) -> PHResult:
        return self._last

    def reset(self) -> None:
        self._n = 0
        self._mean = 0.0
        self._u = 0.0
        self._min_u = 0.0
        self._last = PHResult(False, 0.0, self.threshold)
