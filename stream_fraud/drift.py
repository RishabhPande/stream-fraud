"""Drift detection on the score distribution via PSI.

When fraud patterns shift (see TransactionStream.inject_drift), the mix of
scores the scorer emits shifts too. DriftMonitor compares the current
window's score histogram against a frozen reference window using the
Population Stability Index (PSI):

    PSI = sum((cur% - ref%) * ln(cur% / ref%))

Rule of thumb: PSI < 0.1 no change, 0.1-0.25 moderate shift, > 0.25
significant shift. check() returns (drifted, psi).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass
class DriftResult:
    drifted: bool
    psi: float


class DriftMonitor:
    def __init__(
        self,
        n_bins: int = 10,
        reference_window: int = 500,
        current_window: int = 500,
        psi_threshold: float = 0.25,
    ):
        self.n_bins = n_bins
        self.reference_window = reference_window
        self.current_window = current_window
        self.psi_threshold = psi_threshold
        self._ref: list[float] = []
        self._cur: deque[float] = deque(maxlen=current_window)
        self.reference_ready = False

    def _bin(self, scores: list[float]) -> list[float]:
        counts = [0] * self.n_bins
        for s in scores:
            idx = min(int(s * self.n_bins), self.n_bins - 1)
            counts[idx] += 1
        total = len(scores) or 1
        return [c / total for c in counts]

    @staticmethod
    def _psi(ref: list[float], cur: list[float], eps: float = 1e-4) -> float:
        total = 0.0
        for r, c in zip(ref, cur):
            r = max(r, eps)
            c = max(c, eps)
            total += (c - r) * math.log(c / r)
        return total

    def update(self, score: float) -> None:
        if not self.reference_ready:
            self._ref.append(score)
            if len(self._ref) >= self.reference_window:
                self.reference_ready = True
        else:
            self._cur.append(score)

    def check(self) -> DriftResult:
        if not self.reference_ready or len(self._cur) < self.current_window:
            return DriftResult(drifted=False, psi=0.0)
        psi = self._psi(self._bin(self._ref), self._bin(list(self._cur)))
        return DriftResult(drifted=psi >= self.psi_threshold, psi=round(psi, 4))

    def reset(self) -> None:
        self._ref = []
        self._cur.clear()
        self.reference_ready = False
