"""Replay a simulated stream with a drift injected at a known point.

Runs the features + scorer once and feeds every score to a PSI monitor and a
KS monitor, recording when each one first alarms. Pure stdlib; the figure
in examples/drift_replay.py is drawn from this output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stream_fraud.ks import KSMonitor
from stream_fraud.drift import DriftMonitor
from stream_fraud.features import VelocityFeatures
from stream_fraud.scorer import RiskScorer
from stream_fraud.stream import TransactionStream


@dataclass
class ReplayResult:
    n: int
    drift_at: int | None
    scores: list[float] = field(default_factory=list)
    psi: list[float] = field(default_factory=list)       # 0.0 until the current window fills
    ks: list[float] = field(default_factory=list)        # 0.0 until the current window fills
    psi_alarm_at: int | None = None
    ks_alarm_at: int | None = None


def replay(
    n: int = 3000,
    drift_at: int | None = 1200,
    seed: int = 42,
    fraud_rate: float = 0.02,
    window: int = 400,
    psi_threshold: float = 0.25,
    ks_alpha: float = 0.001,
) -> ReplayResult:
    """Score `n` transactions, injecting drift at index `drift_at` (None = no drift)."""
    stream = TransactionStream(n=n, fraud_rate=fraud_rate, seed=seed)
    feats, scorer = VelocityFeatures(), RiskScorer()
    psi_mon = DriftMonitor(reference_window=window, current_window=window, psi_threshold=psi_threshold)
    ks_mon = KSMonitor(reference_window=window, current_window=window, alpha=ks_alpha)
    out = ReplayResult(n=n, drift_at=drift_at)
    for i, txn in enumerate(stream):
        if drift_at is not None and i == drift_at:
            stream.inject_drift()
        s = scorer.score(txn, feats.update(txn)).score
        psi_mon.update(s)
        ks_mon.update(s)
        p, c = psi_mon.check(), ks_mon.check()
        out.scores.append(s)
        out.psi.append(p.psi)
        out.ks.append(c.statistic)
        if out.psi_alarm_at is None and p.drifted:
            out.psi_alarm_at = i
        if out.ks_alarm_at is None and c.drifted:
            out.ks_alarm_at = i
    return out


def compare(seeds=range(1, 21), **kw) -> dict:
    """Run `replay` with and without drift over several seeds.

    Only the FIRST alarm per run is recorded. For each detector and seed:
    delay (txns from drift to first alarm, None if that alarm came before the
    drift or never came), early (first alarm before the drift point), never
    (no alarm at all) and false_alarm_no_drift (any alarm on the matching
    no-drift stream). Seeds 1..20 by default, all reported, none dropped.
    """
    drift_at = kw.pop("drift_at", 1200)
    rows = []
    for seed in seeds:
        d = replay(seed=seed, drift_at=drift_at, **kw)
        q = replay(seed=seed, drift_at=None, **kw)
        row = {"seed": seed}
        for name in ("psi", "ks"):
            at = getattr(d, f"{name}_alarm_at")
            row[name] = {
                "delay": None if at is None or at < d.drift_at else at - d.drift_at,
                "early": at is not None and at < d.drift_at,   # first alarm came before the drift point
                "never": at is None,
                "false_alarm_no_drift": getattr(q, f"{name}_alarm_at") is not None,
            }
        rows.append(row)
    return {"rows": rows, "drift_at": drift_at}
