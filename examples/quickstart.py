"""Quickstart: run 2,000 txns through features + scorer, then inject drift."""

import time

from stream_fraud.drift import DriftMonitor
from stream_fraud.features import VelocityFeatures
from stream_fraud.scorer import RiskScorer
from stream_fraud.stream import TransactionStream

stream = TransactionStream(n=2000, fraud_rate=0.02, seed=42)
feats, scorer = VelocityFeatures(), RiskScorer()
monitor = DriftMonitor(reference_window=400, current_window=400)

tp = fp = tn = fn = 0
latencies = []
drift_fired_at = None

t0 = time.perf_counter()
for i, txn in enumerate(stream):
    if i == 1000:
        stream.inject_drift()
        print("--- drift injected: fraud pattern switched "
              "(card-testing bursts -> large cross-border) ---")
    start = time.perf_counter()
    f = feats.update(txn)
    res = scorer.score(txn, f)
    latencies.append((time.perf_counter() - start) * 1e6)
    monitor.update(res.score)

    if res.flagged and txn.is_fraud:
        tp += 1
    elif res.flagged:
        fp += 1
    elif txn.is_fraud:
        fn += 1
    else:
        tn += 1

    if drift_fired_at is None and monitor.check().drifted:
        drift_fired_at = i

total = time.perf_counter() - t0
detection_rate = tp / (tp + fn) if (tp + fn) else 0
fpr = fp / (fp + tn) if (fp + tn) else 0
avg_lat = sum(latencies) / len(latencies)

print(f"txns=2000  frauds flagged: {tp}/{tp + fn}")
print(f"detection_rate={detection_rate:.2%}  false_positive_rate={fpr:.2%}")
print(f"avg scoring latency={avg_lat:.1f}us  total wall={total:.2f}s")
final = monitor.check()
print(f"drift monitor: fired_at_txn={drift_fired_at}  final PSI={final.psi}")
