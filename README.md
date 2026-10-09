# stream-fraud

A pedagogical demo of a **real-time transaction scoring pipeline**: a
simulated card-transaction stream, sliding-window velocity features, an
interpretable risk scorer, and drift detection for when fraud patterns shift.

This is a teaching skeleton, not production infrastructure. Every piece is
pure Python stdlib — no Kafka, no Flink, no dependencies — so you can read
the whole thing in an afternoon and see how the parts fit together.

## The pipeline

```
TransactionStream → VelocityFeatures → RiskScorer → DriftMonitor
   (simulated)       (per-card sliding     (rules + z-score,   (PSI on score
                      windows: 5m/1h/24h)   0-1 + reasons)      distribution)
```

1. **Stream** (`stream_fraud/stream.py`) — generates legit traffic plus two
   fraud patterns: card-testing bursts (5–8 rapid small txns) and large
   cross-border txns. `inject_drift()` switches the fraud pattern mid-stream,
   simulating a ring changing tactics.
2. **Features** (`stream_fraud/features.py`) — per-card sliding windows kept
   in bounded deques: `count_5m`, `count_1h`, `amount_1h`, `amount_24h`,
   `distinct_merchants_1h`, `time_since_last_txn`. Memory stays flat no
   matter how long the stream runs.
3. **Scorer** (`stream_fraud/scorer.py`) — rules (velocity burst, amount
   anomaly via per-card Welford z-score, large cross-border txn,
   single-merchant hammering) producing a 0–1 score with human-readable
   reasons. A skeleton for a real streaming scoring service.
4. **Drift** (`stream_fraud/drift.py`) — tracks PSI (population stability
   index) of the score distribution between a frozen reference window and a
   rolling current window. PSI ≥ 0.25 → drifted.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -q
PYTHONPATH=. .venv/bin/python examples/quickstart.py
```

Expected output: ~2,000 txns scored in well under a second (~10µs/txn),
fraud detection around 50%+ at ~0% false positives with the rules baseline,
and the drift monitor firing a few hundred txns after the pattern switch.

## What's deliberately missing (roadmap)

- [ ] Kafka/Flink adapter replacing the simulated stream
- [ ] Online learning: scorer that retrains incrementally as labels arrive
- [ ] Feature store (offline/online parity) instead of in-memory deques
- [ ] Cost-aware thresholds: tune the flag threshold by $ loss, not F1
- [ ] Multi-pattern drift: detect *which* pattern shifted, not just *that* it did

## Not for production

Amounts, merchants, and fraud patterns are synthetic. The scorer is a
rules baseline — a real system would use a trained model, proper label
feedback loops, and adversarial review. Don't point this at real money.
