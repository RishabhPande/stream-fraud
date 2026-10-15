"""Cost-aware threshold example: tune the flag threshold by $ loss, not F1.

Runs the simulated stream through features + scorer, records (score, label,
amount) observations, then tunes the flag threshold to minimize total
expected dollar cost. Compares three thresholds:

- default 0.6 (the fixed baseline),
- F1-optimal (the classic ML tuning target),
- cost-optimal (CostAwareThreshold).

The cost model: a caught fraud costs a $2 analyst confirmation; a false
alarm costs $8 (review + customer friction from step-up auth/declines);
a missed fraud costs its full amount.

Two regimes, 6,000 txns each:
- pre-drift: card-testing bursts ($1-9 each). Fraud is cheap but false
  alarms are pricey, so the cost-optimal threshold is *precision-first*:
  it rejects the F1-optimal threshold's 500+ false positives.
- post-drift: large cross-border chargebacks ($900-4,000). A single miss
  dwarfs review cost, so the tuned threshold converges with F1-optimal
  and both beat the stale fixed baseline.
"""

from stream_fraud.features import VelocityFeatures
from stream_fraud.scorer import RiskScorer
from stream_fraud.stream import TransactionStream
from stream_fraud.threshold import CostAwareThreshold

N = 6000
FRAUD_RATE = 0.08
TP_COST = 2.0
FP_COST = 8.0


def run(inject_at=None, seed=11):
    stream = TransactionStream(n=N, fraud_rate=FRAUD_RATE, seed=seed)
    vf, scorer = VelocityFeatures(), RiskScorer()
    tuner = CostAwareThreshold(default_threshold=scorer.threshold,
                              tp_cost=TP_COST, fp_cost=FP_COST)
    emitted = 0
    for txn in stream:
        if inject_at is not None and emitted == inject_at:
            stream.inject_drift()
        f = vf.update(txn)
        res = scorer.score(txn, f)
        tuner.observe(res.score, txn.is_fraud, txn.amount)
        emitted += 1
    return tuner


def f1_optimal(tuner):
    """Threshold maximizing F1 over the same observations (for contrast)."""
    obs = tuner._obs
    best_t, best_f1 = 1.0, -1.0
    for t in sorted({o.score for o in obs} | {1.0001}):
        tp, fp, fn, _ = tuner.confusion(t)
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def report(name, tuner):
    res = tuner.tune()
    f1_t, f1 = f1_optimal(tuner)
    print(f"--- {name} ({res.n} observations, TP=${TP_COST:.0f}/FP=${FP_COST:.0f}) ---")
    for label, t in [("default 0.600 ", tuner.default_threshold),
                     (f"F1-optimal {f1_t:.3f} ", f1_t),
                     (f"cost-optimal {res.threshold:.3f}", res.threshold)]:
        tp, fp, fn, _ = tuner.confusion(t)
        cost = tuner.total_cost(t)
        caught = tp / (tp + fn) if (tp + fn) else 0
        print(f"  {label}: cost=${cost:,.0f}  "
              f"frauds caught {tp}/{tp + fn} ({caught:.0%})  false positives {fp}")
    saved = res.baseline_cost - res.expected_cost
    print(f"  -> tuned threshold saves ${saved:,.0f} vs the fixed baseline")


def main():
    report("pre-drift: card-testing bursts ($1-9)", run())
    report("post-drift: cross-border chargebacks ($900-4,000)",
           run(inject_at=3000))


if __name__ == "__main__":
    main()
