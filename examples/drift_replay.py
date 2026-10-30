"""Replay a simulated stream, inject drift at txn 1200, and watch two detectors.

    python examples/drift_replay.py [out.png]      (needs matplotlib; the library itself is stdlib)

All data is synthetic (stream_fraud.stream). Panel 3 repeats the replay over 20 seeds, with and without drift.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stream_fraud.replay import compare, replay

out = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "docs" / "drift_replay.png")
r = replay(seed=2)
cmp = compare()
rows = cmp["rows"]

fig = plt.figure(figsize=(11, 8), dpi=140)
gs = fig.add_gridspec(3, 2, width_ratios=[1.7, 1], hspace=.55, wspace=.28)
ax0, ax1, ax2 = (fig.add_subplot(gs[i, 0]) for i in range(3))
axs = fig.add_subplot(gs[:, 1])
x = range(r.n)
roll = [sum(r.scores[max(0, i - 99):i + 1]) / len(r.scores[max(0, i - 99):i + 1]) for i in x]
ax0.plot(x, roll, color="#2e86ab", lw=1)
ax0.set_title("Mean risk score, rolling 100 txns (seed 2)", fontsize=9, loc="left")
ax1.plot(x, r.psi, color="#e4572e", lw=1); ax1.axhline(.25, color="#999", ls="--", lw=.8)
ax1.set_title("PSI vs frozen reference (alarm at 0.25)", fontsize=9, loc="left")
ax2.plot(x, r.ks, color="#59a14f", lw=1)
from stream_fraud.ks import KSMonitor
ax2.axhline(KSMonitor(400, 400)._crit(400, 400), color="#999", ls="--", lw=.8)
ax2.set_title("KS statistic vs frozen reference (dashed = alpha 0.001 critical value)", fontsize=9, loc="left")
for ax, at, c in ((ax1, r.psi_alarm_at, "#e4572e"), (ax2, r.ks_alarm_at, "#59a14f")):
    if at is not None:
        ax.axvline(at, color=c, ls=":", lw=1.2); ax.text(at + 25, ax.get_ylim()[1] * .45, f"alarm @ {at}", fontsize=8, color=c)
for ax in (ax0, ax1, ax2):
    ax.axvline(r.drift_at, color="k", lw=1); ax.set_xlim(0, r.n)
ax0.text(r.drift_at + 15, ax0.get_ylim()[1] * .9, "drift injected @ 1200", fontsize=8)
ax2.set_xlabel("transaction index")

n = len(rows)
nd = {k: sum(row[k]["false_alarm_no_drift"] for row in rows) for k in ("psi", "ks")}
ea = {k: sum(row[k]["early"] for row in rows) for k in ("psi", "ks")}
for j, (k, c) in enumerate((("psi", "#e4572e"), ("ks", "#59a14f"))):
    ds = [row[k]["delay"] for row in rows]
    axs.scatter([d for d in ds if d is not None], [j + (i % 5 - 2) * .06 for i, d in enumerate(d for d in ds if d is not None)], color=c, s=18, alpha=.8)
    nv = sum(row[k]["never"] for row in rows)
    axs.text(650, j - .45, f"first alarm before the drift: {ea[k]}/{n}\nno alarm at all: {nv}/{n}\nalarm on the no-drift replay: {nd[k]}/{n}", fontsize=7.5, va="top")
axs.set_yticks([0, 1]); axs.set_yticklabels(["PSI", "KS"]); axs.set_ylim(-.9, 1.5); axs.set_xlim(0, 2000)
axs.set_xlabel("txns from drift to first alarm\n(one dot per seed that alarmed after the drift)")
axs.set_title(f"{n} seeds", fontsize=9, loc="left")
fig.suptitle("Drift replay on SYNTHETIC transactions: in this setup PSI alarmed earlier than a KS test, and neither is false-alarm free", fontsize=11, x=.01, ha="left")
fig.text(.01, .005, "Simulated stream (stream_fraud.stream), reference = first 400 txns, window = 400. Not real transaction data.", fontsize=7, color="#666")
Path(out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
for k in ("psi", "ks"):
    ds = sorted(row[k]["delay"] for row in rows if row[k]["delay"] is not None)
    print(k, "delays", ds, "early", ea[k], "no-drift false alarms", nd[k])
