"""Regression tests for the interactive drift-demo builder.

The demo page embeds series computed from stream_fraud.replay; these tests
build a small two-seed page into a temp file and check the embedded data
matches the real replay output, so the published page cannot silently drift
from the library it claims to show.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_drift_demo import build, rolling_mean  # noqa: E402
from stream_fraud.replay import replay  # noqa: E402


def _payload(path):
    html = Path(path).read_text()
    m = re.search(r'<script id="drift-data" type="application/json">(.*?)</script>', html, re.S)
    assert m, "embedded drift-data JSON not found"
    return html, json.loads(m.group(1))


def test_embedded_series_match_replay(tmp_path):
    out = tmp_path / "index.html"
    build(str(out), seeds=[2, 5])
    _, data = _payload(out)
    assert data["n"] == 3000 and data["drift_at"] == 1200
    assert [s["seed"] for s in data["seeds"]] == [2, 5]
    s2 = data["seeds"][0]
    r = replay(n=3000, drift_at=1200, seed=2, window=400)
    assert s2["psi"] == [round(x, 4) for x in r.psi]
    assert s2["ks"] == [round(x, 4) for x in r.ks]
    assert s2["score_roll"] == [round(x, 4) for x in rolling_mean(r.scores, 100)]
    assert s2["summary"]["psi"]["alarm_at"] == r.psi_alarm_at
    assert s2["summary"]["ks"]["alarm_at"] == r.ks_alarm_at


def test_summary_flags_consistent(tmp_path):
    out = tmp_path / "index.html"
    build(str(out), seeds=[2])
    _, data = _payload(out)
    for s in data["seeds"]:
        for k in ("psi", "ks"):
            row = s["summary"][k]
            if row["never"]:
                assert row["alarm_at"] is None and row["delay"] is None
            elif row["early"]:
                assert row["alarm_at"] < data["drift_at"] and row["delay"] is None
            else:
                assert row["delay"] == row["alarm_at"] - data["drift_at"]


def test_page_is_honest_and_self_contained(tmp_path):
    out = tmp_path / "index.html"
    build(str(out), seeds=[2])
    html, _ = _payload(out)
    assert "Catch the drift" in html
    assert "synthetic" in html.lower()
    assert "nothing is hand-drawn" in html
    assert "__DRIFT_DATA__" not in html, "template placeholder was not replaced"
