"""news-inbox / validate-labels commands, offline."""
import json

import numpy as np
import pandas as pd
import pytest

from quantdesk import cli, llm
from quantdesk.data.store import HistoryStore


@pytest.fixture()
def offline(monkeypatch):
    tickers = [f"T{i:03d}" for i in range(120)]
    idx = pd.bdate_range("2026-06-01", "2026-10-09")
    rng = np.random.default_rng(1)
    close = pd.DataFrame(rng.uniform(20, 200, (len(idx), len(tickers))), index=idx, columns=tickers)
    vol = pd.DataFrame(rng.uniform(1e5, 1e7, (len(idx), len(tickers))), index=idx, columns=tickers)
    vol["T007"] = 1e9                                       # the most liquid name
    st = HistoryStore({"close": close, "volume": vol})
    calls = {"sp500": 0, "news": 0}

    def fake_sp500(start, current_only=False, **k):
        calls["sp500"] += 1
        return st, {"sectors": {}}

    def fake_news(t):
        calls["news"] += 1
        now = pd.Timestamp("2026-10-12T12:20:00Z")
        return [{"title": f"{t} beats estimates", "summary": "s", "publisher": "P",
                 "published": (now - pd.Timedelta(hours=3)).isoformat()},
                {"title": f"{t} old story", "summary": "", "publisher": "P",
                 "published": (now - pd.Timedelta(days=9)).isoformat()}]
    monkeypatch.setattr("quantdesk.data.loader.build_sp500_store", fake_sp500)
    monkeypatch.setattr("quantdesk.data.yahoo.news", fake_news)
    return calls


def test_news_inbox_and_label_validation(tmp_path, offline, capsys):
    rc = cli.main(["news-inbox", "--state", str(tmp_path), "--now", "2026-10-12T12:20:00Z"])
    assert rc == 0
    inbox = json.loads(llm.inbox_path(tmp_path, "2026-10-12").read_text())
    assert len(inbox["universe"]) == 100 and inbox["universe"][0] == "T007"
    assert len(inbox["items"]) == 100                       # the 9-day-old headlines are dropped
    assert inbox["prompt_version"] == llm.PROMPT_VERSION
    # second call the same day does nothing; the universe is cached for the month
    assert cli.main(["news-inbox", "--state", str(tmp_path), "--now", "2026-10-12T12:50:00Z"]) == 0
    assert offline["sp500"] == 1
    # labels: missing -> error, partial -> INVALID, complete -> OK
    assert cli.main(["validate-labels", "--state", str(tmp_path), "--date", "2026-10-12"]) == 1
    labels = {"date": "2026-10-12", "prompt_version": llm.PROMPT_VERSION, "labelled_by": "test",
              "labels": [{"id": it["id"], "ticker": it["ticker"], "event_type": "earnings", "direction": 1,
                          "materiality": 2, "confidence": 0.6, "rationale": "beat"} for it in inbox["items"]]}
    lp = llm.labels_path(tmp_path, "2026-10-12")
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps({**labels, "labels": labels["labels"][:-1]}))
    assert cli.main(["validate-labels", "--state", str(tmp_path), "--date", "2026-10-12"]) == 1
    assert "have no label" in capsys.readouterr().out
    bad = json.loads(json.dumps(labels))
    bad["labels"][0]["materiality"] = 5
    lp.write_text(json.dumps(bad))
    assert cli.main(["validate-labels", "--state", str(tmp_path), "--date", "2026-10-12"]) == 1
    lp.write_text(json.dumps(labels))
    assert cli.main(["validate-labels", "--state", str(tmp_path), "--date", "2026-10-12"]) == 0
    assert "OK: 100 labels" in capsys.readouterr().out


def test_news_inbox_skips_market_holidays_and_covers_the_gap(tmp_path, offline):
    # Thanksgiving 2026 (26 Nov) is an NYSE holiday: no inbox
    assert cli.main(["news-inbox", "--state", str(tmp_path), "--now", "2026-11-26T12:20:00Z"]) == 0
    assert not llm.inbox_path(tmp_path, "2026-11-26").exists()
    # an inbox on Friday, the next on Monday: Monday's covers everything since Friday's
    cli.main(["news-inbox", "--state", str(tmp_path), "--now", "2026-10-09T12:20:00Z"])
    cut = llm.news_cutoff(tmp_path, pd.Timestamp("2026-10-12T12:20:00Z"))
    assert cut == pd.Timestamp("2026-10-09T12:20:00Z")
