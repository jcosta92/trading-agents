"""The Grok trader: decides only live, logs every call, executes the stored plan with stop and
target orders, respects the spending cap, and holds cash without an API key."""
import json

import pandas as pd
import pytest

from quantdesk.bots import llm_agent
from quantdesk.bots.runner import load_config, run_cycle

from fakes import FamilySource

CALLS = []


class FakeClient:
    def __init__(self, key, model):
        self.model = model

    def chat(self, system, user):
        CALLS.append((system, user))
        btc = [l for l in user.splitlines() if l.startswith("current_price")][0]
        px = float(btc.split("=")[1].split(",")[0])
        return json.dumps({"decisions": [
            {"coin": "BTC", "signal": "buy_to_enter", "quantity": 300 / px, "profit_target": px * 1.04,
             "stop_loss": px * 0.98, "invalidation_condition": "4h close below EMA50", "confidence": 0.6,
             "risk_eur": 6, "justification": "test"},
            {"coin": "ETH", "signal": "sell_to_enter", "quantity": 1, "profit_target": 1, "stop_loss": 2},
            {"coin": "SOL", "signal": "hold"}]}), {"prompt_tokens": 4000, "completion_tokens": 1500}


@pytest.fixture()
def cfg():
    c = load_config()
    c["bots"] = {"grok_trader": c["bots"]["grok_trader"]}
    c["start"], c["end"] = "2026-10-20T00:00:00Z", "2026-11-20T00:00:00Z"
    return c


def test_decides_live_and_executes_the_plan(tmp_path, cfg, monkeypatch):
    CALLS.clear()
    monkeypatch.setenv("XAI_API_KEY", "test")
    monkeypatch.setattr(llm_agent, "XaiClient", FakeClient)
    state = tmp_path / "bots"
    src = FamilySource()
    # 08:05 UTC: the 07:00 candle has just closed at 08:00, a decision hour (every 4 h)
    s = run_cycle(state, now=pd.Timestamp("2026-10-21T08:05:00Z"), source=src, cfg=cfg, cache_dir="")
    assert not s["errors"], s["errors"]
    assert len(CALLS) == 1
    sys_p, user_p = CALLS[0]
    assert "LONG ONLY" in sys_p and "ALL BTC DATA" in user_p and "Available Cash" in user_p
    rows = [json.loads(l) for l in (state / "llm" / "grok_trader.jsonl").read_text().splitlines()]
    assert rows[0]["candle"] == "2026-10-21T07:00:00+00:00" and rows[0]["cost_usd"] > 0
    # the earlier decision hours (catching up from 20 Oct) were skipped, not decided after the fact
    assert len(rows) == 1
    # the next cycle fills the buy at the 08:00 open and places the stop and the target
    run_cycle(state, now=pd.Timestamp("2026-10-21T09:05:00Z"), source=src, cfg=cfg, cache_dir="")
    f = pd.read_csv(state / "fills.csv")
    buy = f[(f.bot == "grok_trader") & (f.side == "buy")]
    assert list(buy.inst) == ["BTC_SPOT"] and buy.value_eur.iloc[0] == pytest.approx(300, rel=0.02)
    st = json.loads((state / "status.json").read_text())["bots"]["grok_trader"]
    tags = sorted(o["tag"] for o in st["open_orders"])
    assert tags == ["llm:profit target", "llm:stop loss"]
    assert len(CALLS) == 1                                 # 09:00 is not a decision hour


def test_budget_cap_and_missing_key(tmp_path, cfg, monkeypatch):
    CALLS.clear()
    monkeypatch.setattr(llm_agent, "XaiClient", FakeClient)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    state = tmp_path / "bots"
    s = run_cycle(state, now=pd.Timestamp("2026-10-21T08:05:00Z"), source=FamilySource(), cfg=cfg, cache_dir="")
    assert not CALLS and "no XAI_API_KEY" in s["bots"]["grok_trader"]["state"]["llm_status"]
    monkeypatch.setenv("XAI_API_KEY", "test")
    cfg["bots"]["grok_trader"]["params"]["budget_usd"] = 0.0
    s = run_cycle(state, now=pd.Timestamp("2026-10-21T12:05:00Z"), source=FamilySource(), cfg=cfg, cache_dir="")
    assert not CALLS and "budget" in s["bots"]["grok_trader"]["state"]["llm_status"]
