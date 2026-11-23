"""QQQ noise-band + opening-range breakout with 1-day options (added 23 Nov 2026)."""
import csv

import numpy as np
import pandas as pd
import pytest

from fakes import FamilySource
from quantdesk.bots.families import options_orb as oo
from quantdesk.bots.runner import load_config, run_cycle

BOT = "qqq_noise_orb_options"
TREND_DAY = pd.Timestamp("2026-11-24")        # QQQ rallies after the opening range on this session


class MinuteSource(FamilySource):
    """QQQ 1-minute candles: quiet sessions, except a steady rally on TREND_DAY from 9:45 to 12:00.
    An option prints every minute at intrinsic value + 2.00 (calls and puts)."""

    def _qqq(self):
        if "qqq1m" not in self._c:
            sched = self._sched().loc["2026-10-01":"2026-11-27"]
            rng = np.random.default_rng(7)
            frames, px = [], 600.0
            for day, (o, c) in zip(sched.index, zip(sched["open"], sched["close"])):
                idx = pd.date_range(o, c, freq="1min", inclusive="left")
                steps = rng.normal(0, 0.15, len(idx))
                if day == TREND_DAY:
                    mins = np.arange(len(idx))
                    steps = np.where((mins >= 15) & (mins < 150), 0.12, 0.0)   # +16 in 2h15, then flat
                close = px + np.cumsum(steps)
                op = np.r_[px, close[:-1]]
                frames.append(pd.DataFrame({"open": op, "high": np.maximum(op, close) + 0.05,
                                            "low": np.minimum(op, close) - 0.05, "close": close,
                                            "volume": 1e5}, index=idx))
                px = close[-1]
            self._c["qqq1m"] = pd.concat(frames)
        return self._c["qqq1m"]

    def minute(self, symbol, since, now):
        q = self._qqq()
        if symbol != "QQQ":
            right, strike = symbol[9], int(symbol[10:]) / 1000
            iv = (q["close"] - strike) if right == "C" else (strike - q["close"])
            px = iv.clip(lower=0) + 2.0
            q = pd.DataFrame({"open": px, "high": px + 0.02, "low": px - 0.02, "close": px, "volume": 10.0})
        return q[(q.index >= since) & (q.index + pd.Timedelta(minutes=1) <= now)]


def _cfg(**params):
    cfg = load_config()
    spec = cfg["bots"][BOT]
    spec["params"].update({"start": "2026-11-23T00:00:00Z", **params})
    cfg["bots"] = {BOT: spec}
    cfg["initial_capital"] = 5000.0
    return cfg


def _log(state):
    with open(state / "options" / f"{BOT}.csv") as fh:
        return list(csv.DictReader(fh))


def test_rally_buys_one_call_and_sells_at_1545(tmp_path):
    src = MinuteSource()
    s = run_cycle(tmp_path, now=pd.Timestamp("2026-11-25T02:00Z"), source=src, cfg=_cfg(contracts=3))
    assert not s["errors"], s["errors"]
    rows = [r for r in _log(tmp_path) if r["ts_ny"].startswith("2026-11-24")]
    ev = [r["event"] for r in rows]
    assert ev.count("buy") == 1 and ev.count("sell") == 1 and ev[-2:] == ["exit signal", "sell"]   # one trade a day
    sigs = [r for r in rows if r["event"] == "signal"]
    buy, sell = rows[ev.index("buy")], rows[-1]
    sig = sigs[-1]
    for r in sigs:
        assert "10:00" <= r["ts_ny"][-5:] <= "14:00" and int(r["ts_ny"][-2:]) % 10 == 0   # only on the checks
        assert r["right"] == "C"
        assert float(r["qqq_close"]) > float(r["or_high"]) + 0.3 * float(r["atr"])
        assert float(r["qqq_close"]) > float(r["upper"]) and float(r["qqq_close"]) > float(r["vwap"])
    # strike nearest the price, expiry the next session, limit at the last print
    assert buy["contract"] == f"QQQ:2026-11-25:C:{round(float(sig['qqq_close']))}"
    assert float(buy["qty"]) == 3 and float(buy["price"]) <= float(sig["price"])
    assert sell["ts_ny"] == "2026-11-24 15:45" and "time exit" in sell["detail"]
    fills = pd.read_csv(tmp_path / "fills.csv")
    assert (fills["qty"] == 3).all() and list(fills["side"]) == ["buy", "sell"] * (len(fills) // 2)
    # cash moves by qty x price x 100 (options) plus fees: equity = 5000 + round-trip P&L
    b = s["bots"][BOT]
    assert not b["positions"]
    sells, buys = fills[fills["side"] == "sell"], fills[fills["side"] == "buy"]
    fx_fee = b["costs_eur"].get("fx", 0.0)                                     # EUR->USD conversion
    assert b["equity_eur"] == pytest.approx(5000 + sells["pnl_eur"].sum() - fx_fee, abs=0.05)  # P&L net of fees
    assert list(buys["value_eur"]) == pytest.approx(list(3 * 100 * buys["price"] / 1.16), rel=1e-6)


def test_cycles_give_the_same_result_as_one_pass(tmp_path):
    one = run_cycle(tmp_path / "a", now=pd.Timestamp("2026-11-25T02:00Z"), source=MinuteSource(), cfg=_cfg())
    src = MinuteSource()
    for now in ("2026-11-24T15:07Z", "2026-11-24T15:22Z", "2026-11-24T18:00Z", "2026-11-24T20:58Z",
                "2026-11-25T02:00Z"):
        many = run_cycle(tmp_path / "b", now=pd.Timestamp(now), source=src, cfg=_cfg())
    assert one["bots"][BOT]["equity_eur"] == pytest.approx(many["bots"][BOT]["equity_eur"], rel=1e-9)
    assert _log(tmp_path / "a") == _log(tmp_path / "b")


def test_waits_for_delayed_option_prints(tmp_path):
    """Minutes are only replayed once they closed data_delay_minutes ago."""
    s = run_cycle(tmp_path, now=pd.Timestamp("2026-11-24T15:15Z"), source=MinuteSource(), cfg=_cfg())
    assert pd.Timestamp(s["bots"][BOT]["last_candle"]) + pd.Timedelta(minutes=1) <= pd.Timestamp("2026-11-24T14:55Z")


def test_gap_rule_and_band():
    src = MinuteSource()
    m1 = src._qqq()
    t = oo.signal_table(m1, {}, pd.Timestamp("2026-11-24"))
    day = oo.session_frames(m1)
    g = day[TREND_DAY]
    prev_close = float(day[max(d for d in day if d < TREND_DAY)]["close"].iloc[-1])
    op = float(g["open"].iloc[0])
    sig = oo.noise_sigma(day, TREND_DAY)
    r = t.loc[t.index[0]]
    m = int((t.index[0].tz_convert(oo.NY).hour * 60 + t.index[0].tz_convert(oo.NY).minute) - 570)
    assert r["upper"] == pytest.approx(max(op, prev_close) * (1 + sig[m]))
    assert r["lower"] == pytest.approx(min(op, prev_close) * (1 - sig[m]))
    # the first check is the 1-minute candle closing at 10:00 New York
    assert (t.index[0] + pd.Timedelta(minutes=1)).tz_convert(oo.NY).strftime("%H:%M") == "10:00"


def test_option_symbol():
    assert oo.yahoo_option_symbol("QQQ", "2026-11-24", "C", 653) == "QQQ261124C00653000"
