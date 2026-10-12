"""Protective orders: levels from the fill price, bar checks, trailing stops, time exits."""
import pandas as pd
import pytest

from quantdesk import protect
from quantdesk.data.store import HistoryStore
from quantdesk.context import Context
from quantdesk.execution import Fill
from quantdesk.ledger import FxRates, Ledger
from quantdesk.registry import Registry


def _ledger_with(instruments, iid="GOLD", qty=10, px=100.0, ts="2024-03-01", spec=None):
    led = Ledger("t", 5000.0, ts=ts)
    led.trade(ts, Registry(instruments).get(iid), qty, px, {}, FxRates(1.1), px)
    protect.register(led, [Fill(ts, "t", iid, qty, px, px, 0.0, "EUR", qty * px, "entry", None, spec)])
    return led


def test_levels_set_from_fill_price(instruments):
    led = _ledger_with(instruments, spec={"stop_atr": 2.0, "atr": 3.0, "target_pct": 0.1, "max_days": 5})
    pr = led.meta["protect"]["GOLD"]
    assert pr["stop"] == pytest.approx(94.0) and pr["target"] == pytest.approx(110.0)
    led2 = _ledger_with(instruments, spec={"stop_atr": 2.0, "atr": 3.0, "stop_pct": 0.03})
    assert led2.meta["protect"]["GOLD"]["stop"] == pytest.approx(97.0)          # the tighter of the two


def test_no_protection_without_spec_or_for_sells(instruments):
    led = _ledger_with(instruments, spec=None)
    assert "GOLD" not in led.meta.get("protect", {})


def test_check_bar_gap_intrabar_and_both_hit(instruments):
    led = _ledger_with(instruments, spec={"stop_pct": 0.05, "target_pct": 0.10})
    # gap down through the stop: fill at the open
    (o, px), = protect.check_bar(led, {"GOLD": (90.0, 92.0, 88.0)})
    assert px == 90.0 and o.qty == -10 and o.reason.startswith("STOP-LOSS")
    # traded through the stop intrabar: fill at the stop
    (o, px), = protect.check_bar(led, {"GOLD": (99.0, 101.0, 94.0)})
    assert px == pytest.approx(95.0)
    # both levels inside one bar: assume the stop came first (pessimistic)
    (o, px), = protect.check_bar(led, {"GOLD": (100.0, 112.0, 94.0)})
    assert o.reason.startswith("STOP-LOSS")
    # gap ABOVE the target at the open: the take-profit fills at the open, before any later dip
    (o, px), = protect.check_bar(led, {"GOLD": (112.0, 113.0, 94.0)})
    assert o.reason.startswith("TAKE-PROFIT") and px == 112.0
    # target only
    (o, px), = protect.check_bar(led, {"GOLD": (101.0, 111.0, 99.0)})
    assert o.reason.startswith("TAKE-PROFIT") and px == pytest.approx(110.0)
    assert protect.check_bar(led, {"GOLD": (100.0, 105.0, 96.0)}) == []


def test_check_price(instruments):
    led = _ledger_with(instruments, spec={"stop_pct": 0.05, "target_pct": 0.10})
    assert protect.check_price(led, {"GOLD": 100.0}) == []
    assert protect.check_price(led, {"GOLD": 94.9})[0].reason.startswith("STOP-LOSS")
    assert protect.check_price(led, {"GOLD": 110.5})[0].reason.startswith("TAKE-PROFIT")


def _ctx_for(instruments, closes: list[float], start="2024-01-01", asof=None):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx)
    store = HistoryStore({"close": pd.DataFrame({"GOLD": c}), "high": pd.DataFrame({"GOLD": c + 1}),
                          "low": pd.DataFrame({"GOLD": c - 1})})
    t = pd.Timestamp(asof) if asof else idx[-1]
    return Context(t, store, Registry(instruments), None, "backtest", now=t)


def test_trailing_stop_ratchets_up_never_down(instruments):
    closes = [100.0] * 30
    led = _ledger_with(instruments, ts=str(pd.bdate_range("2024-01-01", periods=30)[-1].date()),
                       spec={"stop_atr": 3.0, "atr": 2.0, "trail_atr": 3.0, "atr_n": 5})
    entry_day = pd.bdate_range("2024-01-01", periods=30)[-1]
    assert led.meta["protect"]["GOLD"]["stop"] == pytest.approx(94.0)
    # prices rise to 120 after entry: chandelier stop = highest close - 3 x ATR(5)
    up = closes + [105, 110, 115, 120]
    ctx = _ctx_for(instruments, up)
    protect.update_daily(led, ctx)
    pr = led.meta["protect"]["GOLD"]
    d = ctx.ohlc("GOLD", 150)
    from quantdesk import indicators as ind
    a = ind.last(ind.atr(d["high"], d["low"], d["close"], 5))
    assert pr["hwm"] == 120 and pr["stop"] == pytest.approx(120 - 3 * a) and pr["stop"] > 94.0
    s1 = pr["stop"]
    # prices fall back: the stop must not move down
    protect.update_daily(led, _ctx_for(instruments, up + [110, 100]))
    assert led.meta["protect"]["GOLD"]["stop"] >= s1
    assert entry_day.date().isoformat() == led.meta["protect"]["GOLD"]["opened"]


def test_trailing_stop_ignores_closes_before_entry(instruments):
    closes = [100.0] * 25 + [150.0] + [100.0] * 5         # a spike BEFORE the entry
    ts = str(pd.bdate_range("2024-01-01", periods=len(closes))[-1].date())
    led = _ledger_with(instruments, ts=ts, spec={"stop_atr": 3.0, "atr": 2.0, "trail_atr": 3.0, "atr_n": 5})
    protect.update_daily(led, _ctx_for(instruments, closes))
    assert led.meta["protect"]["GOLD"]["hwm"] == 100.0


def test_target_sma_only_armed_above_entry(instruments):
    led = _ledger_with(instruments, ts="2024-02-09", spec={"target_sma": 5, "sma_value": 97.0, "stop_pct": 0.1})
    assert led.meta["protect"]["GOLD"]["target"] is None           # SMA(5) below the entry: no take-profit
    assert protect.check_price(led, {"GOLD": 100.5}) == []
    protect.update_daily(led, _ctx_for(instruments, [100.0] * 30 + [104, 105, 106, 107, 108]))
    assert led.meta["protect"]["GOLD"]["target"] == pytest.approx(106.0)


def test_split_scales_levels(instruments):
    led = _ledger_with(instruments, spec={"stop_atr": 2.0, "atr": 3.0, "target_pct": 0.1})
    led.split("2024-03-05", "GOLD", 4.0)
    protect.apply_split(led, "GOLD", 4.0)
    pr = led.meta["protect"]["GOLD"]
    assert led.qty("GOLD") == 40 and led.positions["GOLD"].avg_price == pytest.approx(25.0)
    assert (pr["entry"], pr["stop"], pr["target"]) == pytest.approx((25.0, 23.5, 27.5))
    assert led.verify()[0]


def test_target_sma_follows_the_average(instruments):
    closes = [100.0] * 30
    led = _ledger_with(instruments, ts="2024-02-09", spec={"target_sma": 5, "sma_value": 104.0, "stop_pct": 0.1})
    assert led.meta["protect"]["GOLD"]["target"] == 104.0
    protect.update_daily(led, _ctx_for(instruments, closes + [90, 90, 90, 90, 90]))
    assert led.meta["protect"]["GOLD"]["target"] is None           # the average fell below the entry: disarmed


def test_time_exit_counts_business_days(instruments):
    led = _ledger_with(instruments, ts="2024-03-01", spec={"max_days": 5, "stop_pct": 0.1})   # a Friday
    ctx = _ctx_for(instruments, [100.0] * 60)
    ctx.now = pd.Timestamp("2024-03-07")                     # Thursday: 4 business days
    assert protect.time_exits(led, ctx) == []
    ctx.now = pd.Timestamp("2024-03-08")                     # Friday: 5 business days
    (o,) = protect.time_exits(led, ctx)
    assert o.qty == -10 and o.reason.startswith("TIME EXIT")
    assert protect.due_time_exits(led, pd.Timestamp("2024-03-07")) == []
    assert len(protect.due_time_exits(led, pd.Timestamp("2024-03-08"))) == 1


def test_cleanup_after_position_closed(instruments):
    led = _ledger_with(instruments, spec={"stop_pct": 0.05})
    led.trade("2024-03-02", Registry(instruments).get("GOLD"), -10, 101.0, {}, FxRates(1.1), 101.0)
    protect.cleanup(led)
    assert led.meta["protect"] == {}
    assert protect.resting_orders(led) == []
