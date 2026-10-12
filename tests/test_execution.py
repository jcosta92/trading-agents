import pytest

from quantdesk.config import stock_instrument
from quantdesk.costs import CostModel
from quantdesk.execution import Executor, OrderIntent, Quote
from quantdesk.ledger import FxRates, Ledger
from quantdesk.registry import Registry


class Q:
    def __init__(self, prices, fx=1.1):
        self.prices = prices
        self._fx = FxRates(fx)

    def quote(self, inst):
        p = self.prices.get(inst.id)
        if p is None:
            return None
        if isinstance(p, tuple):
            return Quote((p[0] + p[1]) / 2, p[0], p[1])
        return Quote(p)

    def fx(self):
        return self._fx


def test_whole_shares_and_cash_limit(instruments, venues):
    reg = Registry(instruments)
    ex = Executor(CostModel(venues, 1.0), reg)
    led = Ledger("t", 300.0)
    fills, rej = ex.execute(led, [OrderIntent("WORLD_EQ", 3.7, "test")], Q({"WORLD_EQ": 110.0}), "d")
    assert len(fills) == 1 and fills[0].qty == 2           # floor to whole shares, then cash limit
    assert led.cash["EUR"] > 0


def test_auto_fx_for_usd_stock(instruments, venues):
    reg = Registry(instruments)
    reg.stock("AAPL")
    ex = Executor(CostModel(venues, 1.0), reg)
    led = Ledger("t", 1000.0)
    fills, rej = ex.execute(led, [OrderIntent("AAPL", 1.5, "test")], Q({"AAPL": (199.9, 200.1)}), "d")
    assert not rej
    assert led.cash["USD"] == pytest.approx(0.0, abs=1e-9)
    assert led.costs["fx"] > 0
    assert led.verify()[0]


def test_combo_all_or_nothing(instruments, venues):
    reg = Registry(instruments)
    ex = Executor(CostModel(venues, 1.0), reg)
    led = Ledger("t", 1000.0)
    orders = [OrderIntent("BTC_SPOT", 0.001, "a", combo="c1"), OrderIntent("BTC_PERP", -0.001, "b", combo="c1")]
    fills, rej = ex.execute(led, orders, Q({"BTC_SPOT": 60000.0}), "d")      # perp has no quote
    assert fills == [] and len(rej) == 2
    assert led.positions == {}


def test_insufficient_cash_rejected(instruments, venues):
    reg = Registry(instruments)
    ex = Executor(CostModel(venues, 1.0), reg)
    led = Ledger("t", 50.0)
    fills, rej = ex.execute(led, [OrderIntent("WORLD_EQ", 1, "x")], Q({"WORLD_EQ": 110.0}), "d")
    assert not fills and rej[0].why == "insufficient cash"


def test_closing_full_position_not_rounded(instruments, venues):
    reg = Registry(instruments)
    ex = Executor(CostModel(venues, 1.0), reg)
    led = Ledger("t", 1000.0)
    ex.execute(led, [OrderIntent("BTC_SPOT", 0.0123456789, "buy")], Q({"BTC_SPOT": 50000.0}), "d")
    q = led.qty("BTC_SPOT")
    ex.execute(led, [OrderIntent("BTC_SPOT", -q, "sell")], Q({"BTC_SPOT": 50000.0}), "d2")
    assert led.positions == {}


def test_combo_scaled_together_keeps_hedge(instruments, venues):
    reg = Registry(instruments)
    ex = Executor(CostModel(venues, 1.0), reg)
    led = Ledger("t", 300.0)                      # can only afford part of the spot leg
    orders = [OrderIntent("BTC_SPOT", 0.01, "spot", combo="c"), OrderIntent("BTC_PERP", -0.01, "perp", combo="c")]
    fills, rej = ex.execute(led, orders, Q({"BTC_SPOT": 50000.0, "BTC_PERP": 55000.0}), "d")
    assert not rej and len(fills) == 2
    assert led.qty("BTC_SPOT") == pytest.approx(-led.qty("BTC_PERP"))
    assert led.qty("BTC_SPOT") < 0.01
    assert led.cash["EUR"] >= -1e-9


def test_cash_instruments_cannot_be_sold_short(instruments, venues):
    """A stale exit order for an ETF that is no longer held must not open a short."""
    reg = Registry(instruments)
    ex = Executor(CostModel(venues, 1.0), reg)
    led = Ledger("t", 1000.0)
    fills, rejects = ex.execute(led, [OrderIntent("GOLD", -3, "stale stop")], Q({"GOLD": 50.0}), "2024-01-02")
    assert not fills and rejects[0].why == "no position to sell"
    ex.execute(led, [OrderIntent("GOLD", 4, "buy")], Q({"GOLD": 50.0}), "2024-01-02")
    fills, _ = ex.execute(led, [OrderIntent("GOLD", -10, "oversized sell")], Q({"GOLD": 50.0}), "2024-01-03")
    assert fills[0].qty == -4 and led.qty("GOLD") == 0
