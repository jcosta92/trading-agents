import math

import pytest

from quantdesk.config import option_instrument, stock_instrument
from quantdesk.costs import CostModel


def test_ibkr_xetra_small_order_hits_minimums(instruments, venues):
    cm = CostModel(venues, 1.0)
    fb = cm.fees(instruments["WORLD_EQ"], 2, 110.0, "buy")       # EUR 220 order
    # 1.25 IBKR min + 0.90 Xetra min + 0.02 + 0.0018 clearing + 0.01 regulatory
    assert fb.components["ibkr_commission"] == pytest.approx(1.25)
    assert fb.components["xetra_exchange_fee"] == pytest.approx(0.90)
    assert fb.total == pytest.approx(1.25 + 0.90 + 0.02 + 220 * 0.000008 + 0.01, abs=1e-9)


def test_ibkr_xetra_large_order_uses_percentage(instruments, venues):
    cm = CostModel(venues, 1.0)
    fb = cm.fees(instruments["WORLD_EQ"], 100, 110.0, "buy")     # EUR 11,000
    assert fb.components["ibkr_commission"] == pytest.approx(5.5)


def test_t212_is_free_and_fractional(instruments, venues):
    cm = CostModel(venues, 1.0)
    assert cm.fees(instruments["WORLD_EQ"], 2, 110, "buy", override="t212").total == 0
    assert cm.lot_size(instruments["WORLD_EQ"], "t212") == 0.0
    assert cm.lot_size(instruments["WORLD_EQ"]) == 1.0


def test_us_stock_fees_and_sec_fee_only_on_sells(venues):
    cm = CostModel(venues, 1.0)
    inst = stock_instrument("AAPL")
    buy = cm.fees(inst, 0.5, 230.0, "buy")
    sell = cm.fees(inst, 0.5, 230.0, "sell")
    assert buy.components["ibkr_commission"] == pytest.approx(0.35)
    assert "sec_fee" not in buy.components
    assert sell.components["sec_fee"] == pytest.approx(0.5 * 230 * 0.0000206)


def test_option_commission_tiers_and_minimum(instruments, venues):
    cm = CostModel(venues, 1.0)
    cheap = option_instrument(instruments["XSP"], "2026-11-20", "P", 600)
    fb = cm.fees(cheap, 1, 0.04, "sell")
    assert fb.components["ibkr_commission"] == pytest.approx(1.00)   # 0.25 rate but USD 1 minimum
    fb10 = cm.fees(cheap, 10, 2.0, "sell")
    assert fb10.components["ibkr_commission"] == pytest.approx(6.5)


def test_exec_price_crosses_spread_and_stress(instruments, venues):
    inst = instruments["BTC_PERP"]
    cm1, cm2, cm0 = CostModel(venues, 1.0), CostModel(venues, 2.0), CostModel(venues, 0.0)
    b1 = cm1.exec_price(inst, "buy", 100.0, 99.99, 100.01)
    b2 = cm2.exec_price(inst, "buy", 100.0, 99.99, 100.01)
    assert b1 == pytest.approx(100.01 + 100 * 2 / 1e4)
    assert b2 - 100 == pytest.approx(2 * (b1 - 100))
    assert cm0.exec_price(inst, "sell", 100.0, 99.99, 100.01) == pytest.approx(100.0)
    s1 = cm1.exec_price(inst, "sell", 100.0)        # fallback half spread 1bp + 2bp slippage
    assert s1 == pytest.approx(100 * (1 - 3 / 1e4))


def test_exec_price_none_when_bid_is_zero_for_option(instruments, venues):
    cm = CostModel(venues, 1.0)
    opt = option_instrument(instruments["XSP"], "2026-11-20", "P", 500, half_spread_abs=0.05)
    assert cm.exec_price(opt, "sell", 0.03) is None
    assert cm.exec_price(opt, "buy", 0.03) == pytest.approx(0.08)


def test_estimate_cost_includes_spread_and_fees(instruments, venues):
    cm = CostModel(venues, 1.0)
    inst = instruments["WORLD_EQ"]
    est = cm.estimate_cost(inst, 1, 110.0)
    spread = 110 * (5 + 2) / 1e4
    assert est == pytest.approx(spread + cm.fees(inst, 1, 110 + spread, "buy").total)
    assert math.isfinite(est)


def test_wide_quotes_are_crossed_not_mid_filled(instruments, venues):
    cm = CostModel(venues, 1.0)
    opt = option_instrument(instruments["XSP"], "2026-11-20", "P", 640, half_spread_abs=0.025)
    wide = cm.exec_price(opt, "buy", 0.45, 0.30, 0.60)
    tight = cm.exec_price(opt, "buy", 0.50, 0.40, 0.60)
    assert wide == pytest.approx(0.60) and tight == pytest.approx(0.60)
    etf = cm.exec_price(instruments["WORLD_EQ"], "buy", 33.0, 32.0, 34.0)
    assert etf >= 34.0


def test_zero_bid_cannot_be_sold(instruments, venues):
    cm = CostModel(venues, 1.0)
    opt = option_instrument(instruments["XSP"], "2026-11-20", "P", 500)
    assert cm.exec_price(opt, "sell", 0.02, 0.0, 0.05) is None
