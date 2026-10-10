import json

import pytest

from quantdesk.config import option_instrument, stock_instrument
from quantdesk.ledger import FxRates, Ledger

FX = FxRates(1.10)


def test_spot_round_trip_cash_and_pnl(instruments):
    led = Ledger("t", 1000.0)
    inst = instruments["WORLD_EQ"]
    led.trade("d1", inst, 5, 100.0, {"EUR": 2.0}, FX, mid=99.9)
    assert led.cash["EUR"] == pytest.approx(1000 - 500 - 2)
    assert led.equity(FX, {"WORLD_EQ": 110.0}) == pytest.approx(498 + 550)
    led.trade("d2", inst, -5, 110.0, {"EUR": 2.0}, FX, mid=110.1)
    assert led.positions == {}
    assert led.cash["EUR"] == pytest.approx(1000 - 502 + 548)
    assert led.realized[-1]["pnl"] == pytest.approx(50.0)
    assert led.costs["commissions"] == pytest.approx(4.0)
    assert led.costs["spread_slippage"] == pytest.approx(5 * 0.1 + 5 * 0.1)
    assert led.verify()[0]


def test_perp_moves_no_notional_cash_and_realises_pnl(instruments):
    led = Ledger("t", 1000.0)
    perp = instruments["BTC_PERP"]
    led.trade("d1", perp, 0.01, 60000.0, {"USD": 0.3}, FX, mid=60000.0)
    assert led.cash["EUR"] == pytest.approx(1000)
    assert led.cash["USD"] == pytest.approx(-0.3)
    # unrealised +100 USD
    assert led.equity(FX, {"BTC_PERP": 70000.0}) == pytest.approx(1000 + (100 - 0.3) / 1.1)
    led.trade("d2", perp, -0.01, 70000.0, {"USD": 0.35}, FX, mid=70000.0)
    assert led.cash["USD"] == pytest.approx(100 - 0.65)
    assert "BTC_PERP" not in led.positions
    assert led.verify()[0]


def test_short_option_credit_and_close(instruments):
    led = Ledger("t", 1000.0)
    opt = option_instrument(instruments["XSP"], "2026-11-20", "P", 640)
    led.trade("d1", opt, -1, 1.50, {"USD": 1.3}, FX, mid=1.55)
    assert led.cash["USD"] == pytest.approx(150 - 1.3)
    assert led.equity(FX, {opt.id: 1.55}) == pytest.approx(1000 + (150 - 1.3 - 155) / 1.1)
    led.trade("d2", opt, 1, 0.70, {"USD": 1.3}, FX, mid=0.65)
    assert led.realized[-1]["pnl"] == pytest.approx(80.0)
    assert led.verify()[0]


def test_fx_conversion_fee_tracked():
    led = Ledger("t", 1000.0)
    led.convert("d", "EUR", "USD", 110.0, FX, 0.0003)
    assert led.cash["USD"] == pytest.approx(110.0)
    assert led.cash["EUR"] == pytest.approx(1000 - 110 / (1.1 * 0.9997))
    assert led.costs["fx"] == pytest.approx(110 / (1.1 * 0.9997) - 100)
    assert led.equity(FX) == pytest.approx(1000 - led.costs["fx"])


def test_funding_and_charges():
    led = Ledger("t", 1000.0)
    led.funding("d", "BTC_PERP", "USD", -1.1, FX, 0.0001)
    led.charge("d", "USD", 0.55, FX, "inference")
    assert led.costs["funding"] == pytest.approx(1.0)
    assert led.costs["inference"] == pytest.approx(0.5)
    assert led.equity(FX) == pytest.approx(998.5)


def test_roundtrip_serialisation_and_tamper_detection(instruments):
    led = Ledger("t", 1000.0)
    led.trade("d1", instruments["GOLD"], 3, 100.0, {"EUR": 2.2}, FX, mid=100.0)
    led.meta["x"] = {"a": 1}
    d = json.loads(json.dumps(led.to_dict()))
    led2 = Ledger.from_dict(d)
    assert led2.verify()[0]
    assert led2.equity(FX, {"GOLD": 101}) == pytest.approx(led.equity(FX, {"GOLD": 101}))
    d["cash"]["EUR"] += 50                     # tamper with the balance
    assert not Ledger.from_dict(d).verify()[0]


def test_partial_reduce_keeps_average(instruments):
    led = Ledger("t", 10000.0)
    inst = stock_instrument("MSFT")
    led.trade("d1", inst, 2, 100.0, {}, FX, mid=100)
    led.trade("d2", inst, 2, 200.0, {}, FX, mid=200)
    assert led.positions["MSFT"].avg_price == pytest.approx(150)
    led.trade("d3", inst, -1, 300.0, {}, FX, mid=300)
    assert led.positions["MSFT"].qty == pytest.approx(3)
    assert led.positions["MSFT"].avg_price == pytest.approx(150)
    assert led.realized[-1]["pnl"] == pytest.approx(150)
