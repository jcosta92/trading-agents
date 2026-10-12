"""Indicator definitions against hand-computed values."""
import numpy as np
import pandas as pd
import pytest

from quantdesk import indicators as ind


def _s(vals):
    return pd.Series(vals, index=pd.date_range("2024-01-01", periods=len(vals), freq="B"), dtype=float)


def test_sma_and_ema():
    c = _s([1, 2, 3, 4, 5])
    assert ind.sma(c, 3).tolist()[2:] == [2.0, 3.0, 4.0]
    assert ind.sma(c, 3).isna().sum() == 2
    e = ind.ema(c, 3)                                  # alpha = 2/(3+1) = 0.5, seeded with the first value
    assert e.iloc[-1] == pytest.approx(4.0625)


def test_rsi_wilder_known_value():
    # gains/losses: +1, -1, +2  ->  Wilder (alpha 1/2) avg up/down after 3 changes
    c = _s([10, 11, 10, 12])
    up = pd.Series([np.nan, 1, 0, 2.0]).ewm(alpha=0.5, adjust=False, min_periods=2).mean().iloc[-1]
    dn = pd.Series([np.nan, 0, 1, 0.0]).ewm(alpha=0.5, adjust=False, min_periods=2).mean().iloc[-1]
    assert ind.rsi(c, 2).iloc[-1] == pytest.approx(100 - 100 / (1 + up / dn))


def test_rsi_extremes():
    assert ind.rsi(_s(range(1, 30)), 2).iloc[-1] == 100.0          # only up moves
    assert ind.rsi(_s(range(30, 1, -1)), 2).iloc[-1] == pytest.approx(0.0)
    assert ind.rsi(_s([5] * 10), 2).iloc[-1] == 50.0                 # no moves at all


def test_true_range_uses_previous_close():
    h, l, c = _s([10, 12, 11]), _s([9, 11, 7]), _s([9.5, 11.5, 8])
    tr = ind.true_range(h, l, c)
    assert np.isnan(tr.iloc[0]) or tr.iloc[0] == 1.0
    assert tr.iloc[1] == pytest.approx(max(12 - 11, abs(12 - 9.5), abs(11 - 9.5)))   # gap up: 2.5
    assert tr.iloc[2] == pytest.approx(max(11 - 7, abs(11 - 11.5), abs(7 - 11.5)))    # 4.5


def test_atr_constant_range():
    n = 40
    c = _s([100.0] * n)
    a = ind.atr(c + 1, c - 1, c, 14)
    assert a.iloc[-1] == pytest.approx(2.0)


def test_donchian_excludes_today():
    h = _s([1, 2, 3, 10, 4])
    hi = ind.donchian_high(h, 3)
    assert hi.iloc[3] == 3.0                           # day 3 (high 10) compares with days 0-2 only
    assert hi.iloc[4] == 10.0
    assert ind.donchian_low(_s([5, 4, 3, 1, 2]), 2).iloc[4] == 1.0


def test_roc_and_macd_shapes():
    c = _s(np.linspace(100, 200, 60))
    assert ind.roc(c, 10).iloc[-1] == pytest.approx(c.iloc[-1] / c.iloc[-11] - 1)
    m = ind.macd(c)
    assert list(m.columns) == ["macd", "signal", "hist"] and m["macd"].iloc[-1] > 0


def test_ohlc_fill_widens_high_low():
    df = pd.DataFrame({"close": [10.0, 11.0], "open": [np.nan, 12.0], "high": [np.nan, 11.5], "low": [np.nan, 10.5]})
    out = ind.ohlc_fill(df)
    assert out.loc[0, ["open", "high", "low"]].tolist() == [10.0, 10.0, 10.0]
    assert out.loc[1, "high"] == 12.0                  # high can't be below the open
