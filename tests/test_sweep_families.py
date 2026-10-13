"""Families added in the final GitHub sweep (13 Oct 2026): framework-specific indicators match the
originals, and the new order logic (scale-in, ATR trailing stop, OctoBot Dip Analyser) behaves as the
source strategies do."""
import numpy as np
import pandas as pd
import pytest

from quantdesk.bots.engine import BotEngine, Market
from quantdesk.bots.families import custom, etf_taa, frameworks
from quantdesk.bots.generic import SignalBot
from quantdesk.config import load_instruments, load_venues
from quantdesk.costs import CostModel
from quantdesk.ledger import FxRates, Ledger


def _ohlcv(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    h = c * (1 + np.abs(rng.normal(0, 0.005, n)))
    lo = c * (1 - np.abs(rng.normal(0, 0.005, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({"open": c, "high": h, "low": lo, "close": c, "volume": rng.uniform(100, 1000, n)}, index=idx)


# ------------------------------------------------------------------------------- reference algorithms
def _octobot_threshold_change_indexes(data, threshold):
    """OctoBot-Tentacles Evaluator/Util/trend_analysis: TrendAnalysis.get_threshold_change_indexes (verbatim)."""
    sub_threshold_indexes = np.where(data <= threshold)[0]
    threshold_crossing_indexes = []
    current_move_size = 1
    for i, index in enumerate(sub_threshold_indexes):
        if not len(threshold_crossing_indexes):
            threshold_crossing_indexes.append(index)
        else:
            if threshold_crossing_indexes[-1] == index - current_move_size:
                current_move_size += 1
            else:
                if sub_threshold_indexes[i - 1] not in threshold_crossing_indexes:
                    threshold_crossing_indexes.append(sub_threshold_indexes[i - 1])
                if index not in threshold_crossing_indexes:
                    threshold_crossing_indexes.append(index)
                current_move_size = 1
    if len(sub_threshold_indexes) > 0 and sub_threshold_indexes[-1] < len(data) and data[-1] > threshold \
            and sub_threshold_indexes[-1] + 1 not in threshold_crossing_indexes:
        threshold_crossing_indexes.append(sub_threshold_indexes[-1] + 1)
    return threshold_crossing_indexes


def _octobot_min_reached(data, acceptance_window=0.8, delay=1):
    if len(data) > 1:
        min_val = min(data)
        return bool(min_val in data[-(delay + 1):] and data[-1] / acceptance_window > min_val)
    return False


def test_tulip_kvo_matches_tulipy():
    tulipy = pytest.importorskip("tulipy")
    df = _ohlcv()
    ref = tulipy.kvo(*(df[k].to_numpy() for k in ("high", "low", "close", "volume")), 35, 55)
    assert np.allclose(frameworks.tulip_kvo(df, 35, 55).dropna().to_numpy(), ref, rtol=1e-12, atol=1e-9)


def test_klinger_reversal_matches_octobot_per_candle():
    diff = np.random.default_rng(1).normal(0, 1, 600).cumsum() * 0.3
    fast = frameworks._klinger_reversal(diff)
    for k in range(1, len(diff)):
        d = diff[:k + 1]
        z = _octobot_threshold_change_indexes(d, 0)
        ref = _octobot_min_reached(d[-min(7, len(d) - z[-1]):], 0.9, 1) if z else False
        assert fast[k] == ref, k
    assert fast.sum() > 5


def test_min_just_reached_matches_octobot():
    rng = np.random.default_rng(2)
    for _ in range(500):
        w = rng.uniform(10, 90, rng.integers(2, 6))
        assert frameworks._min_just_reached(w, 0.95, 2) == _octobot_min_reached(list(w), 0.95, 2)


def test_dip_weight_uses_the_first_matching_slow_tier_only():
    assert frameworks._dip_weight(28, 18, 18) == (2, 2)
    assert frameworks._dip_weight(28, 25, 25) == (1, 1)
    assert frameworks._dip_weight(28, 32, 32) == (None, None)       # tier 30 matched, no fast match: stop
    assert frameworks._dip_weight(60, 50, 50) == (3, 2)
    assert frameworks._dip_weight(80, 10, 10) == (None, None)


def test_gekko_zone_transitions_ignore_returns_to_neutral():
    x = pd.Series([0, -30, -10, -30, 0, 30, 40, 10, 30, -30], dtype=float)
    ent, ex = frameworks._zone_transitions(x, -25, 25)
    assert list(np.flatnonzero(ent)) == [1, 9]       # the second dip at 3 is the same 'low' direction
    assert list(np.flatnonzero(ex)) == [5]


def test_zenbot_stddev_matches_the_js_loop():
    x = pd.Series(np.random.default_rng(3).normal(0, 1, 200))
    got = frameworks._zenbot_stddev(x, 13)
    for k in range(13, 200):
        cur, look = x[k], [x[k - i] for i in range(1, 14)]
        avg = (cur + sum(look)) / 14
        assert got[k] == pytest.approx(np.sqrt(sum((v - avg) ** 2 for v in look) / 14), rel=1e-9)


def test_zenbot_oversold_buy_blocks_trend_sells_until_the_rate_turns_up():
    n = 400
    c = np.concatenate([np.linspace(100, 120, 200), np.linspace(120, 60, 60), np.linspace(60, 61, 140)])
    df = pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 1.0},
                      index=pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"))
    out = frameworks.zenbot_trend_ema(df, {}, {})
    rsi = frameworks.rsi(df["close"], 14)
    first_os = int(np.flatnonzero((rsi <= 10).to_numpy())[0])
    assert out["entry"].iloc[first_os]
    later = out.iloc[first_os + 1:first_os + 40]
    assert not later["exit"].any()                   # sells blocked while the crash continues


def test_hrp_and_min_variance_basics():
    rng = np.random.default_rng(4)
    rets = pd.DataFrame({"A": rng.normal(0, 0.01, 260), "B": rng.normal(0, 0.02, 260)})
    w = etf_taa.hrp_weights(rets)
    iv = 1 / rets.var()
    assert w.sum() == pytest.approx(1.0) and w["A"] == pytest.approx(iv["A"] / iv.sum(), rel=1e-9)
    mv = etf_taa.min_variance(np.diag([1e-4, 4e-4]))
    assert mv == pytest.approx([0.8, 0.2], abs=1e-6)


# ------------------------------------------------------------------------------- order logic
def _engine():
    inst = load_instruments()["BTC_SPOT"]
    return BotEngine({"BTC_SPOT": Market(inst, 0.0, 0.0, 0.0015, 0.0025)}, CostModel(load_venues()), FxRates(1.16))


def _bars(rows, freq="1h"):
    idx = pd.date_range("2026-09-01", periods=len(rows), freq=freq, tz="UTC")
    return {"BTC_SPOT": pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)}, idx


def _signal_bot(cls, spec, sig):
    bot = cls("t", {"timeframe": "1h", "params": {"instrument": "BTC_SPOT"}, **spec})
    bot.sig = sig
    return bot


def test_scale_in_adds_one_more_stake_on_a_later_fresh_signal():
    bars, idx = _bars([[100, 101, 99, 100, 1]] * 8)
    sig = {idx[0]: {"entry": True, "add": True}, idx[1]: {"entry": True, "add": False},
           idx[3]: {"entry": True, "add": True}, idx[5]: {"entry": True, "add": True}}
    bot = _signal_bot(SignalBot, {"size": 0.495, "scale_in": {"col": "add", "max_entries": 2}}, sig)
    led = Ledger("t", 1000.0)
    eng = _engine()
    eng.run(bot, led, bars, idx[0], idx[-1] + pd.Timedelta(days=1))
    buys = [f for f in eng.fills if f.side == "buy"]
    assert [f.ts for f in buys] == [idx[1].isoformat(), idx[4].isoformat()]   # max 2 entries
    assert buys[0].value_eur == pytest.approx(495, rel=0.01) and buys[1].value_eur == pytest.approx(495, rel=0.02)


def test_trailing_stop_distance_is_fixed_at_entry_and_trails_the_high():
    rows = [[100, 100, 100, 100, 1], [100, 100, 100, 100, 1], [100, 110, 100, 110, 1],
            [110, 110, 104, 106, 1], [106, 106, 103, 104, 1]]
    bars, idx = _bars(rows)
    sig = {idx[0]: {"entry": True, "trail_dist": 5.0}, idx[2]: {"entry": True, "trail_dist": 50.0}}
    bot = _signal_bot(SignalBot, {}, sig)
    led = Ledger("t", 1000.0)
    eng = _engine()
    eng.run(bot, led, bars, idx[0], idx[-1] + pd.Timedelta(days=1))
    sells = [f for f in eng.fills if f.side == "sell"]
    # entry at 100 (idx1); high 110 on idx2 moves the stop to 110 - 5 (not 110 - 50); idx3 trades through it
    assert len(sells) == 1 and sells[0].price == pytest.approx(105.0) and sells[0].ts == idx[3].isoformat()


def test_dip_analyser_limit_buy_then_three_take_profit_sells():
    rows = [[100, 100, 100, 100, 1], [100, 100, 99, 99.6, 1], [100, 101, 99.8, 101, 1],
            [101, 106, 101, 105, 1], [105, 108, 105, 107, 1]]
    bars, idx = _bars(rows, "4h")
    bot = custom.DipAnalyserBot("t", {"timeframe": "4h", "params": {"instrument": "BTC_SPOT"}})
    bot.sig = {idx[0]: {"entry": True, "price_weight": 2.0, "volume_weight": 3.0}}
    led = Ledger("t", 1000.0)
    eng = _engine()
    eng.run(bot, led, bars, idx[0], idx[-1] + pd.Timedelta(days=1))
    buy = [f for f in eng.fills if f.side == "buy"]
    assert len(buy) == 1 and buy[0].price == pytest.approx(99.5) and buy[0].liquidity == "maker"
    assert buy[0].value_eur == pytest.approx(500, rel=0.01)          # 50% of cash x heavy volume weight 1.0
    sells = sorted(f.price for f in eng.fills if f.side == "sell")
    assert sells == pytest.approx([99.5 * (1 + 0.07 * k / 3) for k in (1, 2, 3)])
    assert not led.positions
