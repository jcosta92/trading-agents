"""Trading bots: engine fill rules, each strategy, replay invariance, end-of-test liquidation."""
import json

import numpy as np
import pandas as pd
import pytest

from quantdesk.bots import ta
from quantdesk.bots.data import TF
from quantdesk.bots.engine import BotEngine, Market
from quantdesk.bots.runner import load_config, run_cycle
from quantdesk.bots.strategies import Bot, DCABot, GridBot, build_bot
from quantdesk.config import load_instruments, load_venues
from quantdesk.costs import CostModel
from quantdesk.ledger import FxRates, Ledger

BASE = pd.Timestamp("2026-08-01", tz="UTC")


def _walk(seed: int, n: int, start_px: float, vol: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.normal(0, vol / 6, n // 200 + 1), 200)[:n]
    return start_px * np.exp(np.cumsum(regime + rng.normal(0, vol, n)))


class FakeSource:
    """Deterministic synthetic candles; the full series is fixed, so any slicing is consistent."""
    PX = {"BTC/EUR": 60000.0, "ETH/EUR": 2500.0, "SOL/EUR": 150.0, "XRP/EUR": 2.0, "DOGE/EUR": 0.2,
          "ADA/EUR": 0.7, "LINK/EUR": 15.0, "AVAX/EUR": 25.0}

    def __init__(self):
        self.used = {}
        self._cache = {}

    def _series(self, symbol, timeframe):
        key = (symbol, timeframe)
        if key not in self._cache:
            tf = TF[timeframe]
            idx = pd.date_range(BASE, BASE + pd.Timedelta(days=120), freq=tf, inclusive="left")
            vol = (0.004 if symbol < "C" else 0.009) * (tf / pd.Timedelta(minutes=15)) ** 0.5
            c = _walk(sum(map(ord, symbol + timeframe)), len(idx), self.PX[symbol], vol)
            o = np.r_[c[0], c[:-1]]
            rng = np.random.default_rng(len(symbol))
            hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, len(c))))
            lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, len(c))))
            self._cache[key] = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": 1.0}, index=idx)
        return self._cache[key]

    def crypto(self, symbol, timeframe, since, now):
        df = self._series(symbol, timeframe)
        return df[(df.index >= since) & (df.index + TF[timeframe] <= now)]

    def eurusd(self):
        return 1.16

    # stocks: 3 tickers, NYSE regular sessions only
    def _stock_series(self):
        if "stocks" not in self._cache:
            import exchange_calendars as xcals
            sched = xcals.get_calendar("XNYS").schedule.loc["2026-08-01":"2026-11-30"]
            idx = pd.DatetimeIndex([t for o, c in zip(sched["open"], sched["close"])
                                    for t in pd.date_range(o, c, freq="5min", inclusive="left")])
            out = {}
            for i, t in enumerate(["AAA", "BBB", "CCC"]):
                c = _walk(100 + i, len(idx), 50 + 20 * i, 0.003)
                o = np.r_[c[0], c[:-1]]
                vol = np.where(idx.tz_convert("America/New_York").hour == 9, 3e5, 1e5) * (1 + 0.5 * np.sin(np.arange(len(idx)) + i))
                out[t] = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999,
                                       "close": c, "volume": vol}, index=idx)
            self._cache["stocks"] = out
        return self._cache["stocks"]

    def stocks(self, tickers, timeframe, since, now):
        return {t: df[(df.index >= since) & (df.index + TF["5m"] <= now)]
                for t, df in self._stock_series().items() if t in tickers}

    def stock_daily(self, tickers, days, now):
        out = {}
        for t, df in self._stock_series().items():
            d = df.copy()
            d.index = d.index.tz_convert("America/New_York").tz_localize(None).normalize()
            out[t] = d.groupby(level=0).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        return out


@pytest.fixture()
def cfg(tmp_path):
    c = load_config()
    c["start"], c["end"] = "2026-09-01T00:00:00Z", "2026-09-21T00:00:00Z"
    uni = tmp_path / "llm" / "universe.json"
    uni.parent.mkdir(parents=True)
    uni.write_text(json.dumps({"tickers": ["AAA", "BBB", "CCC"]}))
    return c


def _fills(state):
    return pd.read_csv(state / "fills.csv") if (state / "fills.csv").exists() else pd.DataFrame()


def test_all_bots_trade_and_ledgers_stay_consistent(tmp_path, cfg):
    state = tmp_path / "bots"
    s = run_cycle(state, now=pd.Timestamp("2026-09-15T12:00:00Z"), source=FakeSource(), cfg=cfg)
    assert not s["errors"], s["errors"]
    f = _fills(state)
    for bid in cfg["bots"]:
        led = Ledger.from_dict(json.loads((state / "ledgers" / f"{bid}.json").read_text()))
        assert led.verify()[0]
        assert all(p.qty > 0 for p in led.positions.values()), f"{bid} went short"
        assert led.cash["EUR"] > -1e-6, f"{bid} overdrew cash"
    traded = set(f["bot"])
    assert {"grid_btc", "dca_eth"} <= traded, traded
    assert (f[f.bot == "grid_btc"].side == "sell").sum() > 0


def test_replay_is_independent_of_cycle_timing(tmp_path, cfg):
    """One cycle at T, or many cycles up to T, must give identical trades."""
    a, b = tmp_path / "a", tmp_path / "b"
    T = pd.Timestamp("2026-09-08T09:00:00Z")
    run_cycle(a, now=T, source=FakeSource(), cfg=cfg, only=["grid_btc", "dca_eth", "bb_meanrev", "supertrend", "ema_macd"])
    for h in pd.date_range("2026-09-01T03:17:00Z", T, freq="7h"):
        run_cycle(b, now=h, source=FakeSource(), cfg=cfg, only=["grid_btc", "dca_eth", "bb_meanrev", "supertrend", "ema_macd"])
    run_cycle(b, now=T, source=FakeSource(), cfg=cfg, only=["grid_btc", "dca_eth", "bb_meanrev", "supertrend", "ema_macd"])
    key = ["bot", "ts", "inst", "side", "tag", "qty"]
    fa = _fills(a).sort_values(key, kind="stable").reset_index(drop=True)
    fb = _fills(b).sort_values(key, kind="stable").reset_index(drop=True)
    assert len(fa) > 0
    pd.testing.assert_frame_equal(fa, fb)
    for bid in ("grid_btc", "dca_eth", "bb_meanrev", "supertrend", "ema_macd"):
        ea = json.loads((a / "ledgers" / f"{bid}.json").read_text())["cash"]
        eb = json.loads((b / "ledgers" / f"{bid}.json").read_text())["cash"]
        assert ea == pytest.approx(eb), bid


def test_end_of_test_sells_everything(tmp_path, cfg):
    state = tmp_path / "bots"
    s = run_cycle(state, now=pd.Timestamp("2026-09-25T00:00:00Z"), source=FakeSource(), cfg=cfg,
                  only=["grid_btc", "dca_eth"])
    for bid in ("grid_btc", "dca_eth"):
        assert s["bots"][bid]["finished"]
        assert not s["bots"][bid]["positions"] and not s["bots"][bid]["open_orders"]
    f = _fills(state)
    assert (f.tag == "end of test: sell all").any()
    # later cycles change nothing
    n = len(f)
    run_cycle(state, now=pd.Timestamp("2026-09-26T00:00:00Z"), source=FakeSource(), cfg=cfg, only=["grid_btc", "dca_eth"])
    assert len(_fills(state)) == n


def test_orb_buys_breakouts_and_is_flat_overnight(tmp_path, cfg):
    state = tmp_path / "bots"
    s = run_cycle(state, now=pd.Timestamp("2026-09-12T00:00:00Z"), source=FakeSource(), cfg=cfg, only=["orb_stocks"])
    assert not s["errors"], s["errors"]
    assert not s["bots"]["orb_stocks"]["positions"], "ORB must not hold positions overnight"
    f = _fills(state)
    assert len(f) > 0 and set(f.side) == {"buy", "sell"}
    led = Ledger.from_dict(json.loads((state / "ledgers" / "orb_stocks.json").read_text()))
    assert led.costs.get("fx", 0) > 0 and led.costs.get("commissions", 0) > 0


# ------------------------------------------------------------------------------------- engine rules
class Scripted(Bot):
    timeframe = "15m"

    def __init__(self, plan):
        super().__init__("t", {"params": {"instrument": "BTC_SPOT"}})
        self.plan, self.fills = plan, []

    def on_candle(self, ctx, iid):
        for spec in self.plan.pop(ctx.ts, []):
            ctx.place(iid, *spec[:2], **spec[2])

    def on_fill(self, ctx, order, fill):
        self.fills.append((order["tag"], fill.price, fill.liquidity))


def _engine():
    inst = load_instruments()["BTC_SPOT"]
    return BotEngine({"BTC_SPOT": Market(inst, 0.0002, 0.0003, 0.0015, 0.0025)}, CostModel(load_venues()), FxRates(1.16))


def _bars(rows):
    idx = pd.date_range("2026-09-01", periods=len(rows), freq="15min", tz="UTC")
    return {"BTC_SPOT": pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx)}, idx


def test_limit_fills_at_price_or_better_and_not_in_its_own_candle():
    bars, idx = _bars([[100, 101, 99, 100, 1], [100, 100.5, 97, 98, 1], [95, 96, 94, 95, 1]])
    bot = Scripted({idx[0]: [("buy", "limit", {"qty": 1, "price": 99.5, "tag": "a"}),
                             ("buy", "limit", {"qty": 1, "price": 96, "tag": "b"})]})
    led = Ledger("t", 1000.0)
    _engine().run(bot, led, bars, idx[0], idx[-1] + pd.Timedelta(days=1))
    assert ("a", 99.5, "maker") in bot.fills            # low 97 < 99.5 on the 2nd candle
    assert ("b", 95, "maker") in bot.fills              # 3rd candle opened below 96: filled at the open
    assert led.costs["commissions"] == pytest.approx((99.5 + 95) * 0.0015 / 1.0, rel=1e-6)


def test_stop_is_assumed_before_take_profit_in_the_same_candle():
    bars, idx = _bars([[100, 100, 100, 100, 1], [100, 100, 100, 100, 1], [100, 104, 94, 100, 1]])
    bot = Scripted({idx[0]: [("buy", "market", {"qty": 1, "tag": "entry"})],
                    idx[1]: [("sell", "limit", {"qty": 1, "price": 103, "tag": "tp"}),
                             ("sell", "stop", {"qty": 1, "price": 95, "tag": "sl"})]})
    led = Ledger("t", 1000.0)
    _engine().run(bot, led, bars, idx[0], idx[-1] + pd.Timedelta(days=1))
    tags = [t for t, _, _ in bot.fills]
    assert tags == ["entry", "sl"]                       # tp had nothing left to sell
    sl = bot.fills[-1]
    assert sl[1] == pytest.approx(95 * (1 - 0.0005)) and sl[2] == "taker"


def test_buys_are_capped_by_cash_and_sells_by_holdings():
    bars, idx = _bars([[100, 100, 100, 100, 1]] * 4)
    bot = Scripted({idx[0]: [("buy", "market", {"qty": 50, "tag": "big"})],
                    idx[1]: [("sell", "market", {"qty": 99, "tag": "oversell"})]})
    led = Ledger("t", 1000.0)
    _engine().run(bot, led, bars, idx[0], idx[-1] + pd.Timedelta(days=1))
    assert led.cash["EUR"] >= 0 and not led.positions
    assert 9.8 < led.realized[0]["qty"] <= 10


def test_supertrend_flips_on_trend_change():
    n = 300
    c = np.r_[np.linspace(100, 150, n // 2), np.linspace(150, 90, n // 2)]
    df = pd.DataFrame({"open": c, "high": c * 1.002, "low": c * 0.998, "close": c})
    d = ta.supertrend(df, 10, 3.0)
    assert d.iloc[140] == 1 and d.iloc[-1] == -1


def test_grid_rebuys_one_level_below_after_a_sell():
    cfg = load_config()
    bot = build_bot("grid_btc", cfg["bots"]["grid_btc"])
    assert isinstance(bot, GridBot)
    lv = bot._levels(100.0)
    assert len(lv) == 17 and lv[0] == pytest.approx(92) and lv[-1] == pytest.approx(108)
    steps = [lv[i + 1] / lv[i] - 1 for i in range(16)]
    assert max(steps) - min(steps) < 1e-9 and 0.0099 < steps[0] < 0.0101


def test_dca_take_profit_moves_after_a_safety_order():
    bars, idx = _bars([[100, 100, 100, 100, 1], [100, 100, 100, 100, 1], [100, 100, 97.9, 98, 1],
                       [98, 98.3, 98, 98.2, 1], [98.2, 101, 98.2, 100.5, 1]])
    cfg = load_config()
    spec = {**cfg["bots"]["dca_eth"], "params": {**cfg["bots"]["dca_eth"]["params"], "instrument": "BTC_SPOT"}}
    bot = DCABot("dca", spec)
    led = Ledger("dca", 1000.0)
    eng = _engine()
    eng.run(bot, led, bars, idx[0], idx[-1] + pd.Timedelta(days=1))
    f = pd.DataFrame([x.__dict__ for x in eng.fills])
    assert list(f.tag[:3]) == ["dca:base", "dca:so:1", "dca:tp"]
    avg = (70 + 70) / (70 / f.price[0] + 70 / f.price[1])
    assert f.price[2] == pytest.approx(avg * 1.015, rel=2e-3)


def test_dashboard_and_status_table(tmp_path, cfg):
    from quantdesk.bots import report
    state = tmp_path / "bots"
    run_cycle(state, now=pd.Timestamp("2026-09-10T00:00:00Z"), source=FakeSource(), cfg=cfg)
    t = report.write(state, cfg, tmp_path / "docs", tmp_path / "reports")
    assert set(t["bot"]) == set(cfg["bots"])
    html = (tmp_path / "docs" / "index.html").read_text()
    assert "Scoreboard" in html and "BTC buy-and-hold" in html
    md = (tmp_path / "reports" / "BOTS.md").read_text()
    assert md.startswith("# Trading bots")
    p = pd.read_csv(state / "prices.csv")
    assert {"BTC_SPOT", "ETH_SPOT", "SOL_SPOT"} <= set(p.columns) and p["ts"].is_monotonic_increasing


def test_grid_never_has_two_orders_on_one_level(tmp_path, cfg):
    state = tmp_path / "bots"
    for t in pd.date_range("2026-09-02", "2026-09-14", freq="17h", tz="UTC"):
        s = run_cycle(state, now=t, source=FakeSource(), cfg=cfg, only=["grid_btc"])
        tags = [o["tag"] for o in s["bots"]["grid_btc"]["open_orders"] if o["tag"].count(":") == 2]
        levels = [int(x.split(":")[2]) for x in tags]
        assert len(levels) == len(set(levels)), tags
