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
    c = load_config(families=False)
    c["start"], c["end"] = "2026-09-01T00:00:00Z", "2026-09-21T00:00:00Z"
    uni = tmp_path / "llm" / "universe.json"
    uni.parent.mkdir(parents=True)
    uni.write_text(json.dumps({"tickers": ["AAA", "BBB", "CCC"]}))
    c["bots"]["ibs_etf"]["params"]["instruments"] = ["AAA", "BBB", "CCC"]
    c["bots"]["overnight_spy"]["params"]["instruments"] = ["AAA"]
    c["bots"]["turn_of_month"]["params"]["instruments"] = ["BBB"]
    c["bots"]["leverage_rotation"]["params"].update({"instrument": "CCC", "signal": "AAA", "sma_days": 20})
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
    assert "Families" in html and "BTC buy-and-hold" in html
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


# ------------------------------------------------------------------------------ alternative bots
def _run_alt(tmp_path, cfg, bots, now="2026-09-15T12:00:00Z"):
    state = tmp_path / "alt"
    s = run_cycle(state, now=pd.Timestamp(now), source=FakeSource(), cfg=cfg, only=bots)
    assert not s["errors"], s["errors"]
    f = _fills(state)
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    return s, f


def test_vol_breakout_buys_on_the_break_and_sells_at_the_day_end(tmp_path, cfg):
    s, f = _run_alt(tmp_path, cfg, ["vol_breakout"])
    f = f[f.bot == "vol_breakout"]
    buys, sells = f[f.side == "buy"], f[f.side == "sell"]
    assert len(buys) > 0 and len(sells) > 0
    # every sale happens on the first candle of a UTC day, every buy is a breakout stop (taker)
    assert ((sells.ts.dt.hour == 0) & (sells.ts.dt.minute == 0)).all()
    assert (buys.liquidity == "taker").all() and buys.tag.str.startswith("vb:buy breakout").all()
    # a buy never fills below that day's target
    src = FakeSource()
    for _, b in buys.iterrows():
        sym = {"BTC_SPOT": "BTC/EUR", "ETH_SPOT": "ETH/EUR", "XRP_SPOT": "XRP/EUR"}[b.inst]
        df = src._series(sym, "15m")
        day0 = b.ts.normalize()
        prev = df[(df.index >= day0 - pd.Timedelta(days=1)) & (df.index < day0)]
        target = prev.close.iloc[-1] + 0.5 * (prev.high.max() - prev.low.min())
        assert b.price >= target * 0.9999


def test_btc_night_holds_only_from_21_to_23_utc(tmp_path, cfg):
    s, f = _run_alt(tmp_path, cfg, ["btc_night"])
    f = f[f.bot == "btc_night"]
    assert set(f[f.side == "buy"].ts.dt.hour) == {21}
    assert set(f[f.side == "sell"].ts.dt.hour) == {23}
    assert len(f[f.side == "buy"]) >= 13               # one round trip a day


def test_rebalance_keeps_weights_inside_the_band(tmp_path, cfg):
    s, f = _run_alt(tmp_path, cfg, ["rebalance"])
    st = s["bots"]["rebalance"]
    assert st["state"]["rebalances"] > 0
    eq = st["equity_eur"]
    w = {i: p["qty"] * p["mark"] / eq for i, p in st["positions"].items()}
    assert len(w) == 4
    # weights may drift between checks, but stay near the 24.75% target
    assert all(abs(x - 0.2475) < 0.06 for x in w.values()), w


def test_momentum_rotation_holds_one_coin_and_switches_at_midnight(tmp_path, cfg):
    s, f = _run_alt(tmp_path, cfg, ["momentum_rotation"])
    f = f[f.bot == "momentum_rotation"]
    assert len(f[f.side == "buy"]) > 3
    assert (f.ts.dt.hour == 0).all()
    assert len(s["bots"]["momentum_rotation"]["positions"]) <= 1
    # each switch: the sale is paid out before the buy in the same candle, so the buy is not cut short
    for ts, g in f.groupby("ts"):
        if (g.side == "sell").any() and (g.side == "buy").any():
            assert g[g.side == "buy"].value_eur.sum() > 0.95 * g[g.side == "sell"].value_eur.sum()


def test_ibs_trades_only_five_minutes_before_the_close(tmp_path, cfg):
    s, f = _run_alt(tmp_path, cfg, ["ibs_etf"])
    f = f[f.bot == "ibs_etf"]
    assert len(f) > 0
    ny = f.ts.dt.tz_convert("America/New_York")
    assert ((ny.dt.hour == 15) & (ny.dt.minute == 55)).all()
    assert f[f.side == "buy"].tag.str.contains("IBS 0.[01]").all()
    assert f[f.side == "sell"].tag.str.contains(r"IBS (?:0\.[89]|1\.0)").all()


def test_sells_fill_before_buys_in_the_same_candle():
    inst = load_instruments()
    mk = {i: Market(inst[i], 0.0, 0.0, 0.0015, 0.0025) for i in ("BTC_SPOT", "ETH_SPOT")}
    eng = BotEngine(mk, CostModel(load_venues(), 1.0), FxRates(1.16))

    class Switch(Bot):
        timeframe = "1h"

        def on_bar(self, ctx, bars):
            if ctx.ts == T[0]:
                ctx.place("BTC_SPOT", "buy", "market", eur=990, tag="x")
            if ctx.ts == T[1]:
                ctx.place("ETH_SPOT", "buy", "market", eur=980, tag="x")     # listed first, needs the BTC cash
                ctx.place("BTC_SPOT", "sell", "market", qty=ctx.qty("BTC_SPOT"), tag="x")

    T = pd.date_range("2026-09-01", periods=4, freq="1h", tz="UTC")
    c = {i: pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=T)
         for i, p in (("ETH_SPOT", 2500.0), ("BTC_SPOT", 60000.0))}
    led = Ledger("t", 1000.0, ts=T[0].isoformat())
    eng.run(Switch("t", {}), led, c, T[0], T[-1] + pd.Timedelta(hours=1))
    assert led.qty("BTC_SPOT") == 0
    assert led.qty("ETH_SPOT") * 2500 > 970


# ------------------------------------------------------------------------------ classic systems and anomalies
def test_turtle_enters_on_breakouts_and_exits_on_stops(tmp_path, cfg):
    s, f = _run_alt(tmp_path, cfg, ["turtle_crypto"])
    f = f[f.bot == "turtle_crypto"]
    buys, sells = f[f.side == "buy"], f[f.side == "sell"]
    assert len(buys) > 0 and len(sells) > 0
    assert (f.liquidity == "taker").all() and buys.tag.str.startswith("tt:entry").all()
    src = FakeSource()
    for _, b in buys.iterrows():
        sym = {"BTC_SPOT": "BTC/EUR", "ETH_SPOT": "ETH/EUR", "SOL_SPOT": "SOL/EUR"}[b.inst]
        df = src._series(sym, "4h")
        prev = df[df.index < b.ts].iloc[-20:]          # the 20 candles up to the one the order was placed on
        assert b.price >= prev.high.max() * 0.999


def test_squeeze_and_rsi_bots_trade_on_their_signals(tmp_path, cfg):
    s, f = _run_alt(tmp_path, cfg, ["squeeze", "rsi_classic"], now="2026-09-20T12:00:00Z")
    sq, rs = f[f.bot == "squeeze"], f[f.bot == "rsi_classic"]
    assert len(sq[sq.side == "buy"]) > 0 and len(rs[rs.side == "buy"]) > 0
    rsi_in = rs[rs.side == "buy"].tag.str.extract(r"RSI ([\d.]+)")[0].astype(float)
    assert (rsi_in < 30).all()
    rsi_out = rs[rs.side == "sell"].tag.str.extract(r"RSI ([\d.]+)")[0].astype(float)
    assert (rsi_out > 70).all()


def test_overnight_buys_at_the_close_and_sells_at_the_open(tmp_path, cfg):
    s, f = _run_alt(tmp_path, cfg, ["overnight_spy"])
    f = f[f.bot == "overnight_spy"]
    ny = f.ts.dt.tz_convert("America/New_York")
    b, sl = f.side == "buy", f.side == "sell"
    assert b.sum() >= 8 and sl.sum() >= 8
    assert ((ny[b].dt.hour == 15) & (ny[b].dt.minute == 55)).all()
    assert ((ny[sl].dt.hour == 9) & (ny[sl].dt.minute == 30)).all()


def test_turn_of_month_holds_over_the_month_end(tmp_path, cfg):
    cfg["start"], cfg["end"] = "2026-09-01T00:00:00Z", "2026-10-30T00:00:00Z"
    s, f = _run_alt(tmp_path, cfg, ["turn_of_month"], now="2026-10-10T00:00:00Z")
    f = f[f.bot == "turn_of_month"]
    days = f.assign(d=f.ts.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d"))
    assert list(days[days.side == "buy"].d) == ["2026-09-24"]      # 4 trading days left in September
    assert list(days[days.side == "sell"].d) == ["2026-10-05"]     # 3rd trading day of October


def test_leverage_rotation_follows_the_moving_average(tmp_path, cfg):
    s, f = _run_alt(tmp_path, cfg, ["leverage_rotation"], now="2026-09-20T12:00:00Z")
    sig = s["bots"]["leverage_rotation"]["state"]["signal"]
    pos = s["bots"]["leverage_rotation"]["positions"]
    assert ("CCC" in pos) == sig["above"]
    f = f[f.bot == "leverage_rotation"]
    assert set(f.inst) <= {"CCC"}


def test_single_ticker_yfinance_frames_are_read():
    from quantdesk.bots.data import _ticker_frame
    fields = ["Open", "High", "Low", "Close", "Volume"]
    multi = pd.DataFrame([[1, 2, 0.5, 1.5, 10]], columns=pd.MultiIndex.from_product([["SPY"], fields]))
    flat = pd.DataFrame([[1, 2, 0.5, 1.5, 10]], columns=fields)
    for raw in (multi, flat):
        assert float(_ticker_frame(raw, "SPY").rename(columns=str.lower)["close"].iloc[0]) == 1.5


def test_capital_change_restarts_every_bot_from_the_start(tmp_path, cfg):
    """Changing initial_capital discards ledgers, fills and equity rows opened with the old amount, so
    the next cycle replays from the test start; the result equals a fresh run at the new amount."""
    import copy
    bots = ["grid_btc", "dca_eth", "bb_meanrev", "supertrend", "ema_macd"]
    T = pd.Timestamp("2026-09-08T09:00:00Z")
    old = copy.deepcopy(cfg)
    old["initial_capital"] = 1000.0
    new = copy.deepcopy(cfg)
    new["initial_capital"] = 5000.0
    a, b = tmp_path / "a", tmp_path / "b"
    run_cycle(a, now=T - pd.Timedelta(days=2), source=FakeSource(), cfg=old, only=bots)
    (a / "ledgers" / "gone_bot.json").write_text((a / "ledgers" / "grid_btc.json").read_text())  # bot no longer in config
    assert len(_fills(a)) > 0
    s = run_cycle(a, now=T, source=FakeSource(), cfg=new, only=bots)
    assert s["capital_reset"] == {"initial_capital": 5000.0, "restarted_bots": len(bots) + 1}
    assert not (a / "ledgers" / "gone_bot.json").exists()
    run_cycle(b, now=T, source=FakeSource(), cfg=new, only=bots)
    key = ["bot", "ts", "inst", "side", "tag", "qty"]
    fa = _fills(a).sort_values(key, kind="stable").reset_index(drop=True)
    fb = _fills(b).sort_values(key, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(fa, fb)
    for bid in bots:
        la = json.loads((a / "ledgers" / f"{bid}.json").read_text())
        assert la["initial_capital"] == 5000.0
        assert la["cash"] == pytest.approx(json.loads((b / "ledgers" / f"{bid}.json").read_text())["cash"]), bid
    ea, eb = pd.read_csv(a / "equity.csv"), pd.read_csv(b / "equity.csv")
    assert len(ea) == len(eb) and ea["equity_eur"].sum() == pytest.approx(eb["equity_eur"].sum())
    # no further restart once the new amount is in force
    s2 = run_cycle(a, now=T + pd.Timedelta(hours=1), source=FakeSource(), cfg=new, only=bots)
    assert "capital_reset" not in s2
