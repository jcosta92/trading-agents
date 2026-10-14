"""Gold-futures bots (added 14 Oct 2026): margin accounting for shorts in the bot engine, the IFVG and
level re-entry rules, and that a replay split into cycles gives the same trades as one pass."""
import json

import numpy as np
import pandas as pd
import pytest

from quantdesk.bots.data import TF
from quantdesk.bots.engine import BotEngine, Market
from quantdesk.bots.families import futures as fu
from quantdesk.bots.runner import _futures_market, load_config, run_cycle
from quantdesk.bots.strategies import Bot, build_bot
from quantdesk.config import load_venues
from quantdesk.costs import CostModel
from quantdesk.ledger import FxRates, Ledger

FX = 1.16
SPEC = {"yf": "GC=F", "currency": "USD", "venue": "ibkr_comex_micro", "multiplier": 10, "lot_size": 1,
        "tick": 0.10, "half_spread_bps": 0.12, "data_delay_minutes": 15}
FEE = 0.96                                         # USD per contract per side (0.25 + 0.70 + 0.01)


def _market(spread=True):
    markets = {}
    _futures_market(markets, "MGC", SPEC)
    if not spread:
        markets["MGC"] = Market(markets["MGC"].inst, 0.0, 0.0)
    return markets


def _engine(spread=True):
    return BotEngine(_market(spread), CostModel(load_venues()), FxRates(FX))


def bars5(start: str, rows) -> pd.DataFrame:
    idx = pd.date_range(pd.Timestamp(start), periods=len(rows), freq="5min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


def flat(start: str, end: str, px: float) -> pd.DataFrame:
    """Quiet 5-minute candles (no gaps between candles, so no fair value gaps), skipping the CME
    daily break 17:00-18:00 New York."""
    idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="5min", inclusive="left")
    ny = idx.tz_convert("America/New_York")
    idx = idx[ny.hour != 17]
    return pd.DataFrame({"open": px, "high": px + 0.5, "low": px - 0.5, "close": px, "volume": 1.0}, index=idx)


def joined(*parts) -> pd.DataFrame:
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


class Script(Bot):
    """Places the orders listed for given candle times."""
    timeframe = "5m"

    def __init__(self, plan):
        super().__init__("script", {"params": {"instrument": "MGC"}})
        self.plan = plan

    def on_candle(self, ctx, iid):
        for kw in self.plan.get(ctx.ts, []):
            ctx.place(iid, **kw)


def _run(bot, df, start=None, end=None, led=None, eng=None):
    led = led or Ledger(bot.id, 5000.0)
    eng = eng or _engine()
    if hasattr(bot, "prepare"):
        bot.prepare(df)
    eng.run(bot, led, {"MGC": df}, start or df.index[0], end or df.index[-1] + pd.Timedelta(days=1))
    return led, eng


# ------------------------------------------------------------------------------------------ engine
def test_short_round_trip_pnl_fees_and_cash():
    df = bars5("2026-10-14T14:00Z", [[4000, 4001, 3999, 4000]] * 2 + [[3990, 3991, 3989, 3990]] * 3)
    t = df.index
    bot = Script({t[0]: [dict(side="sell", type_="market", qty=2, tag="open short")],
                  t[2]: [dict(side="buy", type_="market", qty=2, tag="cover", reduce_only=True)]})
    led, eng = _run(bot, df, eng=_engine(spread=False))
    sell, buy = eng.fills
    assert (sell.side, sell.qty, sell.price) == ("sell", 2, 4000) and (buy.side, buy.price) == ("buy", 3990)
    assert sell.value_eur == pytest.approx(2 * 4000 * 10 / FX)
    gross = (4000 - 3990) * 2 * 10
    assert buy.pnl_eur == pytest.approx((gross - 4 * FEE) / FX)
    assert not led.positions and led.cash["USD"] == pytest.approx(gross - 4 * FEE)
    assert led.equity(FxRates(FX)) == pytest.approx(5000 + (gross - 4 * FEE) / FX)
    assert led.verify()[0]


def test_unrealised_short_loss_marks_equity_and_end_of_test_buys_back():
    df = bars5("2026-10-14T14:00Z", [[4000, 4001, 3999, 4000]] * 2 + [[4020, 4021, 4019, 4020]] * 3)
    bot = Script({df.index[0]: [dict(side="sell", type_="market", qty=1, tag="short")]})
    led, eng = _run(bot, df, end=df.index[-1], eng=_engine(spread=False))
    assert led.meta["finished"] and not led.positions
    assert eng.fills[-1].tag == "end of test: buy back the short" and eng.fills[-1].price == 4020
    assert led.equity(FxRates(FX)) == pytest.approx(5000 + (-200 - 2 * FEE) / FX)


def test_margin_cap_and_reduce_only():
    df = bars5("2026-10-14T14:00Z", [[4000, 4001, 3999, 4000]] * 4)
    t = df.index
    bot = Script({t[0]: [dict(side="buy", type_="market", qty=5, tag="too big")],
                  t[1]: [dict(side="sell", type_="market", qty=9, tag="close only", reduce_only=True)],
                  t[2]: [dict(side="buy", type_="market", qty=1, tag="reduce while flat", reduce_only=True)]})
    led, eng = _run(bot, df, eng=_engine(spread=False))
    cap = int(15 * 5000 * FX / (4000 * 10))                 # 15x equity / one contract's notional = 2
    assert [(f.tag, f.qty) for f in eng.fills] == [("too big", cap), ("close only", cap)]
    assert not led.positions


def test_same_bar_bracket_protects_from_the_entry_open():
    rows = [[4000, 4000.5, 3999.5, 4000]] * 2 + [[4000, 4001, 3994, 3995], [3995, 3996, 3994, 3995]]

    class Bracket(Bot):
        timeframe = "5m"

        def on_candle(self, ctx, iid):
            if ctx.ts == df.index[1]:
                ctx.place(iid, "buy", "market", qty=1, tag="entry")

        def on_fill(self, ctx, order, fill):
            if order["tag"] == "entry":
                ctx.place(order["inst"], "sell", "stop", qty=1, price=3996, tag="stop", reduce_only=True, same_bar=True)

    df = bars5("2026-10-14T14:00Z", rows)
    led, eng = _run(Bracket("b", {"params": {"instrument": "MGC"}}), df, eng=_engine(spread=False))
    entry, stop = eng.fills
    assert entry.ts == stop.ts == df.index[2].isoformat() and stop.price == 3996 and not led.positions


# ------------------------------------------------------------------------------------------ IFVG rules
def _m15(rows):
    idx = pd.date_range("2026-10-14T00:00Z", periods=len(rows), freq="15min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_first_ifvg_bullish_gap_closed_below_is_a_sell_with_the_swing_high_stop():
    m = _m15([[4000, 4002, 3999.5, 4001.5], [4001.5, 4011, 4001.5, 4010.5], [4011.5, 4014, 4011, 4012.5],
              [4012.5, 4013, 4001, 4001.5]])
    r = fu.first_ifvg(m)
    assert r == {"pos": 3, "side": "sell", "stop": 4014.0, "gap": (4002.0, 4011.0), "formed": 2}


def test_first_ifvg_bearish_gap_closed_above_is_a_buy_and_no_gap_means_none():
    m = _m15([[4010, 4011, 4008, 4008.5], [4008.5, 4008.5, 4000, 4001], [4000.5, 4002, 3996, 3997],
              [3997, 4012, 3997, 4011.5]])
    r = fu.first_ifvg(m)
    assert r["side"] == "buy" and r["stop"] == 3996.0 and r["gap"] == (4002.0, 4008.0)
    assert fu.first_ifvg(_m15([[4000, 4001, 3999, 4000]] * 6)) is None


def test_session_windows():
    T = pd.Timestamp
    assert fu.session_key(T("2026-10-14T00:30Z"), fu.NY, "20:00", "24:00") == "2026-10-13"
    assert fu.session_key(T("2026-10-14T04:00Z"), fu.NY, "20:00", "24:00") is None
    assert fu.session_key(T("2026-10-14T08:45Z"), "Asia/Tokyo", "09:00", "18:00") == "2026-10-14"
    assert fu.session_key(T("2026-10-14T09:00Z"), "Asia/Tokyo", "09:00", "18:00") is None
    assert fu.session_key(T("2026-10-14T07:00Z"), fu.NY, "19:00", "04:00") == "2026-10-13"
    assert fu.next_clock(T("2026-11-02T01:00Z"), fu.NY, "16:55") == T("2026-11-02T21:55Z")   # after DST ends


def _ifvg_day():
    """Quiet market at 4000, then in the 13 Oct Asia session (20:00 NY = 00:00 UTC) a bullish gap
    that the 00:45 candle closes back below (sell), a fall through the 68% mark and the target,
    and later in the same session a second inversion that must not be traded."""
    pre = flat("2026-10-13T18:00Z", "2026-10-14T00:00Z", 4000)
    pat = bars5("2026-10-14T00:00Z", [
        [4000, 4001, 3999.5, 4000.5], [4000.5, 4001.5, 4000, 4001], [4001, 4002, 4000.5, 4001.5],      # b0
        [4001.5, 4005, 4001.5, 4005], [4005, 4009, 4005, 4009], [4009, 4011, 4009, 4010.5],            # b1
        [4011.5, 4013, 4011, 4012.5], [4012.5, 4014, 4012, 4013], [4013, 4013.5, 4012, 4012.5],        # b2: gap 4002-4011
        [4012.5, 4013, 4008, 4008.5], [4008.5, 4009, 4003, 4003.5], [4003.5, 4004, 4001, 4001.5],      # b3 closes below 4002
        [4001.5, 4002, 3995, 3996], [3996, 3996.5, 3992, 3992.5], [3992.5, 3993, 3988, 3988.5]])       # BE, then target
    mid = flat("2026-10-14T01:15Z", "2026-10-14T02:00Z", 3988.5)
    again = bars5("2026-10-14T02:00Z", [
        [3988.5, 3989, 3988, 3988.5]] * 3 + [[3988.5, 3995, 3988.5, 3995]] * 3 + [[3996, 3998, 3996, 3997]] * 3
        + [[3997, 3997, 3980, 3981]] * 3)
    post = flat("2026-10-14T03:00Z", "2026-10-14T23:00Z", 3981)
    return joined(pre, pat, mid, again, post)


def _ifvg_bot(**kw):
    c = load_config()
    spec = json.loads(json.dumps(c["bots"]["gold_ifvg_asia.asia"]))
    spec["params"].update(kw)
    bot = build_bot("gold_ifvg_asia.asia", spec)
    bot.contract = SPEC
    return bot


def test_ifvg_bot_sells_the_first_inversion_moves_to_breakeven_and_takes_the_1r_target():
    df = _ifvg_day()
    bot = _ifvg_bot()
    led, eng = _run(bot, df, start=pd.Timestamp("2026-10-13T18:00Z"))
    f = eng.fills
    assert [x.tag.split(":")[1].split(" ")[0] for x in f] == ["entry", "target"]
    entry, tp = f
    assert entry.side == "sell" and entry.qty == 1 and entry.ts == "2026-10-14T01:00:00+00:00"
    cost = (0.12 + 0.24) / 1e4
    px = 4001.5 * (1 - cost)
    assert entry.price == pytest.approx(px)
    risk = 4014 - px
    assert tp.side == "buy" and tp.price == pytest.approx(px - risk) and tp.liquidity == "maker"
    assert tp.pnl_eur == pytest.approx((risk * 10 - 2 * FEE) / FX)
    assert led.meta["state"]["done"] == "2026-10-13" and not led.positions


def test_ifvg_bot_breakeven_stop_after_68_percent():
    df = _ifvg_day()
    # after the 68% mark the price comes back up through the entry instead of reaching the target
    up = bars5("2026-10-14T01:10Z", [[3992.5, 3993, 3990, 3992], [3992, 4003, 3992, 4002.5]])
    df = joined(df[df.index < pd.Timestamp("2026-10-14T01:10Z")], up,
                flat("2026-10-14T01:20Z", "2026-10-14T23:00Z", 4002.5))
    led, eng = _run(_ifvg_bot(), df, start=pd.Timestamp("2026-10-13T18:00Z"))
    entry, exit_ = eng.fills
    assert exit_.tag == "fut:stop at breakeven" and exit_.side == "buy"
    assert exit_.price == pytest.approx(entry.price * (1 + (0.12 + 0.24) / 1e4))   # stop at entry + slippage
    loss = (exit_.price - entry.price) * 10 + 2 * FEE                                 # slippage and fees only
    assert exit_.pnl_eur == pytest.approx(-loss / FX, abs=1e-3) and loss < 4


def test_ifvg_bot_time_exit_before_the_cme_break():
    df = _ifvg_day()
    df = joined(df[df.index < pd.Timestamp("2026-10-14T01:00Z")], flat("2026-10-14T01:00Z", "2026-10-15T02:00Z", 4001.5))
    led, eng = _run(_ifvg_bot(), df, start=pd.Timestamp("2026-10-13T18:00Z"))
    entry, out = eng.fills
    assert out.tag == "fut:time exit (16:55 New York)" and out.ts == "2026-10-14T20:55:00+00:00"


def test_ifvg_bot_skips_when_one_contract_risks_too_much():
    led, eng = _run(_ifvg_bot(max_risk=0.01), _ifvg_day(), start=pd.Timestamp("2026-10-13T18:00Z"))
    assert not eng.fills and any("skipped" in n for n in eng.notes["gold_ifvg_asia.asia"])


# ------------------------------------------------------------------------------------------ levels
def _level_day(back_inside=True):
    """Prev trading day (13 Oct) range 3990-4010, overnight 3995-4008; at 09:15 NY a candle entirely
    above both highs, at 09:30 a close back below them (buy), then a rise to the 96-tick target."""
    base = flat("2026-10-12T22:00Z", "2026-10-14T13:00Z", 4000)
    spikes = pd.DataFrame({"open": [4000] * 4, "high": [4010, 4000.5, 4008, 4000.5], "low": [3999.5, 3990, 3999.5, 3995],
                           "close": [4000] * 4, "volume": 1.0},
                          index=pd.DatetimeIndex(["2026-10-13T15:00Z", "2026-10-13T16:00Z", "2026-10-14T03:00Z",
                                                  "2026-10-14T06:00Z"]))
    third = [[4013, 4013.5, 4009, 4009.5], [4009.5, 4010, 4007, 4007.5], [4007.5, 4008, 4006, 4006.5]] if back_inside \
        else [[4013, 4014, 4012, 4013]] * 3
    win = bars5("2026-10-14T13:00Z", [
        [4000, 4006, 3999.5, 4005.5], [4005.5, 4009, 4005, 4008.5], [4008.5, 4010.5, 4008, 4010.2],
        [4011, 4013, 4010.6, 4012.5], [4012.5, 4014, 4012, 4013.5], [4013.5, 4014, 4012.2, 4013]] + third + [
        [4006.5, 4010, 4006.5, 4009.5], [4009.5, 4017, 4009.5, 4016.5], [4016.5, 4017, 4016, 4016.5]])
    after = flat("2026-10-14T13:00Z", "2026-10-14T23:00Z", 4016.5)
    return joined(base, spikes, after[after.index >= win.index[-1] + pd.Timedelta(minutes=5)], win)


def _level_bot():
    c = load_config()
    bot = build_bot("gold_level_reentry", json.loads(json.dumps(c["bots"]["gold_level_reentry"])))
    bot.contract = SPEC
    return bot


def test_levels_and_signal():
    df = _level_day()
    day = pd.Timestamp("2026-10-14")
    lv = fu.day_levels(df[df.index < pd.Timestamp("2026-10-14T13:00Z")], day)
    assert lv == {"yesterday's high": 4010.0, "yesterday's low": 3990.0, "overnight high": 4008.0, "overnight low": 3995.0}
    win = fu.to_15m(df[(df.index >= pd.Timestamp("2026-10-14T13:00Z")) & (df.index < pd.Timestamp("2026-10-14T14:00Z"))])
    assert fu.level_signal(win, lv) == {"pos": 2, "side": "buy", "level": "yesterday's high"}


def test_level_bot_buys_the_close_back_inside_with_48_96_tick_bracket():
    led, eng = _run(_level_bot(), _level_day(), start=pd.Timestamp("2026-10-13T00:00Z"))
    entry, tp = eng.fills
    assert entry.side == "buy" and entry.ts == "2026-10-14T13:45:00+00:00"
    assert entry.price == pytest.approx(4006.5 * (1 + (0.12 + 0.24) / 1e4))
    assert tp.tag == "fut:target" and tp.price == pytest.approx(entry.price + 9.6)
    assert tp.pnl_eur == pytest.approx((96 - 2 * FEE) / FX)


def test_level_bot_no_trade_without_the_close_back_inside_and_time_exit():
    led, eng = _run(_level_bot(), _level_day(back_inside=False), start=pd.Timestamp("2026-10-13T00:00Z"))
    assert not eng.fills
    df = _level_day()
    df = joined(df[df.index < pd.Timestamp("2026-10-14T13:45Z")], flat("2026-10-14T13:45Z", "2026-10-14T23:00Z", 4006.5))
    led, eng = _run(_level_bot(), df, start=pd.Timestamp("2026-10-13T00:00Z"))
    assert eng.fills[-1].tag == "fut:time exit (15:30 New York)" and eng.fills[-1].ts == "2026-10-14T19:30:00+00:00"


# ------------------------------------------------------------------------------------------ replay invariance
@pytest.mark.parametrize("make,builder,start", [(_ifvg_day, _ifvg_bot, "2026-10-13T18:00Z"),
                                                (_level_day, _level_bot, "2026-10-13T00:00Z")])
def test_cycles_give_the_same_trades_as_one_pass(make, builder, start):
    df = make()
    start = pd.Timestamp(start)
    _, one = _run(builder(), df, start=start)
    led, eng = Ledger("x", 5000.0), _engine()
    for now in pd.date_range(start + pd.Timedelta(hours=1), df.index[-1] + pd.Timedelta(minutes=20), freq="35min"):
        bot = builder()                                     # rebuilt every cycle, as in the live loop
        seen = df[df.index + TF["5m"] <= now]
        bot.prepare(seen)
        eng.run(bot, led, {"MGC": seen}, start, df.index[-1] + pd.Timedelta(days=1))
    assert [(f.ts, f.side, round(f.price, 6), f.tag) for f in eng.fills] == \
           [(f.ts, f.side, round(f.price, 6), f.tag) for f in one.fills]


# ------------------------------------------------------------------------------------------ runner
class GoldSource:
    used = {}

    def __init__(self, df):
        self.df = df

    def futures(self, symbol, timeframe, since, now, delay):
        d = self.df
        return d[(d.index >= since) & (d.index + TF[timeframe] <= now - delay)]

    def eurusd(self):
        return FX


def test_run_cycle_trades_the_gold_bots(tmp_path):
    cfg = load_config()
    cfg["bots"] = {k: v for k, v in cfg["bots"].items() if k.startswith("gold_")}
    cfg["start"], cfg["end"] = "2026-10-13T18:00:00Z", "2026-10-31T00:00:00Z"
    ifvg = _ifvg_day()
    pat = ifvg[(ifvg.index >= pd.Timestamp("2026-10-14T00:00Z")) & (ifvg.index < pd.Timestamp("2026-10-14T03:00Z"))]
    pat.index = pat.index + pd.Timedelta(days=1)            # the evening session after the level trade
    lvl = _level_day()
    df = joined(lvl[lvl.index < pd.Timestamp("2026-10-15T00:00Z")], pat,
                flat("2026-10-15T03:00Z", "2026-10-15T23:00Z", 3981))
    src = GoldSource(df)
    for now in ["2026-10-14T16:00:00Z", "2026-10-15T01:30:00Z", "2026-10-15T23:30:00Z"]:
        s = run_cycle(tmp_path / "bots", now=pd.Timestamp(now), source=src, cfg=cfg)
        assert not s["errors"], s["errors"]
    fills = pd.read_csv(tmp_path / "bots" / "fills.csv")
    assert set(fills["bot"]) == {"gold_ifvg_asia.asia", "gold_ifvg_asia.tokyo", "gold_level_reentry"}
    assert fills.groupby("bot").size().to_dict() == {"gold_ifvg_asia.asia": 2, "gold_ifvg_asia.tokyo": 2,
                                                      "gold_level_reentry": 2}
    assert (tmp_path / "bots" / "futures_prices.csv").exists()
