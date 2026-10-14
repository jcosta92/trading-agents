"""Gold-futures day-trading strategies (long and short), traded in whole COMEX Micro Gold contracts.

Both come from r/Daytrading posts on 12 and 14 Oct 2026 and are implemented from the authors' own
words (post text plus their replies in the comments); where the words leave a choice open, the
choice is written in the family's rules on the dashboard.

  GoldIfvgBot   u/Current_Heron3284: "I trade gold during Asia and take the first newly created 15
                minute IFVG I get. I target 1:1RR and move to breakeven once price gets about 68% of
                the way to TP." (gold futures, 15-minute candles, one trade a day)
  GoldLevelBot  u/tiffarooner: yesterday's high/low and the overnight high/low on gold futures; a
                15-minute candle closes fully outside a level, then the first candle that closes back
                inside is the entry (upper level: buy, lower level: sell); target 96 ticks, stop 48
                ticks; trades 9-11 am; anything still open is closed at 3:30 pm.

Signals use 15-minute candles built from the 5-minute candles; stops, targets and the breakeven move
are managed on the 5-minute candles. A 15-minute candle is only used once it has closed.
"""
from __future__ import annotations

import math

import pandas as pd

from ..data import TF
from ..strategies import Bot

NY = "America/New_York"
M15 = pd.Timedelta(minutes=15)
TAG = "fut:"


def to_15m(df5: pd.DataFrame) -> pd.DataFrame:
    """15-minute candles from 5-minute ones (open time index) with `done`, the time the candle is
    complete (its open + 15 minutes)."""
    if df5 is None or df5.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "done"])
    g = df5.groupby(df5.index.floor("15min"))
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                        "close": g["close"].last()})
    out["done"] = out.index + M15
    return out


def _hhmm(s: str) -> pd.Timedelta:
    h, m = s.split(":")
    return pd.Timedelta(hours=int(h), minutes=int(m))


def _wall(ts: pd.Timestamp, tz: str) -> pd.Timestamp:
    """Local wall-clock time (naive) of a UTC timestamp."""
    return ts.tz_convert(tz).tz_localize(None)


def _at(day: pd.Timestamp, hhmm: str, tz: str) -> pd.Timestamp:
    """UTC time at which the clock in `tz` shows hhmm on the (naive) local date `day`."""
    return (day.normalize() + _hhmm(hhmm)).tz_localize(tz, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC")


def session_key(ts: pd.Timestamp, tz: str, start: str, end: str) -> str | None:
    """Local date on which the session containing a candle opening at `ts` starts, or None outside
    it. The window [start, end) may end at 24:00 or run past midnight (e.g. 19:00-04:00)."""
    w = _wall(ts, tz)
    tod = w - w.normalize()
    a, b = _hhmm(start), _hhmm(end)
    if a < b:
        return w.strftime("%Y-%m-%d") if a <= tod < b else None
    if tod >= a:
        return w.strftime("%Y-%m-%d")
    if tod < b:
        return (w.normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return None


def next_clock(ts: pd.Timestamp, tz: str, hhmm: str) -> pd.Timestamp:
    """The first time at or after `ts` (UTC) at which the clock in `tz` shows hhmm."""
    w = _wall(ts, tz)
    t = _at(w.normalize(), hhmm, tz)
    return t if t >= ts else _at(w.normalize() + pd.Timedelta(days=1), hhmm, tz)


# ------------------------------------------------------------------------------------- IFVG
def first_ifvg(bars: pd.DataFrame) -> dict | None:
    """The first inversion of a fair value gap formed within `bars` (one session of 15-minute
    candles, in order).

    Fair value gap formed by candles k-2, k-1, k: bullish if low[k] > high[k-2] (gap high[k-2]..low[k]),
    bearish if high[k] < low[k-2] (gap high[k]..low[k-2]). It inverts when a later candle CLOSES
    beyond its far side: a bullish gap closed below its bottom becomes a bearish IFVG (sell); a
    bearish gap closed above its top becomes a bullish IFVG (buy). Returned: position of the
    inverting candle, side, and the stop = the highest high (sell) / lowest low (buy) from the gap's
    first candle to the inverting candle (the swing the inversion came from). If several gaps invert
    on the same candle, the most recently formed one counts."""
    h, lo, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    gaps: list[tuple[int, str, float, float]] = []
    for j in range(len(c)):
        hits = [g for g in gaps if (g[1] == "bull" and c[j] < g[2]) or (g[1] == "bear" and c[j] > g[3])]
        if hits:
            k, kind, bot, top = max(hits, key=lambda g: g[0])
            side = "sell" if kind == "bull" else "buy"
            seg = slice(k - 2, j + 1)
            stop = float(h[seg].max()) if side == "sell" else float(lo[seg].min())
            return {"pos": j, "side": side, "stop": stop, "gap": (bot, top), "formed": k}
        if j >= 2:
            if lo[j] > h[j - 2]:
                gaps.append((j, "bull", float(h[j - 2]), float(lo[j])))
            elif h[j] < lo[j - 2]:
                gaps.append((j, "bear", float(h[j]), float(lo[j - 2])))
    return None


# ------------------------------------------------------------------------------------- levels
def trading_day(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """CME trading day of each candle: the Globex day runs 18:00-17:00 New York time, so candles from
    18:00 belong to the next day (naive dates)."""
    return (idx.tz_convert(NY).tz_localize(None) + pd.Timedelta(hours=6)).normalize()


def day_levels(df: pd.DataFrame, day: pd.Timestamp, overnight_end: str = "09:00") -> dict:
    """Yesterday's high/low (previous CME trading day) and the overnight high/low (this trading day's
    candles before `overnight_end` New York time)."""
    if df.empty:
        return {}
    td = trading_day(df.index)
    days = sorted(set(td[td < day]))
    out = {}
    if days:
        prev = df[td == days[-1]]
        out["yesterday's high"], out["yesterday's low"] = float(prev["high"].max()), float(prev["low"].min())
    cut = _at(day, overnight_end, NY)
    night = df[(td == day) & (df.index < cut)]
    if len(night):
        out["overnight high"], out["overnight low"] = float(night["high"].max()), float(night["low"].min())
    return out


def level_signal(bars: pd.DataFrame, levels: dict) -> dict | None:
    """First entry in `bars` (the day's 15-minute candles inside the trading window, in order).
    A candle entirely above an upper level (low > level) arms it; the first later candle that closes
    back below it is a BUY. A candle entirely below a lower level (high < level) arms it; the first
    later close back above it is a SELL ("If it's the upper limit it's a buy order, if it's a lower
    limit it's a sell order"). A candle with both a buy and a sell trigger is skipped."""
    hi, lo, c = (bars[k].to_numpy(float) for k in ("high", "low", "close"))
    armed: dict[str, int] = {}
    for j in range(len(c)):
        buys = [n for n, a in armed.items() if a < j and "high" in n and c[j] < levels[n]]
        sells = [n for n, a in armed.items() if a < j and "low" in n and c[j] > levels[n]]
        if buys and not sells:
            return {"pos": j, "side": "buy", "level": buys[0]}
        if sells and not buys:
            return {"pos": j, "side": "sell", "level": sells[0]}
        for n, v in levels.items():
            if n in armed:
                continue
            if ("high" in n and lo[j] > v) or ("low" in n and hi[j] < v):
                armed[n] = j
    return None


# ------------------------------------------------------------------------------------- base bot
class FuturesBot(Bot):
    """One gold-futures trade at a time, in whole contracts, long or short. Risk per trade: `risk`
    (default 1%) of equity at the stop, at least one contract; with only EUR 5,000 one Micro Gold
    contract often risks more than that, and a trade whose one-contract risk is above `max_risk`
    (default 5%) is skipped."""
    timeframe = "5m"
    history_days = 5

    def __init__(self, bot_id: str, spec: dict):
        super().__init__(bot_id, spec)
        self.contract = {"tick": 0.10, "multiplier": 10}
        self.m15 = to_15m(None)

    def instruments(self) -> list[str]:
        return [self.p.get("instrument", "MGC")]

    def warmup(self) -> int:
        return 1

    def prepare(self, df5: pd.DataFrame) -> None:
        self.df5 = df5
        self.m15 = to_15m(df5)

    @property
    def mult(self) -> float:
        return float(self.contract.get("multiplier", 10))

    def size(self, ctx, risk_per_contract_usd: float) -> int:
        if not (risk_per_contract_usd > 0):
            return 0
        eq_usd = ctx.engine.fx.from_eur(ctx.equity(), "USD")
        n = int(math.floor(float(self.p.get("risk", 0.01)) * eq_usd / risk_per_contract_usd + 1e-9))
        if n < 1 and risk_per_contract_usd <= float(self.p.get("max_risk", 0.05)) * eq_usd:
            n = 1
        return n

    # 15-minute candles completed since the last look
    def new_15m(self, ctx) -> pd.DataFrame:
        st = ctx.state
        close_ts = ctx.ts + TF[self.timeframe]
        last = pd.Timestamp(st["last15"]) if st.get("last15") else ctx.ts.floor("15min") - M15
        fresh = self.m15[(self.m15.index > last) & (self.m15["done"] <= close_ts)]
        if len(fresh):
            st["last15"] = fresh.index[-1].isoformat()
        return fresh

    def enter(self, ctx, iid: str, side: str, n: int, plan: dict, why: str) -> None:
        ctx.state["trade"] = {**plan, "side": side}
        ctx.place(iid, side, "market", qty=n, tag=f"{TAG}entry {side}: {why}")

    def on_candle(self, ctx, iid: str) -> None:
        bar = ctx.candle(iid)
        if bar is None:
            return
        if abs(ctx.qty(iid)) > 1e-9:
            self.manage(ctx, iid, bar)
        elif ctx.state.get("trade") and not ctx.orders(TAG + "entry", iid):
            ctx.state.pop("trade", None)                 # entry never filled (no margin): forget it
            ctx.cancel(TAG, iid)
        for b, row in self.new_15m(ctx).iterrows():
            if abs(ctx.qty(iid)) > 1e-9 or ctx.orders(TAG + "entry", iid):
                continue
            self.on_15m(ctx, iid, b, row)

    def on_15m(self, ctx, iid: str, b: pd.Timestamp, row) -> None: ...

    def on_fill(self, ctx, order, fill) -> None:
        iid = order["inst"]
        t = ctx.state.get("trade")
        if order["tag"].startswith(TAG + "entry") and t is not None:
            sgn = 1.0 if order["side"] == "buy" else -1.0
            entry = fill.price
            if t.get("stop") is not None:                # stop at a chart level, target at rr x risk
                risk = (entry - float(t["stop"])) * sgn
                stop, target = float(t["stop"]), entry + sgn * risk * float(t.get("rr", 1.0))
            else:                                        # fixed distances in ticks
                risk = float(t["sl_ticks"]) * float(self.contract.get("tick", 0.10))
                stop = entry - sgn * risk
                target = entry + sgn * float(t["tp_ticks"]) * float(self.contract.get("tick", 0.10))
            t.update({"entry": entry, "risk": risk, "stop_px": stop, "target": target, "filled": ctx.ts.isoformat(),
                      "contracts": fill.qty})
            if t.get("be_frac"):
                t["be_at"] = entry + sgn * risk * float(t.get("rr", 1.0)) * float(t["be_frac"])
            exit_side = "sell" if sgn > 0 else "buy"
            n = abs(ctx.qty(iid))
            if risk <= 0:                                # opened beyond its own stop: close at once
                ctx.place(iid, exit_side, "market", qty=n, tag=f"{TAG}exit: opened beyond the stop",
                          reduce_only=True, same_bar=True)
                return
            ctx.place(iid, exit_side, "stop", qty=n, price=stop, tag=f"{TAG}stop", reduce_only=True, same_bar=True)
            ctx.place(iid, exit_side, "limit", qty=n, price=target, tag=f"{TAG}target", reduce_only=True,
                      same_bar=True)
            return
        if abs(ctx.qty(iid)) <= 1e-9:                    # flat again: the bracket's other leg goes
            ctx.cancel(TAG, iid)
            if t is not None:
                ctx.state["last_trade"] = {"side": t.get("side"), "entry": t.get("entry"), "exit": fill.price,
                                           "why": order["tag"], "day": t.get("day")}
            ctx.state.pop("trade", None)

    def manage(self, ctx, iid: str, bar) -> None:
        t = ctx.state.get("trade")
        if not t or "entry" not in t:
            return
        held = ctx.qty(iid)
        exit_side = "sell" if held > 0 else "buy"
        if t.get("exit_at") and ctx.ts + TF[self.timeframe] >= pd.Timestamp(t["exit_at"]):
            ctx.cancel(TAG, iid)
            ctx.place(iid, exit_side, "market", qty=abs(held), tag=f"{TAG}time exit ({t.get('exit_label', '')})",
                      reduce_only=True)
            t.pop("exit_at", None)
            return
        be = t.get("be_at")
        if be is not None and not t.get("at_breakeven"):
            reached = float(bar["high"]) >= be if held > 0 else float(bar["low"]) <= be
            if reached:
                ctx.cancel(TAG + "stop", iid)
                ctx.place(iid, exit_side, "stop", qty=abs(held), price=float(t["entry"]),
                          tag=f"{TAG}stop at breakeven", reduce_only=True)
                t["at_breakeven"] = True


# ------------------------------------------------------------------------------------- strategies
class GoldIfvgBot(FuturesBot):
    """u/Current_Heron3284 (r/Daytrading, 12 Oct 2026): gold futures, 15-minute candles, the first
    inverse fair value gap created in the Asia session, 1:1 target, stop to breakeven at 68% of the
    way to the target, one trade a day. Params: session tz/start/end, rr, be_frac, exit (New York)."""

    def on_15m(self, ctx, iid: str, b: pd.Timestamp, row) -> None:
        tz, a, z = self.p.get("tz", NY), self.p.get("start", "20:00"), self.p.get("end", "24:00")
        key = session_key(b, tz, a, z)
        if key is None or ctx.state.get("done") == key:
            return
        sess = self.m15[(self.m15.index > b - pd.Timedelta(days=1)) & (self.m15.index <= b)]
        sess = sess[[session_key(t, tz, a, z) == key for t in sess.index]]
        res = first_ifvg(sess)
        if res is None:
            return
        ctx.state["done"] = key                          # one IFVG per session, taken or not
        if res["pos"] != len(sess) - 1:
            return                                       # its inversion happened before this bot ran
        close = float(row["close"])
        side = res["side"]
        risk_pt = (close - res["stop"]) if side == "buy" else (res["stop"] - close)
        n = self.size(ctx, risk_pt * self.mult)
        if n < 1:
            ctx.note(f"IFVG {side} skipped: one contract would risk USD {risk_pt * self.mult:,.0f}")
            return
        exit_at = next_clock(ctx.ts + TF[self.timeframe], NY, self.p.get("exit", "16:55"))
        plan = {"day": key, "stop": res["stop"], "rr": float(self.p.get("rr", 1.0)),
                "be_frac": float(self.p.get("be_frac", 0.68)), "exit_at": exit_at.isoformat(),
                "exit_label": f"{self.p.get('exit', '16:55')} New York"}
        gap = f"{res['gap'][0]:.1f}-{res['gap'][1]:.1f}"
        self.enter(ctx, iid, side, n, plan, f"first IFVG of the session (gap {gap} closed through), "
                                            f"stop {res['stop']:.1f}")


class GoldLevelBot(FuturesBot):
    """u/tiffarooner (r/Daytrading, 14 Oct 2026): gold futures, 15-minute candles, 9:00-11:00 New
    York. Levels: yesterday's high/low and the overnight high/low. A candle closes fully outside a
    level, then the first candle closing back inside is the entry (upper level buy, lower level sell).
    Target 96 ticks, stop 48 ticks, close at 15:30; one trade a day."""

    def on_15m(self, ctx, iid: str, b: pd.Timestamp, row) -> None:
        start, end = self.p.get("start", "09:00"), self.p.get("end", "11:00")
        key = session_key(b, NY, start, end)
        if key is None or ctx.state.get("done") == key:
            return
        day = pd.Timestamp(key)
        w0 = _at(day, start, NY)
        lv = day_levels(self.df5[self.df5.index < w0], day, start)
        if not lv:
            return
        win = self.m15[(self.m15.index >= w0) & (self.m15.index <= b)]
        res = level_signal(win, lv)
        if res is None:
            return
        ctx.state["done"] = key
        if res["pos"] != len(win) - 1:
            return
        sl = float(self.p.get("sl_ticks", 48))
        n = self.size(ctx, sl * float(self.contract.get("tick", 0.10)) * self.mult)
        if n < 1:
            ctx.note("level entry skipped: one contract risks too much")
            return
        exit_at = _at(day, self.p.get("exit", "15:30"), NY)
        plan = {"day": key, "stop": None, "sl_ticks": sl, "tp_ticks": float(self.p.get("tp_ticks", 96)),
                "exit_at": exit_at.isoformat(), "exit_label": f"{self.p.get('exit', '15:30')} New York"}
        word = "back below" if res["side"] == "buy" else "back above"
        self.enter(ctx, iid, res["side"], n, plan,
                   f"closed {word} {res['level']} {lv[res['level']]:.1f} after a candle fully outside it")
