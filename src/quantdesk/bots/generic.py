"""Generic, config-driven bots for the strategy families.

A family is one published strategy (one signal or weight function plus its risk rules). Each of
its bots ("variants") runs the same rules, with the published default parameters, on a different
market or timeframe. Four kinds:

  SignalBot          one crypto market: entry/exit signals on candles, Freqtrade-style risk
                     management (stoploss, ROI table, trailing stop, profit lock, time stop)
  IntradayEtfBot     the same on 5-minute US ETF candles, flat by the close
  EtfDailyBot        daily rules on a US ETF, decided 5 minutes before the close on the day's
                     price so far, orders at 15:55 New York time
  WeightBot          a portfolio: target weights from a function of several markets, rebalanced
                     on a schedule (crypto on its candles, US ETFs 5 minutes before the close)

Signals and weights are computed once per cycle on the whole candle history (vectorised and
causal); the bot then reads the value for the candle being replayed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import TF
from .strategies import Bot, UsEtfBot

SIGNALS: dict[str, callable] = {}
WEIGHTS: dict[str, callable] = {}


def signal(name: str):
    def deco(fn):
        SIGNALS[name] = fn
        return fn
    return deco


def weights(name: str):
    def deco(fn):
        WEIGHTS[name] = fn
        return fn
    return deco


FEE_OPEN, FEE_CLOSE = 0.0025, 0.0015       # used only to place ROI targets net of fees, as Freqtrade does


def _truthy(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v)) and bool(v)


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


# ============================================================================================ risk core
class RiskMixin:
    """Position management shared by the single-market signal bots. Spec keys (all optional):
        stoploss: 0.05                  stop 5% below the entry price
        roi: {0: 0.05, 30: 0.02}        Freqtrade minimal_roi (minutes since entry -> profit)
        trailing: {pct: 0.1} | {positive: 0.002, offset: 0.03, only_offset: true}
        profit_lock: [[0.05, 0.04, 1.0], [0.02, 0.02, 0.667]]   [threshold, locked, slope]
        exits: [{col: exit, min_profit: 0}]   signal exits (default: column "exit")
        stop_col / tp_col: price levels from the signal frame, re-read every candle
        bracket: true                   sl_dist / tp_dist columns fixed at the entry signal
        trail_close_pct: 0.05           exit when the close is 5% below the highest close since entry
        size_col: weight                share of equity from the signal row (e.g. volatility target)
        time_stop: 10                   bars after the entry
        cooldown: 96                    bars after an exit before a new entry
        entry: market | stop | limit    (stop/limit at column entry_price)
        entry_valid: 1                  bars a stop/limit entry order stays live
        size: 0.99                      share of equity per entry
    """

    def risk(self, key, default=None):
        return self.spec.get(key, default)

    def _profit_targets(self, st, ts, tf, r):
        """(stop price, take-profit price) for the open position."""
        entry, hi = st["entry_px"], st["hi"]
        stops, tps = [], []
        sl = self.risk("stoploss")
        if sl:
            stops.append(entry * (1 - float(sl)))
        tr = self.risk("trailing")
        if tr:
            if tr.get("pct"):
                stops.append(hi * (1 - float(tr["pct"])))
            elif tr.get("positive") is not None:
                if hi / entry - 1 > float(tr.get("offset", 0.0)):
                    stops.append(hi * (1 - float(tr["positive"])))
                elif not tr.get("only_offset", False) and sl:
                    stops.append(hi * (1 - float(sl)))
        lock = self.risk("profit_lock")
        if lock:
            pmax = hi / entry - 1
            for thr, base, slope in sorted(lock, key=lambda x: -x[0]):
                if pmax > thr:
                    stops.append(entry * (1 + base + slope * (pmax - thr)))
                    break
        if st.get("fixed_sl"):
            stops.append(st["fixed_sl"])
        if st.get("fixed_tp"):
            tps.append(st["fixed_tp"])
        col = self.risk("stop_col")
        if col and r is not None and _num(r.get(col)) is not None:
            stops.append(_num(r.get(col)))
        col = self.risk("tp_col")
        if col and r is not None and _num(r.get(col)) is not None:
            tps.append(_num(r.get(col)))
        roi = self.risk("roi")
        if roi:
            mins = (ts + tf - pd.Timestamp(st["entry_ts"])).total_seconds() / 60
            keys = sorted((float(k), float(v)) for k, v in roi.items())
            cur = None
            for k, v in keys:
                if mins >= k:
                    cur = v
            if cur is not None and cur < 1.0:
                tps.append(entry * (1 + cur + FEE_OPEN) / (1 - FEE_CLOSE))
        stop = max(stops) if stops else None
        tp = min(tps) if tps else None
        return stop, tp

    def manage(self, ctx, iid: str, r, close: float, high: float, tf, allow_entry: bool = True) -> None:
        st = ctx.state.setdefault("pos", {})
        if ctx.qty(iid) <= 0:
            if ctx.orders("sbx:", iid):
                ctx.cancel("sbx:", iid)
            pend = ctx.orders("sbe:", iid)
            if pend:
                age = (ctx.ts - pd.Timestamp(pend[0]["placed"])) / tf
                if age >= int(self.risk("entry_valid", 1)):
                    ctx.cancel("sbe:", iid)
                else:
                    return
            cool = ctx.state.get("cool_until", {}).get(iid)
            if cool and ctx.ts < pd.Timestamp(cool):
                return
            if not allow_entry or r is None or not _truthy(r.get("entry")):
                return
            eur = self.entry_eur(ctx, iid)
            scol = self.risk("size_col")
            if scol and _num(r.get(scol)) is not None:
                eur = min(eur, ctx.equity() * max(0.0, min(1.0, _num(r.get(scol)))) * 0.99)
                if eur < 5.0:
                    return
            kind = self.risk("entry", "market")
            st[iid] = {"sig": {k: _num(r.get(k)) for k in ("sl_dist", "tp_dist", "sl_pct", "tp_pct")}}
            if kind == "market":
                ctx.place(iid, "buy", "market", eur=eur, tag="sbe:buy")
            else:
                px = _num(r.get("entry_price"))
                if px and px > 0:
                    ctx.place(iid, "buy", kind, eur=eur, price=px, tag=f"sbe:buy {kind}")
            return
        p = st.get(iid)
        if not p or "entry_px" not in p:            # position from a fill we did not see (should not happen)
            p = st[iid] = {"entry_px": ctx.avg_price(iid), "entry_ts": ctx.ts.isoformat(), "hi": high, "bars": 0}
        p["bars"] = int(p.get("bars", 0)) + 1
        p["hi"] = max(float(p["hi"]), high)
        p["hic"] = max(float(p.get("hic", p["entry_px"])), close)
        profit = close / p["entry_px"] - 1
        tc = self.risk("trail_close_pct")
        if tc and close <= p["hic"] * (1 - float(tc)):
            ctx.cancel("sbx:", iid)
            ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag=f"sbx:close {float(tc):.0%} below the highest close")
            return
        for ex in self.risk("exits", [{"col": "exit"}]):
            if r is not None and _truthy(r.get(ex["col"])):
                mp = ex.get("min_profit")
                if mp is None or profit > float(mp):
                    ctx.cancel("sbx:", iid)
                    ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag=f"sbx:exit ({ex['col']})")
                    return
        ts_bars = self.risk("time_stop")
        if ts_bars and p["bars"] >= int(ts_bars):
            ctx.cancel("sbx:", iid)
            ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag=f"sbx:time stop ({ts_bars} bars)")
            return
        stop, tp = self._profit_targets(p, ctx.ts, tf, r)
        ctx.cancel("sbx:", iid)
        if stop and stop > 0:
            if stop >= close:                         # the stop is already through the price: exit now
                ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="sbx:stop (through the price)")
                return
            ctx.place(iid, "sell", "stop", qty=ctx.qty(iid), price=stop, tag="sbx:stop")
        if tp and tp > 0:
            ctx.place(iid, "sell", "limit", qty=ctx.qty(iid), price=tp, tag="sbx:take profit")

    def entry_eur(self, ctx, iid) -> float:
        return min(ctx.equity() * float(self.risk("size", 0.99)), ctx.cash_eur() * 0.995)

    def on_fill(self, ctx, order, fill) -> None:
        iid = order["inst"]
        st = ctx.state.setdefault("pos", {})
        if order["side"] == "buy":
            p = st.setdefault(iid, {})
            if "entry_px" not in p:
                p.update({"entry_px": fill.price, "entry_ts": ctx.ts.isoformat(), "hi": fill.price, "bars": 0})
                sig = p.get("sig") or {}
                if self.risk("bracket"):
                    if sig.get("sl_dist"):
                        p["fixed_sl"] = fill.price - sig["sl_dist"]
                    if sig.get("tp_dist"):
                        p["fixed_tp"] = fill.price + sig["tp_dist"]
                    if sig.get("sl_pct"):
                        p["fixed_sl"] = fill.price * (1 - sig["sl_pct"])
                    if sig.get("tp_pct"):
                        p["fixed_tp"] = fill.price * (1 + sig["tp_pct"])
            else:                                       # scale-in: blend the entry price
                p["entry_px"] = ctx.avg_price(iid) or p["entry_px"]
        elif ctx.qty(iid) <= 1e-12:
            ctx.cancel("sbx:", iid)
            ctx.cancel("sbe:", iid)
            st.pop(iid, None)
            cd = self.risk("cooldown")
            if cd:
                ctx.state.setdefault("cool_until", {})[iid] = (ctx.ts + int(cd) * self.tf).isoformat()


def _records(frame: pd.DataFrame) -> dict:
    """{timestamp: {column: value}} for fast lookup during the replay."""
    if frame is None or frame.empty:
        return {}
    frame = frame.copy()
    for c in frame.columns:
        if frame[c].dtype == bool:
            frame[c] = frame[c].astype(object)
    return frame.to_dict("index")


# ============================================================================================ crypto single market
class SignalBot(RiskMixin, Bot):
    """One crypto market, entries and exits from a signal function, Freqtrade-style risk rules."""

    def __init__(self, bot_id, spec):
        super().__init__(bot_id, spec)
        self.sig: dict = {}
        self.tf = TF[self.timeframe]

    def warmup(self) -> int:
        return 0

    def informative(self) -> list[str]:
        return list(self.spec.get("informative", []))

    def prepare(self, df: pd.DataFrame, info: dict[str, pd.DataFrame]) -> None:
        fn = SIGNALS[self.spec["signal"]]
        out = fn(df, dict(self.p), info) if len(df) else pd.DataFrame()
        self.sig = _records(out)

    def on_candle(self, ctx, iid: str) -> None:
        bar = ctx.candle(iid)
        if bar is None:
            return
        self.manage(ctx, iid, self.sig.get(ctx.ts), float(bar["close"]), float(bar["high"]), self.tf)


# ============================================================================================ US ETFs
class IntradayEtfBot(RiskMixin, UsEtfBot):
    """Signal rules on 5-minute US candles; no new entries in the last 15 minutes, flat at 15:55."""

    def __init__(self, bot_id, spec):
        super().__init__(bot_id, spec)
        self.sig: dict = {}
        self.tf = TF["5m"]

    def prepare_intraday(self, candles: dict[str, pd.DataFrame]) -> None:
        fn = SIGNALS[self.spec["signal"]]
        iid = self.instruments()[0]
        df = candles.get(iid)
        extra = {k: v for k, v in candles.items() if k != iid}
        extra["daily"] = self.daily
        extra["sessions"] = self.sessions
        self.sig = _records(fn(df, dict(self.p), extra)) if df is not None and len(df) else {}

    def on_bar(self, ctx, bars: dict) -> None:
        sess = self._session(ctx.ts)
        if sess is None:
            return
        tf = self.tf
        iid = self.instruments()[0]
        if ctx.ts + 2 * tf == sess["close"]:            # 15:50 candle: go flat at 15:55
            ctx.cancel("sbe:", iid)
            ctx.cancel("sbx:", iid)
            if ctx.qty(iid) > 0:
                ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="sbx:eod flat at the close")
            return
        if ctx.ts + 2 * tf > sess["close"]:
            return
        bar = bars.get(iid)
        if bar is None:
            return
        allow = ctx.ts + 4 * tf <= sess["close"]
        self.manage(ctx, iid, self.sig.get(ctx.ts), float(bar["close"]), float(bar["high"]), tf, allow)


def proxy_daily(daily: pd.DataFrame | None, intraday: pd.DataFrame | None, sess, upto: pd.Timestamp) -> pd.DataFrame:
    """Daily candles known at `upto` (the close of the 15:50 candle): real daily bars before the
    session's date plus today's bar so far from the 5-minute candles."""
    day = sess["open"].tz_convert("America/New_York").normalize().tz_localize(None)
    past = daily[daily.index < day] if daily is not None else pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if intraday is None or intraday.empty:
        return past
    today = intraday[(intraday.index >= sess["open"]) & (intraday.index < upto)]
    if today.empty:
        return past
    row = pd.DataFrame({"open": [float(today["open"].iloc[0])], "high": [float(today["high"].max())],
                        "low": [float(today["low"].min())], "close": [float(today["close"].iloc[-1])],
                        "volume": [float(today["volume"].sum())]}, index=[day])
    return pd.concat([past[["open", "high", "low", "close", "volume"]], row])


class EtfDailyBot(RiskMixin, UsEtfBot):
    """Daily rules on one US ETF, decided at 15:50-15:55 New York time on the day's price so far."""
    needs_daily = True

    def __init__(self, bot_id, spec):
        super().__init__(bot_id, spec)
        self.tf = pd.Timedelta(days=1)
        self.memo: dict = {}

    def daily_signal(self, ctx, sess, day: str) -> dict | None:
        if day in self.memo:
            return self.memo[day]
        iid = self.instruments()[0]
        sig_t = self.p.get("signal_ticker")
        ticker = sig_t or iid
        d = proxy_daily(self.daily.get(ticker), ctx._hist.get(ticker), sess, ctx.ts + TF["5m"])
        if len(d) < 3:
            return None
        extra = {"daily": self.daily, "intraday": ctx._hist, "sess": sess, "day": day, "proxy": proxy_daily,
                 "upto": ctx.ts + TF["5m"]}
        out = SIGNALS[self.spec["signal"]](d.tail(int(self.spec.get("lookback", 400))), dict(self.p), extra)
        r = out.iloc[-1].to_dict() if len(out) else None
        self.memo[day] = r
        return r

    def on_close(self, ctx, bars: dict, sess, day: str) -> None:
        iid = self.instruments()[0]
        bar = bars.get(iid)
        if bar is None:
            return
        r = self.daily_signal(ctx, sess, day)
        # the daily "candle" for risk management is today's bar so far
        today = ctx._hist.get(iid)
        hi = float(today[(today.index >= sess["open"])]["high"].max()) if today is not None else float(bar["high"])
        self.manage(ctx, iid, r, float(bar["close"]), hi, self.tf)
        self.after_close(ctx, iid, r, float(bar["close"]))

    def after_close(self, ctx, iid, r, close) -> None: ...

    def data_tickers(self) -> list[str]:
        t = super().data_tickers()
        st = self.p.get("signal_ticker")
        return list(dict.fromkeys(t + ([st] if st else []) + list(self.p.get("extra_tickers", []))))


# ============================================================================================ portfolios
class Panel:
    """Aligned wide frames of several markets."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        idx = sorted(set().union(*[set(df.index) for df in frames.values()])) if frames else []
        self.index = pd.DatetimeIndex(idx)
        for col in ("open", "high", "low", "close", "volume"):
            setattr(self, col, pd.DataFrame({k: df[col] for k, df in frames.items()}).reindex(self.index))

    @property
    def names(self) -> list[str]:
        return list(self.frames)


class WeightBot(Bot):
    """Target weights from a function of several markets, rebalanced on a schedule. Orders are market
    orders; sales are filled before purchases within a candle. Spec keys:
        weights: function name; universe: list of instruments; schedule: every | daily | weekly | monthly
        | every_n_days; band: minimum weight change to trade (default 0.02)
    """

    def __init__(self, bot_id, spec):
        super().__init__(bot_id, spec)
        self.w: dict = {}
        self.tf = TF[self.timeframe] if self.timeframe in TF else pd.Timedelta(days=1)

    def instruments(self) -> list[str]:
        return list(self.p.get("universe") or self.spec.get("universe") or [])

    def warmup(self) -> int:
        return 0

    def prepare(self, candles: dict[str, pd.DataFrame]) -> None:
        panel = Panel({k: v for k, v in candles.items() if v is not None and len(v)})
        extra = {"tf": self.tf, "_from": getattr(self, "from_ts", None)}
        out = WEIGHTS[self.spec["weights"]](panel, dict(self.p), extra) if panel.frames else pd.DataFrame()
        self.w = _records(out.fillna(0.0)) if len(out) else {}

    def on_schedule(self, ts_close: pd.Timestamp) -> bool:
        sch = self.spec.get("schedule", "every")
        if sch == "every":
            return True
        if sch == "daily":
            return ts_close.hour == int(self.spec.get("hour", 0)) and ts_close.minute == 0
        if sch == "weekly":
            return ts_close.weekday() == int(self.spec.get("weekday", 6)) and ts_close.hour == int(self.spec.get("hour", 0)) \
                and ts_close.minute == 0
        if sch == "every_n_days":
            n = int(self.spec["n_days"])
            days = (ts_close.normalize() - pd.Timestamp("2026-01-04", tz="UTC")).days   # a Sunday anchor
            return days % n == 0 and ts_close.hour == int(self.spec.get("hour", 0)) and ts_close.minute == 0
        return False

    def on_bar(self, ctx, bars: dict) -> None:
        end = ctx.ts + self.tf
        first = not ctx.state.get("started")
        if not (first or self.on_schedule(end)):
            return
        w = self.w.get(ctx.ts)
        if w is None:
            return
        px = {i: float(bars[i]["close"]) for i in self.instruments() if i in bars}
        self.rebalance(ctx, w, px)
        ctx.state["started"] = True

    def rebalance(self, ctx, w: dict, px: dict) -> None:
        eq = ctx.equity()
        band = float(self.spec.get("band", 0.02))
        tot = sum(max(0.0, float(v)) for v in w.values())
        scale = 0.985 / tot if tot > 0.985 else 1.0
        target = {i: max(0.0, float(w.get(i, 0.0))) * scale for i in px}
        proceeds, buys = 0.0, {}
        held = {i for i, p in ctx.ledger.positions.items() if p.qty > 0}
        for i in sorted(set(px) | held):
            if i not in px:
                continue
            cur = ctx.qty(i) * px[i]
            want = target.get(i, 0.0) * eq
            diff = want - cur
            full_exit = want <= 1e-9 and cur >= 5.0
            if not full_exit and abs(diff) < max(5.0, band * eq):
                continue
            if diff < 0:
                q = ctx.qty(i) if want <= 1e-9 else min(ctx.qty(i), -diff / px[i])
                ctx.place(i, "sell", "market", qty=q, tag="wb:sell (rebalance)")
                proceeds += min(cur, -diff)
            elif diff > 0:
                buys[i] = diff
        budget = (ctx.cash_eur() + proceeds * 0.995) * 0.997
        need = sum(buys.values())
        k = min(1.0, budget / need) if need > 0 else 0.0
        for i, eur in buys.items():
            if eur * k >= 5.0:
                ctx.place(i, "buy", "market", eur=eur * k, tag="wb:buy (rebalance)")
        ctx.state["last_target"] = {i: round(v, 4) for i, v in target.items() if v > 0}


class EtfWeightBot(UsEtfBot):
    """Target weights for US ETFs, decided 5 minutes before the close on the day's prices so far
    and traded at 15:55 New York time. schedule: daily | weekly (last session of the week) |
    monthly (last session of the month) | first_day (first session of the month)."""
    needs_daily = True

    def __init__(self, bot_id, spec):
        super().__init__(bot_id, spec)
        self.memo: dict = {}

    def instruments(self) -> list[str]:
        return list(self.p.get("universe") or [])

    def data_tickers(self) -> list[str]:
        return list(dict.fromkeys(self.instruments() + list(self.p.get("signal_tickers", []))))

    def is_rebalance_day(self, day: str) -> bool:
        sch = self.spec.get("schedule", "daily")
        if sch == "daily":
            return True
        d = pd.Timestamp(day)
        idx = self.sessions.index
        if sch == "weekly":
            nxt = idx[idx > d]
            return len(nxt) == 0 or nxt[0].isocalendar()[1] != d.isocalendar()[1]
        if sch == "monthly":
            nth, left = self.trading_day_position(day)
            return left == 0
        if sch == "first_day":
            nth, left = self.trading_day_position(day)
            return nth == 1
        return False

    def on_close(self, ctx, bars: dict, sess, day: str) -> None:
        first = not ctx.state.get("started")
        if not (first or self.is_rebalance_day(day) or self.spec.get("schedule") == "calendar"):
            return
        tickers = self.data_tickers()
        frames = {t: proxy_daily(self.daily.get(t), ctx._hist.get(t), sess, ctx.ts + TF["5m"]) for t in tickers}
        frames = {t: d.tail(int(self.spec.get("lookback", 600))) for t, d in frames.items() if len(d)}
        if not frames:
            return
        panel = Panel(frames)
        extra = {"day": day, "sessions": self.sessions, "pos": self.trading_day_position(day), "bot": self}
        out = WEIGHTS[self.spec["weights"]](panel, dict(self.p), extra)
        if out is None or len(out) == 0:
            return
        w = out.iloc[-1].fillna(0.0).to_dict()
        px = {i: float(bars[i]["close"]) for i in self.instruments() if i in bars}
        WeightBot.rebalance(self, ctx, w, px)
        ctx.state["started"] = True
