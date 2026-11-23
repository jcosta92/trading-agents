"""QQQ noise-band + opening-range breakout traded with 1-day QQQ calls and puts (added 23 Nov 2026).

Rules (from a social-media post the owner sent; the noise band is the one of Zarattini, Aziz & Barbon
2024, 'Beat the Market'):
  1. Opening range: high and low of QQQ from 9:30 to 9:45 New York time.
  2. Noise band: for each of the last 20 sessions, |close / that day's open - 1| at every minute; the
     average per minute of the day x 1.25 = sigma(m). Upper = max(open, yesterday's close) x (1 + sigma),
     lower = min(open, yesterday's close) x (1 - sigma).
  3. ATR: 5-day average true range of QQQ (previous 5 sessions, from the stored 1-minute candles).
  4. No trade before 10:00; from 10:00 to 14:00 check every 10 minutes on the 1-minute close.
  5. Calls when close > OR high + 0.3 ATR, > upper band and > VWAP.
  6. Puts when close < OR low - 0.3 ATR, < lower band and < VWAP.
  7. One trade a day: the strike nearest the price, expiring the next session (1 trading day out),
     limit order at the mid (the option's last 1-minute close), valid until the next check.
  8. Exits on QQQ's 1-minute closes: after a favourable move of 4 ATR, sell when half of the best move
     is given back; emergency stop at 10 ATR against; otherwise sell at 15:45 (15 minutes before the
     close on half days).
  9. Contracts per trade from the config (params.contracts); every signal, buy and sell is logged to
     state/bots/options/<bot>.csv.

VWAP: session VWAP of the 1-minute typical price; Yahoo's 1-minute volumes contain glitches (single
minutes at ~100x the norm), so each minute's volume is capped at 20x the median minute of the previous
20 sessions.

Data: QQQ and option 1-minute candles from Yahoo (option prints are delayed ~15 minutes, so the bot
only processes minutes that closed at least `data_delay_minutes` ago). Yahoo keeps 1-minute data for
30 days, so QQQ's candles are also stored in state/bots/options/QQQ_1m.csv for the 20-day band.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import option_id, option_instrument, stock_instrument
from ..strategies import Bot

NY = "America/New_York"
LOG_COLS = ["ts_utc", "ts_ny", "event", "right", "contract", "qty", "price", "qqq_close", "or_high", "or_low",
            "atr", "upper", "lower", "vwap", "detail"]


def yahoo_option_symbol(underlying: str, expiry: str, right: str, strike: float) -> str:
    return f"{underlying}{pd.Timestamp(expiry):%y%m%d}{right}{int(round(strike * 1000)):08d}"


def _ny_minutes(idx: pd.DatetimeIndex) -> np.ndarray:
    ny = idx.tz_convert(NY)
    return (ny.hour * 60 + ny.minute).to_numpy()


def session_frames(m1: pd.DataFrame) -> dict[pd.Timestamp, pd.DataFrame]:
    """Regular-session 1-minute candles grouped by New York date."""
    if m1.empty:
        return {}
    mins = _ny_minutes(m1.index)
    reg = m1[(mins >= 570) & (mins < 960)]
    days = reg.index.tz_convert(NY).tz_localize(None).normalize()
    return {d: g for d, g in reg.groupby(days)}


def daily_bars(days: dict) -> pd.DataFrame:
    rows = {d: {"open": float(g["open"].iloc[0]), "high": float(g["high"].max()), "low": float(g["low"].min()),
                "close": float(g["close"].iloc[-1])} for d, g in days.items() if len(g)}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def atr(daily: pd.DataFrame, n: int = 5) -> float | None:
    """Simple average of the last n true ranges of `daily` (sessions strictly before the trading day)."""
    if len(daily) < n + 1:
        return None
    prev = daily["close"].shift(1)
    tr = pd.concat([daily["high"] - daily["low"], (daily["high"] - prev).abs(), (daily["low"] - prev).abs()],
                   axis=1).max(axis=1)
    return float(tr.iloc[-n:].mean())


def noise_sigma(days: dict, before: pd.Timestamp, lookback: int = 20, mult: float = 1.25) -> pd.Series | None:
    """sigma per minute of the day (minutes since 9:30, candle open), averaged over the `lookback`
    sessions before `before`, times `mult`."""
    prior = sorted(d for d in days if d < before)[-lookback:]
    if len(prior) < lookback:
        return None
    cols = []
    for d in prior:
        g = days[d]
        mv = (g["close"] / float(g["open"].iloc[0]) - 1.0).abs()
        mv.index = _ny_minutes(g.index) - 570
        cols.append(mv[~mv.index.duplicated()])
    return pd.concat(cols, axis=1).mean(axis=1, skipna=True) * mult


def signal_table(m1: pd.DataFrame, p: dict, from_day: pd.Timestamp) -> pd.DataFrame:
    """One row per 10-minute check (10:00-14:00 NY) of every session from `from_day` on, indexed by the
    open time (UTC) of the 1-minute candle whose close is checked. Only data known at that close is used."""
    days = session_frames(m1)
    dailies = daily_bars(days)
    lb, mult = int(p.get("noise_days", 20)), float(p.get("noise_mult", 1.25))
    k_or, n_atr = float(p.get("or_atr_mult", 0.3)), int(p.get("atr_days", 5))
    rows = []
    for d, g in days.items():
        if d < from_day:
            continue
        sig = noise_sigma(days, d, lb, mult)
        prev = dailies[dailies.index < d]
        a = atr(prev, n_atr)
        if sig is None or a is None:
            rows.append({"ts": g.index[0], "day": d, "skip": f"not enough history ({len(prev)} sessions)"})
            continue
        mins = _ny_minutes(g.index)
        op, pc = float(g["open"].iloc[0]), float(prev["close"].iloc[-1])
        orng = g[(mins >= 570) & (mins < 585)]
        if len(orng) == 0:
            continue
        orh, orl = float(orng["high"].max()), float(orng["low"].min())
        tp = (g["high"] + g["low"] + g["close"]) / 3.0
        # Yahoo's 1-minute volumes have glitches (single minutes at 100x the norm): each minute's volume is
        # capped at `vol_cap` x the median minute volume of the previous sessions (known before the day)
        prior_vol = pd.concat([days[x]["volume"] for x in sorted(x for x in days if x < d)[-lb:]])
        vol = g["volume"].clip(upper=float(p.get("vol_cap", 20)) * max(float(prior_vol.median()), 1.0))
        cv = vol.cumsum()
        vwap = ((tp * vol).cumsum() / cv.where(cv > 0)).fillna(g["close"])
        for i, (ts, bar) in enumerate(g.iterrows()):
            close_min = mins[i] + 1                      # NY minute at the candle's close
            if close_min < 600 or close_min > 840 or close_min % 10:
                continue
            if orng.index[-1] > ts:                      # the opening range must be complete
                continue
            s = float(sig.get(mins[i] - 570, np.nan))
            if not np.isfinite(s):
                continue
            c = float(bar["close"])
            up, lo = max(op, pc) * (1 + s), min(op, pc) * (1 - s)
            vw = float(vwap.iloc[i])
            right = None
            if c > orh + k_or * a and c > up and c > vw:
                right = "C"
            elif c < orl - k_or * a and c < lo and c < vw:
                right = "P"
            rows.append({"ts": ts, "day": d, "close": c, "or_high": orh, "or_low": orl, "atr": a, "upper": up,
                         "lower": lo, "vwap": vw, "right": right})
    if not rows:
        return pd.DataFrame(columns=["day", "close", "right"])
    return pd.DataFrame(rows).set_index("ts").sort_index()


class QqqNoiseOrbOptionsBot(Bot):
    """See the module docstring. Trades whole QQQ option contracts (x100) at IBKR's option fees."""
    timeframe = "1m"
    UNDERLYING = "QQQ"

    def __init__(self, bot_id: str, spec: dict):
        super().__init__(bot_id, spec)
        self.table = pd.DataFrame()
        self.sessions: pd.DataFrame | None = None
        self.contracts: dict[pd.Timestamp, str] = {}       # signal row ts -> option id

    def instruments(self) -> list[str]:
        return [self.UNDERLYING]

    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.p.get("start", "2026-11-23T00:00Z"))

    # ------------------------------------------------------------------ set-up (runner)
    def expiry_for(self, day: pd.Timestamp) -> str:
        """Next session after `day` (1 trading day out; QQQ lists options for every session)."""
        nxt = self.sessions.index[self.sessions.index > pd.Timestamp(day)]
        return nxt[0].strftime("%Y-%m-%d")

    def contract(self, day: pd.Timestamp, close: float, right: str) -> tuple[str, str]:
        exp, strike = self.expiry_for(day), float(round(close))
        return option_id(self.UNDERLYING, exp, right, strike), yahoo_option_symbol(self.UNDERLYING, exp, right, strike)

    def option_inst(self, oid: str):
        und, exp, right, strike = oid.split(":")
        return option_instrument(stock_instrument(und), exp, right, float(strike))

    def session_close(self, day: pd.Timestamp) -> pd.Timestamp:
        return self.sessions.loc[pd.Timestamp(day).tz_localize(None).normalize(), "close"]

    def prepare(self, m1: pd.DataFrame, from_day: pd.Timestamp) -> pd.DataFrame:
        self.table = signal_table(m1, self.p, from_day)
        self.contracts = {}
        if "right" in self.table:
            for ts, r in self.table[self.table["right"].notna()].iterrows():
                self.contracts[ts] = self.contract(r["day"], r["close"], r["right"])[0]
        return self.table

    # ------------------------------------------------------------------ trading
    def _log(self, ctx, event: str, right=None, contract="", qty=None, price=None, row=None, detail="",
             ts: pd.Timestamp | None = None) -> None:
        t = ts if ts is not None else ctx.ts + pd.Timedelta(minutes=1)
        r = row if row is not None else {}
        ctx.state.setdefault("log", []).append({
            "ts_utc": t.isoformat(), "ts_ny": t.tz_convert(NY).strftime("%Y-%m-%d %H:%M"), "event": event,
            "right": right or "", "contract": contract, "qty": qty if qty is not None else "",
            "price": round(price, 4) if price is not None else "",
            "qqq_close": _r(r.get("close")), "or_high": _r(r.get("or_high")), "or_low": _r(r.get("or_low")),
            "atr": _r(r.get("atr")), "upper": _r(r.get("upper")), "lower": _r(r.get("lower")),
            "vwap": _r(r.get("vwap")), "detail": detail})

    def on_bar(self, ctx, bars: dict) -> None:
        bar = bars.get(self.UNDERLYING)
        if bar is None or ctx.ts < self.start():
            return
        st = ctx.state
        close_ts = ctx.ts + pd.Timedelta(minutes=1)
        ny = close_ts.tz_convert(NY)
        day = ny.tz_localize(None).normalize().strftime("%Y-%m-%d")
        if st.get("day") != day:
            st.update({"day": day, "traded": False})
            if st.get("pending"):                       # a limit left from an earlier day (should not happen)
                ctx.cancel("opt:buy")
                st["pending"] = None
        close = float(bar["close"])
        pos = st.get("pos")
        # exits, judged on QQQ's 1-minute closes
        if pos and not pos.get("exiting"):
            sgn = 1.0 if pos["right"] == "C" else -1.0
            fav = (close - pos["entry_u"]) * sgn
            pos["best"] = max(pos["best"], fav)
            a = pos["atr"]
            exit_at = self.session_close(pd.Timestamp(day)) - pd.Timedelta(minutes=15)
            why = None
            if pos["best"] >= float(self.p.get("trail_atr", 4.0)) * a and fav <= pos["best"] * 0.5:
                why = f"gave back half of the best move ({pos['best']:.2f} -> {fav:.2f})"
            elif -fav >= float(self.p.get("stop_atr", 10.0)) * a:
                why = f"emergency stop ({fav:.2f} = {fav / a:.1f} ATR)"
            elif close_ts >= exit_at:
                why = f"time exit {exit_at.tz_convert(NY):%H:%M}"
            if why:
                # market sell: crosses half the quoted spread (USD half_spread_usd, as a fraction of the last print)
                q = ctx.hist(pos["inst"])
                last = float(q["close"].iloc[-1]) if len(q) else 0.0
                hs = min(0.5, float(self.p.get("half_spread_usd", 0.03)) / max(last, 0.01))
                ctx.place(pos["inst"], "sell", "market", qty=ctx.qty(pos["inst"]), tag=f"opt:sell {why}", impact=hs)
                pos["exiting"] = True
                self._log(ctx, "exit signal", pos["right"], pos["inst"], detail=why,
                          row={"close": close, "atr": a})
            return
        # an unfilled limit lapses at the next check
        pend = st.get("pending")
        if pend and close_ts >= pd.Timestamp(pend["expires"]):
            if ctx.cancel("opt:buy"):
                self._log(ctx, "limit not filled", pend["right"], pend["inst"], detail="cancelled at the next check")
            st["pending"] = None
            pend = None
        if pos or pend or st.get("traded") or ctx.ts not in self.table.index:
            return
        row = self.table.loc[ctx.ts]
        if not isinstance(row.get("right"), str):
            return
        right, oid = row["right"], self.contracts[ctx.ts]
        q = ctx.hist(oid)
        if q.empty:
            self._log(ctx, "signal skipped", right, oid, row=row, detail="no option prints yet today")
            return
        mid = float(q["close"].iloc[-1])
        n = int(self.p.get("contracts", 1))
        self._log(ctx, "signal", right, oid, n, mid, row, detail=f"limit at mid (last print {q.index[-1].tz_convert(NY):%H:%M})")
        ctx.place(oid, "buy", "limit", qty=n, price=mid, tag=f"opt:buy {right}")
        st["pending"] = {"inst": oid, "right": right, "expires": (close_ts + pd.Timedelta(minutes=10)).isoformat(),
                         "entry_u": close, "atr": float(row["atr"]), "row": {k: _r(row.get(k)) for k in
                         ("close", "or_high", "or_low", "atr", "upper", "lower", "vwap")}}

    def on_fill(self, ctx, order: dict, fill) -> None:
        st = ctx.state
        if order["side"] == "buy":
            pend = st.get("pending") or {}
            st["pending"] = None
            st["traded"] = True
            st["pos"] = {"inst": fill.inst, "right": pend.get("right"), "entry_u": pend.get("entry_u"),
                         "atr": pend.get("atr"), "best": 0.0}
            self._log(ctx, "buy", pend.get("right"), fill.inst, fill.qty, fill.price, pend.get("row"),
                      detail=f"fee EUR {fill.fee_eur:.2f}", ts=ctx.ts)
        else:
            pos = st.get("pos") or {}
            self._log(ctx, "sell", pos.get("right"), fill.inst, fill.qty, fill.price,
                      detail=f"{order['tag'][9:]}; P&L EUR {fill.pnl_eur:.2f}" if fill.pnl_eur is not None else "",
                      ts=ctx.ts)
            st["pos"] = None

    def after_save(self, ledger, state: Path) -> None:
        """Rewrites the signal/buy/sell log from the ledger (so a cycle that failed never leaves rows)."""
        rows = ledger.meta.get("state", {}).get("log", [])
        path = Path(state) / "options" / f"{self.id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_COLS)
            w.writeheader()
            w.writerows(rows)


def _r(v):
    try:
        return round(float(v), 4) if v is not None and np.isfinite(float(v)) else ""
    except (TypeError, ValueError):
        return ""
