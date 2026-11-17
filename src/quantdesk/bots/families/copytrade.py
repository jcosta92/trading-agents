"""Copy trading: following another trader's account instead of running a strategy.

MQL5 signal copy (Mql5CopyBot)
    MetaTrader 5 copies every trade of the signal provider's account into the subscriber's account,
    scaled by the ratio of the two balances times the "use no more than X% of deposit" setting
    (95% by default). The provider's trade history is behind a login on mql5.com, but the public
    signal page carries the account's balance/equity series and its withdrawals and deposits. From
    those the bot rebuilds what the subscriber's account would have done: between two samples the
    copier gains (ratio x the provider's equity change, deposits and withdrawals removed); the ratio
    is reset to load x copier balance / provider balance whenever the provider's balance changes
    (a trade closed, money paid in or out). The copier's net asset value (1.0 at the start) is published as a synthetic
    instrument, bought with the whole account at the start and held; the monthly subscription is
    charged as a fee. Not modelled: lot rounding (MetaTrader rounds copied volumes down to the
    broker's lot step), the copier's own slippage and the provider's samples being hours apart.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from ..data import TF
from ..strategies import Bot

UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")


# ============================================================================================ MQL5
def parse_mql5_signal(html: str) -> dict:
    """Balance/equity samples, deposits/withdrawals and headline figures from a public MQL5 signal page.

    samples: DataFrame indexed by UTC time with columns balance, equity (account currency).
    flows: DataFrame with columns ts, amount (negative = withdrawal), the balance operations the page marks.
    """
    m = re.search(r"var\s+equityData\s*=\s*\[([^\]]*)\]", html)
    if not m:
        raise ValueError("no equityData on the signal page")
    vals = [float(x) for x in m.group(1).split(",") if x.strip()]
    if len(vals) % 3:
        raise ValueError(f"equityData has {len(vals)} numbers, not triplets")
    s = pd.DataFrame({"ts": vals[0::3], "balance": vals[1::3], "equity": vals[2::3]})
    s.index = pd.to_datetime(s.pop("ts"), unit="s", utc=True)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    flows = []
    fm = re.search(r"stops\s*:\s*window\.buildBalanceData\(\[([^\]]*)\]\)", html)
    if fm:
        q = [float(x) for x in fm.group(1).split(",") if x.strip()]
        for i in range(0, len(q) - 3, 4):            # (index, index, unix time, amount)
            flows.append({"ts": pd.Timestamp(int(q[i + 2]), unit="s", tz="UTC"), "amount": q[i + 3]})
    acc = {}
    am = re.search(r"var\s+acc\s*=\s*\{(.*?)\};", html, re.S)
    if am:
        for k in ("growth", "balance", "equity", "deposits", "withdrawals", "depositLoad"):
            v = re.search(rf"\b{k}\s*:\s*(-?[0-9.eE+-]+)", am.group(1))
            if v:
                acc[k] = float(v.group(1))
        c = re.search(r'currency\s*:\s*"([A-Z]{3})"', am.group(1))
        if c:
            acc["currency"] = c.group(1)
    for label, key in (("Trades:", "trades"), ("Subscribers:", "subscribers")):
        v = re.search(re.escape(label) + r"\s*(?:<[^>]+>\s*)*([0-9][0-9,]*)", html)
        if v:
            acc[key] = int(v.group(1).replace(",", ""))
    return {"samples": s, "flows": pd.DataFrame(flows, columns=["ts", "amount"]), "acc": acc}


def copier_nav(samples: pd.DataFrame, flows: pd.DataFrame, start: pd.Timestamp, load: float = 0.95) -> pd.Series:
    """Net asset value (1.0 at `start`) of an account copying the provider from `start`.

    The copier subscribes at `start` with its open positions synchronised to the provider's, so from
    then on each change in the provider's equity (net of deposits/withdrawals) moves the copier's equity
    by `ratio` times as much, where ratio = load x copier balance / provider balance, recomputed every
    time the provider's balance changes (a trade closed, a deposit or a withdrawal). A copier
    whose equity reaches zero is stopped out and stays at zero."""
    s = samples.sort_index()
    before = s[s.index <= start]
    if before.empty:
        raise ValueError("no provider sample at or before the start")
    base_ts = before.index[-1]
    s = s[s.index >= base_ts]
    fl = flows.copy() if flows is not None and len(flows) else pd.DataFrame(columns=["ts", "amount"])
    if len(fl):
        fl["ts"] = pd.to_datetime(fl["ts"], utc=True)
    e_c = 1.0
    b_p, e_p = float(s["balance"].iloc[0]), float(s["equity"].iloc[0])
    b_c = 1.0                                        # copier balance (no floating P&L at the start)
    ratio = load * b_c / b_p if b_p > 0 else 0.0
    out = {start: 1.0}
    prev_t = base_ts
    for t, row in s.iloc[1:].iterrows():
        b, e = float(row["balance"]), float(row["equity"])
        flow = float(fl.loc[(fl["ts"] > prev_t) & (fl["ts"] <= t), "amount"].sum()) if len(fl) else 0.0
        if e_c > 0:
            e_c += ratio * ((e - e_p) - flow)
            if e_c <= 0:
                e_c = 0.0
            elif abs(b - b_p) > 1e-9 and b > 0:       # a trade closed or money moved: MT5 resizes the copies
                b_c = e_c - ratio * (e - b)
                ratio = load * max(b_c, 0.0) / b
        b_p, e_p, prev_t = b, e, t
        if t > start:
            out[t] = e_c
    return pd.Series(out).sort_index()


def nav_candles(nav: pd.Series, start: pd.Timestamp, now: pd.Timestamp, tf: str = "15m") -> pd.DataFrame:
    """Candles of a step series (indexed by the candle's open time, only candles closed by `now`):
    the close is the last value published by the candle's close, high/low include every value
    published inside the candle."""
    cols = ["open", "high", "low", "close", "volume"]
    step = TF[tf]
    first = pd.Timestamp(start).floor(step)
    idx = pd.date_range(first, now - step, freq=step)
    nav = nav.sort_index()
    if not len(idx) or nav.empty:
        return pd.DataFrame(columns=cols)
    closes = idx + step
    close = nav.reindex(nav.index.union(closes)).ffill().reindex(closes).to_numpy()
    df = pd.DataFrame({"close": close}, index=idx)
    known = nav[nav.index <= first]
    df["open"] = df["close"].shift(1)
    df.iloc[0, df.columns.get_loc("open")] = float(known.iloc[-1]) if len(known) else float(nav.iloc[0])
    inside = nav[(nav.index > first) & (nav.index <= closes[-1])]
    bucket = idx[(((inside.index - first) / step).to_numpy() - 1e-12).astype("int64")] if len(inside) else idx[:0]
    g = pd.Series(inside.to_numpy(), index=bucket).groupby(level=0).agg(["max", "min"]).reindex(idx)
    df["high"] = pd.concat([df["open"], df["close"], g["max"]], axis=1).max(axis=1)
    df["low"] = pd.concat([df["open"], df["close"], g["min"]], axis=1).min(axis=1)
    df["volume"] = 0.0
    return df[cols].dropna(subset=["open", "close"])


class Mql5Store:
    """Signal-page samples kept in the bots' state, so the replay never depends on what the page still
    shows (it is fetched once per cycle; a failed fetch falls back to what was stored)."""

    def __init__(self, state_dir: Path, signal_id: str):
        self.dir = Path(state_dir) / "copy"
        self.path = self.dir / f"mql5_{signal_id}.json"

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        d = json.loads(self.path.read_text())
        s = pd.DataFrame(d["samples"], columns=["ts", "balance", "equity"])
        s.index = pd.to_datetime(s.pop("ts"), utc=True)
        fl = pd.DataFrame(d["flows"], columns=["ts", "amount"])
        if len(fl):
            fl["ts"] = pd.to_datetime(fl["ts"], utc=True)
        return {"samples": s, "flows": fl, "acc": d.get("acc", {}), "fetched": d.get("fetched")}

    def merge(self, new: dict, now: pd.Timestamp) -> dict:
        old = self.load()
        if old is not None:
            s = pd.concat([old["samples"], new["samples"]])
            s = s[~s.index.duplicated(keep="last")].sort_index()
            fl = pd.concat([old["flows"], new["flows"]]) if len(old["flows"]) else new["flows"]
            fl = fl.drop_duplicates(subset=["ts", "amount"]).sort_values("ts") if len(fl) else fl
            new = {**new, "samples": s, "flows": fl}
        self.dir.mkdir(parents=True, exist_ok=True)
        out = {"fetched": now.isoformat(), "acc": new.get("acc", {}),
               "samples": [[t.isoformat(), float(r["balance"]), float(r["equity"])] for t, r in new["samples"].iterrows()],
               "flows": [[pd.Timestamp(r["ts"]).isoformat(), float(r["amount"])] for _, r in new["flows"].iterrows()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out))
        tmp.replace(self.path)
        return {**new, "fetched": out["fetched"]}


class Mql5CopyBot(Bot):
    """Subscriber to a public MQL5 signal: the whole account copies the provider from the start
    (synthetic instrument = the copier's NAV, no fees on it: the provider's spreads, commissions and
    swaps are already inside its equity); `monthly_usd` subscription charged at the start and on each
    monthly renewal."""
    timeframe = "15m"

    def instruments(self) -> list[str]:
        return [f"MQL5_{self.p['signal_id']}"]

    def warmup(self) -> int:
        return 1

    def on_bar(self, ctx, bars: dict) -> None:
        st = ctx.state
        iid = self.instruments()[0]
        self._subscription(ctx)
        if st.get("subscribed") or iid not in bars:
            return
        px = float(bars[iid]["close"])
        # the whole trading account copies; the subscription is paid from the MQL5 wallet, not from it
        ctx.place(iid, "buy", "market", qty=max(ctx.ledger.cash.get("EUR", 0.0), 0.0) / px,
                  tag="copy:subscribe and copy the provider")
        st["subscribed"] = ctx.ts.isoformat()

    def _subscription(self, ctx) -> None:
        fee = float(self.p.get("monthly_usd", 0.0))
        if fee <= 0:
            return
        st = ctx.state
        start = pd.Timestamp(st.setdefault("sub_start", ctx.ts.isoformat()))
        paid = int(st.get("months_paid", 0))
        while start + pd.DateOffset(months=paid) <= ctx.ts + TF[self.timeframe]:
            ctx.ledger.charge(ctx.ts.isoformat(), "USD", fee, ctx.engine.fx, "subscription",
                              note=f"MQL5 signal {self.p['signal_id']} subscription, month {paid + 1}")
            paid += 1
            st["months_paid"] = paid


# ============================================================================================ Sonic wallets
SONIC_COINS = {"S_SONIC": "S", "ETH_SONIC": "ETH"}
COIN_IID = {v: k for k, v in SONIC_COINS.items()}


class SonicCopyBot(Bot):
    """Copies one Sonic wallet's trades between USDC.e, S and WETH in proportion: when the wallet
    pays a share f of its holding of one coin for another, the bot pays the same share f of its own
    holding of that coin (seen at the close of the 5-minute candle of the wallet's trade, filled at the
    next candle's open, at the pool price after the wallet's own trade). Holdings start as USDC.e
    (EUR changed at the start). Costs: the pool fee (0.15%), price impact from the pool's liquidity
    at that moment, gas; a switch between S and ETH goes through USDC.e (two swaps)."""
    timeframe = "5m"

    def __init__(self, bot_id: str, spec: dict):
        super().__init__(bot_id, spec)
        self.wallet = str(self.p["wallet"]).lower()
        self.trades: list[dict] = []

    def instruments(self) -> list[str]:
        return list(SONIC_COINS)

    def warmup(self) -> int:
        return 1

    def prepare(self, trades: list[dict]) -> None:
        self.trades = [{**t, "_ts": pd.Timestamp(t["ts"])} for t in trades if t["wallet"] == self.wallet]

    def on_bar(self, ctx, bars: dict) -> None:
        st = ctx.state
        if not st.get("funded"):
            eur = ctx.ledger.cash.get("EUR", 0.0)
            if eur > 0:                                  # EUR -> USDC.e once, at the start
                rate = float(self.p.get("onramp_fee", 0.003))
                ctx.ledger.convert(ctx.ts.isoformat(), "EUR", "USD", ctx.engine.fx.from_eur(eur, "USD") * (1 - rate),
                                   ctx.engine.fx, rate, reason="EUR -> USDC.e on an exchange, withdrawn to Sonic")
            st["funded"] = ctx.ts.isoformat()
            st["seen"] = [0, -1]                         # nothing before the start is copied
            first = [t for t in self.trades if t["_ts"] < ctx.ts]
            if first:
                st["seen"] = [first[-1]["block"], first[-1]["idx"]]
        close_ts = ctx.ts + TF[self.timeframe]
        seen = tuple(st.get("seen", [0, -1]))
        new = [t for t in self.trades if (t["block"], t["idx"]) > seen and t["_ts"] < close_ts]
        if not new:
            return
        st["seen"] = [new[-1]["block"], new[-1]["idx"]]
        px = {c: float(bars[i]["close"]) for i, c in SONIC_COINS.items() if i in bars}
        if len(px) < len(SONIC_COINS):
            px.update({c: float(ctx.hist(i)["close"].iloc[-1]) for i, c in SONIC_COINS.items()
                       if c not in px and len(ctx.hist(i))})
        px["USD"] = 1.0
        if any(c not in px for c in ("S", "ETH")):
            return
        book = {"USD": max(ctx.ledger.cash.get("USD", 0.0), 0.0), "S": ctx.qty(COIN_IID["S"]), "ETH": ctx.qty(COIN_IID["ETH"])}
        start = dict(book)
        copied = []
        for t in new:
            fl = t["flows"]
            paid = max((k for k, v in fl.items() if v < 0), key=lambda k: -fl[k] * px.get(k, 0.0), default=None)
            got = max((k for k, v in fl.items() if v > 0), key=lambda k: fl[k] * px.get(k, 0.0), default=None)
            if paid is None or got is None or paid == got:
                continue
            have = float(t.get("before", {}).get(paid, 0.0))
            f = min(1.0, -fl[paid] / have) if have > 0 else 1.0
            amt = f * book[paid]
            if amt * px[paid] < 1.0:
                continue
            book[paid] -= amt
            book[got] += amt * px[paid] / px[got]
            copied.append(f"{f:.0%} of {paid} -> {got}")
        tag = f"copy {self.wallet[:8]}: " + ", ".join(copied)[:120]
        fx = ctx.engine.fx
        for coin in ("S", "ETH"):
            iid, d = COIN_IID[coin], book[coin] - start[coin]
            if d < 0 and -d * px[coin] >= 5.0:
                usd = -d * px[coin]
                ctx.place(iid, "sell", "market", qty=-d, tag=tag, impact=self._impact(bars, iid, coin, "sell", usd))
        for coin in ("S", "ETH"):
            iid, d = COIN_IID[coin], book[coin] - start[coin]
            if d > 0 and d * px[coin] >= 5.0:
                usd = d * px[coin] / (1 + (ctx.engine.markets[iid].taker or 0.0))   # the pool fee comes out of it
                ctx.place(iid, "buy", "market", eur=fx.to_eur(usd, "USD"), tag=tag,
                          impact=self._impact(bars, iid, coin, "buy", usd))
        if copied:
            ctx.note(tag)

    @staticmethod
    def _impact(bars, iid, coin, side, usd) -> float:
        from ..sonic_chain import price_impact
        b = bars.get(iid)
        if b is None or "liq" not in b or not (b["liq"] > 0):
            return 0.0
        return min(price_impact(coin, side, usd, float(b["sqrt"]), float(b["liq"])), 0.5)

    def on_fill(self, ctx, order, fill) -> None:
        gas = float(self.p.get("gas_usd", 0.01))
        if gas > 0:
            ctx.ledger.charge(ctx.ts.isoformat(), "USD", gas, ctx.engine.fx, "gas", note="Sonic network fee for one swap")
