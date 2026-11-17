"""Deterministic synthetic market data for the family tests (any coin, ticker or timeframe)."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from quantdesk.bots.data import TF

START = {"5m": "2026-07-20", "15m": "2026-06-01", "1h": "2026-01-01", "4h": "2025-01-01", "1d": "2022-01-01"}
END = pd.Timestamp("2026-11-30", tz="UTC")


def _seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


def _walk(seed: int, n: int, start_px: float, vol: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.normal(0, vol / 5, n // 150 + 1), 150)[:n]
    return start_px * np.exp(np.cumsum(regime + rng.normal(0, vol, n)))


def _ohlcv(idx, seed, px0, vol):
    c = _walk(seed, len(idx), px0, vol)
    o = np.r_[c[0], c[:-1]]
    rng = np.random.default_rng(seed + 1)
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, len(c))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, len(c))))
    v = np.exp(rng.normal(10, 0.6, len(c)))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=idx)


class FamilySource:
    def __init__(self):
        self.used = {}
        self._c = {}

    def crypto(self, symbol, timeframe, since, now):
        key = (symbol, timeframe)
        if key not in self._c:
            tf = TF[timeframe]
            idx = pd.date_range(pd.Timestamp(START[timeframe], tz="UTC"), END, freq=tf, inclusive="left")
            vol = 0.02 * (tf / pd.Timedelta(days=1)) ** 0.5
            self._c[key] = _ohlcv(idx, _seed(symbol), 100.0 + _seed(symbol) % 900, vol)
        df = self._c[key]
        return df[(df.index >= since) & (df.index + TF[timeframe] <= now)]

    def eurusd(self):
        return 1.16

    def futures(self, symbol, timeframe, since, now, delay=pd.Timedelta(0)):
        """Gold-futures-like candles: every Globex hour (no 17:00-18:00 New York break, no weekend)."""
        key = ("fut", symbol, timeframe)
        if key not in self._c:
            idx = pd.date_range(pd.Timestamp(START[timeframe], tz="UTC"), END, freq=TF[timeframe], inclusive="left")
            ny = idx.tz_convert("America/New_York")
            wk = ny.dayofweek
            open_ = (ny.hour != 17) & (wk != 5) & ~((wk == 4) & (ny.hour >= 17)) & ~((wk == 6) & (ny.hour < 18))
            self._c[key] = _ohlcv(idx[open_], _seed(symbol, timeframe), 4100.0, 0.0009)
        df = self._c[key]
        return df[(df.index >= since) & (df.index + TF[timeframe] <= now - delay)]

    def sonic_rpc(self):
        if "sonic" not in self._c:
            self._c["sonic"] = FakeSonicRpc()
        return self._c["sonic"]

    def mql5_page(self, signal_id):
        """A page shaped like an MQL5 signal page: balance/equity about every 5 hours (trades closing
        now and then), one withdrawal on 20 Oct 2026."""
        return mql5_html(_seed("mql5", signal_id))

    def _sched(self):
        if "sched" not in self._c:
            import exchange_calendars as xcals
            self._c["sched"] = xcals.get_calendar("XNYS").schedule.loc["2016-01-01":"2026-12-31"]
        return self._c["sched"]

    def _stock(self, t):
        key = ("stock", t)
        if key not in self._c:
            sched = self._sched()
            s5 = sched.loc["2026-08-01":]
            idx = pd.DatetimeIndex([x for o, c in zip(s5["open"], s5["close"])
                                    for x in pd.date_range(o, c, freq="5min", inclusive="left")])
            base = 20.0 + _seed(t) % 300
            intr = _ohlcv(idx, _seed(t, "5m"), base, 0.0015)
            didx = pd.DatetimeIndex(sched.index[sched.index < pd.Timestamp("2026-08-01")])
            daily = _ohlcv(didx, _seed(t, "1d"), base * 0.6, 0.012)
            daily = daily * (intr["open"].iloc[0] / daily["close"].iloc[-1])
            day5 = intr.copy()
            day5.index = day5.index.tz_convert("America/New_York").tz_localize(None).normalize()
            agg = day5.groupby(level=0).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                             "volume": "sum"})
            self._c[key] = (intr, pd.concat([daily, agg]))
        return self._c[key]

    def stocks(self, tickers, timeframe, since, now):
        out = {}
        for t in tickers:
            intr, _ = self._stock(t)
            out[t] = intr[(intr.index >= since) & (intr.index + TF["5m"] <= now)]
        return out

    def stock_daily(self, tickers, days, now):
        day = pd.Timestamp(now).tz_convert("America/New_York").tz_localize(None).normalize()
        return {t: self._stock(t)[1][lambda d: d.index < day] for t in tickers}


def mql5_html(seed: int, start="2026-09-01", end=END, withdrawal=("2026-10-20 06:00", -300.0)) -> str:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), end, freq="5h")
    wts = pd.Timestamp(withdrawal[0], tz="UTC")
    b, fl, done, rows = 1000.0, 0.0, False, []
    for t in idx:
        if not done and t > wts:
            b += withdrawal[1]
            done = True
        fl += rng.normal(0.4, 12.0)
        if rng.random() < 0.3:               # a trade closes: its P&L moves to the balance
            b, fl = b + fl, 0.0
        rows.append((int(t.timestamp()), round(b, 2), round(b + fl, 2)))
    eq = ",".join(f"{a},{c},{d}" for a, c, d in rows)
    return (f"<html><script>var equityData=[{eq}];</script><div>Trades: <b>120</b></div>"
            f"<script>var acc={{growth:0.1,balance:{rows[-1][1]},equity:{rows[-1][2]},drawdown:0,deposits:0,"
            f"withdrawals:{-withdrawal[1]},currency:\"USD\",digits:2,stops:window.buildBalanceData("
            f"[5,5,{int(wts.timestamp())},{withdrawal[1]}]),}};</script></html>")


class FakeSonicRpc:
    """A deterministic stand-in for the Sonic RPC: a block every 2 seconds from 1 Oct 2026, a swap in
    the S/USDC.e pool every 150 blocks and in the USDC.e/WETH pool 75 blocks later (random-walk prices,
    constant liquidity), and for every wallet asked about, a trade about every 6 hours (alternately
    buying S or ETH with a share of its USDC.e and selling a share of it back)."""
    BASE = int(pd.Timestamp("2026-10-01", tz="UTC").timestamp())
    LIMIT = 5000

    def __init__(self, head_ts=None, liq=7e17):
        from quantdesk.bots import sonic_chain as sc
        self.sc = sc
        self.head = ((int(pd.Timestamp(head_ts).timestamp()) if head_ts is not None else int(END.timestamp())) - self.BASE) // 2
        self.liq = liq
        self.calls = 0

    # chain
    def ts(self, n):
        return self.BASE + 2 * n

    def px(self, coin, n):
        k = n // 150
        rng = np.random.default_rng(_seed("sonicpx", coin, k // 500))
        steps = rng.normal(0, 0.004, 500)
        base = 0.5 if coin == "S" else 4000.0
        return base * float(np.exp(np.cumsum(steps)[k % 500] + 0.05 * np.sin(k / 300)))

    def sqrt_of(self, pool, human):
        _, t0, t1 = self.sc.POOLS[pool]
        raw = human * 10 ** (self.sc.TOKENS[t1][1] - self.sc.TOKENS[t0][1])
        return int((raw ** 0.5) * self.sc.Q96)

    def _word(self, v):
        return format(v % 2 ** 256, "064x")

    def swap_log(self, pool, n, a0, a1, tx=None, idx=0):
        addr, t0, t1 = self.sc.POOLS[pool]
        coin = "S" if pool == "S/USD" else "ETH"
        p = self.px(coin, n)
        human = p if t0 == coin else 1.0 / p
        data = "0x" + "".join(self._word(v) for v in (a0, a1, self.sqrt_of(pool, human), int(self.liq), 0))
        return {"address": addr, "topics": [self.sc.SWAP, "0x" + "0" * 64, "0x" + "0" * 64], "data": data,
                "blockNumber": hex(n), "blockTimestamp": hex(self.ts(n)), "logIndex": hex(idx),
                "transactionHash": tx or f"0xswap{pool}{n}", "transactionIndex": "0x0"}

    def trades_of(self, w, a, b):
        """(block, n-th trade) of wallet w in [a, b]: one about every 6 hours."""
        out = []
        period = 10800
        off = _seed("off", w) % period
        k0 = max(0, (a - off) // period)
        for k in range(k0, (b - off) // period + 2):
            n = off + k * period
            if a <= n <= b:
                out.append((n, k))
        return out

    def trade(self, w, n, k):
        """Trade k of wallet w: even k buys S (k % 4 == 0) or ETH with 20-60% of its USDC.e, odd k sells
        half of what it bought back."""
        rng = np.random.default_rng(_seed("trade", w, k))
        coin = "S" if (k // 2) % 2 == 0 else "ETH"
        pool = "S/USD" if coin == "S" else "USD/ETH"
        usd_have = 10000.0
        if k % 2 == 0:
            usd = usd_have * float(rng.uniform(0.2, 0.6))
            amt = usd / self.px(coin, n)
            before = {"USD": usd_have, "S": 1000.0, "ETH": 0.5}
            paid, got = ("USD", usd), (coin, amt)
        else:
            have = 1000.0 if coin == "S" else 0.5
            amt = have * 0.5
            usd = amt * self.px(coin, n)
            before = {"USD": usd_have, "S": 1000.0, "ETH": 0.5}
            paid, got = (coin, amt), ("USD", usd)
        _, t0, t1 = self.sc.POOLS[pool]
        raw = {c: v * 10 ** self.sc.TOKENS[c][1] for c, v in (paid, got)}
        a0 = int(raw[t0]) if t0 == paid[0] else -int(raw[t0])
        a1 = int(raw[t1]) if t1 == paid[0] else -int(raw[t1])
        tx = f"0xtx{w[2:10]}{n}"
        return {"tx": tx, "block": n, "log": self.swap_log(pool, n, a0, a1, tx=tx, idx=1), "before": before}

    # RPC surface used by SonicStore
    def block_number(self):
        return self.head

    def block_ts(self, n):
        return self.ts(n)

    def block_at(self, t):
        return max(0, -(-(int(t) - self.BASE) // 2))

    def logs(self, flt, a, b):
        self.calls += 1
        b = min(b, self.head)
        out = []
        if flt.get("address"):
            addrs = flt["address"] if isinstance(flt["address"], list) else [flt["address"]]
            for pool, (addr, _, _) in self.sc.POOLS.items():
                if addr not in addrs:
                    continue
                off = 0 if pool == "S/USD" else 75
                first = a + ((off - a) % 150)
                for n in range(first, b + 1, 150):
                    out.append(self.swap_log(pool, n, 10 ** 18, -(10 ** 5)))
        else:
            t = flt["topics"]
            wallets = t[1] if len(t) > 1 and t[1] else t[2]
            self._wallets = sorted(set(getattr(self, "_wallets", [])) | {"0x" + wt[-40:] for wt in wallets})
            for wt in wallets:
                w = "0x" + wt[-40:]
                for n, k in self.trades_of(w, a, b):
                    tr = self.trade(w, n, k)
                    out.append({"transactionHash": tr["tx"], "blockNumber": hex(n), "topics": t})
        if len(out) > self.LIMIT:
            raise self.sc.TooManyResults("too many results")
        return out

    def batch(self, reqs):
        res = []
        for m, p in reqs:
            if m == "eth_getTransactionReceipt":
                h = p[0]
                w8, n = h[4:12], int(h[12:])
                w = next(x for x in self._wallets if x[2:10] == w8)
                k = next(k for nn, k in self.trades_of(w, n, n))
                tr = self.trade(w, n, k)
                res.append({"from": w, "status": "0x1", "blockNumber": hex(n), "transactionIndex": "0x1",
                            "logs": [tr["log"]]})
            elif m == "eth_call":
                tok = p[0]["to"]
                w = "0x" + p[0]["data"][-40:]
                n = int(p[1], 16) + 1
                k = next(k for nn, k in self.trades_of(w, n, n))
                coin = self.sc.TOKEN_BY_ADDR[tok]
                v = self.trade(w, n, k)["before"][coin]
                res.append(hex(int(v * 10 ** self.sc.TOKENS[coin][1])))
            elif m == "eth_getBalance":
                res.append("0x0")
            elif m == "eth_getBlockByNumber":
                res.append({"timestamp": hex(self.ts(int(p[0], 16)))})
            else:
                raise ValueError(m)
        return res
