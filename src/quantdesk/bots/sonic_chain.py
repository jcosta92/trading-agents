"""On-chain data from the Sonic blockchain (ex-Fantom) for the wallet copy-trading bots.

Everything comes from Sonic's free public RPC (https://rpc.soniclabs.com, no key): swap events of the
three most liquid Shadow Exchange pools (prices, pool liquidity), the copied wallets' token
transfers (which transactions to look at), transaction receipts (the swaps in them) and the wallets'
balances just before each trade (historical state is served by the public node).

Pools are Uniswap-v3 style (concentrated liquidity): Swap(sender, recipient, amount0, amount1,
sqrtPriceX96, liquidity, tick), amounts signed from the pool's side (positive = paid into the pool).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RPC = "https://rpc.soniclabs.com"
SWAP = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
Q96 = 2 ** 96

# token -> (address, decimals); wS is wrapped S, counted together with native S
TOKENS = {
    "S": ("0x039e2fb66102314ce7b64ce5ce3e5183bc94ad38", 18),
    "USD": ("0x29219dd400f2bf60e5a23d13be72b486d4038894", 6),          # USDC.e
    "ETH": ("0x50c42deacd8fc9773493ed674b675be577f2634b", 18),         # WETH
}
# Shadow Exchange concentrated-liquidity pools, 0.15% fee tier: name -> (address, token0, token1)
POOLS = {
    "S/USD": ("0x324963c267c354c7660ce8ca3f5f167e05649970", "S", "USD"),
    "S/ETH": ("0xb6d9b069f6b96a507243d501d1a23b3fccfc85d3", "S", "ETH"),
    "USD/ETH": ("0x6fb30f3fcb864d49cdff15061ed5c6adfee40b40", "USD", "ETH"),
}
POOL_BY_ADDR = {a: (n, t0, t1) for n, (a, t0, t1) in POOLS.items()}
TOKEN_BY_ADDR = {a: k for k, (a, _) in TOKENS.items()}
PRICE_POOLS = {"S": "S/USD", "ETH": "USD/ETH"}       # USD price of each coin


class TooManyResults(Exception):
    pass


class SonicRpc:
    def __init__(self, url: str = RPC, timeout: float = 90.0):
        import requests
        self.url, self.timeout = url, timeout
        self.s = requests.Session()
        self.calls = 0

    def call(self, method: str, params: list):
        err = None
        for i in range(5):
            try:
                self.calls += 1
                r = self.s.post(self.url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                                timeout=self.timeout).json()
            except Exception as exc:              # network error or a non-JSON answer (rate limit page)
                err = exc
                time.sleep(1.5 * (i + 1))
                continue
            if "error" in r:
                msg = str(r["error"].get("message", r["error"]))
                if "too many results" in msg or "limit" in msg and "block range" in msg:
                    raise TooManyResults(msg)
                err = RuntimeError(msg)
                time.sleep(1.5 * (i + 1))
                continue
            return r["result"]
        raise RuntimeError(f"Sonic RPC {method} failed: {err}")

    def batch(self, reqs: list[tuple[str, list]], size: int = 100) -> list:
        out = []
        for i in range(0, len(reqs), size):
            chunk = reqs[i:i + size]
            err = None
            for k in range(5):
                try:
                    self.calls += 1
                    r = self.s.post(self.url, json=[{"jsonrpc": "2.0", "id": j, "method": m, "params": p}
                                                    for j, (m, p) in enumerate(chunk)], timeout=self.timeout * 2).json()
                    if not isinstance(r, list):
                        raise RuntimeError(str(r)[:200])
                    r = sorted(r, key=lambda x: x["id"])
                    bad = [x["error"] for x in r if "error" in x]
                    if bad or len(r) != len(chunk):
                        raise RuntimeError(f"batch errors: {bad[:2]}")
                    out += [x["result"] for x in r]
                    break
                except Exception as exc:
                    err = exc
                    time.sleep(2.0 * (k + 1))
            else:
                raise RuntimeError(f"Sonic RPC batch failed: {err}")
        return out

    def block_number(self) -> int:
        return int(self.call("eth_blockNumber", []), 16)

    def block_ts(self, n: int) -> int:
        return int(self.call("eth_getBlockByNumber", [hex(n), False])["timestamp"], 16)

    def block_at(self, ts: int) -> int:
        """First block with timestamp >= ts."""
        lo, hi = 1, self.block_number()
        while lo < hi:
            mid = (lo + hi) // 2
            if self.block_ts(mid) < ts:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def logs(self, flt: dict, a: int, b: int) -> list[dict]:
        """eth_getLogs over [a, b], split in halves while the node says the answer is too big."""
        try:
            return self.call("eth_getLogs", [{**flt, "fromBlock": hex(a), "toBlock": hex(b)}])
        except TooManyResults:
            if a >= b:
                raise
            m = (a + b) // 2
            return self.logs(flt, a, m) + self.logs(flt, m + 1, b)


# ---------------------------------------------------------------------------------------- decoding
def _int(word: str, signed: bool = False) -> int:
    v = int(word, 16)
    if signed and v >= 2 ** 255:
        v -= 2 ** 256
    return v


def decode_swap(lg: dict) -> dict:
    d = lg["data"][2:]
    w = [d[i:i + 64] for i in range(0, len(d), 64)]
    name, t0, t1 = POOL_BY_ADDR[lg["address"].lower()]
    return {"pool": name, "t0": t0, "t1": t1, "a0": _int(w[0], True), "a1": _int(w[1], True),
            "sqrt": _int(w[2]), "liq": _int(w[3]), "block": int(lg["blockNumber"], 16),
            "ts": int(lg["blockTimestamp"], 16), "tx": lg["transactionHash"], "idx": int(lg["logIndex"], 16)}


def pool_price(pool: str, sqrt: int) -> float:
    """Price of token0 in token1 (human units)."""
    _, t0, t1 = POOLS[pool]
    return (sqrt / Q96) ** 2 * 10 ** (TOKENS[t0][1] - TOKENS[t1][1])


def usd_price(coin: str, sqrt: int) -> float:
    p = pool_price(PRICE_POOLS[coin], sqrt)
    return p if POOLS[PRICE_POOLS[coin]][1] == coin else 1.0 / p


def price_impact(coin: str, side: str, usd: float, sqrt: float, liq: float) -> float:
    """Price impact of buying (side="buy") or selling `usd` worth of `coin` against the pool's active
    liquidity, as a fraction: how much more a buy pays per coin, or how much less a sale receives,
    than the pool price before the trade. Concentrated-liquidity maths for a trade that stays inside
    the current price range: token1 in moves sqrtP up by dy/L, token0 in moves 1/sqrtP up by dx/L, so
    the input token's average price worsens by k = dy/(L sqrtP) or dx sqrtP/L. Real impact is larger
    when the trade crosses into thinner ranges."""
    pool = PRICE_POOLS[coin]
    _, t0, t1 = POOLS[pool]
    if liq <= 0 or sqrt <= 0 or usd <= 0:
        return 0.0
    sp = sqrt / Q96
    tin = "USD" if side == "buy" else coin
    amt = usd if tin == "USD" else usd / usd_price(coin, sqrt)
    raw = amt * 10 ** TOKENS[tin][1]
    k = raw / (liq * sp) if tin == t1 else raw * sp / liq
    # a buy pays k more per coin; a sale receives 1/(1+k) of the pool price
    return k if side == "buy" else 1.0 - 1.0 / (1.0 + k)


def trade_flows(receipt: dict, wallet: str) -> dict[str, float]:
    """The wallet's net amounts per coin (human units, + = received) from the swaps of the three pools
    in one transaction."""
    out: dict[str, float] = {}
    for lg in receipt.get("logs", []):
        if lg["address"].lower() not in POOL_BY_ADDR or not lg["topics"] or lg["topics"][0] != SWAP:
            continue
        s = decode_swap({**lg, "blockTimestamp": lg.get("blockTimestamp", "0x0")})
        for tok, amt in ((s["t0"], s["a0"]), (s["t1"], s["a1"])):
            out[tok] = out.get(tok, 0.0) - amt / 10 ** TOKENS[tok][1]
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


# ---------------------------------------------------------------------------------------- live sync
@dataclass
class SyncResult:
    candles: dict[str, pd.DataFrame]     # coin -> 5-minute candles (open time) with close liquidity/sqrt
    trades: list[dict]                   # copied wallets' trades in the three pools, oldest first
    synced_ts: pd.Timestamp              # data complete up to this time
    error: str | None = None


class SonicStore:
    """Prices (5-minute candles) and the copied wallets' trades from `start` on, kept in the bots'
    state and extended every cycle from the last block read (time-boxed, so a long backfill spreads
    over several cycles instead of stalling one)."""

    def __init__(self, state_dir: Path, wallets: list[str], start: pd.Timestamp, rpc: SonicRpc | None = None):
        self.dir = Path(state_dir) / "copy" / "sonic"
        self.wallets = sorted(w.lower() for w in wallets)
        self.start = start
        self.rpc = rpc

    # files
    def _meta_path(self):
        return self.dir / "sync.json"

    def load(self) -> tuple[dict, dict[str, pd.DataFrame], list[dict]]:
        meta = json.loads(self._meta_path().read_text()) if self._meta_path().exists() else {}
        candles = {}
        for coin in PRICE_POOLS:
            p = self.dir / f"{coin}_5m.csv"
            if p.exists():
                df = pd.read_csv(p, index_col=0)
                df.index = pd.to_datetime(df.index, utc=True, format="ISO8601")
                candles[coin] = df
        tp = self.dir / "trades.json"
        trades = json.loads(tp.read_text()) if tp.exists() else []
        return meta, candles, trades

    def _save(self, meta, candles, trades) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        for coin, df in candles.items():
            df.to_csv(self.dir / f"{coin}_5m.csv")
        (self.dir / "trades.json").write_text(json.dumps(trades))
        tmp = self._meta_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(meta))
        tmp.replace(self._meta_path())

    def sync(self, now: pd.Timestamp, budget_s: float = 240.0, chunk: int = 50_000) -> SyncResult:
        meta, candles, trades = self.load()
        if meta.get("wallets") and meta["wallets"] != self.wallets:
            raise RuntimeError("the copied wallets changed: the stored Sonic data belongs to another selection")
        rpc = self.rpc or SonicRpc()
        tic = time.time()
        err = None
        try:
            if "next_block" not in meta:
                # from a day before the start: candles need a previous close at the start
                meta = {"wallets": self.wallets, "next_block": rpc.block_at(int((self.start - pd.Timedelta(days=1)).timestamp())),
                        "synced_ts": None, "last_px": {}}
            head = rpc.block_number()
            while (meta["next_block"] <= head and time.time() - tic < budget_s
                   and (meta.get("synced_ts") is None or pd.Timestamp(meta["synced_ts"]) < now)):
                a = meta["next_block"]
                b = min(a + chunk - 1, head)
                swaps = [decode_swap(lg) for lg in rpc.logs({"address": [POOLS[p][0] for p in PRICE_POOLS.values()],
                                                            "topics": [SWAP]}, a, b)]
                new_trades = self._wallet_trades(rpc, a, b)
                end_ts = rpc.block_ts(b)
                candles = _add_candles(candles, swaps, meta, end_ts)
                trades += new_trades
                meta["next_block"] = b + 1
                meta["synced_ts"] = pd.Timestamp(end_ts, unit="s", tz="UTC").isoformat()
                self._save(meta, candles, trades)          # progress survives a cycle cut short
        except Exception as exc:
            err = f"Sonic RPC: {exc}"
            log.warning("Sonic sync stopped: %s", exc)
        synced = pd.Timestamp(meta["synced_ts"]) if meta.get("synced_ts") else self.start - pd.Timedelta(days=1)
        return SyncResult(candles, trades, synced, err)

    def _wallet_trades(self, rpc: SonicRpc, a: int, b: int) -> list[dict]:
        if not self.wallets:
            return []
        topics = ["0x" + w[2:].rjust(64, "0") for w in self.wallets]
        txs = {}
        for flt in ({"topics": [TRANSFER, topics]}, {"topics": [TRANSFER, None, topics]}):
            for lg in rpc.logs(flt, a, b):
                txs[lg["transactionHash"]] = int(lg["blockNumber"], 16)
        if not txs:
            return []
        hashes = sorted(txs, key=lambda h: txs[h])
        receipts = rpc.batch([("eth_getTransactionReceipt", [h]) for h in hashes])
        found = []
        for h, rc in zip(hashes, receipts):
            w = (rc.get("from") or "").lower()
            if w not in self.wallets or rc.get("status") not in ("0x1", 1):
                continue
            fl = trade_flows(rc, w)
            paid = [k for k, v in fl.items() if v < 0]
            got = [k for k, v in fl.items() if v > 0]
            if not paid or not got:
                continue
            blk = int(rc["blockNumber"], 16)
            found.append({"tx": h, "wallet": w, "block": blk, "idx": int(rc["transactionIndex"], 16), "flows": fl})
        if not found:
            return []
        # the wallet's balances just before the trade's block (native S + wS together)
        reqs = []
        for t in found:
            blk, w = hex(t["block"] - 1), t["wallet"]
            for coin in ("S", "USD", "ETH"):
                reqs.append(("eth_call", [{"to": TOKENS[coin][0], "data": "0x70a08231" + w[2:].rjust(64, "0")}, blk]))
            reqs.append(("eth_getBalance", [w, blk]))
        res = rpc.batch(reqs)
        blocks = sorted({t["block"] for t in found})
        heads = rpc.batch([("eth_getBlockByNumber", [hex(n), False]) for n in blocks])
        ts = {n: int(h["timestamp"], 16) for n, h in zip(blocks, heads)}
        for i, t in enumerate(found):
            ws, usd, eth, native = (int(x, 16) for x in res[4 * i:4 * i + 4])
            t["before"] = {"S": (ws + native) / 1e18, "USD": usd / 10 ** TOKENS["USD"][1], "ETH": eth / 1e18}
            t["ts"] = pd.Timestamp(ts[t["block"]], unit="s", tz="UTC").isoformat()
        return sorted(found, key=lambda t: (t["block"], t["idx"]))


def _add_candles(candles: dict, swaps: list[dict], meta: dict, end_ts: int) -> dict:
    """Extend the 5-minute USD price candles of S and ETH with new pool swaps. A 5-minute slot with no
    swap repeats the last price (the pool price did not move). Only slots that ended by `end_ts`
    (the last block read) are written."""
    step = pd.Timedelta(minutes=5)
    last_px = meta.setdefault("last_px", {})
    end = pd.Timestamp(end_ts, unit="s", tz="UTC").floor(step)          # slots strictly before this are complete
    out = dict(candles)
    for coin, pool in PRICE_POOLS.items():
        rows = [s for s in swaps if s["pool"] == pool]
        old = out.get(coin)
        first_slot = (old.index[-1] + step) if old is not None and len(old) else None
        pts = pd.DataFrame([{"ts": pd.Timestamp(s["ts"], unit="s", tz="UTC"), "px": usd_price(coin, s["sqrt"]),
                             "sqrt": s["sqrt"], "liq": s["liq"]} for s in sorted(rows, key=lambda s: (s["block"], s["idx"]))])
        pend = meta.setdefault("pending", {}).get(coin, [])
        if pend:
            pp = pd.DataFrame(pend)
            pp["ts"] = pd.to_datetime(pp["ts"], utc=True)
            pts = pd.concat([pp, pts]) if len(pts) else pp
        if first_slot is None:
            if not len(pts):
                continue
            first_slot = pts["ts"].iloc[0].floor(step)
        done = pts[pts["ts"] < end] if len(pts) else pts
        later = pts[pts["ts"] >= end] if len(pts) else pts
        meta["pending"][coin] = [{"ts": r.ts.isoformat(), "px": r.px, "sqrt": int(r.sqrt), "liq": int(r.liq)}
                                 for r in later.itertuples()] if len(later) else []
        slots = pd.date_range(first_slot, end - step, freq=step)
        if not len(slots):
            continue
        prev = last_px.get(coin)
        recs = []
        g = {k: v for k, v in done.groupby(done["ts"].dt.floor(step))} if len(done) else {}
        for t in slots:
            v = g.get(t)
            if v is None or not len(v):
                if prev is None:
                    continue
                c = prev
                recs.append({"ts": t, "open": c["px"], "high": c["px"], "low": c["px"], "close": c["px"],
                             "volume": 0.0, "sqrt": float(c["sqrt"]), "liq": float(c["liq"])})
                continue
            o = prev["px"] if prev is not None else float(v["px"].iloc[0])
            prev = {"px": float(v["px"].iloc[-1]), "sqrt": int(v["sqrt"].iloc[-1]), "liq": int(v["liq"].iloc[-1])}
            recs.append({"ts": t, "open": o, "high": max(o, float(v["px"].max())), "low": min(o, float(v["px"].min())),
                         "close": prev["px"], "volume": float(len(v)), "sqrt": float(prev["sqrt"]), "liq": float(prev["liq"])})
        if prev is not None:
            last_px[coin] = {"px": prev["px"], "sqrt": int(prev["sqrt"]), "liq": int(prev["liq"])}
        if recs:
            new = pd.DataFrame(recs).set_index("ts")
            out[coin] = pd.concat([old, new]) if old is not None and len(old) else new
    return out
