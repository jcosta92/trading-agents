"""Pick the Sonic wallets the copy-trading bots follow, using ONLY data from before the test start.

Run once (17 Nov 2026); the result is config/sonic_leaders.json, fixed for the whole test. Re-running
reproduces it from the chain (slow: it reads every swap transaction of the window).

Method, written down before looking at any wallet's results after the start:
  1. Window: the 30 days before 12 Oct 2026 17:00 UTC.
  2. Every swap in the three most liquid Shadow Exchange pools (S/USDC.e, WETH/S, USDC.e/WETH, 0.15%).
  3. The trader of a swap = the account that signed the transaction. Kept: ordinary wallets (no
     contract code) trading through a public router (a contract that at least 20 different wallets
     sent swaps through) -- this drops arbitrage and market-making bots, which call their own contracts.
  4. A wallet's flows per transaction = what it paid and received in those pools (several pools in one
     transaction are netted, so S bought and sold on the way cancels out).
  5. Trading P&L = the value at the start (pool prices at 12 Oct 17:00 UTC) of everything bought minus
     everything paid over the 30 days, i.e. the gain over not trading. Volume = USD value of what was
     paid, at the time.
  6. Eligible: at least 10 trades on at least 5 different days, at most 20 trades a day on average,
     at least USD 1,000 traded. Ranked by trading P&L; the top 5 are copied (one bot each).
This is how copy-trading leaderboards pick traders (recent P&L), so the test asks whether last
month's best traders stay good once copied.

    python scripts/select_sonic_leaders.py --cache /tmp/sonic-cache
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantdesk.bots.sonic_chain import POOLS, SWAP, TOKENS, SonicRpc, decode_swap, usd_price  # noqa: E402

T0 = pd.Timestamp("2026-10-12T17:00Z")
DAYS = 30
TOP = 5
RULES = {"min_trades": 10, "min_days": 5, "max_trades_per_day": 20, "min_volume_usd": 1000.0, "router_min_wallets": 20}


def cached(path: Path, fn):
    if path.exists():
        return pickle.loads(path.read_bytes())
    v = fn()
    path.write_bytes(pickle.dumps(v))
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/tmp/sonic-cache")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "config" / "sonic_leaders.json"))
    args = ap.parse_args()
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    rpc = SonicRpc()

    b0, bs = cached(cache / "blocks.pkl", lambda: (rpc.block_at(int(T0.timestamp())),
                                                   rpc.block_at(int((T0 - pd.Timedelta(days=DAYS)).timestamp()))))
    print("window blocks", bs, b0 - 1)

    def get_logs():
        out = []
        for name, (addr, _, _) in POOLS.items():
            for a in range(bs, b0, 200_000):
                out += rpc.logs({"address": addr, "topics": [SWAP]}, a, min(a + 199_999, b0 - 1))
            print(name, len(out), flush=True)
        return out
    swaps = [decode_swap(lg) for lg in cached(cache / "logs.pkl", get_logs)]
    assert all(s["ts"] < T0.timestamp() for s in swaps), "a swap from after the start slipped in"

    def get_senders():
        out = pickle.loads((cache / "txfrom.pkl").read_bytes()) if (cache / "txfrom.pkl").exists() else {}
        todo = sorted({s["tx"] for s in swaps} - set(out))
        for i in range(0, len(todo), 1000):
            ch = todo[i:i + 1000]
            for h, t in zip(ch, rpc.batch([("eth_getTransactionByHash", [h]) for h in ch])):
                out[h] = (t["from"].lower(), (t.get("to") or "").lower(), int(t["value"], 16), int(t["blockNumber"], 16))
            if i % 20000 == 0:
                (cache / "txfrom.pkl").write_bytes(pickle.dumps(out))
                print("senders", i, len(todo), time.strftime("%H:%M:%S"), flush=True)
        return out
    senders = get_senders()                       # resumes from the cache, fetches what is missing
    (cache / "txfrom.pkl").write_bytes(pickle.dumps(senders))

    # public routers: contracts that many different wallets send their swaps through
    by_to = collections.defaultdict(set)
    for h, (frm, to, _, _) in senders.items():
        by_to[to].add(frm)
    routers = {to for to, w in by_to.items() if len(w) >= RULES["router_min_wallets"]}
    wallets = sorted({senders[s["tx"]][0] for s in swaps if senders[s["tx"]][1] in routers})
    codes = cached(cache / "codes.pkl", lambda: dict(zip(wallets, rpc.batch([("eth_getCode", [w, hex(b0)]) for w in wallets]))))
    eoa = {w for w in wallets if codes.get(w) in ("0x", "0x0", None)}
    print("routers", len(routers), "wallets via routers", len(wallets), "of which ordinary wallets", len(eoa))

    # USD prices over the window (from the S/USDC.e and USDC.e/WETH pools) and at the start
    series = {}
    for coin, pool in (("S", "S/USD"), ("ETH", "USD/ETH")):
        pts = sorted((s["block"], s["idx"], usd_price(coin, s["sqrt"])) for s in swaps if s["pool"] == pool)
        series[coin] = ([(b, i) for b, i, _ in pts], [p for _, _, p in pts])
    px_t0 = {"USD": 1.0, "S": series["S"][1][-1], "ETH": series["ETH"][1][-1]}

    def px_at(coin, block, idx):
        if coin == "USD":
            return 1.0
        keys, vals = series[coin]
        k = bisect.bisect_right(keys, (block, idx)) - 1
        return vals[max(k, 0)]

    flows = collections.defaultdict(lambda: collections.defaultdict(float))
    meta = {}
    for s in swaps:
        frm, to, _, _ = senders[s["tx"]]
        if frm not in eoa or to not in routers:
            continue
        f = flows[s["tx"]]
        f[s["t0"]] -= s["a0"] / 10 ** TOKENS[s["t0"]][1]
        f[s["t1"]] -= s["a1"] / 10 ** TOKENS[s["t1"]][1]
        meta.setdefault(s["tx"], (frm, s["block"], s["idx"], s["ts"]))
    stats = collections.defaultdict(lambda: {"trades": 0, "days": set(), "volume_usd": 0.0, "pnl_usd": 0.0,
                                             "net": collections.defaultdict(float)})
    for h, f in flows.items():
        f = {k: v for k, v in f.items() if abs(v) > 1e-12}
        paid = {k: -v for k, v in f.items() if v < 0}
        if not paid or not any(v > 0 for v in f.values()):
            continue
        w, blk, idx, ts = meta[h]
        st = stats[w]
        st["trades"] += 1
        st["days"].add(pd.Timestamp(ts, unit="s", tz="UTC").date())
        st["volume_usd"] += sum(v * px_at(k, blk, idx) for k, v in paid.items())
        for k, v in f.items():
            st["pnl_usd"] += v * px_t0[k]
            st["net"][k] += v
    table = []
    for w, st in stats.items():
        table.append({"wallet": w, "trades": st["trades"], "days": len(st["days"]),
                      "volume_usd": round(st["volume_usd"], 2), "pnl_usd": round(st["pnl_usd"], 2),
                      "pnl_per_volume": round(st["pnl_usd"] / st["volume_usd"], 5) if st["volume_usd"] else None,
                      "net": {k: round(v, 6) for k, v in st["net"].items()}})
    ok = [r for r in table if r["trades"] >= RULES["min_trades"] and r["days"] >= RULES["min_days"]
          and r["trades"] <= RULES["max_trades_per_day"] * DAYS and r["volume_usd"] >= RULES["min_volume_usd"]]
    ok.sort(key=lambda r: -r["pnl_usd"])
    print(f"{len(table)} wallets traded, {len(ok)} eligible")
    for r in ok[:15]:
        print(r)
    out = {
        "selected_on": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
        "method": __doc__.split("\n\n")[2].strip(),
        "window": {"from": (T0 - pd.Timedelta(days=DAYS)).isoformat(), "to": T0.isoformat(), "blocks": [bs, b0 - 1]},
        "rules": RULES, "prices_at_start_usd": px_t0,
        "counts": {"swaps": len(swaps), "transactions": len(senders), "public_routers": len(routers),
                   "wallets_via_routers": len(wallets), "ordinary_wallets": len(eoa), "wallets_with_trades": len(table),
                   "eligible": len(ok)},
        "leaders": [{"rank": i + 1, **r} for i, r in enumerate(ok[:TOP])],
        "runners_up": [{"rank": i + 1 + TOP, **r} for i, r in enumerate(ok[TOP:TOP + 15])],
    }
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + "\n")
    print("written", args.out)


if __name__ == "__main__":
    main()
