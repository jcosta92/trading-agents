"""An LLM trader in the style of nof1's Alpha Arena, where xAI's Grok was among the top models
(Season 1, Oct-Nov 2025; Grok 4.20 won Season 1.5 on US stocks, Dec 2025).

Every few hours the model receives the same kind of prompt Alpha Arena used (per coin: price,
EMA20, MACD, RSI7/RSI14 series on 1-hour candles and 4-hour context; the account and open
positions with their exit plans) and answers in JSON: buy_to_enter / hold / close per coin, with a
profit target, a stop loss and an invalidation condition. Adapted to this test: Bitvavo EUR spot,
long only, no leverage, at most 40% of the account per coin.

Decisions are made live only, once per decision time, and stored in state/bots/llm/<bot>.jsonl;
the replay engine then executes the stored decision at that candle like any other order. Nothing
is decided for past candles (no look-ahead, no cost when the loop catches up after downtime).
A spending cap stops the calls; without an API key the bot simply holds cash.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import lib
from .data import TF
from .strategies import Bot

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an autonomous crypto trading agent managing a EUR {capital:,.0f} account on Bitvavo spot markets.
Mission: maximise risk-adjusted returns through systematic, disciplined trading.

Assets: {coins} (spot, quoted in EUR). LONG ONLY: you can buy and later sell; there is no shorting and no leverage.
Costs: 0.25% taker fee per market order plus the bid-ask spread (round trip about 0.6%). Trade only when the expected
move clearly exceeds the costs.

Every {interval} hours you choose ONE action per coin:
- "buy_to_enter": open a long position (only if you hold none in that coin; no pyramiding)
- "close": sell the whole position in that coin
- "hold": do nothing

Position rules: no single position above 40% of the account value; size from available cash. Every entry needs an exit
plan: a profit_target above the current price and a stop_loss below it, sized so the loss at the stop is 1-3% of the
account value, with a reward-to-risk of at least 2:1, plus an invalidation_condition (a market condition that would
make you close early - you will see your exit plan again at every decision). Stops and targets are placed as orders.
When in doubt, hold rather than force a trade.

Answer with JSON only:
{{"decisions": [{{"coin": "BTC", "signal": "buy_to_enter|hold|close", "quantity": <coins to buy, 0 otherwise>,
 "profit_target": <EUR price or 0>, "stop_loss": <EUR price or 0>, "invalidation_condition": "<text>",
 "confidence": <0-1>, "risk_eur": <loss at the stop in EUR>, "justification": "<max 300 characters>"}}]}}
Give one entry per coin."""


def _series(s: pd.Series, n: int = 10, d: int = 2) -> list:
    return [None if pd.isna(x) else round(float(x), d) for x in s.iloc[-n:]]


def _coin_block(coin: str, h1: pd.DataFrame, h4: pd.DataFrame) -> str:
    c = h1["close"]
    e20 = lib.ema(c, 20)
    m, _, _ = lib.macd(c)
    r7, r14 = lib.rsi(c, 7), lib.rsi(c, 14)
    px = float(c.iloc[-1])
    dec = 2 if px >= 10 else (4 if px >= 0.1 else 6)
    out = [f"ALL {coin} DATA", f"current_price = {px:.{dec}f}, current_ema20 = {float(e20.iloc[-1]):.{dec}f}, "
           f"current_macd = {float(m.iloc[-1]):.{dec + 1}f}, current_rsi (7 period) = {float(r7.iloc[-1]):.2f}",
           "Intraday series (1-hour candles, oldest -> latest):",
           f"Close prices: {_series(c, 10, dec)}", f"EMA indicators (20-period): {_series(e20, 10, dec)}",
           f"MACD indicators: {_series(m, 10, dec + 1)}", f"RSI indicators (7-Period): {_series(r7)}",
           f"RSI indicators (14-Period): {_series(r14)}"]
    if h4 is not None and len(h4) > 60:
        c4 = h4["close"]
        a3, a14 = lib.atr(h4, 3), lib.atr(h4, 14)
        m4, _, _ = lib.macd(c4)
        out += ["Longer-term context (4-hour timeframe):",
                f"20-Period EMA: {float(lib.ema(c4, 20).iloc[-1]):.{dec}f} vs. 50-Period EMA: {float(lib.ema(c4, 50).iloc[-1]):.{dec}f}",
                f"3-Period ATR: {float(a3.iloc[-1]):.{dec}f} vs. 14-Period ATR: {float(a14.iloc[-1]):.{dec}f}",
                f"Current Volume: {float(h4['volume'].iloc[-1]):.2f} vs. Average Volume: {float(h4['volume'].iloc[-20:].mean()):.2f}",
                f"MACD indicators: {_series(m4, 10, dec + 1)}", f"RSI indicators (14-Period): {_series(lib.rsi(c4, 14))}"]
    return "\n".join(out)


class LLMAgentBot(Bot):
    """Alpha-Arena-style LLM trader. Spec params: coins, interval_hours, provider (xai), model,
    budget_usd, price_in / price_out (USD per 1M tokens), api_key_env."""
    timeframe = "1h"

    def __init__(self, bot_id, spec):
        super().__init__(bot_id, spec)
        self.decisions: dict[str, dict] = {}

    def instruments(self) -> list[str]:
        return list(self.p.get("coins", []))

    def warmup(self) -> int:
        return 0

    # ------------------------------------------------------------------ decision log
    def log_path(self, state: Path) -> Path:
        return Path(state) / "llm" / f"{self.id}.jsonl"

    def load_decisions(self, state: Path) -> list[dict]:
        p = self.log_path(state)
        rows = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        self.decisions = {r["candle"]: r for r in rows if r.get("decisions") is not None}
        return rows

    def spent_usd(self, rows) -> float:
        return float(sum(r.get("cost_usd", 0.0) for r in rows))

    # ------------------------------------------------------------------ live decision
    def maybe_decide(self, state: Path, now: pd.Timestamp, h1: dict, h4: dict, led, fx, client=None,
                     start: pd.Timestamp | None = None) -> str:
        """Ask the model for a decision on the latest closed 1-hour candle when a decision time is due."""
        rows = self.load_decisions(state)
        if not h1:
            return "no candles"
        latest = min(df.index[-1] for df in h1.values() if len(df))
        close_t = latest + TF["1h"]
        interval = int(self.p.get("interval_hours", 4))
        if close_t.hour % interval != 0:
            return "not a decision hour"
        if latest.isoformat() in self.decisions or any(r.get("candle") == latest.isoformat() for r in rows):
            return "already decided"
        if now - close_t > pd.Timedelta(minutes=int(self.p.get("max_delay_minutes", 45))):
            return "too late for this candle (catching up): no decision"
        if self.spent_usd(rows) >= float(self.p.get("budget_usd", 8.0)):
            return "budget used up: holding"
        key = os.environ.get(self.p.get("api_key_env", "XAI_API_KEY"), "")
        if client is None and not key:
            return f"no {self.p.get('api_key_env', 'XAI_API_KEY')} set: holding cash"
        user = self.user_prompt(now, h1, h4, led, fx, len(rows), start)
        system = SYSTEM_PROMPT.format(capital=led.initial_capital, interval=interval,
                                      coins=", ".join(c.replace("_SPOT", "") for c in self.instruments()))
        rec = {"candle": latest.isoformat(), "asked_at": now.isoformat(), "model": self.p.get("model"),
               "prompt_sha": hashlib.sha256((system + user).encode()).hexdigest()[:16]}
        try:
            client = client or XaiClient(key, self.p.get("model", "grok-4.3"))
            raw, usage = client.chat(system, user)
            rec["raw"] = raw[:6000]
            rec["usage"] = usage
            rec["cost_usd"] = round(usage.get("prompt_tokens", 0) * float(self.p.get("price_in", 1.25)) / 1e6
                                    + usage.get("completion_tokens", 0) * float(self.p.get("price_out", 2.5)) / 1e6, 6)
            rec["decisions"] = self.parse(raw)
        except Exception as exc:                          # a failed call = hold this time
            rec["error"] = repr(exc)[:500]
            rec["decisions"] = []
        p = self.log_path(state)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.decisions[rec["candle"]] = rec
        return f"decided: {[(d['coin'], d['signal']) for d in rec['decisions']]}" if not rec.get("error") else rec["error"]

    def user_prompt(self, now, h1, h4, led, fx, n_calls, start=None) -> str:
        mins = int((now - start).total_seconds() // 60) if start is not None else 0
        eq = led.equity(fx)
        blocks = [f"It has been {mins} minutes since you started trading. You have been invoked {n_calls} times.",
                  "ALL OF THE PRICE OR SIGNAL DATA BELOW IS ORDERED: OLDEST -> NEWEST", "",
                  "CURRENT MARKET STATE FOR ALL COINS"]
        for iid in self.instruments():
            if iid in h1 and len(h1[iid]) > 50:
                blocks += ["", _coin_block(iid.replace("_SPOT", ""), h1[iid], h4.get(iid))]
        pos = []
        plans = led.meta.get("state", {}).get("plans", {})
        for iid, p in led.positions.items():
            if p.qty <= 0:
                continue
            plan = plans.get(iid, {})
            pos.append({"coin": iid.replace("_SPOT", ""), "quantity": round(p.qty, 8), "entry_price": round(p.avg_price, 6),
                        "current_price": round(p.last_mark or p.avg_price, 6),
                        "unrealized_pnl_eur": round((p.last_mark - p.avg_price) * p.qty, 2) if p.last_mark else 0.0,
                        "exit_plan": {k: plan.get(k) for k in ("profit_target", "stop_loss", "invalidation_condition")},
                        "confidence": plan.get("confidence")})
        blocks += ["", "HERE IS YOUR ACCOUNT INFORMATION & PERFORMANCE",
                   f"Current Total Return (percent): {(eq / led.initial_capital - 1) * 100:.2f}%",
                   f"Available Cash: {led.cash_eur(fx):.2f} EUR", f"Current Account Value: {eq:.2f} EUR",
                   f"Current live positions & performance: {json.dumps(pos)}", "",
                   "Return your decision as the JSON described in the instructions."]
        return "\n".join(blocks)

    def parse(self, raw: str) -> list[dict]:
        txt = raw.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").split("\n", 1)[-1]
        a, b = txt.find("{"), txt.rfind("}")
        data = json.loads(txt[a:b + 1])
        out = []
        valid = {c.replace("_SPOT", ""): c for c in self.instruments()}
        for d in data.get("decisions", []):
            coin = str(d.get("coin", "")).upper().replace("/EUR", "").replace("-EUR", "")
            if coin not in valid:
                continue
            sig = str(d.get("signal", "hold")).lower()
            out.append({"coin": coin, "iid": valid[coin], "signal": sig,
                        "quantity": _num(d.get("quantity")), "profit_target": _num(d.get("profit_target")),
                        "stop_loss": _num(d.get("stop_loss")), "confidence": _num(d.get("confidence")),
                        "invalidation_condition": str(d.get("invalidation_condition", ""))[:300],
                        "justification": str(d.get("justification", ""))[:400]})
        return out

    # ------------------------------------------------------------------ execution in the replay
    def on_bar(self, ctx, bars: dict) -> None:
        rec = self.decisions.get(ctx.ts.isoformat())
        if not rec or ctx.state.get("applied") == rec["candle"]:
            return
        ctx.state["applied"] = rec["candle"]
        plans = ctx.state.setdefault("plans", {})
        eq = ctx.equity()
        for d in rec.get("decisions") or []:
            iid = d["iid"]
            bar = bars.get(iid)
            if bar is None:
                continue
            px = float(bar["close"])
            if d["signal"] == "close" and ctx.qty(iid) > 0:
                ctx.cancel("llm:", iid)
                ctx.place(iid, "sell", "market", qty=ctx.qty(iid), tag="llm:close")
                plans.pop(iid, None)
            elif d["signal"] == "buy_to_enter" and ctx.qty(iid) <= 0 and not ctx.orders("llm:", iid):
                tgt, stop = d.get("profit_target") or 0, d.get("stop_loss") or 0
                if not (stop and tgt and stop < px < tgt):
                    ctx.note(f"{d['coin']}: invalid exit plan (stop {stop}, target {tgt}, price {px}); skipped")
                    continue
                want = (d.get("quantity") or 0) * px
                eur = min(want if want > 0 else 0.2 * eq, 0.4 * eq, ctx.cash_eur() * 0.995)
                if eur < 5:
                    continue
                ctx.place(iid, "buy", "market", eur=eur, tag=f"llm:buy ({d.get('confidence')})")
                plans[iid] = {k: d.get(k) for k in ("profit_target", "stop_loss", "invalidation_condition",
                                                    "confidence", "justification")}
            elif d["signal"] == "sell_to_enter":
                ctx.note(f"{d['coin']}: short requested; not possible on spot, ignored")
        ctx.state["last_decision"] = {"candle": rec["candle"], "actions": [(d["coin"], d["signal"]) for d in rec.get("decisions") or []]}

    def on_fill(self, ctx, order, fill) -> None:
        iid = order["inst"]
        plan = ctx.state.get("plans", {}).get(iid)
        if order["side"] == "buy" and plan:
            ctx.place(iid, "sell", "stop", qty=fill.qty, price=float(plan["stop_loss"]), tag="llm:stop loss")
            ctx.place(iid, "sell", "limit", qty=fill.qty, price=float(plan["profit_target"]), tag="llm:profit target")
        elif order["side"] == "sell" and ctx.qty(iid) <= 1e-12:
            ctx.cancel("llm:", iid)
            ctx.state.get("plans", {}).pop(iid, None)


def _num(v):
    try:
        x = float(v)
        return None if np.isnan(x) else x
    except (TypeError, ValueError):
        return None


class XaiClient:
    """xAI's OpenAI-compatible chat completions API."""
    URL = "https://api.x.ai/v1/chat/completions"

    def __init__(self, key: str, model: str):
        self.key, self.model = key, model

    def chat(self, system: str, user: str) -> tuple[str, dict]:
        import requests
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}, "temperature": 0.2}
        last = None
        for attempt in range(3):
            try:
                r = requests.post(self.URL, json=body, timeout=120,
                                  headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                if r.status_code >= 500 or r.status_code == 429:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                j = r.json()
                usage = j.get("usage", {}) or {}
                out = {k: usage.get(k) or 0 for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
                # reasoning tokens are billed as output; count them even if the API also includes them in
                # completion_tokens (over-counting only makes the spending cap stricter)
                out["reasoning_tokens"] = ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
                out["completion_tokens"] = out["completion_tokens"] + out["reasoning_tokens"]
                return j["choices"][0]["message"]["content"], out
            except Exception as exc:
                last = exc
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"xAI call failed: {last}")
