"""A6: bounded LLM news challenger.

prepare_live() (paper run only): pick the monthly universe, pull fresh headlines, have the LLM
label them, and charge the inference cost to this agent's ledger.
decide(): fixed rules on the labels - buy up to 3 names with material positive news and positive
1-month momentum, hold 5 trading days, 7% stop."""
from __future__ import annotations

import logging
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from ..context import Context
from ..execution import OrderIntent
from ..ledger import Ledger
from ..portfolio import rebalance_to_weights
from .base import Agent

log = logging.getLogger(__name__)


class LLMNews(Agent):
    backtestable = False

    def lookback_days(self) -> int:
        return 90

    def extra_needs(self) -> set[str]:
        return {"news"}

    def decision_due(self, asof, ledger) -> bool:
        return ledger.meta.get("last_decision") != pd.Timestamp(asof).date().isoformat()

    def needs_universe_refresh(self, asof, ledger) -> bool:
        return ledger.meta.get("universe_month") != self.month_key(asof) or not ledger.meta.get("universe")

    # ------------------------------------------------------------------ live preparation
    def select_universe(self, close: pd.DataFrame, volume: pd.DataFrame) -> list[str]:
        n = int(self.p.get("universe_size", 100))
        dv = (close * volume).iloc[-60:].median()
        return list(dv.dropna().sort_values(ascending=False).index[:n])

    def prepare_live(self, runner, ledger: Ledger, asof: pd.Timestamp) -> dict:
        """Fetch news, label it with the LLM, charge the cost. Returns extras for decide()."""
        from ..llm import NewsClassifier
        out = {"llm_events": [], "llm_status": "ok"}
        if self.needs_universe_refresh(asof, ledger):
            runner.ensure_sp500_panel()
            ctx = runner.context(asof, {})
            tickers = list(runner.sp500_sectors)
            close = ctx.panel(tickers, "close", lookback=60)
            vol = ctx.panel(tickers, "volume", lookback=60)
            uni = self.select_universe(close, vol.reindex(columns=close.columns))
            if len(uni) >= 20:
                ledger.meta["universe"] = uni
                ledger.meta["universe_month"] = self.month_key(asof)
        clf = NewsClassifier(self.p["model"], float(self.p["price_input_per_mtok"]),
                             float(self.p["price_output_per_mtok"]))
        if not clf.available:
            out["llm_status"] = "disabled: ANTHROPIC_API_KEY not set"
            return out
        month = self.month_key(runner.now)
        spend = ledger.meta.setdefault("llm_spend_usd", {})
        if spend.get(month, 0.0) >= float(self.p.get("monthly_llm_budget_usd", 15.0)):
            out["llm_status"] = "monthly LLM budget exhausted"
            return out
        universe = ledger.meta.get("universe") or []
        cutoff = runner.now - timedelta(hours=int(self.p.get("lookback_hours", 24)))
        seen = set(ledger.meta.get("seen_ids", []))
        items = []
        for t in universe:
            for it in runner.fetch_news(t):
                pub = pd.Timestamp(it["published"]) if it.get("published") else None
                if pub is not None and pub.tzinfo is None:
                    pub = pub.tz_localize("UTC")
                if pub is None or pub < cutoff or it["id"] in seen:
                    continue
                it["ticker"] = t
                items.append(it)
        # de-duplicate identical headlines across tickers (keep first)
        uniq, titles = [], set()
        for it in items:
            key = (it["ticker"], it["title"].lower())
            if key not in titles:
                titles.add(key)
                uniq.append(it)
        bs = int(self.p.get("batch_size", 25))
        events, n_calls = [], 0
        for i in range(0, len(uniq), bs):
            if spend.get(month, 0.0) >= float(self.p.get("monthly_llm_budget_usd", 15.0)):
                out["llm_status"] = "budget reached mid-run"
                break
            batch = uniq[i:i + bs]
            try:
                labels, usage = clf.classify(batch)
            except Exception as exc:
                out["llm_status"] = f"LLM error: {exc}"[:200]
                break
            n_calls += 1
            ledger.charge(runner.ts, "USD", usage["cost_usd"], runner.fx, "inference",
                          note=f"{usage['input_tokens']} in / {usage['output_tokens']} out tokens, {usage['model']}")
            spend[month] = spend.get(month, 0.0) + usage["cost_usd"]
            by_id = {b["id"][:64]: b for b in batch}
            for lab in labels:
                src = by_id.get(lab["id"])
                if src is None:
                    continue
                lab["ticker"] = src["ticker"]
                lab["title"] = src["title"]
                lab["published"] = src.get("published")
                events.append(lab)
            runner.log_llm({"ts": runner.ts, "agent": self.id, "items": batch, "labels": labels, "usage": usage})
        ledger.meta["seen_ids"] = (list(seen) + [it["id"] for it in uniq])[-3000:]
        # price history for tickers that could become entries (momentum filter) and for holdings
        cands = sorted({e["ticker"] for e in events if e["direction"] == 1
                        and e["materiality"] >= int(self.p.get("min_materiality", 2))})
        held = [iid for iid, p in ledger.positions.items() if p.kind == "stock"]
        runner.ensure_history(cands + held, days=int(self.lookback_days()))
        out["llm_events"] = events
        out["llm_calls"] = n_calls
        out["headlines"] = len(uniq)
        return out

    # ------------------------------------------------------------------ rules
    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        day = ctx.asof.date().isoformat()
        if ledger.meta.get("last_decision") == day:
            return []
        ledger.meta["last_decision"] = day
        hold_days = int(self.p.get("hold_days", 5))
        stop = float(self.p.get("stop_loss", 0.07))
        entries = ledger.meta.setdefault("entries", {})
        for stale in [t for t in entries if t not in ledger.positions]:
            entries.pop(stale)
        kept: dict[str, float] = {}
        exits = set()
        today = ctx.now.normalize()
        for iid, pos in list(ledger.positions.items()):
            if pos.kind != "stock":
                continue
            e = entries.get(iid, {})
            opened = pd.Timestamp(e.get("date", pos.opened[:10]))
            held_days = int(np.busday_count(opened.date(), today.date()))
            m = ctx.mark(iid) or pos.last_mark or pos.avg_price
            if held_days >= hold_days:
                exits.add(iid)
                ctx.note(f"{iid}: holding period over")
            elif m <= pos.avg_price * (1 - stop):
                exits.add(iid)
                ctx.note(f"{iid}: stop loss")
            else:
                kept[iid] = pos.qty                     # keep as is (never resized)
        w = (1.0 - float(self.p.get("cash_buffer", 0.01))) / int(self.p.get("max_positions", 3))
        slots = int(self.p.get("max_positions", 3)) - len(kept)
        events = ctx.extras.get("llm_events", [])
        universe = set(ledger.meta.get("universe", []))
        cands = {}
        for ev in events:
            t = ev["ticker"]
            if (ev["direction"] != 1 or ev["materiality"] < int(self.p.get("min_materiality", 2))
                    or t not in universe or t in kept or t in exits):
                continue
            score = (ev["materiality"], ev["confidence"])
            if t not in cands or score > cands[t]["score"]:
                cands[t] = {"score": score, "event": ev}
        picks = []
        mw = int(self.p.get("momentum_window", 21))
        for t, c in sorted(cands.items(), key=lambda kv: kv[1]["score"], reverse=True):
            if len(picks) >= slots:
                break
            px = ctx.history(t)
            if len(px) < mw + 1 or px.iloc[-1] <= px.iloc[-mw - 1]:
                continue
            picks.append(t)
            self.registry.stock(t)
            entries[t] = {"date": today.date().isoformat(), "event": c["event"]}
        ledger.meta["last_candidates"] = {t: c["event"] for t, c in list(cands.items())[:10]}
        orders = [OrderIntent(iid, -ledger.qty(iid), "llm news exit") for iid in exits]
        if picks:
            tw = {t: w for t in picks}
            orders += rebalance_to_weights(ctx, ledger, tw, self.cm, venue_override=self.venue_override,
                                           drift_band=math.inf, max_cost_frac=0.02, min_trade_eur=20.0,
                                           reason="llm news entry")
        for iid in exits:
            entries.pop(iid, None)
        return orders
