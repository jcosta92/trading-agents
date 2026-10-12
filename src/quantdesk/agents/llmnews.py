"""A6: bounded LLM news challenger.

Headlines for the 100 most liquid S&P 500 names are written to a daily inbox; a scheduled Claude
task labels them (event type, direction, materiality 0-3) - no paid API. This agent then applies
fixed rules on the paper run's first market-open cycle after the labels arrive:

BUY  up to 3 names with materiality >= 2, direction +1, and a positive 1-month return.
EXIT stop-loss 7% below entry, take-profit 10% above entry, or after 5 business days
     (protective orders, checked every hourly cycle).
Not backtestable honestly (no point-in-time news archive; the model has seen the past).
"""
from __future__ import annotations

import logging

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
        return 60

    def extra_needs(self) -> set[str]:
        return {"news"}

    def decision_due(self, asof, ledger) -> bool:
        return ledger.meta.get("last_decision") != pd.Timestamp(asof).date().isoformat()

    def prepare_live(self, runner, ledger: Ledger, asof: pd.Timestamp) -> dict:
        from ..llm import load_labels
        today = runner.now.strftime("%Y-%m-%d")
        labels, status = load_labels(runner.state_dir, today)
        out = {"llm_labels": labels, "llm_status": status}
        if labels:
            cands = sorted({l["ticker"] for l in labels if l["direction"] == 1
                            and l["materiality"] >= int(self.p.get("min_materiality", 2))})
            runner.ensure_history(cands, days=int(self.lookback_days()))
        return out

    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        labels = ctx.extras.get("llm_labels")
        if labels is None:
            ctx.note(f"no trading today yet: {ctx.extras.get('llm_status', 'no labels')}")
            return []                                  # not marked as decided: retried next cycle
        if not self.once_per_day(ctx, ledger):
            return []
        p = self.p
        held = {iid for iid, pos in ledger.positions.items() if pos.kind == "stock" and pos.qty > 0}
        slots = int(p["max_positions"]) - len(held)
        mw = int(p["momentum_window"])
        cands = {}
        for ev in labels:
            t = ev["ticker"]
            if ev["direction"] != 1 or ev["materiality"] < int(p["min_materiality"]) or t in held:
                continue
            score = (ev["materiality"], ev["confidence"])
            if t not in cands or score > cands[t]["score"]:
                cands[t] = {"score": score, "event": ev}
        picks, checked = [], {}
        for t, c in sorted(cands.items(), key=lambda kv: kv[1]["score"], reverse=True):
            if len(picks) >= slots:
                break
            px = ctx.history(t)
            if len(px) < mw + 1:
                checked[t] = "no price history"
                continue
            mom = float(px.iloc[-1] / px.iloc[-mw - 1] - 1)
            if mom <= 0:
                checked[t] = f"1-month return {mom:+.1%} - skipped"
                continue
            checked[t] = f"1-month return {mom:+.1%} - BUY"
            picks.append((t, c["event"], mom))
        ledger.meta["last_candidates"] = {t: {"event": c["event"], "check": checked.get(t)} for t, c in list(cands.items())[:10]}
        bought = {t for t, _, _ in picks}
        ledger.meta["signals"] = {
            t: {"state": "BUY" if t in bought else ("SKIP" if t in checked else "NO SLOT"),
                "event": c["event"]["event_type"], "materiality": c["event"]["materiality"],
                "confidence": c["event"]["confidence"], "check": checked.get(t, "not checked (positions full)")}
            for t, c in sorted(cands.items(), key=lambda kv: kv[1]["score"], reverse=True)[:8]}
        if not picks:
            return []
        w = (1.0 - float(p.get("cash_buffer", 0.01))) / int(p["max_positions"])
        for t, _, _ in picks:
            self.registry.stock(t)
        orders = rebalance_to_weights(ctx, ledger, {t: w for t, _, _ in picks}, self.cm, venue_override=self.venue_override,
                                      drift_band=float("inf"), max_cost_frac=0.02, min_trade_eur=20.0,
                                      manage=set(), reason="news")
        by_t = {t: (ev, mom) for t, ev, mom in picks}
        for o in orders:
            if o.qty > 0 and o.inst_id in by_t:
                ev, mom = by_t[o.inst_id]
                o.reason = (f"BUY signal: {ev['event_type']} news, materiality {ev['materiality']}, direction +1 "
                            f"(\"{ev['title'][:80]}\"), 1-month return {mom:+.1%}")
                o.meta["protect"] = {"stop_pct": float(p["stop_loss"]), "target_pct": float(p["take_profit"]),
                                     "max_days": int(p["hold_days"]),
                                     "label": f"stop -{float(p['stop_loss']):.0%}, take-profit +{float(p['take_profit']):.0%}, "
                                              f"time exit {p['hold_days']} days"}
        return orders
