"""A5: defined-risk volatility risk premium - XSP put credit spreads."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..context import Context
from ..execution import OrderIntent
from ..ledger import Ledger
from .base import Agent


class PutCreditSpreads(Agent):
    """Sell a ~16-delta XSP put and buy a put ~0.5% of the index lower, 30-45 DTE.

    Exits: 50% of credit captured, 21 DTE, or the index closing at/below the short strike.
    Size: total max loss of open spreads <= max_risk_frac x equity. Entries at most weekly and only
    when VIX is between vix_min and its trailing one-year 95th percentile.
    Fills cross the full quoted spread (sell at bid, buy at ask) - the 'natural' price. In backtests the
    orders decided on a close fill at the NEXT day's close (one day of delay, like a real end-of-day trader)."""

    fill_timing = "next_close"      # backtest: decide on today's close, fill at tomorrow's close quotes

    def static_instruments(self) -> list[str]:
        return [self.p.get("underlying", "XSP"), "VIX", "TBILL", "EURUSD"]

    def calendar_reference(self) -> list[str]:
        return [self.p.get("underlying", "XSP")]          # US trading days only

    def lookback_days(self) -> int:
        return 300

    def extra_needs(self) -> set[str]:
        return {"options"}

    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        day = ctx.asof.date().isoformat()
        if ledger.meta.get("last_decision") == day:
            return []
        ledger.meta["last_decision"] = day
        und = self.p.get("underlying", "XSP")
        spreads = [s for s in ledger.meta.get("spreads", []) if ledger.qty(s["short"]) < 0]
        S = ctx.mark(und)
        orders: list[OrderIntent] = []
        still_open = []
        for s in spreads:
            qs, ql = ctx.quote(s["short"]), ctx.quote(s["long"])
            dte = (pd.Timestamp(s["expiry"]) - ctx.asof).days
            why = None
            if qs is not None and ql is not None:
                debit = qs.mid - ql.mid
                s["last_debit_mid"] = round(debit, 4)
                if debit <= (1.0 - float(self.p["take_profit"])) * s["credit"]:
                    why = "take profit"
            if why is None and dte <= int(self.p["exit_dte"]):
                why = "time exit"
            if why is None and self.p.get("close_on_breach", True) and S is not None and S <= s["short_strike"]:
                why = "short strike breached"
            if why:
                combo = f"close-{s['short']}-{day}"
                orders.append(OrderIntent(s["short"], -ledger.qty(s["short"]), f"spread exit: {why}", combo=combo))
                lq = ledger.qty(s["long"])
                if lq:
                    orders.append(OrderIntent(s["long"], -lq, f"spread exit: {why}", combo=combo))
            else:
                still_open.append(s)
        ledger.meta["spreads"] = spreads           # positions are re-checked against the book next time
        entry = self._entry(ctx, ledger, still_open, S, day)
        if entry:
            orders += entry
        return orders

    # ------------------------------------------------------------------ entries
    def _entry(self, ctx: Context, ledger: Ledger, open_spreads: list[dict], S: float | None, day: str):
        p = self.p
        if S is None or len(open_spreads) >= int(p["max_open"]):
            return None
        last = ledger.meta.get("last_entry")
        if last and np.busday_count(pd.Timestamp(last).date(), ctx.asof.date()) < int(p["entry_gap_days"]):
            return None
        vix = ctx.history("VIX", lookback=252)
        if len(vix) < 200:
            ctx.note("VIX history too short")
            return None
        v = float(vix.iloc[-1])
        cap = float(vix.quantile(float(p["vix_max_pctile"])))
        if v < float(p["vix_min"]) or v > cap:
            ctx.note(f"VIX {v:.1f} outside [{p['vix_min']}, {cap:.1f}] - no entry")
            return None
        chain = ctx.option_chain(int(p["dte_min"]), int(p["dte_max"]), "P")
        if chain is None or chain.empty:
            ctx.note("no option chain")
            return None
        chain = chain.dropna(subset=["delta"])
        mid_dte = (int(p["dte_min"]) + int(p["dte_max"])) / 2
        exp = min(chain["expiry"].unique(), key=lambda e: abs((pd.Timestamp(e) - ctx.asof).days - mid_dte))
        c = chain[chain["expiry"] == exp].set_index("strike").sort_index()
        c = c[(c["bid"] > 0) & (c["ask"] > 0)]
        if c.empty:
            return None
        short_k = (c["delta"] + float(p["short_delta"])).abs().idxmin()
        width = max(1.0, float(round(float(p["width_pct"]) * S)))
        long_k = short_k - width
        if long_k not in c.index:
            ctx.note(f"long strike {long_k} not listed")
            return None
        sh, lg = c.loc[short_k], c.loc[long_k]
        credit = float(sh["bid"] - lg["ask"])
        hs = float(p.get("model_half_spread_min", 0.025))
        und_id = self.p.get("underlying", "XSP")
        si = self.registry.option(und_id, exp, "P", float(short_k), hs)
        li = self.registry.option(und_id, exp, "P", float(long_k), hs)
        # fees to open and close one spread (4 option orders), in USD
        rt_fees = 2 * (self.cm.fees(si, 1, float(sh["mid"]), "sell").total + self.cm.fees(li, 1, float(lg["mid"]), "buy").total)
        if credit * 100.0 < float(p.get("min_credit_cost_multiple", 2.0)) * rt_fees:
            ctx.note(f"credit {credit:.2f} < {p.get('min_credit_cost_multiple', 2.0)}x round-trip fees ({rt_fees:.2f} USD)")
            return None
        max_loss_usd = (width - credit) * 100.0
        eq_usd = ctx.fx.from_eur(ledger.equity(ctx.fx), "USD")
        used = sum(s["max_loss_usd"] * s["qty"] for s in open_spreads)
        qty = math.floor((float(p["max_risk_frac"]) * eq_usd - used) / max_loss_usd)
        if qty < 1:
            ctx.note(f"risk budget too small: max loss {max_loss_usd:.0f} USD/spread")
            return None
        rec = {"short": si.id, "long": li.id, "expiry": exp, "short_strike": float(short_k),
               "long_strike": float(long_k), "width": width, "qty": qty, "credit": round(credit, 4),
               "max_loss_usd": round(max_loss_usd, 2), "opened": day, "short_delta": round(float(sh["delta"]), 3),
               "vix": round(v, 2)}
        ledger.meta.setdefault("spreads", []).append(rec)
        ledger.meta["last_entry"] = day
        combo = f"open-{si.id}-{day}"
        return [OrderIntent(si.id, -qty, f"sell {short_k:g}P ({sh['delta']:.2f} delta)", combo=combo),
                OrderIntent(li.id, qty, f"buy {long_k:g}P wing", combo=combo)]
