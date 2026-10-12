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

    Exits (checked every hourly cycle in the paper run): take-profit when 50% of the credit is captured,
    stop-loss when the spread costs 2x the credit to close or the index trades at/below the short strike,
    time exit at 21 DTE.
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

    # ------------------------------------------------------------------ exits (checked every cycle)
    def _open_spreads(self, ledger: Ledger) -> list[dict]:
        spreads = [s for s in ledger.meta.get("spreads", []) if ledger.qty(s["short"]) < 0]
        for s in spreads:                    # use the credit actually received, not the decision-day estimate
            sp, lp = ledger.positions.get(s["short"]), ledger.positions.get(s["long"])
            if sp is not None and lp is not None:
                s["credit"] = round(sp.avg_price - lp.avg_price, 4)
        ledger.meta["spreads"] = spreads
        return spreads

    def _exit_reason(self, ctx: Context, s: dict, S: float | None) -> tuple[str | None, object, object]:
        p = self.p
        qs, ql = ctx.quote(s["short"]), ctx.quote(s["long"])
        today = (ctx.now.tz_convert(None) if ctx.now.tzinfo else ctx.now).normalize()
        dte = (pd.Timestamp(s["expiry"]) - today).days
        if qs is not None and ql is not None and s["credit"] > 0:
            debit = qs.mid - ql.mid
            s["last_debit_mid"] = round(debit, 4)
            if debit <= (1.0 - float(p["take_profit"])) * s["credit"]:
                return f"TAKE-PROFIT: spread at {debit:.2f} <= {1 - float(p['take_profit']):.0%} of credit {s['credit']:.2f}", qs, ql
            if debit >= float(p["stop_multiple"]) * s["credit"]:
                return f"STOP-LOSS: spread at {debit:.2f} >= {p['stop_multiple']}x credit {s['credit']:.2f}", qs, ql
        if dte <= int(p["exit_dte"]):
            return f"TIME EXIT: {dte} days to expiry <= {p['exit_dte']}", qs, ql
        if p.get("close_on_breach", True) and S is not None and S <= s["short_strike"]:
            return f"STOP-LOSS: index {S:.2f} <= short strike {s['short_strike']:g}", qs, ql
        return None, qs, ql

    def intraday(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        S = ctx.mark(self.p.get("underlying", "XSP"))
        orders: list[OrderIntent] = []
        tag = ctx.now.strftime("%Y-%m-%dT%H")
        for s in self._open_spreads(ledger):
            why, qs, ql = self._exit_reason(ctx, s, S)
            if not why:
                continue
            lq = ledger.qty(s["long"])
            wing_bid = ql.bid if ql is not None else None
            if lq and wing_bid is not None and wing_bid <= 0:
                # the wing is quoted at zero: buy back the short alone, let the wing expire
                orders.append(OrderIntent(s["short"], -ledger.qty(s["short"]), f"{why} (wing worthless, kept)"))
            else:
                combo = f"close-{s['short']}-{tag}"
                orders.append(OrderIntent(s["short"], -ledger.qty(s["short"]), why, combo=combo))
                if lq:
                    orders.append(OrderIntent(s["long"], -lq, why, combo=combo))
        return orders

    # ------------------------------------------------------------------ daily decision: entries only
    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        if not self.once_per_day(ctx, ledger):
            return []
        S = ctx.mark(self.p.get("underlying", "XSP"))
        still_open = [s for s in self._open_spreads(ledger) if self._exit_reason(ctx, s, S)[0] is None]
        p = self.p
        self._sig = {"state": "WAIT", "index": round(S, 2) if S else None, "open_spreads": len(still_open),
                     "buy_rule": f"VIX between {p['vix_min']} and its 1-year {float(p['vix_max_pctile']):.0%} percentile, "
                                 f"at most {p['max_open']} open, {p['entry_gap_days']} business days apart",
                     "sell_rule": f"take-profit at {float(p['take_profit']):.0%} of credit, stop at {p['stop_multiple']}x credit "
                                  f"or index <= short strike, time exit at {p['exit_dte']} DTE"}
        orders = self._entry(ctx, ledger, still_open, S, self.day_key(ctx)) or []
        if orders:
            self._sig["state"] = "ENTRY"
        ledger.meta["signals"] = {f"{p.get('underlying', 'XSP')} put spread": self._sig}
        return orders

    # ------------------------------------------------------------------ entries
    def _entry(self, ctx: Context, ledger: Ledger, open_spreads: list[dict], S: float | None, day: str):
        p = self.p
        sig = getattr(self, "_sig", {})
        if S is None or len(open_spreads) >= int(p["max_open"]):
            sig["state"] = "NO DATA" if S is None else "FULL"
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
        sig["vix"], sig["vix_cap"] = round(v, 2), round(cap, 2)
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
        why = (f"ENTRY signal: VIX {v:.1f} within [{p['vix_min']:g}, {cap:.1f}], {exp} expiry, "
               f"credit {credit:.2f} for {width:g} wide")
        return [OrderIntent(si.id, -qty, f"{why} - SELL {short_k:g}P ({abs(sh['delta']):.2f} delta)", combo=combo),
                OrderIntent(li.id, qty, f"{why} - BUY {long_k:g}P wing", combo=combo)]
