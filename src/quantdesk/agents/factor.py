"""A2: long-only US large-cap factor tilt (momentum + value + quality), ~10 names, monthly buffer."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..context import Context
from ..data.universe import members_on
from ..execution import OrderIntent
from ..ledger import Ledger
from ..portfolio import rebalance_to_weights
from .base import Agent


def _z(s: pd.Series, winsor: float) -> pd.Series:
    s = s.astype(float)
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 5:
        return pd.Series(np.nan, index=s.index)
    lo, hi = s.quantile(winsor), s.quantile(1 - winsor)
    s = s.clip(lo, hi)
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


class FactorEquity(Agent):
    def lookback_days(self) -> int:
        return int(self.p.get("momentum_lookback", 252)) + 40

    def extra_needs(self) -> set[str]:
        return {"sp500_panel", "fundamentals"}

    def decision_due(self, asof, ledger) -> bool:
        return ledger.meta.get("last_rebalance") != self.month_key(asof)

    # ------------------------------------------------------------------ universe
    def candidates(self, ctx: Context) -> tuple[list[str], dict[str, str]]:
        excl = set(self.p.get("exclude_sectors", []))
        sectors: dict[str, str] = dict(ctx.extras.get("sectors", {}))
        if ctx.mode == "backtest":
            members = [t for t in members_on(ctx.extras["membership"], ctx.asof) if "~" not in t]
        else:
            members = list(sectors)
        return [t for t in members if sectors.get(t) not in excl], sectors

    def eligible(self, ctx: Context) -> pd.DataFrame:
        tickers, sectors = self.candidates(ctx)
        lb = int(self.p.get("momentum_lookback", 252))
        adv_w = int(self.p.get("adv_window", 60))
        close = ctx.panel(tickers, "close", lookback=lb + 5)
        vol = ctx.panel(tickers, "volume", lookback=adv_w)
        rows = []
        for t in close.columns:
            px = close[t].dropna()
            if len(px) < lb + 1 or (ctx.asof - px.index[-1]).days > 7:
                continue
            dv = (close[t] * vol[t]).dropna().iloc[-adv_w:] if t in vol.columns else pd.Series(dtype=float)
            mdv = float(dv.median()) if len(dv) >= adv_w // 2 else 0.0
            rows.append({"ticker": t, "price": float(px.iloc[-1]), "mdv": mdv,
                         "mom": float(px.iloc[-1 - int(self.p.get("momentum_skip", 21))] / px.iloc[-1 - lb] - 1.0),
                         "sector": sectors.get(t)})
        df = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame(columns=["price", "mdv", "mom", "sector"])
        cov = {"members": len(tickers), "with_data": int(close.shape[1]), "with_history": len(df)}
        df = df[(df["price"] >= float(self.p["min_price"])) & (df["mdv"] >= float(self.p["min_median_dollar_volume"]))]
        cov["eligible"] = len(df)
        ctx.extras.setdefault("_diag", {})["coverage"] = cov
        return df

    # ------------------------------------------------------------------ scoring
    def score(self, ctx: Context, elig: pd.DataFrame) -> tuple[pd.Series, str]:
        w = float(self.p.get("winsor", 0.05))
        parts = {"momentum": _z(elig["mom"], w)}
        mode = "momentum_only"
        fund = ctx.extras.get("fundamentals")
        wanted = set(self.p.get("factors", ["momentum"]))
        if ctx.mode == "live" and fund is not None and not fund.empty and wanted & {"value", "quality"}:
            f = fund.reindex(elig.index)
            ev_ebitda = pd.to_numeric(f.get("enterpriseToEbitda"), errors="coerce")
            value = 1.0 / ev_ebitda.where(ev_ebitda > 0)
            roe = pd.to_numeric(f.get("returnOnEquity"), errors="coerce")
            gm = pd.to_numeric(f.get("grossMargins"), errors="coerce")
            de = pd.to_numeric(f.get("debtToEquity"), errors="coerce")
            coverage = float(pd.concat([value, roe, gm], axis=1).notna().all(axis=1).mean()) if len(f) else 0.0
            ctx.extras.setdefault("_diag", {})["fundamentals_coverage"] = round(coverage, 3)
            if coverage >= float(self.p.get("min_fundamentals_coverage", 0.7)):
                if "value" in wanted:
                    # negative EBITDA -> worst value score rather than dropped
                    vz = _z(value, w)
                    parts["value"] = vz.fillna(vz.min() if vz.notna().any() else 0.0)
                if "quality" in wanted:
                    q = pd.concat([_z(roe, w), _z(gm, w), _z(-de, w)], axis=1).mean(axis=1)
                    parts["quality"] = q
                mode = "+".join(sorted(parts))
            else:
                ctx.note(f"fundamentals coverage {coverage:.0%} below threshold - momentum only")
        comp = pd.concat(parts, axis=1).mean(axis=1, skipna=True).dropna()
        return comp.sort_values(ascending=False), mode

    # ------------------------------------------------------------------ live preparation
    def prepare_live(self, runner, ledger: Ledger, asof: pd.Timestamp) -> dict:
        """On rebalance days: load the S&P 500 panel and fetch fundamentals for eligible names."""
        if not self.decision_due(asof, ledger):
            return {}
        sectors = runner.ensure_sp500_panel()
        extras = {"sectors": sectors}
        ctx = runner.context(asof, extras)
        elig = self.eligible(ctx)
        if "value" in self.p.get("factors", []) or "quality" in self.p.get("factors", []):
            extras["fundamentals"] = runner.fetch_fundamentals(list(elig.index))
        return extras

    # ------------------------------------------------------------------ decisions
    def decide(self, ctx: Context, ledger: Ledger) -> list[OrderIntent]:
        key = self.month_key(ctx.asof)
        if ledger.meta.get("last_rebalance") == key:
            return []
        elig = self.eligible(ctx)
        if len(elig) < int(self.p["n_holdings"]) * 3:
            ctx.note(f"only {len(elig)} eligible stocks - skipping rebalance")
            return []
        scores, mode = self.score(ctx, elig)
        ranks = scores.rank(ascending=False, pct=True)
        held = [iid for iid, p in ledger.positions.items() if p.kind == "stock"]
        sell_pct = float(self.p.get("sell_rank_pct", 0.25))
        keep = [t for t in held if t in ranks.index and ranks[t] <= sell_pct]
        n = int(self.p["n_holdings"])
        buys = [t for t in scores.index if t not in keep][: max(0, n - len(keep))]
        w = (1.0 - float(self.p.get("cash_buffer", 0.01))) / n
        targets = {t: w for t in keep + buys}
        sectors = ctx.extras.get("sectors", {})
        for t in targets:
            self.registry.stock(t, sectors.get(t))
        quarterly = ctx.asof.month in (1, 4, 7, 10) or not held
        ledger.meta["last_rebalance"] = key
        ledger.meta["mode"] = mode
        ledger.meta["holdings_target"] = sorted(targets)
        ledger.meta["top_scores"] = {t: round(float(scores[t]), 3) for t in scores.index[:15]}
        diag = ctx.extras.get("_diag", {})
        ledger.meta.setdefault("coverage_log", []).append({"month": key, **diag.get("coverage", {}),
                                                           "mode": mode})
        ledger.meta["coverage_log"] = ledger.meta["coverage_log"][-400:]
        return rebalance_to_weights(ctx, ledger, targets, self.cm, venue_override=self.venue_override,
                                    drift_band=float(self.p.get("quarterly_resize_band", 0.3)) if quarterly else math.inf,
                                    max_cost_frac=0.02, min_trade_eur=20.0, manage=set(held),
                                    reason=f"factor rebalance ({mode})")
