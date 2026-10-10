"""Run every pre-registered agent (main config at 0x/1x/2x costs, variants at 1x), compute metrics,
log trials, apply the gate and write reports/backtest/*."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..agents import build_agent
from ..config import AgentSpec, Settings
from ..costs import CostModel
from ..data.store import HistoryStore
from ..registry import Registry
from . import metrics as M
from .engine import BacktestResult, run_backtest

log = logging.getLogger(__name__)

BENCH_FOR_CALENDAR = {"XETR": "SP500", "XNYS": "SP500", "CRYPTO": "BTC_SPOT"}


def config_hash(spec: AgentSpec) -> str:
    blob = json.dumps({"cls": spec.cls, "params": spec.params, "venue": spec.venue_override}, sort_keys=True,
                      default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_all(settings: Settings, store: HistoryStore, instruments: dict, venues: dict, *,
            extras: dict | None = None, only: list[str] | None = None, out_dir: str | Path = "reports/backtest",
            trials_path: str | Path = "reports/trials.csv", end: str | None = None,
            fallback_funding_annual: float = 0.10, write: bool = True) -> dict:
    out_dir = Path(out_dir)
    stresses = list(settings.backtest.get("stress_levels", [0.0, 1.0, 2.0]))
    end = end or settings.backtest.get("end")
    rows, curves, details = [], {}, {}
    t_all = time.time()
    for aid, spec in settings.agents.items():
        if not spec.backtest or (only and aid not in only):
            continue
        for variant in ["main"] + list(spec.variants):
            sp = spec.with_variant(variant)
            for stress in (stresses if variant == "main" else [1.0]):
                registry = Registry(instruments)
                cm = CostModel(venues, stress)                        # execution costs at this stress level
                agent = build_agent(sp, registry, CostModel(venues, 1.0))  # decisions always use x1 costs
                t0 = time.time()
                try:
                    res = run_backtest(agent, store, registry, cm, start=sp.backtest_start, end=end,
                                       capital=settings.initial_capital, extras=extras,
                                       fallback_funding_annual=fallback_funding_annual)
                except Exception as exc:
                    log.exception("backtest failed: %s %s %s", aid, variant, stress)
                    rows.append({"agent": aid, "variant": variant, "stress": stress, "error": repr(exc)})
                    continue
                st = M.perf_stats(res.equity, agent.ppy)
                tr = M.trade_stats(res)
                bench_id = BENCH_FOR_CALENDAR[sp.calendar]
                rel = {}
                if store.has(bench_id) and aid not in ("bench_sp500", "bench_btc"):
                    b = store.close(bench_id).reindex(res.equity.index).ffill()
                    rel = M.relative_stats(res.equity, b, agent.ppy)
                ci = M.bootstrap_sharpe_ci(res.equity, agent.ppy) if stress == 1.0 else (np.nan, np.nan)
                row = {"agent": aid, "name": sp.name, "variant": variant, "stress": stress, "hash": config_hash(sp),
                       "calendar": sp.calendar, "ppy": agent.ppy, **st, **tr, **{f"rel_{k}": v for k, v in rel.items()},
                       "sharpe_ci_lo": ci[0], "sharpe_ci_hi": ci[1],
                       "kill_switch_month_freq": M.kill_switch_month_frequency(res.equity, sp.risk.get("max_drawdown")),
                       "drawdown_events": len(res.kill_events), "liquidations": len(res.diagnostics.get("liquidations", [])),
                       "ledger_verified": res.diagnostics["ledger_verified"],
                       "fallback_funding_days": res.diagnostics["fallback_funding_days"],
                       "runtime_s": round(time.time() - t0, 1)}
                rows.append(row)
                log.info("%s/%s x%.0f: CAGR %.2f%% Sharpe %.2f MDD %.1f%% (%.0fs)", aid, variant, stress,
                         100 * st.get("cagr", np.nan), st.get("sharpe", np.nan), 100 * st.get("max_drawdown", np.nan),
                         row["runtime_s"])
                if variant == "main":
                    curves[f"{aid}@x{stress:g}"] = res.equity
                    if stress == 1.0:
                        details[aid] = _details(res, sp)
    table = pd.DataFrame(rows)
    trials = _update_trials(table, trials_path if write else None)
    table = _apply_gate(table, trials, settings)
    summary = {"generated": datetime.now(timezone.utc).isoformat(), "git": git_rev(),
               "data_end": str(store.last_date().date()) if store.last_date() is not None else None,
               "synthetic_data": bool(store.meta.get("synthetic")), "runtime_s": round(time.time() - t_all, 1),
               "n_trials": int(trials["hash"].nunique()) if len(trials) else 0,
               "data_meta": {k: v for k, v in store.meta.items() if k != "synthetic"}}
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "results.csv", index=False)
        pd.DataFrame(curves).sort_index().to_csv(out_dir / "equity_curves.csv")
        (out_dir / "details.json").write_text(json.dumps(details, indent=1, default=_json_default))
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
        (out_dir / "summary.md").write_text(summary_markdown(table, summary, settings))
    return {"table": table, "curves": curves, "details": details, "summary": summary, "trials": trials}


def _details(res: BacktestResult, spec: AgentSpec) -> dict:
    fills = res.fills.tail(40).to_dict("records") if len(res.fills) else []
    return {"name": spec.name, "thesis": spec.thesis, "note": spec.backtest_note, "drawdown_events": res.kill_events,
            "liquidations": res.diagnostics.get("liquidations", []),
            "diagnostics": {k: v for k, v in res.diagnostics.items() if k not in ("coverage_log", "liquidations")},
            "coverage_tail": res.diagnostics.get("coverage_log", [])[-24:],
            "coverage_summary": _coverage_summary(res.diagnostics.get("coverage_log", [])),
            "costs": res.costs, "last_fills": fills, "final_positions": list(res.ledger.positions)}


def _coverage_summary(log_: list[dict]) -> dict:
    if not log_:
        return {}
    df = pd.DataFrame(log_)
    if "members" not in df or df["members"].sum() == 0:
        return {}
    ratio = (df["with_data"] / df["members"]).replace([np.inf], np.nan)
    return {"months": int(len(df)), "avg_members": float(df["members"].mean()),
            "avg_share_with_price_data": float(ratio.mean()), "min_share_with_price_data": float(ratio.min())}


def _update_trials(table: pd.DataFrame, path) -> pd.DataFrame:
    new = table[(table.get("stress") == 1.0) & table.get("sharpe").notna()] if "sharpe" in table else table.iloc[0:0]
    new = new[["agent", "variant", "hash", "sharpe", "n_obs", "start", "end", "calendar"]].copy() if len(new) else \
        pd.DataFrame(columns=["agent", "variant", "hash", "sharpe", "n_obs", "start", "end", "calendar"])
    new.insert(0, "run_utc", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    new.insert(1, "git", git_rev())
    if path is None:
        return new
    path = Path(path)
    if path.exists():
        old = pd.read_csv(path)
        allt = pd.concat([old, new], ignore_index=True)
    else:
        allt = new
    path.parent.mkdir(parents=True, exist_ok=True)
    allt.to_csv(path, index=False)
    return allt


def _apply_gate(table: pd.DataFrame, trials: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    if table.empty or "sharpe" not in table:
        return table
    gate = settings.backtest.get("gate", {})
    # one Sharpe per distinct configuration ever tested (latest run), excluding the passive benchmarks
    t = trials[~trials["agent"].isin(["bench_sp500", "bench_btc"])] if len(trials) else trials
    latest = t.sort_values("run_utc").groupby("hash").tail(1) if len(t) else t
    trial_srs = latest["sharpe"].astype(float).tolist() if len(latest) else []
    table = table.copy()
    table["dsr"] = np.nan
    table["sr_star"] = np.nan
    table["gate"] = ""
    for i, r in table.iterrows():
        if r.get("variant") != "main" or r.get("stress") != 1.0 or not isinstance(r.get("sharpe"), float):
            continue
        if r["agent"] in ("bench_sp500", "bench_btc"):
            table.at[i, "gate"] = "benchmark"
            continue
        dsr, sr_star = M.deflated_sharpe(r["sharpe_pp"], int(r["n_obs"]), r["skew"], r["kurtosis"], trial_srs, r["ppy"])
        table.at[i, "dsr"] = dsr
        table.at[i, "sr_star"] = sr_star
        x2 = table[(table["agent"] == r["agent"]) & (table["variant"] == "main") & (table["stress"] == 2.0)]
        sr2 = float(x2["sharpe"].iloc[0]) if len(x2) else np.nan
        ex_best = r.get("ex_best_year_cagr", np.nan)
        if not (sr2 > gate.get("min_sharpe_at_2x_costs", 0.0)):
            verdict = "FAIL (Sharpe at 2x costs <= 0)"
        elif not (ex_best > gate.get("ex_best_year_cagr_min", 0.0)):
            verdict = "FAIL (no profit without best year)"
        elif dsr >= gate.get("pass_dsr", 0.95):
            verdict = "PASS"
        else:
            verdict = f"WEAK (DSR {dsr:.2f} < {gate.get('pass_dsr', 0.95)})"
        table.at[i, "gate"] = verdict
    return table


def _json_default(o):
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def _pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.{d}f}%"


def _ks(x):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{100 * x:.0f}% of months"


def _num(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{d}f}"


def summary_markdown(table: pd.DataFrame, summary: dict, settings: Settings) -> str:
    lines = ["# Backtest results", ""]
    if summary.get("synthetic_data"):
        lines += ["> **SYNTHETIC DATA - these numbers mean nothing.** Generated only to test the code.", ""]
    lines += [f"Generated {summary['generated']} from commit `{summary['git']}`; data through {summary['data_end']}. "
              f"Distinct configurations tested so far (for the Deflated Sharpe Ratio): {summary['n_trials']}.", "",
              "Costs: x0 = no trading costs, x1 = modelled fees + spread + slippage + FX, x2 = those trading costs doubled. "
              "Decisions are identical at every level (made with x1 costs); perp funding is market data and applies "
              "unscaled at every level.", "",
              "| Agent | Period | CAGR x0 | CAGR x1 | CAGR x2 | Sharpe x1 (90% CI) | Max DD x1 | Cost drag/yr | "
              "vs benchmark (excess CAGR) | Kill switch would fire in | DSR | Gate |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if table.empty or "cagr" not in table:
        return "\n".join(lines + ["", "_no results_"])
    main = table[table["variant"] == "main"]
    for aid in main["agent"].unique():
        g = main[main["agent"] == aid].set_index("stress")
        if 1.0 not in g.index:
            continue
        r1 = g.loc[1.0]
        c = lambda s: _pct(g.loc[s, "cagr"]) if s in g.index else "n/a"
        ci = f"{_num(r1['sharpe'])} ({_num(r1.get('sharpe_ci_lo'))} to {_num(r1.get('sharpe_ci_hi'))})"
        lines.append(f"| {r1['name']} | {r1['start']} to {r1['end']} | {c(0.0)} | {c(1.0)} | {c(2.0)} | {ci} | "
                     f"{_pct(r1['max_drawdown'])} | {_pct(r1['cost_drag_per_year'], 2)} | "
                     f"{_pct(r1.get('rel_excess_cagr'))} | {_ks(r1.get('kill_switch_month_freq'))} | "
                     f"{_num(r1.get('dsr'))} | {r1.get('gate', '')} |")
    var = table[table["variant"] != "main"]
    if len(var):
        lines += ["", "## Robustness variants (x1 costs, pre-registered; never used to pick the paper config)", "",
                  "| Agent | Variant | CAGR | Sharpe | Max DD |", "|---|---|---|---|---|"]
        for _, r in var.iterrows():
            lines.append(f"| {r['agent']} | {r['variant']} | {_pct(r.get('cagr'))} | {_num(r.get('sharpe'))} | "
                         f"{_pct(r.get('max_drawdown'))} |")
    errs = table[table.get("error").notna()] if "error" in table else []
    if len(errs):
        lines += ["", "## Errors", ""] + [f"- {r['agent']}/{r['variant']} x{r['stress']}: {r['error']}" for _, r in errs.iterrows()]
    notes = [s for s in settings.agents.values() if s.backtest_note]
    if notes:
        lines += ["", "## Caveats per agent", ""] + [f"- **{s.name}**: {s.backtest_note}" for s in notes]
    return "\n".join(lines) + "\n"
