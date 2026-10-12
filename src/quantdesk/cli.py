"""Command line entry point.

  python -m quantdesk.cli fetch-history            # download history for backtests -> data/history/
  python -m quantdesk.cli backtest                 # run all pre-registered agents -> reports/backtest/
  python -m quantdesk.cli paper                    # one paper-trading cycle -> state/ (run hourly)
  python -m quantdesk.cli news-inbox               # today's headlines for the LLM agent -> state/llm/inbox/
  python -m quantdesk.cli validate-labels          # check Claude's labels file before it is committed
  python -m quantdesk.cli dashboard                # rebuild docs/index.html and reports/STATUS.md
  python -m quantdesk.cli selftest                 # quick end-to-end run on synthetic data
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from .config import ROOT, load_instruments, load_settings, load_venues

log = logging.getLogger("quantdesk")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.ERROR)
    logging.getLogger("quantdesk").setLevel(logging.INFO)


def cmd_fetch_history(args) -> int:
    from .data import loader, universe
    inst = load_instruments()
    out = Path(args.out)
    core = loader.build_core_store(inst, start=args.start, use_proxies=True, funding=True, funding_since="2018-01-01")
    core.save(out / "core")
    log.info("core store: %s", {k: v.shape for k, v in core.frames.items()})
    log.info("missing instruments: %s", core.meta.get("missing"))
    if not args.no_sp500:
        sp, extras = loader.build_sp500_store(start=args.sp500_start)
        sp.save(out / "sp500")
        hist = extras["membership"]
        pd.DataFrame({"date": hist.index.date, "tickers": [",".join(t) for t in hist["tickers"]]}).to_csv(
            out / "sp500" / "membership.csv", index=False)
        (out / "sp500" / "sectors.json").write_text(json.dumps(extras["sectors"], indent=0))
        log.info("sp500 store: %s", sp.meta.get("sp500"))
    return 0


def load_backtest_data(path: str | Path):
    from .data.store import HistoryStore
    path = Path(path)
    store = HistoryStore.load(path / "core")
    extras = {}
    sp = path / "sp500"
    if (sp / "close.parquet").exists():
        store.merge(HistoryStore.load(sp))
        m = pd.read_csv(sp / "membership.csv")
        extras["membership"] = pd.DataFrame({"tickers": [str(t).split(",") for t in m["tickers"]]},
                                            index=pd.to_datetime(m["date"]))
        extras["sectors"] = json.loads((sp / "sectors.json").read_text())
    return store, extras


def cmd_backtest(args) -> int:
    from .backtest.runner import run_all
    settings, inst, venues = load_settings(), load_instruments(), load_venues()
    if args.synthetic:
        from .synthetic import make_store
        store, extras = make_store()
        out_dir, trials = Path(args.out) / "synthetic", Path(args.out) / "synthetic" / "trials.csv"
    else:
        store, extras = load_backtest_data(args.data)
        out_dir, trials = Path(args.out), Path(args.trials)
    res = run_all(settings, store, inst, venues, extras=extras, only=args.agents or None, out_dir=out_dir,
                  trials_path=trials, end=args.end)
    print((out_dir / "summary.md").read_text())
    from .reporting import write_backtest_html
    write_backtest_html(out_dir, settings)
    bad = res["table"].get("ledger_verified")
    return 0 if bad is None or bool(bad.fillna(True).all()) else 2


def cmd_paper(args) -> int:
    from .live.runner import PaperRunner
    from .reporting import write_dashboard
    settings, inst, venues = load_settings(), load_instruments(), load_venues()
    if not settings.paper.get("enabled", False) and not args.force:
        print("paper trading is disabled in config/agents.yaml (paper.enabled: false); use --force to run anyway")
        return 0
    runner = PaperRunner(settings, inst, venues, state_dir=args.state, now=pd.Timestamp(args.now) if args.now else None,
                         agents=args.agents or None)
    summary = runner.run()
    write_dashboard(Path(args.state), settings, Path(args.docs), Path(args.reports))
    print(json.dumps({k: v for k, v in summary.items() if k != "agents"}, indent=1, default=str))
    for aid, a in summary["agents"].items():
        print(f"- {aid}: equity {a.get('equity_eur')} open={a.get('market_open')} decided={a.get('decided')} "
              f"fills={len(a.get('fills', []))} rejects={len(a.get('rejects', []))} {a.get('error', '')}")
        for n in a.get("notes", [])[:8]:
            print(f"    . {n}")
    return 1 if summary["errors"] else 0


def _now(args) -> pd.Timestamp:
    now = pd.Timestamp(args.now) if getattr(args, "now", None) else pd.Timestamp.now(tz="UTC")
    return now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")


def cmd_news_inbox(args) -> int:
    """Write state/llm/inbox/<date>.json: headlines since the previous inbox for the most liquid
    S&P 500 names (top N by 60-day median dollar volume, recomputed once a month)."""
    import exchange_calendars as xcals

    from . import llm
    from .data import loader, yahoo
    settings = load_settings()
    spec = settings.agents.get("llm_news")
    p = spec.params if spec else {}
    now = _now(args)
    state = Path(args.state)
    date = now.strftime("%Y-%m-%d")
    ny_day = pd.Timestamp(now.tz_convert("America/New_York").date())
    if not args.force and not xcals.get_calendar("XNYS").is_session(ny_day):
        print(f"{ny_day.date()} is not a NYSE trading day - no inbox")
        return 0
    out = llm.inbox_path(state, date)
    if out.exists() and not args.force:
        print(f"inbox for {date} already exists")
        return 0
    uni_p = state / "llm" / "universe.json"
    month = now.strftime("%Y-%m")
    uni = None
    if uni_p.exists():
        u = json.loads(uni_p.read_text())
        if u.get("month") == month and u.get("tickers"):
            uni = u["tickers"]
    if uni is None:
        st, _ = loader.build_sp500_store(start=str((now - pd.Timedelta(days=120)).date()), current_only=True)
        uni = llm.liquid_universe(st.frames["close"], st.frames["volume"], n=int(p.get("universe_size", 100)))
        if len(uni) < 50:
            print(f"universe too small ({len(uni)} names) - price download failed?")
            return 1
        uni_p.parent.mkdir(parents=True, exist_ok=True)
        uni_p.write_text(json.dumps({"month": month, "created": now.isoformat(), "rule":
                                     "top N S&P 500 members by 60-day median dollar volume", "tickers": uni}, indent=0))
    cutoff = llm.news_cutoff(state, now, int(p.get("lookback_hours", 24)))
    inbox = llm.build_inbox(uni, yahoo.news, now, cutoff=cutoff)
    print(f"news sources: {yahoo.NEWS_STATS} (tickers served by each source; errors)")
    if not inbox["items"]:
        # 100 large caps never have zero headlines in a trading day: the feed is broken. Write
        # nothing, so the labelling task waits and a re-run (or the fallback cron) can retry.
        print(f"::error::0 headlines for {len(uni)} tickers since {cutoff.isoformat()} - news feed failure")
        return 1
    inbox["sources"] = dict(yahoo.NEWS_STATS)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inbox, indent=1))
    print(f"inbox {date}: {len(inbox['items'])} headlines for {len(uni)} tickers since {cutoff.isoformat()}")
    return 0


def cmd_validate_labels(args) -> int:
    from . import llm
    state = Path(args.state)
    date = args.date or _now(args).strftime("%Y-%m-%d")
    ip, lp = llm.inbox_path(state, date), llm.labels_path(state, date)
    if not ip.exists():
        print(f"no inbox for {date} ({ip})")
        return 1
    if not lp.exists():
        print(f"no labels file for {date} ({lp})")
        return 1
    try:
        labels = json.loads(lp.read_text())
    except json.JSONDecodeError as exc:
        print(f"INVALID: {lp} is not valid JSON ({exc})")
        return 1
    probs = llm.validate_labels(json.loads(ip.read_text()), labels)
    if probs:
        print(f"INVALID: {len(probs)} problems")
        for pr in probs[:50]:
            print(" -", pr)
        return 1
    print(f"OK: {len(labels['labels'])} labels for {date}")
    return 0


def cmd_dashboard(args) -> int:
    from .reporting import write_backtest_html, write_dashboard
    settings = load_settings()
    write_dashboard(Path(args.state), settings, Path(args.docs), Path(args.reports))
    if (Path(args.reports) / "backtest" / "results.csv").exists():
        write_backtest_html(Path(args.reports) / "backtest", settings)
    return 0


def cmd_selftest(args) -> int:
    from .backtest.runner import run_all
    from .synthetic import make_store
    settings, inst, venues = load_settings(), load_instruments(), load_venues()
    store, extras = make_store(start="2015-01-01", n_stocks=40)
    for s in settings.agents.values():
        s.backtest_start = max(s.backtest_start or "2016-06-01", "2016-06-01")
        s.variants = {}
    settings.backtest["stress_levels"] = [1.0]
    res = run_all(settings, store, inst, venues, extras=extras, write=False)
    t = res["table"]
    print(t[["agent", "stress", "cagr", "sharpe", "max_drawdown", "n_fills", "ledger_verified"]].to_string())
    ok = bool(t["ledger_verified"].all()) and "error" not in t
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


def cmd_bots(args) -> int:
    """One cycle of the trading bots (replays every candle closed since the last cycle) + dashboard."""
    from .bots import report
    from .bots.runner import load_config, run_cycle
    cfg = load_config()
    s = run_cycle(Path(args.state), now=pd.Timestamp(args.now) if args.now else None, cfg=cfg,
                  only=args.bots or None, start=args.start, end=args.end)
    if not args.no_report:
        if args.start:                                    # preview replay of a past period
            cfg = {**cfg, "start": args.start, "end": args.end or cfg["end"]}
            report.write(Path(args.state), cfg, Path(args.docs), Path(args.reports), "preview.html",
                         "BOTS_PREVIEW.md", preview=True)
        else:
            report.write(Path(args.state), cfg, Path(args.docs), Path(args.reports))
    print(f"bots cycle {s['ts']}: {s['n_fills']} fills, sources {s.get('data_sources')}")
    for bid, b in s["bots"].items():
        print(f"- {bid}: equity {b.get('equity_eur')} positions {len(b.get('positions') or {})} "
              f"orders {len(b.get('open_orders') or [])} {b.get('error', '')}")
        for n in (b.get("notes") or [])[-3:]:
            print(f"    . {n}")
    for e in s["errors"]:
        print(f"::error::{e}")
    return 1 if s["errors"] else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="quantdesk")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch-history")
    f.add_argument("--out", default=str(ROOT / "data" / "history"))
    f.add_argument("--start", default="2003-01-01")
    f.add_argument("--sp500-start", default="2006-06-01")
    f.add_argument("--no-sp500", action="store_true")
    b = sub.add_parser("backtest")
    b.add_argument("--data", default=str(ROOT / "data" / "history"))
    b.add_argument("--out", default=str(ROOT / "reports" / "backtest"))
    b.add_argument("--trials", default=str(ROOT / "reports" / "trials.csv"))
    b.add_argument("--agents", nargs="*")
    b.add_argument("--end")
    b.add_argument("--synthetic", action="store_true")
    r = sub.add_parser("paper")
    r.add_argument("--state", default=str(ROOT / "state"))
    r.add_argument("--docs", default=str(ROOT / "docs"))
    r.add_argument("--reports", default=str(ROOT / "reports"))
    r.add_argument("--now")
    r.add_argument("--agents", nargs="*")
    r.add_argument("--force", action="store_true", help="run even if paper.enabled is false")
    d = sub.add_parser("dashboard")
    d.add_argument("--state", default=str(ROOT / "state"))
    d.add_argument("--docs", default=str(ROOT / "docs"))
    d.add_argument("--reports", default=str(ROOT / "reports"))
    sub.add_parser("selftest")
    n = sub.add_parser("news-inbox")
    n.add_argument("--state", default=str(ROOT / "state"))
    n.add_argument("--now")
    n.add_argument("--force", action="store_true", help="rebuild even if it exists / not a trading day")
    bt = sub.add_parser("bots")
    bt.add_argument("--state", default=str(ROOT / "state" / "bots"))
    bt.add_argument("--docs", default=str(ROOT / "docs"))
    bt.add_argument("--reports", default=str(ROOT / "reports"))
    bt.add_argument("--now")
    bt.add_argument("--start", help="override the test start (preview replay of a past period)")
    bt.add_argument("--end")
    bt.add_argument("--bots", nargs="*")
    bt.add_argument("--no-report", action="store_true")
    v = sub.add_parser("validate-labels")
    v.add_argument("--state", default=str(ROOT / "state"))
    v.add_argument("--date", help="YYYY-MM-DD (default: today UTC)")
    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return {"fetch-history": cmd_fetch_history, "backtest": cmd_backtest, "paper": cmd_paper,
            "dashboard": cmd_dashboard, "selftest": cmd_selftest, "news-inbox": cmd_news_inbox, "bots": cmd_bots,
            "validate-labels": cmd_validate_labels}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
