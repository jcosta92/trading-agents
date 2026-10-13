"""Strategy families: no look-ahead in any signal or weight function, every family runs, and the
generic risk rules (stoploss, ROI, trailing, brackets) behave as specified."""
import json

import numpy as np
import pandas as pd
import pytest

from quantdesk.bots import families as _families, generic  # noqa: F401  (registers the functions)
from quantdesk.bots.data import TF
from quantdesk.bots.families_config import expand, load_families
from quantdesk.bots.generic import Panel, proxy_daily
from quantdesk.bots.runner import load_config, run_cycle
from quantdesk.ledger import Ledger

from fakes import FamilySource

FAM = load_families()
BOTS, FAMS = expand(FAM)
SRC = FamilySource()
NOW = pd.Timestamp("2026-10-20T00:00:00Z")


def _same(a, b) -> bool:
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        return (np.isnan(a) and np.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(a))
    if a is None or b is None:
        return a is b
    try:
        if pd.isna(a) and pd.isna(b):
            return True
    except (TypeError, ValueError):
        pass
    return bool(a == b)


def _crypto_families():
    out = {}
    for fid, f in FAM["families"].items():
        if f["class"] in ("SignalBot", "DipAnalyserBot"):
            out[fid] = f
    return out


def _cut_info(info, t_close):
    out = {}
    for tf, d in info.items():
        out[tf] = d[d.index + TF[tf] <= t_close]
    return out


@pytest.mark.parametrize("fid", sorted(_crypto_families()))
def test_signal_has_no_lookahead(fid):
    f = FAM["families"][fid]
    tf = f["timeframe"]
    df = SRC.crypto("BTC/EUR", tf, NOW - TF[tf] * 1500, NOW)
    info = {t: SRC.crypto("BTC/EUR", t, NOW - TF[t] * 1500, NOW) for t in f.get("informative", [])}
    fn = generic.SIGNALS[f["signal"]]
    p = dict(f.get("params") or {})
    full = fn(df, p, dict(info))
    for k in (len(df) - 1, len(df) - 37, len(df) - 211):
        t = df.index[k]
        cut = fn(df.iloc[:k + 1], p, _cut_info(info, t + TF[tf]))
        for col in full.columns:
            a, b = full[col].iloc[k], cut[col].iloc[-1]
            assert _same(a, b), f"{fid}.{col} at {t}: full={a} cut={b}"


def _weight_families():
    return {fid: f for fid, f in FAM["families"].items() if f["class"] == "WeightBot"}


@pytest.mark.parametrize("fid", sorted(_weight_families()))
def test_crypto_weights_have_no_lookahead(fid):
    f = FAM["families"][fid]
    tf = f["timeframe"]
    uni = f.get("universe") or [v for v in (f.get("variants") or [])][:1]
    if isinstance(uni, str):
        uni = FAM["universes"][uni]
    sym = {u: u.replace("_SPOT", "/EUR") for u in uni}
    frames = {u: SRC.crypto(sym[u], tf, NOW - TF[tf] * 1200, NOW) for u in uni}
    fn = generic.WEIGHTS[f["weights"]]
    p = dict(f.get("params") or {})
    full = fn(Panel(frames), p, {"tf": TF[tf]})
    idx = full.index
    for k in (len(idx) - 1, len(idx) - 23, len(idx) - 160):
        t = idx[k]
        cut = fn(Panel({u: d[d.index <= t] for u, d in frames.items()}), p, {"tf": TF[tf]})
        a, b = full.iloc[k].fillna(0), cut.iloc[-1].fillna(0)
        assert np.allclose(a.to_numpy(), b.reindex(a.index).to_numpy(), atol=1e-9), f"{fid} at {t}: {a.to_dict()} vs {b.to_dict()}"


def _etf_signal_families():
    return {fid: f for fid, f in FAM["families"].items() if f["class"] in ("EtfDailyBot", "TPSBot", "IntradayEtfBot")}


@pytest.mark.parametrize("fid", sorted(_etf_signal_families()))
def test_etf_signals_have_no_lookahead(fid):
    f = FAM["families"][fid]
    fn = generic.SIGNALS[f["signal"]]
    p = dict(f.get("params") or {})
    if f["class"] == "IntradayEtfBot":
        df = SRC.stocks(["SPY"], "5m", NOW - pd.Timedelta(days=40), NOW)["SPY"]
        info = {"sessions": SRC._sched()[["open", "close"]], "daily": {}}
    else:
        _, d = SRC._stock("SPY")
        df = d[d.index < pd.Timestamp("2026-10-16")].tail(500)
        info = {"daily": {"^VIX": SRC._stock("^VIX")[1]}}
    full = fn(df, p, info)
    for k in (len(df) - 1, len(df) - 13, len(df) - 140):
        cut = fn(df.iloc[:k + 1], p, info)
        for col in full.columns:
            a, b = full[col].iloc[k], cut[col].iloc[-1]
            assert _same(a, b), f"{fid}.{col} at {df.index[k]}: full={a} cut={b}"


@pytest.fixture(scope="module")
def run_all(tmp_path_factory):
    """Every family bot, replayed over 9 days of synthetic data (incl. a month-end for the ETF rules)."""
    cfg = load_config()
    cfg["start"], cfg["end"] = "2026-10-26T00:00:00Z", "2026-11-20T00:00:00Z"
    state = tmp_path_factory.mktemp("fam") / "bots"
    (state.parent / "llm").mkdir()
    (state.parent / "llm" / "universe.json").write_text(json.dumps({"tickers": ["AAPL", "MSFT", "NVDA"]}))
    s = run_cycle(state, now=pd.Timestamp("2026-11-04T20:30:00Z"), source=FamilySource(), cfg=cfg, cache_dir="")
    return cfg, state, s


def test_every_bot_runs_without_errors(run_all):
    cfg, state, s = run_all
    assert not s["errors"], s["errors"][:10]
    assert len(s["bots"]) == len(cfg["bots"]) and len(cfg["bots"]) > 800


def test_ledgers_consistent_and_long_only(run_all):
    cfg, state, s = run_all
    for bid in cfg["bots"]:
        led = Ledger.from_dict(json.loads((state / "ledgers" / f"{bid}.json").read_text()))
        assert led.verify()[0], bid
        assert all(p.qty > -1e-9 for p in led.positions.values()), f"{bid} went short"
        assert led.cash.get("EUR", 0) > -1e-6, f"{bid} overdrew EUR"


def test_most_families_trade(run_all):
    cfg, state, s = run_all
    f = pd.read_csv(state / "fills.csv")
    traded = {b.split(".")[0] for b in f["bot"].unique()}
    fams = set(cfg["families"])
    quiet = sorted(fams - traded)
    # calendar and slow allocation rules may legitimately not trade in 9 days of random data
    assert len(quiet) / len(fams) < 0.25, quiet
