"""Buffett and famous-investor families: every bot runs and trades on synthetic data, 13F clones
switch portfolio only after each filing date, and the screens/13F books hold only point-in-time data."""
import json

import pandas as pd
import pytest

from quantdesk.bots import families as _families, generic  # noqa: F401  (registers the functions)
from quantdesk.bots.generic import Panel
from quantdesk.bots.runner import load_config, run_cycle
from quantdesk.config import CONFIG_DIR
from quantdesk.ledger import Ledger

from fakes import FamilySource

GROUPS = ("buffett", "investors")
# one bot of every rule type (the full set is replayed on real data before a push; synthetic 5-minute
# candles for 500 stocks would make this test take minutes)
SAMPLE = ["berkshire_13f", "munger_daily_journal", "cathie_wood_ark", "buffett_9010", "buffett_greedy_fearful.VIX30",
          "buffett_greedy_fearful.DD20", "buffett_indicator", "brk_buyback_rule", "buffett_small_value.IJS",
          "dalio_all_weather", "magic_formula", "graham_defensive", "dogs_of_the_dow", "jt_momentum_12_1", "bab_long_sp500"]


def _mine(cfg):
    return [b for b, s in cfg["bots"].items() if s.get("group") in GROUPS]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    cfg = load_config()
    cfg["start"], cfg["end"] = "2026-10-12T17:00:00Z", None
    for b in ("jt_momentum_12_1", "bab_long_sp500"):
        cfg["bots"][b]["params"]["universe"] = cfg["bots"][b]["params"]["universe"][:40]
    state = tmp_path_factory.mktemp("inv") / "bots"
    s = run_cycle(state, now=pd.Timestamp("2026-11-18T21:30:00Z"), source=FamilySource(), cfg=cfg,
                  only=SAMPLE, cache_dir="")
    return cfg, state, s


def test_investor_bots_run_and_trade(run):
    cfg, state, s = run
    assert len(_mine(cfg)) >= 60 and set(SAMPLE) <= set(_mine(cfg))
    assert not s["errors"], s["errors"][:10]
    fills = pd.read_csv(state / "fills.csv")
    quiet = [b for b in SAMPLE if b not in set(fills["bot"])]
    assert not quiet, quiet                      # every rule buys something on the first day (T-bills included)
    for b in SAMPLE:
        led = Ledger.from_dict(json.loads((state / "ledgers" / f"{b}.json").read_text()))
        assert led.verify()[0], b
        assert all(p.qty > -1e-9 for p in led.positions.values()), b
        assert led.cash.get("EUR", 0) > -1e-6, b


def test_clone_trades_only_on_start_and_after_filings(run):
    cfg, state, s = run
    fills = pd.read_csv(state / "fills.csv")
    f = fills[fills["bot"] == "berkshire_13f"]
    days = sorted({pd.Timestamp(t).tz_convert("America/New_York").strftime("%Y-%m-%d") for t in f["ts"]})
    # bought on the start, then (maybe) rebalanced on the first session after the Q3 filing (16 Nov)
    assert days[0] == "2026-10-12", days
    assert set(days) <= {"2026-10-12", "2026-11-17"}, days


def test_books_are_point_in_time():
    b13 = json.loads((CONFIG_DIR / "investors" / "13f_2026.json").read_text())
    for k, v in b13.items():
        if k.startswith("_"):
            continue
        for e in v["portfolios"]:
            assert pd.Timestamp(e["effective"]) > pd.Timestamp(e["filed"]), (k, e["report"])
            assert abs(sum(e["weights"].values()) - 1) < 1e-3, k
        first = min(v["portfolios"], key=lambda e: e["effective"])
        assert pd.Timestamp(first["filed"]) < pd.Timestamp("2026-10-12"), k
    scr = json.loads((CONFIG_DIR / "investors" / "screens_2026.json").read_text())
    for k, v in scr.items():
        if not k.startswith("_"):
            assert all(pd.Timestamp(e["asof"]) < pd.Timestamp("2026-10-12") for e in v), k


def test_xs_rank_uses_only_past_prices():
    idx = pd.bdate_range("2025-06-02", "2026-10-30")
    import numpy as np
    rng = np.random.default_rng(3)
    names = [f"S{i}" for i in range(30)] + ["SPY"]
    frames = {n: pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) for c in
                               ("open", "high", "low", "close")} | {"volume": 1e6}, index=idx) for n in names}
    sess = pd.DataFrame(index=pd.bdate_range("2026-09-02", "2026-12-31"))
    fn = generic.WEIGHTS["inv_xs_rank"]
    day = "2026-10-30"
    for metric in ("mom12_1", "high52", "rev1m", "lowvol", "lowbeta"):
        p = {"universe": names[:-1], "metric": metric, "n": 5}
        cut = {n: d[d.index <= day] for n, d in frames.items()}
        a = fn(Panel(cut), p, {"day": day, "sessions": sess, "pos": (22, 0)})
        later = {n: d.copy() for n, d in cut.items()}
        assert a.shape[0] == 1 and abs(a.iloc[-1].sum() - 1) < 1e-9, metric
        b = fn(Panel(later), p, {"day": day, "sessions": sess, "pos": (22, 0)})
        assert (a.iloc[-1] == b.iloc[-1]).all(), metric
        assert fn(Panel(cut), p, {"day": "2026-10-29", "sessions": sess, "pos": (21, 1)}).empty
