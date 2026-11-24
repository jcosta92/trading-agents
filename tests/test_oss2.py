"""Second batch of open-source bot families (config/families.d/20_oss2.yaml): every bot runs on
synthetic candles without errors and stays long only, and the Gekko / Zenbot state machines behave
as their sources describe."""
import json

import pandas as pd

from quantdesk.bots import families as _families  # noqa: F401  (registers the functions)
from quantdesk.bots.families.oss2 import _persist, _trend_pulses
from quantdesk.bots.runner import load_config, run_cycle
from quantdesk.ledger import Ledger

from fakes import FamilySource


def _oss2_cfg():
    cfg = load_config()
    cfg["bots"] = {b: s for b, s in cfg["bots"].items() if str(s.get("signal", "")).startswith("oss2_")}
    cfg["start"], cfg["end"] = "2026-10-26T00:00:00Z", "2026-11-20T00:00:00Z"
    return cfg


def test_oss2_bots_run_long_only(tmp_path):
    cfg = _oss2_cfg()
    assert len(cfg["bots"]) == 400
    state = tmp_path / "bots"
    s = run_cycle(state, now=pd.Timestamp("2026-11-04T20:30:00Z"), source=FamilySource(), cfg=cfg, cache_dir="")
    assert not s["errors"], s["errors"][:5]
    for bid in cfg["bots"]:
        led = Ledger.from_dict(json.loads((state / "ledgers" / f"{bid}.json").read_text()))
        assert led.verify()[0], bid
        assert all(p.qty > -1e-9 for p in led.positions.values()), bid
        assert led.cash.get("EUR", 0) > -1e-6, bid
    fills = pd.read_csv(state / "fills.csv")
    traded = {b.split(".")[0] for b in fills["bot"].unique()}
    assert len(traded) >= 25, sorted(traded)


def test_gekko_persistence_advises_once_per_trend():
    idx = pd.RangeIndex(8)
    up = pd.Series([0, 1, 1, 1, 0, 1, 0, 0], index=idx).astype(bool)
    down = pd.Series([0, 0, 0, 0, 0, 0, 1, 1], index=idx).astype(bool)
    ent, ex = _persist(up, down, 2)
    assert list(ent[ent].index) == [2]            # second candle in the up zone, then never again
    assert list(ex[ex].index) == [7]
    ent, _ = _persist(up, down, 0)
    assert list(ent[ent].index) == [1]


def test_zenbot_trend_pulses():
    idx = pd.RangeIndex(6)
    up = pd.Series([1, 1, 0, 0, 1, 1], index=idx).astype(bool)
    down = ~up
    ent, ex = _trend_pulses(up, down)
    assert list(ent[ent].index) == [0, 4] and list(ex[ex].index) == [2]
