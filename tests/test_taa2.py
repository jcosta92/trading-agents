"""taa2 families (config/families.d/40_taa2.yaml): every US ETF allocation returns valid long-only
weights inside its universe, and its decision only depends on data up to the decision day."""
import numpy as np
import pandas as pd
import pytest
import yaml

from quantdesk.bots import families as _families, generic  # noqa: F401  (registers the functions)
from quantdesk.bots.families_config import expand, load_families
from quantdesk.bots.generic import Panel
from quantdesk.config import CONFIG_DIR

from fakes import FamilySource

SRC = FamilySource()
MINE = yaml.safe_load(open(CONFIG_DIR / "families.d" / "40_taa2.yaml"))["families"]
BOTS, _ = expand(load_families())
ETF = {b: s for b, s in BOTS.items() if s["family"] in MINE and s["class"] == "EtfWeightBot"}


def _panel(spec, day):
    p = spec["params"]
    tickers = list(dict.fromkeys(list(p["universe"]) + list(p.get("signal_tickers", []))))
    frames = {t: SRC._stock(t)[1] for t in tickers}
    frames = {t: d[d.index <= day].tail(int(spec.get("lookback", 600))) for t, d in frames.items()}
    return Panel(frames)


@pytest.mark.parametrize("bid", sorted(ETF))
@pytest.mark.parametrize("day", ["2026-09-30", "2026-10-30", "2026-11-03"])
def test_etf_weights_valid_and_causal(bid, day):
    spec = ETF[bid]
    fn = generic.WEIGHTS[spec["weights"]]
    sched = SRC._sched()
    d = pd.Timestamp(day)
    extra = {"day": day, "sessions": sched, "pos": (1, 1), "bot": None}
    out = fn(_panel(spec, d), dict(spec["params"]), extra)
    w = out.iloc[-1].fillna(0.0)
    assert (w >= -1e-12).all(), w[w < 0]
    assert w.sum() <= 1.0 + 1e-6, w.sum()
    held = set(w[w > 1e-9].index)
    assert held <= set(spec["params"]["universe"]), held - set(spec["params"]["universe"])
    # future data (appended rows after `day`) must not change today's decision
    later = pd.Timestamp("2026-11-20")
    full = _panel(spec, later)
    cut = Panel({t: f[f.index <= d] for t, f in full.frames.items()})
    w2 = fn(cut, dict(spec["params"]), extra).iloc[-1].fillna(0.0)
    assert np.allclose(w.reindex(w2.index).fillna(0).to_numpy(), w2.to_numpy(), atol=1e-4)   # EWMA windows start at different rows
