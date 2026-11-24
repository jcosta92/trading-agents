"""Expands config/families.yaml into one bot spec per (family, variant)."""
from __future__ import annotations

import copy

from ..config import CONFIG_DIR, load_yaml

SINGLE_CRYPTO = {"SignalBot", "InfinityGridBot", "AvellanedaBot", "GridBot", "DipAnalyserBot", "OctoGridBot",
                 "OctoDcaBot"}
SINGLE_ETF = {"EtfDailyBot", "IntradayEtfBot", "TPSBot", "PreFomcBot"}


def short(market: str) -> str:
    return market.replace("_SPOT", "")


def load_families(path=None) -> dict:
    """config/families.yaml plus every config/families.d/*.yaml (families added later, one file per
    batch). Later files add families, groups, universes and variant sets; an id may not repeat."""
    cfg = load_yaml(path or CONFIG_DIR / "families.yaml")
    if path is not None:
        return cfg
    for extra in sorted((CONFIG_DIR / "families.d").glob("*.yaml")):
        add = load_yaml(extra) or {}
        for key in ("families", "groups", "universes", "variant_sets"):
            cur = cfg.setdefault(key, {}) or {}
            cfg[key] = cur
            for k, v in (add.get(key) or {}).items():
                if key == "families" and k in cur:
                    raise ValueError(f"family {k!r} in {extra.name} already exists")
                cur[k] = v
    return cfg


def expand(fam_cfg: dict) -> tuple[dict, dict]:
    """Returns (bots, families): bot specs keyed 'family.variant', and family metadata."""
    sets = fam_cfg.get("variant_sets", {})
    unis = fam_cfg.get("universes", {})
    bots, fams = {}, {}
    for fid, f in (fam_cfg.get("families") or {}).items():
        base = {k: copy.deepcopy(v) for k, v in f.items() if k not in ("variants", "risk", "universe")}
        base.update(copy.deepcopy(f.get("risk") or {}))
        base["family"] = fid
        base.setdefault("params", {})
        cls = f["class"]
        if cls in SINGLE_ETF | {"EtfWeightBot", "NineSigBot"}:
            base.setdefault("timeframe", "5m")
        uni = f.get("universe")
        if isinstance(uni, str):
            uni = unis[uni]
        pu = base["params"].get("universe")
        if isinstance(pu, str):
            base["params"]["universe"] = list(unis[pu])
        variants = f.get("variants")
        if isinstance(variants, str):
            variants = sets[variants]
        out = []
        if not variants:
            spec = copy.deepcopy(base)
            if uni:
                spec["params"]["universe"] = list(uni)
            out.append((fid, spec))
        else:
            for v in variants:
                spec = copy.deepcopy(base)
                if isinstance(v, dict):
                    spec["params"].update(copy.deepcopy(v.get("params") or {}))
                    vid = v["id"]
                else:
                    vid = short(v)
                    if cls in SINGLE_CRYPTO:
                        spec["params"]["instrument"] = v
                    elif cls in SINGLE_ETF:
                        spec["params"]["instruments"] = [v]
                    elif cls == "WeightBot":
                        spec["params"]["universe"] = [v]
                    else:
                        spec["params"]["instruments"] = [v]
                    if uni and "universe" not in spec["params"]:
                        spec["params"]["universe"] = list(uni)
                spec["variant"] = vid
                spec["name"] = f"{f['name']} - {vid}"
                out.append((f"{fid}.{vid}", spec))
        fams[fid] = {"name": f["name"], "group": f.get("group"), "source": f.get("source", ""),
                     "rules": f.get("rules", ""), "class": cls, "timeframe": base.get("timeframe"),
                     "bots": [b for b, _ in out]}
        bots.update(out)
    return bots, fams
