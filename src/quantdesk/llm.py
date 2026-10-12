"""Headline inbox and label files for the LLM news agent.

No paid API is used. Every weekday the `news` workflow writes the day's headlines to
state/llm/inbox/<date>.json; a scheduled Claude task labels them with config/llm_prompt.md and
commits state/llm/labels/<date>.json; the hourly paper run trades on the labels with fixed rules.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

PROMPT_VERSION = "news-v2"
EVENT_TYPES = {"earnings", "guidance", "m&a", "product", "regulatory_legal", "management", "capital_return",
               "analyst", "macro", "other"}


def item_id(ticker: str, title: str, published: str | None) -> str:
    return hashlib.sha1(f"{ticker}|{title}|{published}".encode()).hexdigest()[:12]


def inbox_path(state_dir: Path, date: str) -> Path:
    return Path(state_dir) / "llm" / "inbox" / f"{date}.json"


def labels_path(state_dir: Path, date: str) -> Path:
    return Path(state_dir) / "llm" / "labels" / f"{date}.json"


def liquid_universe(close: pd.DataFrame, volume: pd.DataFrame, n: int = 100, window: int = 60) -> list[str]:
    """The `n` names with the highest median daily dollar volume over the last `window` sessions."""
    dv = (close * volume).iloc[-window:]
    med = dv.median(skipna=True).dropna()
    med = med[med > 0]
    return list(med.sort_values(ascending=False).index[:n])


def news_cutoff(state_dir: Path, now: pd.Timestamp, default_hours: int = 24, max_hours: int = 96) -> pd.Timestamp:
    """Headlines are collected since the previous inbox was built (so Monday's inbox covers the
    weekend and a day after a holiday covers the holiday), at least `default_hours` and at most
    `max_hours` back."""
    today = now.strftime("%Y-%m-%d")
    prev = sorted(p for p in (Path(state_dir) / "llm" / "inbox").glob("*.json") if p.stem < today)
    cutoff = now - timedelta(hours=default_hours)
    if prev:
        try:
            created = pd.Timestamp(json.loads(prev[-1].read_text())["created"])
            created = created.tz_localize("UTC") if created.tzinfo is None else created.tz_convert("UTC")
            cutoff = min(cutoff, created)
        except Exception:                              # unreadable file: fall back to the default
            pass
    return max(cutoff, now - timedelta(hours=max_hours))


def build_inbox(universe: list[str], fetch_news, now: pd.Timestamp, lookback_hours: int = 24,
                cutoff: pd.Timestamp | None = None) -> dict:
    """Headlines published since `cutoff` (default: the last `lookback_hours`) for every universe
    ticker, de-duplicated."""
    cutoff = cutoff if cutoff is not None else now - timedelta(hours=lookback_hours)
    lookback_hours = round((now - cutoff).total_seconds() / 3600, 1)
    items, seen = [], set()
    for t in universe:
        for it in fetch_news(t) or []:
            pub = pd.Timestamp(it["published"]) if it.get("published") else None
            if pub is None:
                continue
            pub = pub.tz_localize("UTC") if pub.tzinfo is None else pub.tz_convert("UTC")
            if pub < cutoff or pub > now:
                continue
            key = (t, it["title"].strip().lower())
            if key in seen:
                continue
            seen.add(key)
            items.append({"id": item_id(t, it["title"], pub.isoformat()), "ticker": t, "title": it["title"][:300],
                          "summary": (it.get("summary") or "")[:400], "publisher": it.get("publisher"),
                          "published": pub.isoformat()})
    return {"date": now.strftime("%Y-%m-%d"), "created": now.isoformat(), "prompt_version": PROMPT_VERSION,
            "since": cutoff.isoformat(), "lookback_hours": lookback_hours, "universe": universe, "items": items}


def validate_labels(inbox: dict, labels: dict) -> list[str]:
    """Problems with a labels file (empty list = valid)."""
    probs = []
    if labels.get("date") != inbox.get("date"):
        probs.append(f"date mismatch: labels {labels.get('date')} vs inbox {inbox.get('date')}")
    if labels.get("prompt_version") != inbox.get("prompt_version", PROMPT_VERSION):
        probs.append(f"prompt_version {labels.get('prompt_version')} != {inbox.get('prompt_version', PROMPT_VERSION)}")
    if not isinstance(labels.get("labels"), list):
        probs.append("'labels' must be a list")
        return probs
    ids = {it["id"]: it for it in inbox.get("items", [])}
    seen = set()
    for i, lab in enumerate(labels.get("labels", [])):
        lid = lab.get("id")
        if lid not in ids:
            probs.append(f"label {i}: unknown id {lid}")
            continue
        if lid in seen:
            probs.append(f"label {i}: duplicate id {lid}")
        seen.add(lid)
        if lab.get("ticker") != ids[lid]["ticker"]:
            probs.append(f"label {lid}: ticker {lab.get('ticker')} != inbox {ids[lid]['ticker']}")
        if lab.get("event_type") not in EVENT_TYPES:
            probs.append(f"label {lid}: bad event_type {lab.get('event_type')}")
        if lab.get("direction") not in (-1, 0, 1):
            probs.append(f"label {lid}: bad direction {lab.get('direction')}")
        if lab.get("materiality") not in (0, 1, 2, 3):
            probs.append(f"label {lid}: bad materiality {lab.get('materiality')}")
        try:
            c = float(lab.get("confidence"))
            if not 0 <= c <= 1:
                raise ValueError
        except (TypeError, ValueError):
            probs.append(f"label {lid}: bad confidence {lab.get('confidence')}")
    missing = set(ids) - seen
    if missing:
        probs.append(f"{len(missing)} inbox items have no label")
    return probs


def load_labels(state_dir: Path, date: str) -> tuple[list[dict] | None, str]:
    """Valid labels for `date` joined with their headlines, or (None, reason)."""
    ip, lp = inbox_path(state_dir, date), labels_path(state_dir, date)
    if not ip.exists():
        return None, f"no headline inbox for {date}"
    if not lp.exists():
        return None, f"no labels yet for {date}"
    inbox, labels = json.loads(ip.read_text()), json.loads(lp.read_text())
    probs = validate_labels(inbox, labels)
    if probs:
        return None, "labels file invalid: " + "; ".join(probs[:3])
    by_id = {it["id"]: it for it in inbox["items"]}
    out = []
    for lab in labels["labels"]:
        it = by_id[lab["id"]]
        out.append({**lab, "title": it["title"], "published": it["published"]})
    return out, f"{len(out)} labels from {labels.get('labelled_by', 'unknown')}"
