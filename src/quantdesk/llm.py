"""Minimal Anthropic Messages API client used by the LLM news agent, with token cost accounting.

The LLM only labels headlines (event type, direction, materiality). It never sees prices, never
sizes positions and never places orders; fixed rules in the agent do that.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import requests

log = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
PROMPT_VERSION = "news-v1"

SYSTEM = """You label financial news headlines for a systematic trading experiment.
For each item, judge the likely effect of the news itself on the named company's stock over the
next 5 trading days, relative to the market. You cannot see prices. Be conservative: most
headlines (market recaps, opinion pieces, listicles, analyst chatter, old news) are immaterial.

materiality: 0 = noise/opinion/recap, 1 = minor company news, 2 = material company-specific news
(earnings or guidance surprise, major contract, regulatory decision, being acquired, large buyback),
3 = major and unexpected (transformative deal, guidance shock, drug approval/rejection, fraud).
direction: 1 positive, -1 negative, 0 unclear/neutral.
event_type: one of earnings, guidance, m&a, product, regulatory_legal, management, capital_return,
analyst, macro, other.

Return ONLY a JSON array, one object per input item, in the same order:
[{"id": "...", "ticker": "...", "event_type": "...", "direction": 0, "materiality": 0,
  "confidence": 0.0, "rationale": "max 15 words"}]"""


class NewsClassifier:
    def __init__(self, model: str, price_in_per_mtok: float, price_out_per_mtok: float,
                 api_key: str | None = None, max_tokens: int = 4000, timeout: int = 90):
        self.model = model
        self.price_in = price_in_per_mtok
        self.price_out = price_out_per_mtok
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def classify(self, items: list[dict]) -> tuple[list[dict], dict]:
        """Returns (labels, usage) where usage = {input_tokens, output_tokens, cost_usd, raw}."""
        payload_items = [{"id": str(it["id"])[:64], "ticker": it["ticker"], "title": it["title"][:300],
                          "summary": (it.get("summary") or "")[:300], "publisher": it.get("publisher"),
                          "published": it.get("published")} for it in items]
        body = {
            "model": self.model, "max_tokens": self.max_tokens, "temperature": 0, "system": SYSTEM,
            "messages": [{"role": "user", "content": "Items:\n" + json.dumps(payload_items, ensure_ascii=False)}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        resp = None
        for attempt in range(3):
            try:
                resp = requests.post(API_URL, headers=headers, json=body, timeout=self.timeout)
                if resp.status_code in (429, 500, 502, 503, 529):
                    time.sleep(5 * (attempt + 1))
                    continue
                break
            except requests.RequestException as exc:
                log.warning("LLM request failed (attempt %d): %s", attempt + 1, exc)
                time.sleep(5 * (attempt + 1))
        if resp is None or resp.status_code != 200:
            raise RuntimeError(f"LLM call failed: {getattr(resp, 'status_code', None)} {getattr(resp, 'text', '')[:300]}")
        data = resp.json()
        usage = data.get("usage", {})
        tin, tout = int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
        cost = tin / 1e6 * self.price_in + tout / 1e6 * self.price_out
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        labels = parse_labels(text)
        return labels, {"input_tokens": tin, "output_tokens": tout, "cost_usd": cost, "model": data.get("model"),
                        "prompt_version": PROMPT_VERSION}


def parse_labels(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for o in arr:
        try:
            out.append({
                "id": str(o["id"]), "ticker": str(o.get("ticker", "")),
                "event_type": str(o.get("event_type", "other")),
                "direction": int(max(-1, min(1, int(o.get("direction", 0))))),
                "materiality": int(max(0, min(3, int(o.get("materiality", 0))))),
                "confidence": float(max(0.0, min(1.0, float(o.get("confidence", 0.0))))),
                "rationale": str(o.get("rationale", ""))[:200],
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out
