# Headline labelling instructions (prompt version news-v2)

You label financial news headlines for a systematic trading experiment. A daily scheduled Claude
task applies these instructions to `state/llm/inbox/<date>.json` and writes
`state/llm/labels/<date>.json`. The labels are the ONLY thing the LLM contributes: fixed rules in
`src/quantdesk/agents/llmnews.py` decide what to buy, how much, and when to sell.

## For every item in the inbox

Judge the likely effect of the news itself on the named company's stock over the next 5 trading
days, relative to the market. You cannot see prices and must not try to infer them. Do not use any
knowledge of what happened after the headline's publication time. Be conservative: most headlines
(market recaps, opinion pieces, listicles, analyst chatter, stories that merely mention the company,
old news) are immaterial.

- `materiality`:
  - 0 = noise, opinion, recap, or the company is only mentioned in passing
  - 1 = minor company news
  - 2 = material company-specific news: an earnings or guidance surprise, a major contract, a
    regulatory decision, being acquired, a large buyback
  - 3 = major and unexpected: a transformative deal, a guidance shock, a drug approval or
    rejection, fraud
- `direction`: 1 positive, -1 negative, 0 unclear or neutral
- `event_type`: one of `earnings`, `guidance`, `m&a`, `product`, `regulatory_legal`, `management`,
  `capital_return`, `analyst`, `macro`, `other`
- `confidence`: 0.0 to 1.0, your confidence in the direction
- `rationale`: at most 15 words

## Output file

Write `state/llm/labels/<date>.json`:

```json
{
  "date": "<date>",
  "prompt_version": "news-v2",
  "labelled_by": "Claude scheduled task",
  "labelled_at": "<UTC ISO timestamp>",
  "labels": [
    {"id": "<inbox item id>", "ticker": "<ticker>", "event_type": "earnings", "direction": 1,
     "materiality": 2, "confidence": 0.7, "rationale": "Beat and raised full-year guidance"}
  ]
}
```

Every inbox item must get exactly one label, with the same `id` and `ticker`. Then run
`python -m quantdesk.cli validate-labels --date <date>`; it must print `OK` before you commit.
