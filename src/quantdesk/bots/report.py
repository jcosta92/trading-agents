"""Bots dashboard (docs/index.html), status table (reports/BOTS.md) and the end-of-month results."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..reporting import _eur, _esc, _page, _pct, _cls

COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s7)"]
SINGLE_POSITION = {"BinHV45Bot", "SupertrendBot", "EmaMacdBot", "ORBBot", "VolBreakoutBot", "TimeOfDayBot",
                   "MomentumRotationBot", "IBSBot"}
GROUPS = {"popular": "Most used bot strategies", "alternative": "Less common strategies"}


def colors(cfg: dict) -> dict[str, str]:
    """Colour per bot, counted within its group (each group has its own chart)."""
    out, seen = {}, {}
    for bid, spec in cfg["bots"].items():
        g = spec.get("group", "popular")
        out[bid] = COLORS[seen.get(g, 0) % len(COLORS)]
        seen[g] = seen.get(g, 0) + 1
    return out


def groups(cfg: dict) -> list[str]:
    return list(dict.fromkeys(spec.get("group", "popular") for spec in cfg["bots"].values()))


def load(state: Path) -> dict:
    out = {"status": {}, "fills": pd.DataFrame(), "equity": pd.DataFrame(), "prices": pd.DataFrame()}
    if (state / "status.json").exists():
        out["status"] = json.loads((state / "status.json").read_text())
    if (state / "fills.csv").exists():
        out["fills"] = pd.read_csv(state / "fills.csv")
    if (state / "equity.csv").exists():
        e = pd.read_csv(state / "equity.csv")
        e["ts"] = pd.to_datetime(e["ts"], utc=True, format="ISO8601")
        out["equity"] = e.drop_duplicates(["ts", "bot"], keep="last").sort_values("ts")
    if (state / "prices.csv").exists():
        p = pd.read_csv(state / "prices.csv")
        p["ts"] = pd.to_datetime(p["ts"], utc=True, format="ISO8601")
        out["prices"] = p.drop_duplicates("ts", keep="last").set_index("ts").sort_index()
    return out


def summary_table(data: dict, cfg: dict) -> pd.DataFrame:
    st, fills, eq = data["status"], data["fills"], data["equity"]
    rows = []
    for bid, spec in cfg["bots"].items():
        s = (st.get("bots") or {}).get(bid, {})
        f = fills[fills["bot"] == bid] if len(fills) else fills
        e = eq[eq["bot"] == bid]["equity_eur"] if len(eq) else pd.Series(dtype=float)
        curve = pd.concat([pd.Series([float(cfg.get("initial_capital", 1000.0))]), e])
        mdd = float((curve / curve.cummax() - 1).min()) if len(curve) else float("nan")
        sells = f[f["side"] == "sell"] if len(f) else f
        realised = float(sells["pnl_eur"].sum()) if len(sells) else 0.0
        win = float((sells["pnl_eur"] > 0).mean()) if len(sells) and spec["class"] in SINGLE_POSITION else float("nan")
        equity = s.get("equity_eur", float(cfg.get("initial_capital", 1000.0)))
        rows.append({
            "bot": bid, "name": spec.get("name", bid), "class": spec["class"], "group": spec.get("group", "popular"),
            "equity": equity,
            "return": equity / float(cfg.get("initial_capital", 1000.0)) - 1, "max_dd": mdd,
            "buys": int((f["side"] == "buy").sum()) if len(f) else 0, "sells": int(len(sells)),
            "realised": realised, "fees": float(f["fee_eur"].sum()) if len(f) else 0.0, "win_rate": win,
            "positions": len(s.get("positions") or {}), "open_orders": len(s.get("open_orders") or []),
            "error": s.get("error"),
        })
    return pd.DataFrame(rows)


def hodl(prices: pd.DataFrame, cols: list[str], start: pd.Timestamp, capital: float) -> pd.Series:
    if prices.empty or not all(c in prices for c in cols):
        return pd.Series(dtype=float)
    p = prices.loc[prices.index >= start, cols].dropna()
    if p.empty:
        return pd.Series(dtype=float)
    return (p / p.iloc[0]).mean(axis=1) * capital


def write(state: Path, cfg: dict, docs: Path, reports: Path, page_name: str = "index.html",
          md_name: str = "BOTS.md", preview: bool = False) -> pd.DataFrame:
    data = load(state)
    table = summary_table(data, cfg)
    st = data["status"]
    start, end = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    now = pd.Timestamp(st.get("ts")) if st.get("ts") else None
    cap = float(cfg.get("initial_capital", 1000.0))
    refs = {"BTC buy-and-hold": hodl(data["prices"], ["BTC_SPOT"], start, cap),
            "BTC/ETH/SOL buy-and-hold": hodl(data["prices"], ["BTC_SPOT", "ETH_SPOT", "SOL_SPOT"], start, cap)}

    col = colors(cfg)
    bench = [{"id": label, "label": label, "color": "var(--bench)", "bench": True,
              "points": _thin([[int(t.timestamp() * 1000), round(float(v), 2)] for t, v in ser.items()])}
             for label, ser in refs.items() if len(ser)]
    charts = []
    for g in groups(cfg):
        series = []
        for bid, spec in cfg["bots"].items():
            if spec.get("group", "popular") != g:
                continue
            e = data["equity"][data["equity"]["bot"] == bid] if len(data["equity"]) else pd.DataFrame()
            pts = [[int(start.timestamp() * 1000), cap]] + [[int(t.timestamp() * 1000), round(float(v), 2)]
                                                          for t, v in zip(e.get("ts", []), e.get("equity_eur", []))]
            series.append({"id": bid, "label": _short(spec), "color": col[bid], "points": _thin(pts)})
        charts.append({"id": f"chart-{g}", "title": GROUPS.get(g, g), "series": series + bench})

    ref_now = {k: (float(v.iloc[-1]) / cap - 1 if len(v) else float("nan")) for k, v in refs.items()}
    rows = []
    for i, r in table.iterrows():
        if i == 0 or table.loc[i - 1, "group"] != r["group"]:
            rows.append(f"<tr><td colspan='11' class='l'><b>{_esc(GROUPS.get(r['group'], r['group']))}</b></td></tr>")
        rows.append(
            f"<tr><td><span class='key' style='background:{col[r['bot']]}'></span>{_esc(r['name'])}"
            + (f" <span class='badge fail'>error</span>" if r["error"] else "") + "</td>"
            f"<td class='num'>{_eur(r['equity'])}</td><td class='{_cls(r['return'])}'>{_pct(r['return'], 2)}</td>"
            f"<td class='num'>{r['buys']}</td><td class='num'>{r['sells']}</td>"
            f"<td class='{_cls(r['realised'])}'>{_eur(r['realised'])}</td><td class='num'>{_eur(r['fees'])}</td>"
            f"<td class='num'>{_pct(r['win_rate'], 0, False)}</td><td class='num'>{_pct(r['max_dd'], 1, False)}</td>"
            f"<td class='num'>{r['positions']}</td><td class='num'>{r['open_orders']}</td></tr>")
    ref_rows = "".join(f"<tr><td class='muted'>{_esc(k)} (reference, no costs)</td><td class='num'>"
                       f"{_eur(cap * (1 + v)) if math.isfinite(v) else '–'}</td><td class='{_cls(v)}'>{_pct(v, 2)}</td>"
                       + "<td></td>" * 8 + "</tr>" for k, v in ref_now.items())

    cards = []
    fills = data["fills"]
    for i, (bid, spec) in enumerate(cfg["bots"].items()):
        if i == 0 or list(cfg["bots"].values())[i - 1].get("group", "popular") != spec.get("group", "popular"):
            cards.append(f"<h3 style='margin-top:20px'>{_esc(GROUPS.get(spec.get('group', 'popular'), ''))}</h3>")
        s = (st.get("bots") or {}).get(bid, {})
        f = fills[fills["bot"] == bid].tail(12).iloc[::-1] if len(fills) else pd.DataFrame()
        frows = "".join(
            f"<tr><td class='l'>{_esc(str(x['ts'])[:16].replace('T', ' '))}</td><td class='l'>{_esc(x['inst'])}</td>"
            f"<td class='l'>{_esc(x['side'])}</td><td class='num'>{x['qty']:.6g}</td><td class='num'>{x['price']:,.4f}</td>"
            f"<td class='num'>{_eur(x['value_eur'])}</td><td class='num'>{_eur(x['fee_eur'], 3)}</td>"
            f"<td class='{_cls(x['pnl_eur']) if pd.notna(x['pnl_eur']) else 'num'}'>"
            f"{_eur(x['pnl_eur']) if pd.notna(x['pnl_eur']) else ''}</td><td class='l small'>{_esc(x['tag'])}</td></tr>"
            for _, x in f.iterrows()) or "<tr><td class='l muted' colspan='9'>No trades yet</td></tr>"
        orows = "".join(f"<li>{_esc(o['side'])} {_esc(o['type'])} {_esc(o['inst'])}"
                        + (f" @ {o['price']:,.4f}" if o.get("price") else "") + f" <span class='muted'>({_esc(o['tag'])})</span></li>"
                        for o in (s.get("open_orders") or [])[:12])
        more = len(s.get("open_orders") or []) - 12
        notes = "".join(f"<li>{_esc(n)}</li>" for n in (s.get("notes") or [])[-6:])
        cards.append(f"""
<details class='card'{' open' if i == 0 else ''}><summary><span class='key' style='background:{col[bid]}'></span>
<b>{_esc(spec.get('name', bid))}</b> · {_esc(spec.get('timeframe', ''))} candles</summary>
<p class='small'>{_esc(spec.get('source', ''))}</p>
<pre class='small' style='white-space:pre-wrap'>{_esc(spec.get('rules', ''))}</pre>
<h3>Resting orders</h3><ul class='small'>{orows or "<li class='muted'>none</li>"}{f"<li class='muted'>... and {more} more</li>" if more > 0 else ''}</ul>
{f"<h3>Latest events</h3><ul class='small'>{notes}</ul>" if notes else ''}
<h3>Latest trades</h3><div class='tablewrap'><table><thead><tr><th class='l'>Candle (UTC)</th><th class='l'>Market</th>
<th class='l'>Side</th><th>Qty</th><th>Price</th><th>Value</th><th>Fee</th><th>P&amp;L</th><th class='l'>Why</th></tr></thead>
<tbody>{frows}</tbody></table></div></details>""")

    days = (min(now, end) - start).days if now is not None and now > start else 0
    total_days = (end - start).days
    best = table.sort_values("return", ascending=False).iloc[0] if len(table) else None
    errs = "".join(f"<div class='warn'>{_esc(e)}</div>" for e in st.get("errors", []))
    finished = now is not None and now >= end
    body = f"""
<button class='toggle' onclick='toggleTheme()'>Light / dark</button>
<h1>{'Preview: what the bots would have done last month' if preview else 'Trading bots: one-month demo'}</h1>
{"<div class='warn'>Replay of the month <b>before</b> the live test, with the same code, data and costs. It shows how the bots behave; it is not the test.</div>" if preview else ''}
<p class='sub'>{len(cfg['bots'])} bots, each trading its own simulated €1,000: {sum(1 for b in cfg['bots'].values() if b.get('group', 'popular') == 'popular')} on the strategies most used and
recommended online and {sum(1 for b in cfg['bots'].values() if b.get('group') == 'alternative')} on less common, published rule-based strategies. They trade from {start:%d %b %Y %H:%M} to {end:%d %b %Y %H:%M} UTC. Every real market candle is replayed in order: orders fill
when the market trades through them, with Bitvavo or IBKR fees, the spread and slippage. On the last day every bot sells
what it holds, so the final result is realised. Last update: {f"{now:%d %b %Y %H:%M} UTC" if now is not None else 'never'}.</p>
{errs}
<div class='tiles'>
 <div class='tile'><div class='label'>{'Test finished' if finished else 'Day'}</div><div class='value'>{min(days, total_days)} / {total_days}</div><div class='note'>{'results are final' if finished else 'results are provisional'}</div></div>
 <div class='tile'><div class='label'>Trades so far</div><div class='value'>{int(table['buys'].sum() + table['sells'].sum()) if len(table) else 0}</div><div class='note'>buys and sells, all bots</div></div>
 <div class='tile'><div class='label'>Fees paid</div><div class='value'>{_eur(float(table['fees'].sum()) if len(table) else 0)}</div><div class='note'>exchange and broker fees</div></div>
 <div class='tile'><div class='label'>Leading bot</div><div class='value'>{_pct(best['return'], 1) if best is not None else '–'}</div><div class='note'>{_esc(_short(cfg['bots'][best['bot']])) if best is not None else ''}</div></div>
</div>
<h2>Equity</h2>{''.join(f"<h3>{_esc(c['title'])}</h3><div class='card' id='{c['id']}'></div>" for c in charts)}
<h2>Scoreboard</h2>
<div class='tablewrap'><table><thead><tr><th>Bot</th><th>Equity</th><th>Return</th><th>Buys</th><th>Sells</th>
<th>Realised P&amp;L</th><th>Fees</th><th>Win rate</th><th>Max drawdown</th><th>Positions</th><th>Resting orders</th></tr></thead>
<tbody>{''.join(rows)}{ref_rows}</tbody></table></div>
<p class='small muted'>Realised P&amp;L is net of all fees of the sold units (average-cost basis). Win rate is shown for bots that
hold one position per market; for the grid and DCA bots it is not meaningful. The buy-and-hold lines pay no fees.</p>
<h2>The bots</h2>{''.join(cards)}
<p class='small muted' style='margin-top:28px'>Simulated trading on real market data; not investment advice. Every fill:
<a href='https://github.com/jcosta92/trading-agents/blob/main/state/bots/fills.csv'>state/bots/fills.csv</a> ·
<a href='index.html'>Live test</a> · <a href='preview.html'>Last-month preview</a> ·
<a href='classic.html'>Research agents</a> · <a href='backtest.html'>Backtest report</a></p>
"""
    page = _page("Trading bots", body, {"charts": charts}).replace(
        "(window.__init||function(){})(D);",
        "D.charts.forEach(c=>lineChart(document.getElementById(c.id),{series:c.series,title:c.title,prefix:'€',"
        "decimals:2,baseline:1000,intraday:true}));")
    docs.mkdir(parents=True, exist_ok=True)
    (docs / page_name).write_text(page, encoding="utf-8")
    (docs / ".nojekyll").write_text("")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / md_name).write_text(markdown(table, cfg, st, ref_now), encoding="utf-8")
    return table


def markdown(table: pd.DataFrame, cfg: dict, st: dict, ref_now: dict) -> str:
    lines = [f"# Trading bots: {pd.Timestamp(cfg['start']):%d %b} - {pd.Timestamp(cfg['end']):%d %b %Y}", "",
             f"Last update: {st.get('ts', '–')[:16].replace('T', ' ')} UTC. Each bot started with EUR 1,000.", "",
             "| Bot | Equity | Return | Buys | Sells | Realised P&L | Fees | Win rate | Max drawdown |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, r in table.iterrows():
        if i == 0 or table.loc[i - 1, "group"] != r["group"]:
            lines.append(f"| **{GROUPS.get(r['group'], r['group'])}** | | | | | | | | |")
        lines.append(f"| {r['name']} | {_eur(r['equity'])} | {_pct(r['return'], 2)} | {r['buys']} | {r['sells']} | "
                     f"{_eur(r['realised'])} | {_eur(r['fees'])} | {_pct(r['win_rate'], 0, False)} | {_pct(r['max_dd'], 1, False)} |")
    for k, v in ref_now.items():
        lines.append(f"| {k} (reference, no costs) | | {_pct(v, 2)} | | | | | | |")
    if st.get("errors"):
        lines += ["", "Errors in the last cycle:", ""] + [f"- {e}" for e in st["errors"]]
    return "\n".join(lines) + "\n"


def _short(spec: dict) -> str:
    return spec.get("name", "").split(" (")[0].split(":")[0][:48]


def _thin(pts: list, max_points: int = 1500) -> list:
    if len(pts) <= max_points:
        return pts
    step = math.ceil(len(pts) / max_points)
    return pts[::step] + [pts[-1]]
