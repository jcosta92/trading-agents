"""Bots dashboard (docs/index.html) and status table (reports/BOTS.md).

With ~900 bots the page is organised by strategy family: a family's bots run the same published
rules on different markets, so the page shows each family's median result and how many of its
variants made money, plus a luck yardstick for the whole league table."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..reporting import _eur, _esc, _page, _pct, _cls

COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--s6)", "var(--s7)", "var(--s8)"]
SINGLE_POSITION = {"BinHV45Bot", "SupertrendBot", "EmaMacdBot", "ORBBot", "VolBreakoutBot", "TimeOfDayBot",
                   "MomentumRotationBot", "IBSBot", "OvernightBot", "TurnOfMonthBot", "LeverageRotationBot",
                   "TurtleBot", "SqueezeBot", "RSIBot", "SignalBot", "EtfDailyBot", "IntradayEtfBot", "TPSBot",
                   "PreFomcBot"}
GROUP_ORDER = ["popular", "alternative", "classic", "repos", "ta", "crypto_research", "equity_research", "ml", "social"]


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
    cap = float(cfg.get("initial_capital", 1000.0))
    fb = fills.groupby("bot") if len(fills) else None
    eb = eq.groupby("bot")["equity_eur"] if len(eq) else None
    rows = []
    for bid, spec in cfg["bots"].items():
        s = (st.get("bots") or {}).get(bid, {})
        f = fb.get_group(bid) if fb is not None and bid in fb.groups else pd.DataFrame(columns=["side", "pnl_eur", "fee_eur"])
        e = eb.get_group(bid) if eb is not None and bid in eb.groups else pd.Series(dtype=float)
        curve = pd.concat([pd.Series([cap]), e])
        mdd = float((curve / curve.cummax() - 1).min()) if len(curve) else float("nan")
        sells = f[f["side"] == "sell"]
        win = float((sells["pnl_eur"] > 0).mean()) if len(sells) and spec["class"] in SINGLE_POSITION else float("nan")
        equity = s.get("equity_eur", cap)
        rows.append({
            "bot": bid, "family": spec.get("family", bid), "variant": spec.get("variant", ""),
            "group": spec.get("group", "popular"), "name": spec.get("name", bid), "class": spec["class"],
            "equity": equity, "return": equity / cap - 1, "max_dd": mdd,
            "buys": int((f["side"] == "buy").sum()), "sells": int(len(sells)),
            "realised": float(sells["pnl_eur"].sum()) if len(sells) else 0.0,
            "fees": float(f["fee_eur"].sum()) if len(f) else 0.0, "win_rate": win,
            "positions": len(s.get("positions") or {}), "error": s.get("error"),
        })
    return pd.DataFrame(rows)


def family_table(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    g = table.groupby("family", sort=False)
    out = pd.DataFrame({
        "group": g["group"].first(), "bots": g.size(), "median": g["return"].median(), "mean": g["return"].mean(),
        "best": g["return"].max(), "worst": g["return"].min(),
        "share_up": g["return"].apply(lambda r: float((r > 0).mean())),
        "trades": g["buys"].sum() + g["sells"].sum(), "fees": g["fees"].sum(),
        "errors": g["error"].apply(lambda e: int(e.notna().sum())),
    })
    fams = cfg.get("families", {})
    out["name"] = [fams.get(f, {}).get("name", f) for f in out.index]
    out["source"] = [fams.get(f, {}).get("source", "") for f in out.index]
    out["rules"] = [fams.get(f, {}).get("rules", "") for f in out.index]
    out["timeframe"] = [fams.get(f, {}).get("timeframe") or "" for f in out.index]
    return out


def hodl(prices: pd.DataFrame, cols: list[str], start: pd.Timestamp, capital: float) -> pd.Series:
    if prices.empty or not all(c in prices for c in cols):
        return pd.Series(dtype=float)
    p = prices.loc[prices.index >= start, cols].dropna()
    if p.empty:
        return pd.Series(dtype=float)
    return (p / p.iloc[0]).mean(axis=1) * capital


def _group_curves(eq: pd.DataFrame, table: pd.DataFrame, start, cap) -> dict[str, pd.Series]:
    """Median equity of each group's bots at each 4-hour snapshot (forward-filled per bot)."""
    if eq.empty:
        return {}
    wide = eq.pivot_table(index="ts", columns="bot", values="equity_eur", aggfunc="last").sort_index()
    wide = wide[wide.index >= start].ffill().fillna(cap)
    out = {}
    for grp, bots in table.groupby("group")["bot"]:
        cols = [b for b in bots if b in wide.columns]
        if cols:
            out[grp] = wide[cols].median(axis=1)
    return out


def _pts(s: pd.Series, start, cap) -> list:
    pts = [[int(start.timestamp() * 1000), cap]] + [[int(t.timestamp() * 1000), round(float(v), 2)] for t, v in s.items()]
    return _thin(pts)


def write(state: Path, cfg: dict, docs: Path, reports: Path, page_name: str = "index.html",
          md_name: str = "BOTS.md", preview: bool = False) -> pd.DataFrame:
    data = load(state)
    table = summary_table(data, cfg)
    fam = family_table(table, cfg)
    st = data["status"]
    start, end = pd.Timestamp(cfg["start"]), pd.Timestamp(cfg["end"])
    now = pd.Timestamp(st.get("ts")) if st.get("ts") else None
    cap = float(cfg.get("initial_capital", 1000.0))
    gnames = cfg.get("group_names", {})
    refs = {"BTC buy-and-hold": hodl(data["prices"], ["BTC_SPOT"], start, cap),
            "BTC/ETH/SOL buy-and-hold": hodl(data["prices"], ["BTC_SPOT", "ETH_SPOT", "SOL_SPOT"], start, cap)}
    ref_now = {k: (float(v.iloc[-1]) / cap - 1 if len(v) else float("nan")) for k, v in refs.items()}

    curves = _group_curves(data["equity"], table, start, cap)
    groups = [g for g in GROUP_ORDER if g in curves] + [g for g in curves if g not in GROUP_ORDER]
    series = [{"id": g, "label": f"{gnames.get(g, g)} (median bot)", "color": COLORS[i % len(COLORS)],
               "points": _pts(curves[g], start, cap)} for i, g in enumerate(groups)]
    for label, s in refs.items():
        if len(s):
            series.append({"id": label, "label": label, "color": "var(--bench)", "bench": True,
                           "points": _thin([[int(t.timestamp() * 1000), round(float(v), 2)] for t, v in s.items()])})

    n = len(table)
    up = int((table["return"] > 0).sum()) if n else 0
    btc = ref_now.get("BTC buy-and-hold", float("nan"))
    beat = int((table["return"] > btc).sum()) if n and math.isfinite(btc) else 0
    trades = int(table["buys"].sum() + table["sells"].sum()) if n else 0
    fees = float(table["fees"].sum()) if n else 0.0
    big = fam[fam["bots"] >= 3] if len(fam) else fam
    best_f = (big if len(big) else fam).sort_values("median", ascending=False).iloc[0] if len(fam) else None
    days = (min(now, end) - start).days if now is not None and now > start else 0
    total_days = (end - start).days
    finished = now is not None and now >= end
    errs = st.get("errors", [])
    err_html = (f"<div class='warn'>{len(errs)} bot error(s) in the last cycle: " +
                _esc("; ".join(errs[:5])) + ("…" if len(errs) > 5 else "") + "</div>") if errs else ""
    order = groups or [g for g in GROUP_ORDER if len(fam) and g in set(fam["group"])]

    frows = []
    for g in order:
        sub = fam[fam["group"] == g].sort_values("median", ascending=False)
        if sub.empty:
            continue
        frows.append(f"<tr class='grp'><td colspan='10' class='l'><b>{_esc(gnames.get(g, g))}</b> "
                     f"<span class='muted'>({len(sub)} families, {int(sub['bots'].sum())} bots)</span></td></tr>")
        for fid, r in sub.iterrows():
            frows.append(
                f"<tr><td class='l'><a href='#f-{_esc(fid)}'>{_esc(r['name'])}</a>"
                + (" <span class='badge fail'>error</span>" if r["errors"] else "") + "</td>"
                f"<td class='l small'>{_esc(r['timeframe'])}</td><td class='num'>{int(r['bots'])}</td>"
                f"<td class='{_cls(r['median'])}'>{_pct(r['median'], 2)}</td>"
                f"<td class='num'>{_pct(r['share_up'], 0, False)}</td>"
                f"<td class='{_cls(r['best'])}'>{_pct(r['best'], 1)}</td><td class='{_cls(r['worst'])}'>{_pct(r['worst'], 1)}</td>"
                f"<td class='num'>{int(r['trades'])}</td><td class='num'>{_eur(r['fees'])}</td>"
                f"<td class='num'>{_eur(r['fees'] / max(1, r['bots']))}</td></tr>")
    ref_rows = "".join(f"<tr><td class='l muted'>{_esc(k)} (no costs)</td><td></td><td></td><td class='{_cls(v)}'>{_pct(v, 2)}</td>"
                       + "<td></td>" * 6 + "</tr>" for k, v in ref_now.items())

    details = []
    tb = table.set_index("bot")
    for g in order:
        sub = fam[fam["group"] == g].sort_values("median", ascending=False)
        if sub.empty:
            continue
        details.append(f"<h3 style='margin-top:22px'>{_esc(gnames.get(g, g))}</h3>")
        for fid, r in sub.iterrows():
            bots = cfg.get("families", {}).get(fid, {}).get("bots", [])
            vrows = "".join(
                f"<tr><td class='l'>{_esc(tb.at[b, 'variant'] or b)}</td><td class='num'>{_eur(tb.at[b, 'equity'])}</td>"
                f"<td class='{_cls(tb.at[b, 'return'])}'>{_pct(tb.at[b, 'return'], 2)}</td>"
                f"<td class='num'>{int(tb.at[b, 'buys'])}</td><td class='num'>{int(tb.at[b, 'sells'])}</td>"
                f"<td class='num'>{_pct(tb.at[b, 'win_rate'], 0, False)}</td><td class='num'>{_eur(tb.at[b, 'fees'])}</td>"
                f"<td class='num'>{_pct(tb.at[b, 'max_dd'], 1, False)}</td></tr>"
                for b in bots if b in tb.index)
            src = str(r["source"])
            src_html = f"<a href='{_esc(src)}'>{_esc(src)}</a>" if src.startswith("http") else _esc(src)
            details.append(f"""
<details class='card' id='f-{_esc(fid)}'><summary><b>{_esc(r['name'])}</b> · {_esc(r['timeframe'])} ·
median {_pct(r['median'], 2)} · {_pct(r['share_up'], 0, False)} of {int(r['bots'])} up</summary>
<p class='small'>{_esc(r['rules'])}</p><p class='small muted'>Source: {src_html}</p>
<div class='tablewrap'><table><thead><tr><th class='l'>Market</th><th>Equity</th><th>Return</th><th>Buys</th><th>Sells</th>
<th>Win rate</th><th>Fees</th><th>Max DD</th></tr></thead><tbody>{vrows}</tbody></table></div></details>""")

    fills = data["fills"]
    last = fills.tail(40).iloc[::-1] if len(fills) else pd.DataFrame()
    trows = "".join(
        f"<tr><td class='l'>{_esc(str(x['ts'])[:16].replace('T', ' '))}</td><td class='l'>{_esc(x['bot'])}</td>"
        f"<td class='l'>{_esc(x['side'])}</td><td class='num'>{x['price']:,.4f}</td><td class='num'>{_eur(x['value_eur'])}</td>"
        f"<td class='{_cls(x['pnl_eur']) if pd.notna(x['pnl_eur']) else 'num'}'>{_eur(x['pnl_eur']) if pd.notna(x['pnl_eur']) else ''}</td>"
        f"<td class='l small'>{_esc(x['tag'])}</td></tr>" for _, x in last.iterrows()) \
        or "<tr><td class='l muted' colspan='7'>No trades yet</td></tr>"

    # luck yardstick: the expected best of n draws of pure noise sits about sqrt(2 ln n) standard
    # deviations above the average, so a top bot that far ahead is no evidence of skill
    spread = float(table["return"].std()) if n > 2 else float("nan")
    luck = spread * math.sqrt(2 * math.log(max(n, 2))) if math.isfinite(spread) else float("nan")

    body = f"""
<button class='toggle' onclick='toggleTheme()'>Light / dark</button>
<h1>{'Preview: what the bots would have done in the month before the test' if preview else 'Trading bots: live demo'}</h1>
{"<div class='warn'>Replay of the month <b>before</b> the live test, same code, data and costs. It shows behaviour; it is not the test.</div>" if preview else ''}
<p class='sub'>{n} bots in {len(fam)} strategy families, each trading its own simulated €1,000 from {start:%d %b %Y %H:%M} to
{end:%d %b %Y %H:%M} UTC. A family is one published strategy (open-source bot, research paper or classic system), run with
its default rules on several markets: judge a family by its median and by how many of its bots made money, not by its best
bot. Real candles, Bitvavo / IBKR fees, spread and slippage on every fill. Last update:
{f"{now:%d %b %Y %H:%M} UTC" if now is not None else 'never'}.</p>
{err_html}
<div class='tiles'>
 <div class='tile'><div class='label'>{'Test finished' if finished else 'Day'}</div><div class='value'>{min(days, total_days)} / {total_days}</div><div class='note'>{'results are final' if finished else 'results are provisional'}</div></div>
 <div class='tile'><div class='label'>Bots in profit</div><div class='value'>{up} / {n}</div><div class='note'>{beat} ahead of BTC buy-and-hold</div></div>
 <div class='tile'><div class='label'>Trades / fees</div><div class='value'>{trades:,}</div><div class='note'>{_eur(fees)} paid in fees</div></div>
 <div class='tile'><div class='label'>Best family (median)</div><div class='value'>{_pct(best_f['median'], 1) if best_f is not None else '–'}</div><div class='note'>{_esc(str(best_f['name'])[:60]) if best_f is not None else ''}</div></div>
 <div class='tile'><div class='label'>Luck yardstick</div><div class='value'>{_pct(luck, 1)}</div><div class='note'>how far the best of {n} bots would sit above the average by chance alone</div></div>
</div>
<h2>Median bot of each group</h2><div class='card' id='chart'></div>
<h2>Families</h2>
<p class='small muted'>Median, best and worst are the family's bots' returns after all costs; "up" = share of its bots in profit.
Click a family for its rules, source and every bot.</p>
<div class='tablewrap'><table><thead><tr><th class='l'>Family</th><th class='l'>Candles</th><th>Bots</th><th>Median</th>
<th>Up</th><th>Best</th><th>Worst</th><th>Trades</th><th>Fees</th><th>Fees / bot</th></tr></thead>
<tbody>{''.join(frows)}{ref_rows}</tbody></table></div>
<h2>Latest trades</h2>
<div class='tablewrap'><table><thead><tr><th class='l'>Candle (UTC)</th><th class='l'>Bot</th><th class='l'>Side</th><th>Price</th>
<th>Value</th><th>P&amp;L</th><th class='l'>Why</th></tr></thead><tbody>{trows}</tbody></table></div>
<h2>Family details</h2>{''.join(details)}
<p class='small muted' style='margin-top:28px'>Simulated trading on real market data; not investment advice. Every fill:
<a href='https://github.com/jcosta92/trading-agents/blob/bot-state/state/bots/fills.csv'>state/bots/fills.csv</a> (bot-state branch) ·
<a href='index.html'>Live test</a> · <a href='preview.html'>Last-month preview</a> ·
<a href='classic.html'>Research agents</a> · <a href='backtest.html'>Backtest report</a></p>
"""
    page = _page("Trading bots", body, {"series": series}).replace(
        "(window.__init||function(){})(D);",
        "lineChart(document.getElementById('chart'),{series:D.series,title:'Equity',prefix:'€',decimals:2,baseline:1000,intraday:true});")
    page = page.replace("</style>", "tr.grp td{background:var(--page)} details summary{cursor:pointer}</style>", 1)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / page_name).write_text(page, encoding="utf-8")
    (docs / ".nojekyll").write_text("")
    reports.mkdir(parents=True, exist_ok=True)
    (reports / md_name).write_text(markdown(table, fam, cfg, st, ref_now), encoding="utf-8")
    return table


def markdown(table: pd.DataFrame, fam: pd.DataFrame, cfg: dict, st: dict, ref_now: dict) -> str:
    gn = cfg.get("group_names", {})
    lines = [f"# Trading bots: {pd.Timestamp(cfg['start']):%d %b} - {pd.Timestamp(cfg['end']):%d %b %Y}", "",
             f"Last update: {str(st.get('ts', '–'))[:16].replace('T', ' ')} UTC. {len(table)} bots in {len(fam)} families, "
             "each started with EUR 1,000.", ""]
    for k, v in ref_now.items():
        lines.append(f"- {k} (reference, no costs): {_pct(v, 2)}")
    lines += ["", "| Family | Group | Bots | Median | Up | Best | Worst | Trades | Fees |", "|---|---|---|---|---|---|---|---|---|"]
    if len(fam):
        for fid, r in fam.sort_values("median", ascending=False).iterrows():
            lines.append(f"| {r['name']} | {gn.get(r['group'], r['group'])} | {int(r['bots'])} | {_pct(r['median'], 2)} | "
                         f"{_pct(r['share_up'], 0, False)} | {_pct(r['best'], 1)} | {_pct(r['worst'], 1)} | {int(r['trades'])} | "
                         f"{_eur(r['fees'])} |")
    if st.get("errors"):
        lines += ["", f"Errors in the last cycle ({len(st['errors'])}):", ""] + [f"- {e}" for e in st["errors"][:30]]
    return "\n".join(lines) + "\n"


def _thin(pts: list, max_points: int = 1500) -> list:
    if len(pts) <= max_points:
        return pts
    step = math.ceil(len(pts) / max_points)
    return pts[::step] + [pts[-1]]
