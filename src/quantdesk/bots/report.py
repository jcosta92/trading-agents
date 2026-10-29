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
                   "PreFomcBot", "GoldIfvgBot", "GoldLevelBot"}
BENCH = "benchmarks"           # buy-and-hold references: shown, but left out of every strategy statistic
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
        closes = f[f["pnl_eur"].notna()] if "pnl_eur" in f else sells      # closing fills (a short closes with a buy)
        win = float((closes["pnl_eur"] > 0).mean()) if len(closes) and spec["class"] in SINGLE_POSITION else float("nan")
        equity = float(s.get("equity_eur", cap))
        fees = float(f["fee_eur"].sum()) if len(f) else 0.0
        rows.append({
            "bot": bid, "family": spec.get("family", bid), "variant": spec.get("variant", ""),
            "group": spec.get("group", "popular"), "name": spec.get("name", bid), "class": spec["class"],
            "equity": equity, "return": equity / cap - 1, "max_dd": mdd,
            "buys": int((f["side"] == "buy").sum()), "sells": int(len(sells)),
            "realised": float(closes["pnl_eur"].sum()) if len(closes) else 0.0,
            # equity is net of every fee paid, so P/L before fees adds them back; spread and slippage
            # are inside the fill prices and stay in both figures
            "fees": fees, "pnl_net": equity - cap, "pnl_gross": equity - cap + fees, "win_rate": win,
            "positions": len(s.get("positions") or {}), "error": s.get("error"),
        })
    return pd.DataFrame(rows)


def money(t: pd.DataFrame, cap: float) -> dict:
    """Average bot of a set of bots since the start: P/L before fees, fees, P/L after fees (EUR per bot)
    and the average return after fees. Every bot starts with the same capital, so the average return
    is also the set's total P/L / total capital. `*_total` keep the sums."""
    n = len(t)
    if not n:
        return {"bots": 0, "capital": 0.0, "gross": 0.0, "fees": 0.0, "net": 0.0, "pct": float("nan"),
                "gross_total": 0.0, "fees_total": 0.0, "net_total": 0.0}
    return {"bots": n, "capital": n * cap, "gross": float(t["pnl_gross"].mean()), "fees": float(t["fees"].mean()),
            "net": float(t["pnl_net"].mean()), "pct": float(t["pnl_net"].mean()) / cap,
            "gross_total": float(t["pnl_gross"].sum()), "fees_total": float(t["fees"].sum()),
            "net_total": float(t["pnl_net"].sum())}


def family_table(table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    cap = float(cfg.get("initial_capital", 1000.0))
    g = table.groupby("family", sort=False)
    out = pd.DataFrame({
        "group": g["group"].first(), "bots": g.size(), "median": g["return"].median(), "mean": g["return"].mean(),
        "best": g["return"].max(), "worst": g["return"].min(),
        "share_up": g["return"].apply(lambda r: float((r > 0).mean())),
        "trades": g["buys"].sum() + g["sells"].sum(), "fees": g["fees"].sum(),
        "pnl_gross": g["pnl_gross"].sum(), "pnl_net": g["pnl_net"].sum(),
        "avg_gross": g["pnl_gross"].mean(), "avg_fees": g["fees"].mean(), "avg_net": g["pnl_net"].mean(),
        "errors": g["error"].apply(lambda e: int(e.notna().sum())),
    })
    out["pl_pct"] = out["avg_net"] / cap              # the family's average return after fees
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


def _money_cells(pct, gross, fees, net, strong: bool = False) -> str:
    """P/L % after fees, P/L before fees, fees (shown negative, so the row reads before + fees = after),
    P/L after fees."""
    b = (lambda s: f"<b>{s}</b>") if strong else (lambda s: s)
    return (f"<td class='{_cls(pct)}'>{b(_pct(pct, 2))}</td><td class='{_cls(gross)}'>{b(_eur(gross))}</td>"
            f"<td class='num'>{b(_eur(-fees))}</td><td class='{_cls(net)}'>{b(_eur(net))}</td>")


def _pts(s: pd.Series, start, cap) -> list:
    pts = [[int(start.timestamp() * 1000), cap]] + [[int(t.timestamp() * 1000), round(float(v), 2)] for t, v in s.items()]
    return _thin(pts)


def write(state: Path, cfg: dict, docs: Path, reports: Path, page_name: str = "index.html",
          md_name: str = "BOTS.md", preview: bool = False) -> pd.DataFrame:
    data = load(state)
    table_all = summary_table(data, cfg)
    fam = family_table(table_all, cfg)
    is_bench = table_all["group"] == BENCH if len(table_all) else pd.Series(dtype=bool)
    bench = table_all[is_bench] if len(table_all) else table_all
    table = table_all[~is_bench] if len(table_all) else table_all          # the strategies
    sfam = fam[fam["group"] != BENCH] if len(fam) else fam
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
    eq = data["equity"]
    for i, (_, b) in enumerate(bench.iterrows()):
        e = eq[eq["bot"] == b["bot"]] if len(eq) else eq
        if len(e):
            label = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq-100 (QQQ)", "VGK": "Europe (VGK)"}.get(b["variant"], b["variant"])
            series.append({"id": b["bot"], "label": f"{label} buy-and-hold",
                           "color": COLORS[(len(groups) + i) % len(COLORS)], "bench": True,
                           "points": _pts(e[e["ts"] >= start].set_index("ts")["equity_eur"], start, cap)})

    n = len(table)
    up = int((table["return"] > 0).sum()) if n else 0
    btc = ref_now.get("BTC buy-and-hold", float("nan"))
    beat = int((table["return"] > btc).sum()) if n and math.isfinite(btc) else 0
    trades = int(table["buys"].sum() + table["sells"].sum()) if n else 0
    fees = float(table["fees"].sum()) if n else 0.0
    big = sfam[sfam["bots"] >= 3] if len(sfam) else sfam
    best_f = (big if len(big) else sfam).sort_values("median", ascending=False).iloc[0] if len(sfam) else None
    days = (min(now, end) - start).days if now is not None and now > start else 0
    total_days = (end - start).days
    finished = now is not None and now >= end
    errs = st.get("errors", [])
    err_html = (f"<div class='warn'>{len(errs)} bot error(s) in the last cycle: " +
                _esc("; ".join(errs[:5])) + ("…" if len(errs) > 5 else "") + "</div>") if errs else ""
    order = groups or [g for g in GROUP_ORDER if len(fam) and g in set(fam["group"])]
    if len(bench):
        order = [BENCH] + [g for g in order if g != BENCH]

    tot = money(table, cap)
    frows = []
    for g in order:
        sub = fam[fam["group"] == g].sort_values("median", ascending=False)
        if sub.empty:
            continue
        gm = money(table_all[table_all["group"] == g], cap)
        frows.append(f"<tr class='grp'><td colspan='3' class='l'><b>{_esc(gnames.get(g, g))}</b> "
                     f"<span class='muted'>({len(sub)} families, {int(sub['bots'].sum())} bots)</span></td>"
                     + _money_cells(gm["pct"], gm["gross"], gm["fees"], gm["net"], strong=True)
                     + "<td colspan='5'></td></tr>")
        for fid, r in sub.iterrows():
            frows.append(
                f"<tr><td class='l'><a href='#f-{_esc(fid)}'>{_esc(r['name'])}</a>"
                + (" <span class='badge fail'>error</span>" if r["errors"] else "") + "</td>"
                f"<td class='l small'>{_esc(r['timeframe'])}</td><td class='num'>{int(r['bots'])}</td>"
                + _money_cells(r["pl_pct"], r["avg_gross"], r["avg_fees"], r["avg_net"]) +
                f"<td class='{_cls(r['median'])}'>{_pct(r['median'], 2)}</td>"
                f"<td class='num'>{_pct(r['share_up'], 0, False)}</td>"
                f"<td class='{_cls(r['best'])}'>{_pct(r['best'], 1)}</td><td class='{_cls(r['worst'])}'>{_pct(r['worst'], 1)}</td>"
                f"<td class='num'>{int(r['trades'])}</td></tr>")
    if n:
        frows.append(f"<tr class='grp'><td colspan='2' class='l'><b>All families</b></td><td class='num'><b>{n}</b></td>"
                     + _money_cells(tot["pct"], tot["gross"], tot["fees"], tot["net"], strong=True)
                     + f"<td class='{_cls(float(table['return'].median()))}'><b>{_pct(float(table['return'].median()), 2)}</b></td>"
                     f"<td class='num'><b>{_pct(up / n, 0, False)}</b></td><td></td><td></td>"
                     f"<td class='num'><b>{trades:,}</b></td></tr>")
    ref_rows = "".join(f"<tr><td class='l muted'>{_esc(k)} (no costs)</td><td></td><td></td><td class='{_cls(v)}'>{_pct(v, 2)}</td>"
                       f"<td></td><td></td><td></td><td class='{_cls(v)}'>{_pct(v, 2)}</td>" + "<td></td>" * 4 + "</tr>"
                       for k, v in ref_now.items())

    details = []
    tb = table_all.set_index("bot")
    for g in order:
        sub = fam[fam["group"] == g].sort_values("median", ascending=False)
        if sub.empty:
            continue
        details.append(f"<h3 style='margin-top:22px'>{_esc(gnames.get(g, g))}</h3>")
        for fid, r in sub.iterrows():
            bots = [b for b in cfg.get("families", {}).get(fid, {}).get("bots", []) if b in tb.index]
            vrows = "".join(
                f"<tr><td class='l'>{_esc(tb.at[b, 'variant'] or b)}</td><td class='num'>{_eur(tb.at[b, 'equity'])}</td>"
                + _money_cells(tb.at[b, "return"], tb.at[b, "pnl_gross"], tb.at[b, "fees"], tb.at[b, "pnl_net"]) +
                f"<td class='num'>{int(tb.at[b, 'buys'])}</td><td class='num'>{int(tb.at[b, 'sells'])}</td>"
                f"<td class='num'>{_pct(tb.at[b, 'win_rate'], 0, False)}</td>"
                f"<td class='num'>{_pct(tb.at[b, 'max_dd'], 1, False)}</td></tr>"
                for b in bots)
            vrows += (f"<tr class='grp'><td class='l'><b>Average bot</b></td><td class='num'><b>{_eur(cap + r['avg_net'])}</b></td>"
                      + _money_cells(r["pl_pct"], r["avg_gross"], r["avg_fees"], r["avg_net"], strong=True)
                      + f"<td class='num'><b>{int(tb.loc[bots, 'buys'].sum())}</b></td>"
                      f"<td class='num'><b>{int(tb.loc[bots, 'sells'].sum())}</b></td><td></td><td></td></tr>")
            src = str(r["source"])
            src_html = f"<a href='{_esc(src)}'>{_esc(src)}</a>" if src.startswith("http") else _esc(src)
            details.append(f"""
<details class='card' id='f-{_esc(fid)}'><summary><b>{_esc(r['name'])}</b> · {_esc(r['timeframe'])} ·
average bot {_eur(r['avg_net'])} after fees ({_pct(r['pl_pct'], 2)}) · median {_pct(r['median'], 2)} ·
{_pct(r['share_up'], 0, False)} of {int(r['bots'])} up</summary>
<p class='small'>{_esc(r['rules'])}</p><p class='small muted'>Source: {src_html}</p>
<div class='tablewrap'><table><thead><tr><th class='l'>Market</th><th>Equity</th><th>P/L %<br><span class='small'>after fees</span></th>
<th>P/L €<br><span class='small'>before fees</span></th><th>Fees €</th><th>P/L €<br><span class='small'>after fees</span></th>
<th>Buys</th><th>Sells</th><th>Win rate</th><th>Max DD</th></tr></thead><tbody>{vrows}</tbody></table></div></details>""")

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
<p class='sub'>{n} bots in {len(sfam)} strategy families, each trading its own simulated €{float(cfg.get('initial_capital', 1000.0)):,.0f} from {start:%d %b %Y %H:%M} to
{end:%d %b %Y %H:%M} UTC. A family is one published strategy (open-source bot, research paper or classic system), run with
its default rules on several markets: judge a family by its median and by how many of its bots made money, not by its best
bot. Real candles, Bitvavo / IBKR fees, spread and slippage on every fill. Last update:
{f"{now:%d %b %Y %H:%M} UTC" if now is not None else 'never'}.</p>
{err_html}
<div class='tiles'>
 <div class='tile'><div class='label'>{'Test finished' if finished else 'Day'}</div><div class='value'>{min(days, total_days)} / {total_days}</div><div class='note'>{'results are final' if finished else 'results are provisional'}</div></div>
 <div class='tile'><div class='label'>Bots in profit</div><div class='value'>{up} / {n}</div><div class='note'>{beat} ahead of BTC buy-and-hold</div></div>
 <div class='tile'><div class='label'>Average bot: P/L after fees</div><div class='value {_cls(tot['net']).replace('num', '').strip()}'>{_eur(tot['net'])}</div><div class='note'>{_pct(tot['pct'], 2)} of €{cap:,.0f}. Before fees {_eur(tot['gross'])}; fees {_eur(-tot['fees'])}</div></div>
 <div class='tile'><div class='label'>Trades / fees</div><div class='value'>{trades:,}</div><div class='note'>{_eur(fees)} paid in fees ({_eur(fees / n if n else 0)} per bot)</div></div>
 <div class='tile'><div class='label'>Best family (median)</div><div class='value'>{_pct(best_f['median'], 1) if best_f is not None else '–'}</div><div class='note'>{_esc(_short(str(best_f['name']), 60)) if best_f is not None else ''}</div></div>
 <div class='tile'><div class='label'>Luck yardstick</div><div class='value'>{_pct(luck, 1)}</div><div class='note'>how far the best of {n} bots would sit above the average by chance alone</div></div>
</div>
<h2>Median bot of each group</h2><div class='card' id='chart'></div>
<h2>Families</h2>
<p class='small muted'>P/L columns: the family's average bot since the start ({start:%d %b %H:%M} UTC), per €{cap:,.0f}
account; group and "All families" rows average over all their bots. P/L before fees + fees = P/L after fees; P/L % = the
average return after fees. "Before fees" means before Bitvavo / IBKR commissions; spread and slippage are inside the fill
prices, so they count in both P/L columns. Median, best and worst are single bots' returns after all costs; "up" = share of its
bots in profit. Sorted by median within each group. Click a family for its rules, source and every bot.</p>
<div class='tablewrap'><table class='fam'><thead><tr><th class='l'>Family</th><th class='l'>Candles</th><th>Bots</th>
<th>P/L %<br><span class='small'>after fees</span></th><th>P/L €<br><span class='small'>before fees</span></th><th>Fees €</th>
<th>P/L €<br><span class='small'>after fees</span></th><th>Median</th>
<th>Up</th><th>Best</th><th>Worst</th><th>Trades</th></tr></thead>
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
        "lineChart(document.getElementById('chart'),{series:D.series,title:'Equity',prefix:'€',decimals:2,baseline:" + str(float(cfg.get("initial_capital", 1000.0))) + ",intraday:true});")
    page = page.replace("</style>", "tr.grp td{background:var(--page)} details summary{cursor:pointer} table.fam th,table.fam td{padding:8px 6px} "
        "@media (max-width:600px){table.fam th:first-child,table.fam tr:not(.grp) td:first-child{position:sticky;left:0;"
        "z-index:1;background:var(--surface);min-width:140px;max-width:140px;box-shadow:1px 0 0 var(--grid)}}</style>", 1)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / page_name).write_text(page, encoding="utf-8")
    (docs / ".nojekyll").write_text("")
    reports.mkdir(parents=True, exist_ok=True)
    bench_now = {f"{b['variant']} buy-and-hold (EUR {cap:,.0f} bought at the start, with costs)": b["return"]
                 for _, b in bench.iterrows()}
    (reports / md_name).write_text(markdown(table, sfam, cfg, st, {**bench_now, **ref_now}), encoding="utf-8")
    return table_all


def markdown(table: pd.DataFrame, fam: pd.DataFrame, cfg: dict, st: dict, ref_now: dict) -> str:
    gn = cfg.get("group_names", {})
    cap = float(cfg.get("initial_capital", 1000.0))
    tot = money(table, cap)
    lines = [f"# Trading bots: {pd.Timestamp(cfg['start']):%d %b} - {pd.Timestamp(cfg['end']):%d %b %Y}", "",
             f"Last update: {str(st.get('ts', '–'))[:16].replace('T', ' ')} UTC. {len(table)} bots in {len(fam)} families, "
             f"each started with EUR {cap:,.0f}.", "",
             f"- Average bot since the start: {_eur(tot['gross'])} before fees, {_eur(-tot['fees'])} fees, "
             f"**{_eur(tot['net'])} after fees** ({_pct(tot['pct'], 2)} of EUR {cap:,.0f})"]
    for k, v in ref_now.items():
        lines.append(f"- {k}{'' if 'costs' in k else ' (reference, no costs)'}: {_pct(v, 2)}")
    lines += ["", f"P/L columns are the family's average bot (per EUR {cap:,.0f} account) since the start: before fees + "
              "fees = after fees; P/L % = the average return after fees. Before fees = before commissions; spread "
              "and slippage are in the fill prices, so they count in both. Median, best and worst are single bots' returns "
              "after all costs. Sorted by median.", "",
              "| Family | Group | Bots | P/L % after fees | P/L before fees | Fees | P/L after fees | Median | Up | Best | Worst | Trades |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if len(fam):
        for fid, r in fam.sort_values("median", ascending=False).iterrows():
            lines.append(f"| {r['name']} | {gn.get(r['group'], r['group'])} | {int(r['bots'])} | {_pct(r['pl_pct'], 2)} | "
                         f"{_eur(r['avg_gross'])} | {_eur(-r['avg_fees'])} | {_eur(r['avg_net'])} | {_pct(r['median'], 2)} | "
                         f"{_pct(r['share_up'], 0, False)} | {_pct(r['best'], 1)} | {_pct(r['worst'], 1)} | {int(r['trades'])} |")
        lines.append(f"| **All families** | | {tot['bots']} | **{_pct(tot['pct'], 2)}** | **{_eur(tot['gross'])}** | "
                     f"**{_eur(-tot['fees'])}** | **{_eur(tot['net'])}** | {_pct(float(table['return'].median()), 2)} | "
                     f"{_pct(float((table['return'] > 0).mean()), 0, False)} | | | "
                     f"{int(table['buys'].sum() + table['sells'].sum())} |")
    if st.get("errors"):
        lines += ["", f"Errors in the last cycle ({len(st['errors'])}):", ""] + [f"- {e}" for e in st["errors"][:30]]
    return "\n".join(lines) + "\n"


def _short(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


def _thin(pts: list, max_points: int = 1500) -> list:
    if len(pts) <= max_points:
        return pts
    step = math.ceil(len(pts) / max_points)
    return pts[::step] + [pts[-1]]
