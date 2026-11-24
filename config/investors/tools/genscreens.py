"""Point-in-time fundamental screens for 12 Oct 2026: statements for periods ending by 30 Jun 2026,
prices of 9 Oct 2026 (unadjusted close)."""
import json, pickle, math
import numpy as np, pandas as pd
F = pickle.load(open("fund.pkl", "rb"))
PX = pickle.load(open("px_to_20261009.pkl", "rb"))
ASOF = pd.Timestamp("2026-10-09"); CUT = pd.Timestamp("2026-06-30")
SP = [t.strip() for t in open("sp500_20260818.txt") if t.strip() and t.strip() not in ("WBD", "KVUE", "PSKY")]
def cols(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty: return []
    return sorted([c for c in df.columns if pd.Timestamp(c) <= CUT], reverse=True)
def g(df, names, c):
    for n in ([names] if isinstance(names, str) else names):
        if df is not None and n in df.index:
            v = df.loc[n, c]
            if v is not None and not (isinstance(v, float) and math.isnan(v)): return float(v)
    return np.nan
rows = {}
spy = PX["SPY"]["Close"].dropna()
rm = spy.pct_change().loc[:ASOF].iloc[-252:]
for t in SP:
    d = F.get(t); p = PX.get(t)
    if d is None or p is None: continue
    c = p["Close"].dropna().loc[:ASOF]
    if len(c) < 150 or c.index[-1] < ASOF - pd.Timedelta(days=3): continue
    bs, inc, cf, qbs = d["bs"], d["is"], d["cf"], d["qbs"]
    A, Q = cols(bs), cols(qbs)
    I, C = cols(inc), cols(cf)
    if not A or not I or not Q: continue
    a0, i0, q0 = A[0], I[0], Q[0]
    price = float(c.iloc[-1])
    sh = g(qbs, ["Ordinary Shares Number", "Share Issued"], q0)
    if not sh or np.isnan(sh): sh = g(bs, ["Ordinary Shares Number", "Share Issued"], a0)
    mcap = price * sh
    r = {"sector": d.get("sector"), "price": price, "mcap": mcap}
    def ia(k, n): return g(inc, n, I[k]) if len(I) > k else np.nan
    def ba(k, n): return g(bs, n, A[k]) if len(A) > k else np.nan
    def ca(k, n): return g(cf, n, C[k]) if len(C) > k else np.nan
    r["ni"] = [ia(k, "Net Income") for k in range(4)]
    r["eps"] = [ia(k, ["Diluted EPS", "Basic EPS"]) for k in range(4)]
    r["rev"] = [ia(k, ["Total Revenue", "Operating Revenue"]) for k in range(4)]
    r["gp"] = [ia(k, "Gross Profit") for k in range(4)]
    r["ebit"] = ia(0, ["EBIT", "Operating Income"])
    r["ebitda"] = ia(0, ["EBITDA", "Normalized EBITDA"])
    r["ta"] = [ba(k, "Total Assets") for k in range(4)]
    r["eq"] = [ba(k, ["Stockholders Equity", "Common Stock Equity"]) for k in range(4)]
    r["ca"] = [ba(k, "Current Assets") for k in range(2)]
    r["cl"] = [ba(k, "Current Liabilities") for k in range(2)]
    r["ltd"] = [ba(k, "Long Term Debt") for k in range(2)]
    r["debt"] = g(qbs, "Total Debt", q0)
    r["cash"] = g(qbs, ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"], q0)
    r["ppe"] = ba(0, "Net PPE")
    r["shares_a"] = [ba(k, ["Ordinary Shares Number", "Share Issued"]) for k in range(2)]
    r["cfo"] = [ca(k, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]) for k in range(4)]
    r["capex"] = ca(0, "Capital Expenditure"); r["da"] = ca(0, ["Depreciation And Amortization", "Depreciation Amortization Depletion"])
    r["divpaid"] = ca(0, ["Cash Dividends Paid", "Common Stock Dividend Paid"])
    r["buyback"] = ca(0, ["Repurchase Of Capital Stock", "Common Stock Payments"])
    r["issue"] = ca(0, ["Issuance Of Capital Stock", "Common Stock Issuance"])
    r["debt_rep"] = ca(0, ["Repayment Of Debt", "Long Term Debt Payments"]); r["debt_iss"] = ca(0, ["Issuance Of Debt", "Long Term Debt Issuance"])
    div = d["div"]
    if isinstance(div, pd.Series) and len(div):
        di = div.copy(); di.index = pd.DatetimeIndex(di.index).tz_localize(None)
        r["div_ttm"] = float(di[(di.index > ASOF - pd.Timedelta(days=365)) & (di.index <= ASOF)].sum())
        yrs = set(di[di > 0].index.year)
        r["div_years"] = sum(1 for y in range(2006, 2026) if y in yrs)
    else:
        r["div_ttm"], r["div_years"] = 0.0, 0
    ri = c.pct_change().iloc[-252:]
    j = pd.concat([ri, rm], axis=1).dropna()
    r["beta"] = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var()) if len(j) > 200 else np.nan
    r["mom6"] = float(c.iloc[-1] / c.iloc[-127] - 1) if len(c) > 127 else np.nan
    r["fy_end"] = str(a0.date())
    rows[t] = r
print(len(rows), "stocks with data")
fin = lambda r: r["sector"] in ("Financial Services",)
utl = lambda r: r["sector"] in ("Utilities",)
def z(s):
    s = pd.Series(s, dtype=float); s = s.clip(s.quantile(.02), s.quantile(.98)); return (s - s.mean()) / s.std()
def ok(*v): return all(x is not None and not (isinstance(x, float) and np.isnan(x)) for x in v)
out, meta = {}, {}
def put(fid, picks, note, fallback="SHY"):
    picks = list(picks)
    w = {k: round(1 / len(picks), 6) for k in picks} if picks else {fallback: 1.0}
    out[fid] = [{"asof": "2026-10-09", "effective": "2026-10-12", "note": note, "weights": w}]
    print(fid, len(picks), picks[:12])

# ---- Graham defensive (simplified to the data available)
gr = []
for t, r in rows.items():
    if fin(r): continue
    eps3 = np.nanmean(r["eps"][:3]); cr = r["ca"][0] / r["cl"][0] if ok(r["ca"][0], r["cl"][0]) and r["cl"][0] else np.nan
    ncav = r["ca"][0] - r["cl"][0] if ok(r["ca"][0], r["cl"][0]) else np.nan
    bvps = r["eq"][0] / (r["mcap"] / r["price"]) if ok(r["eq"][0]) else np.nan
    if not ok(eps3, cr, ncav, bvps) or eps3 <= 0 or bvps <= 0: continue
    pe, pb = r["price"] / eps3, r["price"] / bvps
    ltd = r["ltd"][0] if ok(r["ltd"][0]) else 0.0
    ni = [x for x in r["ni"] if ok(x)]
    e_old = r["eps"][3] if ok(r["eps"][3]) else (r["eps"][2] if ok(r["eps"][2]) else np.nan)
    if (cr >= 2 and ltd <= ncav and len(ni) >= 3 and min(ni) > 0 and r["div_years"] >= 20 and ok(e_old)
            and r["eps"][0] > e_old and pe <= 15 and (pb <= 1.5 or pe * pb <= 22.5)):
        gr.append((pe * pb, t))
put("graham_defensive", [t for _, t in sorted(gr)[:30]], "qualifiers, at most 30 by lowest P/E x P/B")

# ---- Greenblatt magic formula
mf = {}
for t, r in rows.items():
    if fin(r) or utl(r): continue
    ev = r["mcap"] + (r["debt"] if ok(r["debt"]) else 0) - (r["cash"] if ok(r["cash"]) else 0)
    cap = (r["ca"][0] - r["cl"][0] if ok(r["ca"][0], r["cl"][0]) else 0) + (r["ppe"] if ok(r["ppe"]) else 0)
    if not ok(r["ebit"]) or ev <= 0 or cap <= 0: continue
    mf[t] = (r["ebit"] / ev, r["ebit"] / cap)
d = pd.DataFrame(mf, index=["ey", "roc"]).T
d["rank"] = d["ey"].rank(ascending=False) + d["roc"].rank(ascending=False)
put("magic_formula", list(d.sort_values("rank").index[:30]), "top 30 by EY rank + ROC rank, ex financials/utilities")

# ---- Piotroski F-score in the cheapest B/M quintile
bm = {t: r["eq"][0] / r["mcap"] for t, r in rows.items() if ok(r["eq"][0]) and r["eq"][0] > 0 and r["mcap"] > 0}
bm = pd.Series(bm); hi = bm[bm >= bm.quantile(0.8)].index
fs = {}
for t in hi:
    r = rows[t]
    try:
        ta0, ta1, ta2 = r["ta"][0], r["ta"][1], r["ta"][2] if ok(r["ta"][2]) else r["ta"][1]
        roa0, roa1 = r["ni"][0] / ta1, r["ni"][1] / ta2
        cfo = r["cfo"][0]
        lev0 = (r["ltd"][0] if ok(r["ltd"][0]) else 0) / ta0; lev1 = (r["ltd"][1] if ok(r["ltd"][1]) else 0) / ta1
        cr0 = r["ca"][0] / r["cl"][0]; cr1 = r["ca"][1] / r["cl"][1]
        gm0 = r["gp"][0] / r["rev"][0]; gm1 = r["gp"][1] / r["rev"][1]
        at0 = r["rev"][0] / ta1; at1 = r["rev"][1] / ta2
        s = [roa0 > 0, cfo > 0, roa0 > roa1, cfo / ta1 > roa0, lev0 <= lev1, cr0 > cr1,
             r["shares_a"][0] <= r["shares_a"][1], gm0 > gm1, at0 > at1]
        if any(isinstance(x, (float, np.floating)) and np.isnan(x) for x in (roa0, roa1, cfo, cr0, cr1, gm0, gm1, at0, at1)):
            continue
        fs[t] = int(sum(bool(x) for x in s))
    except (TypeError, ZeroDivisionError, IndexError):
        continue
fs = pd.Series(fs)
pick = list(fs[fs >= 8].index)
note = "B/M top quintile with F-score 8-9"
if len(pick) < 5:
    pick = list(fs[fs >= 7].index); note = "B/M top quintile with F-score >= 7 (fewer than 5 scored 8-9)"
put("piotroski_fscore", sorted(pick), note)

# ---- Lynch PEG
pg = {}
for t, r in rows.items():
    e0 = r["eps"][0]; e3 = r["eps"][3] if ok(r["eps"][3]) else np.nan
    if not ok(e0, e3) or e0 <= 0 or e3 <= 0: continue
    gro = (e0 / e3) ** (1 / 3) - 1
    pe = r["price"] / e0
    if 0.10 <= gro <= 0.50:
        peg = pe / (gro * 100)
        if peg < 1: pg[t] = peg
put("lynch_peg", [t for t, _ in sorted(pg.items(), key=lambda kv: kv[1])[:20]], "PEG < 1 with 3-year EPS growth 10-50%, 20 lowest PEG")

# ---- O'Shaughnessy trending value
vc = {}
for t, r in rows.items():
    m = r["mcap"]
    ev = m + (r["debt"] if ok(r["debt"]) else 0) - (r["cash"] if ok(r["cash"]) else 0)
    sy = -((r["divpaid"] if ok(r["divpaid"]) else 0) + (r["buyback"] if ok(r["buyback"]) else 0) + (r["issue"] if ok(r["issue"]) else 0)) / m
    vc[t] = {"bp": r["eq"][0] / m if ok(r["eq"][0]) else np.nan, "ep": r["ni"][0] / m if ok(r["ni"][0]) else np.nan,
             "sp": r["rev"][0] / m if ok(r["rev"][0]) else np.nan, "ebitda_ev": r["ebitda"] / ev if ok(r["ebitda"]) and ev > 0 else np.nan,
             "cfp": r["cfo"][0] / m if ok(r["cfo"][0]) else np.nan, "sy": sy, "mom6": r["mom6"]}
v = pd.DataFrame(vc).T
score = sum(v[k].rank(pct=True).fillna(0.5) for k in ("bp", "ep", "sp", "ebitda_ev", "cfp", "sy"))
top = score.sort_values(ascending=False).index[:int(round(len(score) * 0.1))]
put("oshaughnessy_trending_value", list(v.loc[top, "mom6"].dropna().sort_values(ascending=False).index[:25]),
    "value composite (P/B, P/E, P/S, EBITDA/EV, P/CF, shareholder yield) top decile, then the 25 best 6-month returns")

# ---- Faber shareholder yield (dividends + net buybacks + net debt paydown)
sy = {}
for t, r in rows.items():
    divs = -(r["divpaid"] if ok(r["divpaid"]) else 0); bb = -((r["buyback"] if ok(r["buyback"]) else 0) + (r["issue"] if ok(r["issue"]) else 0))
    dp = -((r["debt_rep"] if ok(r["debt_rep"]) else 0) + (r["debt_iss"] if ok(r["debt_iss"]) else 0))
    if fin(r): dp = 0.0
    sy[t] = (divs + bb + dp) / r["mcap"]
sy = pd.Series(sy).dropna()
put("faber_shareholder_yield", list(sy.sort_values(ascending=False).index[:50]), "top 10% (50) by (dividends + net buybacks + net debt paydown) / market value; financials without the debt term")

# ---- Novy-Marx gross profitability
gp = pd.Series({t: r["gp"][0] / r["ta"][0] for t, r in rows.items() if not fin(r) and ok(r["gp"][0], r["ta"][0]) and r["ta"][0] > 0})
put("novy_marx_gross_profit", list(gp.sort_values(ascending=False).index[:50]), "top 10% (50) by gross profit / total assets, ex financials")

# ---- Asset growth (Cooper, Gulen & Schill 2008): lowest growth
ag = pd.Series({t: r["ta"][0] / r["ta"][1] - 1 for t, r in rows.items() if ok(r["ta"][0], r["ta"][1]) and r["ta"][1] > 0})
put("asset_growth_low", list(ag.sort_values().index[:50]), "lowest 10% (50) total-asset growth over the last fiscal year")

# ---- Sloan accruals: lowest (NI - CFO) / average assets
ac = pd.Series({t: (r["ni"][0] - r["cfo"][0]) / ((r["ta"][0] + r["ta"][1]) / 2) for t, r in rows.items()
                if not fin(r) and ok(r["ni"][0], r["cfo"][0], r["ta"][0], r["ta"][1])})
put("sloan_accruals_low", list(ac.sort_values().index[:50]), "lowest 10% (50) accruals (net income - operating cash flow) / average assets, ex financials")

# ---- Net share issuance (Pontiff & Woodgate 2008): biggest share-count reduction
ns = pd.Series({t: math.log(r["shares_a"][0] / r["shares_a"][1]) for t, r in rows.items()
                if ok(r["shares_a"][0], r["shares_a"][1]) and r["shares_a"][1] > 0 and r["shares_a"][0] > 0})
put("net_share_issuance_low", list(ns.sort_values().index[:50]), "lowest 10% (50) log change in shares outstanding over the last fiscal year")

# ---- Quality minus junk (long leg) and Buffett's alpha screen and Hagstrom tenets
q = {}
for t, r in rows.items():
    if not ok(r["ta"][0], r["ta"][1], r["ni"][0]) or r["ta"][1] <= 0: continue
    eq = r["eq"][0] if ok(r["eq"][0]) else np.nan
    q[t] = {"gpoa": r["gp"][0] / r["ta"][0] if ok(r["gp"][0]) else np.nan,
            "roe": r["ni"][0] / eq if ok(eq) and eq > 0 else np.nan, "roa": r["ni"][0] / r["ta"][0],
            "cfoa": r["cfo"][0] / r["ta"][0] if ok(r["cfo"][0]) else np.nan,
            "gmar": r["gp"][0] / r["rev"][0] if ok(r["gp"][0], r["rev"][0]) and r["rev"][0] else np.nan,
            "acc": -((r["ni"][0] - r["cfo"][0]) / r["ta"][0]) if ok(r["cfo"][0]) else np.nan,
            "g_roa": (r["ni"][0] - r["ni"][3]) / r["ta"][3] if ok(r["ni"][3], r["ta"][3]) and r["ta"][3] else np.nan,
            "g_gpoa": (r["gp"][0] - r["gp"][3]) / r["ta"][3] if ok(r["gp"][0], r["gp"][3], r["ta"][3]) and r["ta"][3] else np.nan,
            "lowbeta": -r["beta"], "lowlev": -((r["debt"] if ok(r["debt"]) else 0) / r["ta"][0]),
            "bm": eq / r["mcap"] if ok(eq) else np.nan}
Q = pd.DataFrame(q).T.astype(float)
zz = lambda cols_: sum(z(Q[c]).fillna(0) for c in cols_) / len(cols_)
prof = zz(["gpoa", "roe", "roa", "cfoa", "gmar", "acc"]); grow = zz(["g_roa", "g_gpoa"]); safe = zz(["lowbeta", "lowlev"])
qmj = z(prof) + z(grow) + z(safe)
put("qmj_quality", list(qmj.sort_values(ascending=False).index[:50]), "top 10% (50) by quality = z(profitability) + z(growth) + z(safety)")
ba = z(Q["lowbeta"]).fillna(0) + z(qmj).fillna(0) + z(Q["bm"]).fillna(0)
put("buffett_alpha_screen", list(ba.sort_values(ascending=False).index[:20]), "20 best by z(low beta) + z(quality) + z(book/market)")
hg = {}
med_margin = np.nanmedian([r["ni"][0] / r["rev"][0] for r in rows.values() if ok(r["ni"][0], r["rev"][0]) and r["rev"][0]])
for t, r in rows.items():
    roes = [r["ni"][k] / r["eq"][k] for k in range(3) if ok(r["ni"][k], r["eq"][k]) and r["eq"][k] > 0]
    if len(roes) < 3 or min(roes) < 0.15: continue
    if not ok(r["rev"][0]) or not r["rev"][0] or r["ni"][0] / r["rev"][0] < med_margin: continue
    oe = r["ni"][0] + (r["da"] if ok(r["da"]) else 0) + (r["capex"] if ok(r["capex"]) else 0)
    ltd = r["ltd"][0] if ok(r["ltd"][0]) else 0
    if oe <= 0 or ltd > 5 * r["ni"][0]: continue
    hg[t] = oe / r["mcap"]
put("buffett_tenets_screen", [t for t, _ in sorted(hg.items(), key=lambda kv: -kv[1])[:15]],
    "ROE >= 15% in each of the last 3 years, net margin above the S&P 500 median, long-term debt <= 5 years of earnings; 15 best owner-earnings yields")

# ---- Dogs of the Dow
DOW = "MMM AXP AMGN AAPL BA CAT CVX CSCO KO GOOGL GS HD HON IBM JNJ JPM MCD MRK MSFT NKE NVDA PG CRM SHW TRV UNH V WMT DIS AMZN".split()
dy = {}
for t in DOW:
    d = F.get(t); c = PX[t]["Close"].dropna().loc[:ASOF]
    div = d["div"] if d else None
    if div is None or not len(div): dy[t] = (0.0, float(c.iloc[-1])); continue
    di = div.copy(); di.index = pd.DatetimeIndex(di.index).tz_localize(None)
    dy[t] = (float(di[(di.index > ASOF - pd.Timedelta(days=365)) & (di.index <= ASOF)].sum()) / float(c.iloc[-1]), float(c.iloc[-1]))
dogs = sorted(dy, key=lambda t: -dy[t][0])[:10]
put("dogs_of_the_dow", dogs, "10 highest trailing-12-month dividend yields of the Dow 30 on 9 Oct 2026: " + ", ".join(f"{t} {dy[t][0]*100:.2f}%" for t in dogs))
small = sorted(dogs, key=lambda t: dy[t][1])[:5]
put("small_dogs_of_the_dow", small, "the 5 lowest-priced of the 10 Dogs: " + ", ".join(f"{t} ${dy[t][1]:.2f}" for t in small))
json.dump({"_about": "Fundamental screens for the 12 Oct 2026 start. Universe: S&P 500 members on 18 Aug 2026 (Dow 30 for the Dogs). Statements: Yahoo Finance annual / quarterly statements for fiscal periods ending by 30 Jun 2026 (all filed with the SEC before 12 Oct); prices: unadjusted closes of 9 Oct 2026; dividends by ex-date. Sectors are Yahoo's current classification. Built 24 Nov 2026.", **out},
          open("/home/claude/trading-agents/config/investors/screens_2026.json", "w"), indent=1)
