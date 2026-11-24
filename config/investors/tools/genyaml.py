import json, yaml
SP = [t.strip() for t in open("sp500_20260818.txt") if t.strip() and t.strip() not in ("WBD", "KVUE", "PSKY")]
b13 = json.load(open("/home/claude/trading-agents/config/investors/13f_2026.json"))
def src13(k):
    return " ; ".join(f"{e['report']} 13F-HR filed {e['filed']}: {e['url']}" for e in b13[k]["portfolios"])
CLONE_RULES = ("Copy the fund's latest 13F-HR filing: the 20 largest long stock/ETF positions by reported value, "
               "re-weighted to 100%, bought on the first session after the SEC filing date and held until the next "
               "filing (options, notes and warrants left out). Start: the Q2 2026 filing (public before 12 Oct); "
               "then each new filing on the day after it is filed.")
CLONE_SRC = "13F cloning: Martin & Puthenpurackal, 'Imitation is the Sincerest Form of Flattery: Warren Buffett and Berkshire Hathaway' (SSRN 806246, https://papers.ssrn.com/abstract=806246)"
fam = {}
def add(fid, **kw):
    assert fid not in fam, fid
    fam[fid] = kw
def mix(fid, group, name, source, rules, m, reb="yearly"):
    add(fid, group=group, name=name, source=source, rules=rules, **{"class": "EtfWeightBot"}, weights="inv_fixed_mix",
        schedule="calendar", params={"universe": list(m), "mix": m, "rebalance": reb})
def clone(fid, group, key, label):
    m = b13[key]
    add(fid, group=group, name=f"{label}: 13F clone of {m['filer']}", source=f"{src13(key)} ; {CLONE_SRC}",
        rules=CLONE_RULES + (" Oaktree's Q3 2026 filing was not on EDGAR by 24 Nov 2026." if key == "oaktree" else ""),
        **{"class": "EtfWeightBot"}, weights="inv_dated_portfolio", schedule="calendar",
        params={"universe": sorted({t for e in m["portfolios"] for t in e["weights"]}), "file": "13f_2026.json", "key": key})

# ------------------------------------------------------------------ price-only stock anomalies on the S&P 500
for fid, name, src, metric, rules in [
    ("jt_momentum_12_1", "Stock momentum 12-1 on the S&P 500 (Jegadeesh & Titman 1993)", "https://doi.org/10.1111/j.1540-6261.1993.tb04702.x", "mom12_1",
     "At each month-end rank the S&P 500 on the 12-month return skipping the last month; hold the top 10% (50 stocks) equally weighted for a month."),
    ("george_hwang_52wh", "52-week-high momentum on the S&P 500 (George & Hwang 2004)", "https://doi.org/10.1111/j.1540-6261.2004.00695.x", "high52",
     "At each month-end rank the S&P 500 on price / 52-week high; hold the top 10% (50 stocks) equally weighted."),
    ("st_reversal_1m", "Short-term reversal: last month's losers on the S&P 500 (Jegadeesh 1990; Lehmann 1990)", "https://doi.org/10.1111/j.1540-6261.1990.tb05110.x", "rev1m",
     "At each month-end buy the 10% (50) S&P 500 stocks with the worst 1-month return, equal weight, held for a month."),
    ("low_volatility_sp500", "Low-volatility stocks on the S&P 500 (Ang, Hodrick, Xing & Zhang 2006; Blitz & van Vliet 2007)", "https://papers.ssrn.com/abstract=980865", "lowvol",
     "At each month-end hold the 10% (50) S&P 500 stocks with the lowest 1-year daily volatility, equal weight."),
    ("bab_long_sp500", "Betting against beta, long leg, on the S&P 500 (Frazzini & Pedersen 2014)", "https://doi.org/10.1016/j.jfineco.2013.10.005", "lowbeta",
     "At each month-end hold the 10% (50) S&P 500 stocks with the lowest shrunk beta (0.6 x 1-year daily beta vs SPY + 0.4), equal weight. Long only, unlevered (the paper levers the low-beta leg to beta 1 and shorts high beta)."),
]:
    add(fid, group="investors", name=name, source=src, rules=rules + " Universe: S&P 500 members on 18 Aug 2026 (the last membership list before the start).",
        **{"class": "EtfWeightBot"}, weights="inv_xs_rank", schedule="calendar", lookback=300,
        params={"universe": "sp500_20260818", "signal_tickers": ["SPY"], "metric": metric, "n": 50})

# ------------------------------------------------------------------ Buffett
mix("buffett_9010", "buffett", "Buffett's 90/10: 90% S&P 500 index fund, 10% short-term Treasuries (2013 letter)",
    "https://www.berkshirehathaway.com/letters/2013ltr.pdf (the instructions for his wife's trust)",
    "10% of the cash in short-term government bonds and 90% in a very low-cost S&P 500 index fund (Vanguard): VOO 90% / SHY 10%, rebalanced on the first trading day of each year.",
    {"VOO": 0.9, "SHY": 0.1})
mix("buffett_brk", "buffett", "Own Berkshire Hathaway (BRK-B), Buffett's own vehicle",
    "https://www.berkshirehathaway.com/ ; https://www.berkshirehathaway.com/letters/letters.html",
    "Buy BRK-B at the start and hold.", {"BRK-B": 1.0}, reb="never")
clone("berkshire_13f", "buffett", "berkshire", "Warren Buffett")
add("buffett_greedy_fearful", group="buffett", name="'Greedy when others are fearful': all in stocks when fear spikes",
    source="https://www.berkshirehathaway.com/letters/1986.html ; 'Buy American. I Am.' NYT 17 Oct 2008 https://www.nytimes.com/2008/10/17/opinion/17buffett.html ; the VIX-above-30 buy signal as quoted in Haghani & White, 'When Fear Spikes, Should You Buy?' (Elm Wealth, Jul 2026) https://elmwealth.com/wp-content/uploads/2026/06/Elm-When-Fear-Spikes-Should-You-Buy.pdf",
    rules="Normally 60% SPY / 40% short Treasuries (SHY); 100% SPY while fear is high: the VIX closes above 30 (variant VIX30) or the S&P 500 is 20% or more below its 52-week closing high (variant DD20, Buffett's Oct 2008 'Buy American' moment). Checked daily 5 minutes before the close. (The 60/40 base is this test's choice.)",
    **{"class": "EtfWeightBot"}, weights="inv_greedy_fearful", schedule="daily", band=0.05,
    params={"universe": ["SPY", "SHY"], "base": 0.6},
    variants=[{"id": "VIX30", "params": {"mode": "vix", "vix": 30, "signal_tickers": ["^VIX"]}},
              {"id": "DD20", "params": {"mode": "drawdown", "dd": 0.2}}])
add("buffett_small_value", group="buffett", name="Small-cap value, buy and hold (Buffett's Graham-and-Doddsville value; Fama-French small value)",
    source="Buffett, 'The Superinvestors of Graham-and-Doddsville' (1984) https://www8.gsb.columbia.edu/sites/valueinvesting/files/files/Buffett1984.pdf ; Fama & French (1992) 'The Cross-Section of Expected Stock Returns', JF ; Buffett 1999 (BusinessWeek): with small sums he could earn 50% a year in small companies",
    rules="Buy one US small-cap value ETF at the start and hold; one bot per ETF: IJS (S&P 600 value), VBR (CRSP small value), AVUV (Avantis), SLYV (SPDR S&P 600 value), DFSV (Dimensional).",
    **{"class": "EtfWeightBot"}, weights="inv_fixed_mix", schedule="calendar",
    params={"rebalance": "never"},
    variants=[{"id": t, "params": {"universe": [t], "mix": {t: 1.0}}} for t in ("IJS", "VBR", "AVUV", "SLYV", "DFSV")])
mix("buffett_alpha_etf", "buffett", "Buffett's Alpha with factor ETFs: low beta + quality + value (Frazzini, Kabiller & Pedersen 2018)",
    "https://papers.ssrn.com/abstract=3197185 (Financial Analysts Journal 74(4), 2018)",
    "The paper explains Berkshire's returns by its loadings on betting-against-beta, quality-minus-junk and value (plus ~1.6x leverage). Long-only, unlevered ETF version: 1/3 each in USMV (minimum volatility), QUAL (quality) and VLUE (value), rebalanced quarterly.",
    {"USMV": 1/3, "QUAL": 1/3, "VLUE": 1/3}, reb="quarterly")
add("buffett_indicator", group="buffett", name="The Buffett indicator: total market below 200% of GDP, else T-bills",
    source="Buffett & Loomis, 'Warren Buffett on the Stock Market', Fortune 10 Dec 2001 https://archive.fortune.com/magazines/fortune/fortune_archive/2001/12/10/314691/index.htm ; reading of 240.24% on 12 Aug 2026: https://www.mitrade.com/insights/news/live-news/article-8-2102462-20260921",
    rules="Market value / GDP = 240.24% on 12 Aug 2026 (published before the start), carried forward with VTI's price. Hold VTI while the ratio is below 200% ('approaching 200% ... you are playing with fire'), otherwise T-bills (SHY). Checked daily.",
    **{"class": "EtfWeightBot"}, weights="inv_buffett_indicator", schedule="daily",
    params={"universe": ["VTI", "SHY"], "ref_ratio": 2.4024, "ref_date": "2026-08-12", "max_ratio": 2.0})
add("brk_buyback_rule", group="buffett", name="Berkshire's buyback rule: BRK-B below 1.2x book, else the S&P 500 fund",
    source="https://fortune.com/2012/12/12/buffetts-buy-changes-berkshire-rules/ (limit raised from 110% to 120% of book, 2012) ; book value: Berkshire 10-Qs https://www.sec.gov/Archives/edgar/data/1067983/000119312526341032/ (Q2 2026, filed 10 Aug) and https://www.sec.gov/Archives/edgar/data/1067983/000119312526452412/ (Q3 2026, filed 9 Nov)",
    rules="Buffett and Munger authorised buybacks at up to 120% of book value (2012-2018), the level at which they judged the shares clearly undervalued. Hold BRK-B while its price is at most 1.2x the last reported book value per class-B share (Q2 2026: $348.26; Q3 2026: $357.73, each usable the day after its 10-Q), otherwise VOO. Checked daily.",
    **{"class": "EtfWeightBot"}, weights="inv_brk_book", schedule="daily",
    params={"universe": ["BRK-B", "VOO"], "max_pb": 1.2,
            "book": [{"effective": "2026-08-11", "bvps_b": 348.26}, {"effective": "2026-11-10", "bvps_b": 357.73}]})
clone("munger_daily_journal", "buffett", "daily_journal", "Charlie Munger (Buffett's partner)")

# ------------------------------------------------------------------ other famous investors: 13F clones
for fid, key, label in [("li_lu_himalaya", "himalaya", "Li Lu"), ("pabrai_dalal_street", "dalal_pabrai", "Mohnish Pabrai"),
                        ("akre_capital", "akre", "Chuck Akre"), ("gayner_markel", "markel", "Tom Gayner"),
                        ("klarman_baupost", "baupost", "Seth Klarman"), ("tepper_appaloosa", "appaloosa", "David Tepper"),
                        ("loeb_third_point", "third_point", "Dan Loeb"), ("druckenmiller_duquesne", "duquesne", "Stanley Druckenmiller"),
                        ("terry_smith_fundsmith", "fundsmith", "Terry Smith"), ("cathie_wood_ark", "ark", "Cathie Wood"),
                        ("icahn_13f", "icahn", "Carl Icahn"), ("hohn_tci", "tci", "Chris Hohn"), ("singer_elliott", "elliott", "Paul Singer"),
                        ("mandel_lone_pine", "lone_pine", "Lone Pine (Stephen Mandel's firm)"), ("peltz_trian", "trian", "Nelson Peltz"),
                        ("valueact_13f", "valueact", "ValueAct (Jeff Ubben's firm)"), ("marks_oaktree", "oaktree", "Howard Marks"),
                        ("ackman_pershing", "pershing_inc", "Bill Ackman")]:
    clone(fid, "investors", key, label)

# ------------------------------------------------------------------ lazy portfolios
LP = "https://www.lazyportfolioetf.com/allocation/"
for fid, name, slug, m in [
    ("dalio_all_weather", "Ray Dalio All Weather (as given to Tony Robbins, 'Money: Master the Game')", "ray-dalio-all-weather", {"VTI": .30, "TLT": .40, "IEI": .15, "DBC": .075, "GLD": .075}),
    ("browne_permanent", "Harry Browne Permanent Portfolio", "harry-browne-permanent", {"VTI": .25, "TLT": .25, "BIL": .25, "GLD": .25}),
    ("bogleheads_three_fund", "Bogleheads three-fund portfolio (Taylor Larimore, after John Bogle)", "bogleheads-three-funds", {"VTI": .5, "VEU": .3, "BND": .2}),
    ("swensen_lazy", "David Swensen's lazy portfolio ('Unconventional Success', Yale CIO)", "david-swensen-lazy", {"VTI": .30, "VNQ": .20, "VEU": .15, "EEM": .05, "IEI": .15, "TIP": .15}),
    ("bernstein_no_brainer", "William Bernstein No-Brainer portfolio", "bill-bernstein-no-brainer", {"IJR": .25, "VEU": .25, "VV": .25, "SHY": .25}),
    ("schultheis_coffeehouse", "Bill Schultheis Coffeehouse portfolio", "bill-schultheis-coffee-house", {"IJS": .1, "IJR": .1, "VTV": .1, "VEU": .1, "VNQ": .1, "VV": .1, "BND": .4}),
    ("faber_ivy_static", "Mebane Faber Ivy portfolio, buy and hold (Harvard/Yale endowment style)", "mebane-faber-ivy", {"VTI": .2, "VEU": .2, "VNQ": .2, "BND": .2, "GSG": .2}),
    ("ferri_core_four", "Rick Ferri Core Four", "rick-ferri-core-four", {"VTI": .48, "VEU": .24, "VNQ": .08, "BND": .20}),
    ("merriman_ultimate_bh", "Paul Merriman Ultimate Buy and Hold (all-equity, 10 asset classes)", "paul-merriman-ultimate-buy-and-hold-strategy", {k: .1 for k in ("IJS", "DLS", "IJR", "EEM", "SPY", "VTV", "VEA", "SCZ", "EFV", "VNQ")}),
    ("swedroe_larry", "Larry Swedroe 'Larry Portfolio' (small value tilt, 70% bonds)", "larry-swedroe-larry-portfolio", {"IJS": .15, "EEM": .075, "DLS": .075, "IEI": .70}),
    ("burns_couch_potato", "Scott Burns Couch Potato", "scott-burns-couch", {"VTI": .5, "TIP": .5}),
    ("marc_faber_portfolio", "Marc Faber's 25/25/25/25 (stocks, bonds, gold, real estate)", "marc-faber-portfolio", {"VNQ": .25, "VV": .13, "VEA": .08, "EEM": .04, "BND": .25, "GLD": .25}),
]:
    mix(fid, "investors", name, LP + slug + "/", "Fixed weights " + ", ".join(f"{k} {v*100:g}%" for k, v in m.items()) + "; bought at the start and rebalanced on the first trading day of each year.", m)

# screens added by genscreens (fundamentals), merged later
scr = json.load(open("screens_meta.json")) if __import__("os").path.exists("screens_meta.json") else {}
for fid, kw in scr.items():
    add(fid, **kw)

# The first bot of this file also asks for every other ticker the file uses, so the cycle's candle
# download (DataHub) fetches them all at once instead of once per bot with new tickers.
allt = sorted({t for f in fam.values() for v in ([f] + [x for x in f.get("variants", []) if isinstance(x, dict)])
               for t in (lambda u: u if isinstance(u, list) else [])(v.get("params", {}).get("universe"))} | set(SP) | {"^VIX"})
for f in fam.values():                 # positions of 2% must not fall under the default 2% trade band
    if f.get("schedule") == "calendar":
        f["band"] = 0.001
first = next(iter(fam))
fam[first]["params"]["signal_tickers"] = sorted(set(allt) - set(SP)) + ["SPY"]
doc = {"groups": {"buffett": "Warren Buffett's strategies", "investors": "Famous investors and stock-factor strategies"},
       "universes": {"sp500_20260818": SP}, "families": fam}
hdr = """# WARREN BUFFETT AND OTHER FAMOUS INVESTORS (added 24 Nov 2026; replayed from the common start 12 Oct 17:00 UTC)
# ---------------------------------------------------------------------------------------------------------------
# Every bot: EUR 5,000, long only, IBKR fees, orders 5 minutes before the US close (EtfWeightBot).
# Point in time: 13F clones start from the Q2 2026 filings (public in Jul-Aug 2026) and switch to each later
# filing on the session after its SEC filing date (config/investors/13f_2026.json). Fundamental screens use
# only statements for periods ending by 30 Jun 2026 (filed before the start) and prices of 9 Oct 2026
# (config/investors/screens_2026.json); they are held for a year (until a refreshed screen is added).
# The price-only S&P 500 anomalies are recomputed live at each month-end.
"""
open("/home/claude/trading-agents/config/families.d/10_investors.yaml", "w").write(hdr + yaml.safe_dump(doc, sort_keys=False, width=140, allow_unicode=True))
print(len(fam), "families")
