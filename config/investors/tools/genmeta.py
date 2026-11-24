import json
S = json.load(open("/home/claude/trading-agents/config/investors/screens_2026.json"))
PIT = (" Screened once on the start's data (statements for periods ending by 30 Jun 2026, prices of 9 Oct 2026; "
       "config/investors/screens_2026.json), bought equally weighted on 12 Oct and held for a year (until a refreshed "
       "screen is added). If nothing qualifies the bot holds T-bills (SHY).")
U = " Universe: S&P 500 members on 18 Aug 2026."
D = {
 "buffett_alpha_screen": ("buffett", "Buffett's Alpha stock screen: low beta + quality + value (Frazzini, Kabiller & Pedersen 2018)",
    "https://papers.ssrn.com/abstract=3197185",
    "Rank on z(low 1-year beta) + z(quality: profitability, growth, safety as in QMJ) + z(book/market) and hold the 20 best (Berkshire's factor profile, long only, unlevered)." + U),
 "buffett_tenets_screen": ("buffett", "Buffett tenets screen (Robert Hagstrom, 'The Warren Buffett Way'; AAII Buffett-Hagstrom screen)",
    "https://www.aaii.com/journal/article/14973-buy-great-companies-for-the-long-term-buffett-hagstrom-screening-strategy",
    "Financial tenets: return on equity of at least 15% in each of the last 3 fiscal years, net margin above the median, long-term debt no more than 5 years of net income; value tenet: the 15 highest owner-earnings yields (net income + depreciation - capex, over market value)." + U),
 "graham_defensive": ("investors", "Benjamin Graham's defensive investor (The Intelligent Investor ch. 14), simplified",
    "https://en.wikipedia.org/wiki/The_Intelligent_Investor ; Graham, 'The Intelligent Investor' (4th ed. 1973), ch. 14",
    "Non-financial S&P 500 stocks with current ratio >= 2, long-term debt <= net current assets, positive earnings in each of the last 3-4 fiscal years, dividends in each of the last 20 calendar years, EPS higher than 3 years earlier, P/E on 3-year average EPS <= 15 and P/B <= 1.5 (or P/E x P/B <= 22.5). Graham asks for 10 years of earnings and a one-third EPS rise over 10 years: only 4 years of statements were available, hence 'simplified'. Hold all qualifiers (at most 30)." + U),
 "magic_formula": ("investors", "Joel Greenblatt's Magic Formula ('The Little Book That Beats the Market')",
    "https://en.wikipedia.org/wiki/Magic_formula_investing ; https://www.aaii.com/journal/article/greenblatts-magic-formula",
    "Exclude financials and utilities; rank on earnings yield (EBIT / enterprise value) and return on capital (EBIT / (net working capital + net fixed assets)); hold the 30 best combined ranks for a year." + U),
 "piotroski_fscore": ("investors", "Piotroski F-score on cheap stocks (Piotroski 2000)",
    "https://econpapers.repec.org/RePEc:bla:joares:v:38:y:2000:i::p:1-41 ; https://alphaarchitect.com/value-investing-the-use-of-historical-financial-statement-information-to-separate-winners-from-losers/",
    "Highest book-to-market quintile; nine binary signals (ROA > 0, CFO > 0, rising ROA, CFO > ROA, falling leverage, rising current ratio, no new shares, rising gross margin, rising asset turnover); hold the stocks scoring 8-9 (7+ if fewer than five)." + U),
 "lynch_peg": ("investors", "Peter Lynch PEG growth at a reasonable price ('One Up on Wall Street')",
    "Lynch, 'One Up on Wall Street' (1989), ch. 13 ; https://www.validea.com/peter-lynch",
    "P/E divided by the EPS growth rate (3-year EPS CAGR, in %) below 1, growth between 10% and 50% a year (Lynch distrusts faster growth), positive EPS; hold the 20 lowest PEGs." + U),
 "oshaughnessy_trending_value": ("investors", "James O'Shaughnessy Trending Value ('What Works on Wall Street', 4th ed.)",
    "https://www.quant-investing.com/strategies/oshaughnessy-trending-value-investment-strategy ; https://thereformedbroker.com/2011/12/19/trending-value-or-how-to-crush-the-market/",
    "Value Composite Two: percentile ranks on P/B, P/E, P/S, EBITDA/EV, price/cash flow and shareholder yield (missing = 50); take the top decile, then hold the 25 with the best 6-month price change." + U),
 "faber_shareholder_yield": ("investors", "Meb Faber shareholder yield ('Shareholder Yield', 2013)",
    "https://mebfaber.com/2013/05/29/shareholder-yield-a-better-approach-to-dividend-investing/ ; Faber, 'Shareholder Yield' (2013)",
    "Shareholder yield = (dividends + net share buybacks + net debt paydown) / market value; hold the top 10% (50)." + U),
 "novy_marx_gross_profit": ("investors", "Gross profitability premium (Novy-Marx 2013)",
    "https://doi.org/10.1016/j.jfineco.2013.01.003", "Hold the 10% (50) non-financial stocks with the highest gross profit / total assets." + U),
 "asset_growth_low": ("investors", "Asset growth anomaly: low asset growth (Cooper, Gulen & Schill 2008)",
    "https://doi.org/10.1111/j.1540-6261.2008.01370.x", "Hold the 10% (50) stocks with the lowest total-asset growth over the last fiscal year." + U),
 "sloan_accruals_low": ("investors", "Accrual anomaly: low accruals (Sloan 1996)",
    "https://www.jstor.org/stable/248290", "Hold the 10% (50) non-financial stocks with the lowest accruals: (net income - operating cash flow) / average total assets." + U),
 "net_share_issuance_low": ("investors", "Net share issuance anomaly: shrinking share counts (Pontiff & Woodgate 2008)",
    "https://doi.org/10.1111/j.1540-6261.2008.01329.x", "Hold the 10% (50) stocks whose share count fell most (log change over the last fiscal year): the buyback side." + U),
 "qmj_quality": ("investors", "Quality minus junk, long leg (Asness, Frazzini & Pedersen 2019)",
    "https://doi.org/10.1007/s11142-018-9470-2", "Quality = z(profitability: GPOA, ROE, ROA, CFOA, gross margin, low accruals) + z(3-year growth in ROA and GPOA) + z(safety: low beta, low leverage); hold the top 10% (50). Long only." + U),
 "dogs_of_the_dow": ("investors", "Dogs of the Dow (Michael O'Higgins, 'Beating the Dow' 1991)",
    "https://www.dogsofthedow.com/ ; https://en.wikipedia.org/wiki/Dogs_of_the_Dow",
    "Buy the 10 Dow Jones Industrial Average stocks with the highest dividend yield (trailing 12 months, Dow members since the 29 Jun 2026 change), equal weight, hold a year (the classic rebalance is on the first trading day of the year; this test starts on 12 Oct)."),
 "small_dogs_of_the_dow": ("investors", "Small Dogs of the Dow (O'Higgins)",
    "https://www.dogsofthedow.com/ ; https://en.wikipedia.org/wiki/Dogs_of_the_Dow",
    "The 5 lowest-priced of the 10 Dogs of the Dow, equal weight, held a year."),
}
meta = {}
for fid, (grp, name, src, rules) in D.items():
    e = S[fid][0]
    meta[fid] = {"group": grp, "name": name, "source": src, "rules": rules + PIT + f" Picks: {e['note']}.",
                 "class": "EtfWeightBot", "weights": "inv_dated_portfolio", "schedule": "calendar",
                 "params": {"universe": sorted(e["weights"]), "file": "screens_2026.json", "key": fid}}
json.dump(meta, open("screens_meta.json", "w"), indent=1)
print(len(meta))
