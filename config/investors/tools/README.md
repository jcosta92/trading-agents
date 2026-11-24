Scripts that built 13f_2026.json and screens_2026.json (24 Nov 2026), kept to refresh them:
f13.py (13F-HR information tables from SEC EDGAR) -> map13.py (CUSIP -> ticker, OpenFIGI) -> build13.py (top 20
long positions, re-weighted); fund.py (Yahoo statements for the S&P 500) -> genscreens.py (screens, point in time:
periods ending by 30 Jun 2026, prices of 9 Oct 2026). They write into the working directory; the family YAML is
config/families.d/10_investors.yaml (written by genmeta.py + genyaml.py).
