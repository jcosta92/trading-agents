import json, sys, time, pickle
import pandas as pd, yfinance as yf
tick = [t.strip() for t in open("sp500_20260818.txt") if t.strip()]
out = {}
try: out = pickle.load(open("fund.pkl","rb"))
except Exception: pass
for i, t in enumerate(tick):
    if t in out: continue
    y = t.replace(".", "-")
    for k in range(3):
        try:
            tk = yf.Ticker(y)
            d = {"bs": tk.balance_sheet, "is": tk.income_stmt, "cf": tk.cashflow,
                 "qbs": tk.quarterly_balance_sheet, "qis": tk.quarterly_income_stmt, "qcf": tk.quarterly_cashflow,
                 "div": tk.dividends}
            try: d["sector"] = tk.info.get("sector")
            except Exception: d["sector"] = None
            out[t] = d; break
        except Exception as e:
            time.sleep(3)
    if i % 25 == 0:
        pickle.dump(out, open("fund.pkl","wb")); print(i, t, flush=True)
pickle.dump(out, open("fund.pkl","wb")); print("done", len(out))
