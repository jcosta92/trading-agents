import json, sys, time, re, requests, xml.etree.ElementTree as ET
UA = {"User-Agent": "trading-agents research bot noreply@example.com"}
FUNDS = {
 "berkshire": 1067983, "pershing": 1336528, "dalal_pabrai": 1549575, "akre": 1112520,
 "markel": 1096343, "himalaya": 1709323, "baupost": 1061768, "appaloosa": 1656456, "greenlight": 1079114, "daily_journal": 783412, "trian": 1345471, "valueact": 1418814,
 "third_point": 1040273, "duquesne": 1536411, "gates": 1166559, "fundsmith": 1569205, "ark": 1697748,
 "oaktree": 949509, "icahn": 921669, "tci": 1647251, "elliott": 1791786, "lone_pine": 1061165,
}
def get(url):
    for k in range(4):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 200: return r
        time.sleep(1 + k)
    r.raise_for_status()
out = {}
for name, cik in FUNDS.items():
    try:
        sub = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        rec = sub["filings"]["recent"]
        fl = []
        for i, f in enumerate(rec["form"]):
            if f in ("13F-HR", "13F-HR/A") and rec["reportDate"][i] in ("2026-06-30", "2026-09-30", "2026-03-31"):
                fl.append({"form": f, "report": rec["reportDate"][i], "filed": rec["filingDate"][i], "acc": rec["accessionNumber"][i]})
        res = []
        for f in fl:
            acc = f["acc"].replace("-", "")
            idx = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json").json()
            xmls = [it["name"] for it in idx["directory"]["item"] if it["name"].lower().endswith(".xml") and "primary" not in it["name"].lower()]
            rows = []
            for xn in xmls:
                t = get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{xn}").text
                t = re.sub(r"\s[\w]+:[\w]+=\"[^\"]*\"", "", t); t = re.sub(r"\sxmlns(:\w+)?=\"[^\"]+\"", "", t); t = re.sub(r'<(/?)\w+:', r'<\1', t)
                root = ET.fromstring(t)
                for it in root.iter("infoTable"):
                    g = lambda k: (it.findtext(k) or "").strip()
                    rows.append({"issuer": g("nameOfIssuer"), "class": g("titleOfClass"), "cusip": g("cusip"),
                                 "value": float(g("value") or 0), "shares": float(it.findtext("shrsOrPrnAmt/sshPrnamt") or 0),
                                 "putcall": g("putCall")})
            f["url"] = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
            f["rows"] = rows
            res.append(f)
            time.sleep(0.3)
        out[name] = {"cik": cik, "entity": sub["name"], "filings": res}
        print(name, sub["name"], [(f["form"], f["report"], f["filed"], len(f["rows"])) for f in res], flush=True)
    except Exception as e:
        print("FAIL", name, e, flush=True)
json.dump(out, open("raw13f.json", "w"))
