import json, collections
a = json.load(open("raw13f.json")); a.update(json.load(open("raw13f_b.json"))); a.pop("pershing", None); a.pop("gates", None)
figi = json.load(open("figi.json"))
META = {
 "berkshire": ("Warren Buffett", "Berkshire Hathaway"), "daily_journal": ("Charlie Munger (Daily Journal, run by Munger until 2023)", "Daily Journal Corp"),
 "himalaya": ("Li Lu", "Himalaya Capital"), "dalal_pabrai": ("Mohnish Pabrai", "Dalal Street LLC (Pabrai Investment Funds)"),
 "akre": ("Chuck Akre", "Akre Capital Management"), "markel": ("Tom Gayner", "Markel Group"), "baupost": ("Seth Klarman", "Baupost Group"),
 "appaloosa": ("David Tepper", "Appaloosa LP"), "third_point": ("Dan Loeb", "Third Point"), "duquesne": ("Stanley Druckenmiller", "Duquesne Family Office"),
 "fundsmith": ("Terry Smith", "Fundsmith LLP"), "ark": ("Cathie Wood", "ARK Investment Management"), "icahn": ("Carl Icahn", "Icahn (Carl C)"),
 "tci": ("Chris Hohn", "TCI Fund Management"), "elliott": ("Paul Singer", "Elliott Investment Management"), "lone_pine": ("Stephen Mandel (founder)", "Lone Pine Capital"),
 "trian": ("Nelson Peltz", "Trian Fund Management"), "valueact": ("Jeff Ubben (founder) / Mason Morfit", "ValueAct Holdings"),
 "oaktree": ("Howard Marks", "Oaktree Capital Management"), "pershing_inc": ("Bill Ackman", "Pershing Square"),
 "greenlight": None,
}
BAD = ("NOTE", "DEBT", "BOND", "WARRANT", "WTS", "RIGHT", "PFD", "PREF", "DBCV", "SDCV", "UNIT")
out = {}
for fund, f in a.items():
    if not f["filings"] or META.get(fund) is None:
        continue
    fil = {}
    for fl in sorted(f["filings"], key=lambda x: x["filed"]):
        if fl["report"] not in ("2026-06-30", "2026-09-30") or fl["form"] != "13F-HR":
            continue
        agg = collections.defaultdict(float); unm = 0.0; tot = 0.0
        for r in fl["rows"]:
            if r["putcall"]: continue
            if any(b in r["class"].upper() for b in BAD): continue
            m = figi.get(r["cusip"].upper())
            tot += r["value"]
            if not m or not m["ticker"] or (m.get("type") or "") in ("Warrant", "Right"):
                unm += r["value"]; continue
            t = m["ticker"].replace("/", "-").replace(" ", "-")
            agg[t] += r["value"]
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:20]
        s = sum(v for _, v in top)
        fil[fl["report"]] = {"filed": fl["filed"], "url": fl["url"], "covered_share": round(s / tot, 4) if tot else 0,
                             "weights": {k: round(v / s, 5) for k, v in top}}
    if "2026-06-30" not in fil:
        print("no Q2", fund); continue
    out[fund] = {"manager": META[fund][0], "filer": f["entity"], "cik": f["cik"], "filings": fil}
    print(fund, {k: (v["filed"], len(v["weights"]), v["covered_share"], list(v["weights"])[:6]) for k, v in fil.items()})
json.dump(out, open("13f_clean.json", "w"), indent=1)
