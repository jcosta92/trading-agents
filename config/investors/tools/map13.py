import json, time, requests, collections
a = json.load(open("raw13f.json")); a.update(json.load(open("raw13f_b.json")))
a.pop("pershing", None)
cus = set()
for f in a.values():
    for fl in f["filings"]:
        for r in fl["rows"]:
            cus.add(r["cusip"].upper())
cus = sorted(cus); print(len(cus))
try: m = json.load(open("figi.json"))
except Exception: m = {}
todo = [c for c in cus if c not in m]
for i in range(0, len(todo), 10):
    batch = todo[i:i+10]
    for k in range(5):
        r = requests.post("https://api.openfigi.com/v3/mapping", json=[{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in batch], timeout=30)
        if r.status_code == 429: time.sleep(10); continue
        break
    for c, res in zip(batch, r.json()):
        d = (res.get("data") or [])
        m[c] = {"ticker": d[0]["ticker"], "type": d[0].get("securityType"), "name": d[0].get("name")} if d else None
    time.sleep(2.5)
    if i % 100 == 0: json.dump(m, open("figi.json","w")); print(i, flush=True)
json.dump(m, open("figi.json","w")); print("mapped", sum(1 for v in m.values() if v))
