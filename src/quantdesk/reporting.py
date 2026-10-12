"""Static HTML reports (GitHub Pages) and a Markdown status file.

docs/classic.html        paper-run dashboard of the research agents (docs/index.html is the bots page)
docs/backtest.html       backtest report (rebuilt after every backtest)
reports/STATUS.md        the same headline table, readable directly on GitHub
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings

# fixed categorical slot per agent (colour follows the entity everywhere); benchmarks are neutral gray
# (R1, a backtest-only research proxy, shares A2's colour: same factor idea, never on the live dashboard)
SLOTS = {"trend_etf": 1, "trend_etf_t212": 2, "factor_equity": 3, "vrp_spreads": 4, "llm_news": 5,
         "crypto_trend": 6, "funding_carry": 7, "rsi2_meanrev": 8, "factor_ucits_proxy": 3}
BENCH = {"bench_sp500", "bench_btc"}
SHORT = {"bench_sp500": "B0 S&P 500 (benchmark)", "bench_btc": "B1 Bitcoin (benchmark)",
         "trend_etf": "A1 ETF trend, IBKR", "trend_etf_t212": "A1b ETF trend, Trading 212",
         "factor_equity": "A2 Factor tilt", "crypto_trend": "A3 Crypto trend", "funding_carry": "A4 Funding carry",
         "vrp_spreads": "A5 XSP put spreads", "llm_news": "A6 LLM news", "rsi2_meanrev": "A7 RSI(2) dip buying",
         "factor_ucits_proxy": "R1 UCITS factor ETFs"}
GROUPS = [
    ("EUR-funded market agents vs the S&P 500 UCITS benchmark", "bench_sp500",
     ["trend_etf", "trend_etf_t212", "factor_equity", "vrp_spreads", "llm_news", "rsi2_meanrev"]),
    ("Crypto agents vs Bitcoin buy-and-hold", "bench_btc", ["crypto_trend", "funding_carry"]),
]

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6d68;
--grid:#e1e0d9;--axis:#c3c2b7;--ring:rgba(11,11,11,.10);--good:#006300;--bad:#b42f2f;--bench:#898781;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#6250d6;--s8:#e34948}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;
--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a988f;--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);
--good:#0ca30c;--bad:#e66767;--bench:#9a988f;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;
--s6:#008300;--s7:#9085e9;--s8:#e66767}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#9a988f;
--grid:#2c2c2a;--axis:#383835;--ring:rgba(255,255,255,.10);--good:#0ca30c;--bad:#e66767;--bench:#9a988f;--s1:#3987e5;
--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9;--s8:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px;font-weight:650;letter-spacing:-.01em} h2{font-size:18px;margin:36px 0 10px;font-weight:620}
h3{font-size:15px;margin:0 0 6px;font-weight:600}
.sub{color:var(--ink2);margin:0 0 18px;max-width:80ch} .muted{color:var(--muted)} .small{font-size:13px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:16px 16px 10px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 16px}
.tile .label{color:var(--ink2);font-size:13px} .tile .value{font-size:26px;font-weight:620;margin-top:2px}
.tile .note{color:var(--muted);font-size:12px}
.tablewrap{overflow-x:auto;border:1px solid var(--ring);border-radius:12px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:8px 10px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600;background:var(--surface);position:sticky;top:0}
td:first-child,th:first-child{text-align:left;white-space:normal;min-width:200px}
td.l,th.l{text-align:left;white-space:normal} tbody tr:last-child td{border-bottom:none}
.num{font-variant-numeric:tabular-nums} .pos{color:var(--good)} .neg{color:var(--bad)}
.key{display:inline-block;width:14px;height:2px;border-radius:1px;vertical-align:middle;margin-right:7px}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;color:var(--ink2);font-size:13px;margin:2px 0 8px}
.chart{position:relative;width:100%;min-height:40px} .chart svg{display:block;width:100%;overflow:visible}
.chart svg:focus{outline:2px solid var(--s1);outline-offset:2px;border-radius:4px}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--ring);border-radius:8px;
padding:8px 10px;font-size:12.5px;box-shadow:0 4px 16px rgba(0,0,0,.12);min-width:170px;display:none;z-index:2}
.tip .d{color:var(--ink2);margin-bottom:4px} .tip .row{display:flex;align-items:center;gap:6px;line-height:1.6}
.tip .v{font-weight:620;font-variant-numeric:tabular-nums;min-width:62px} .tip .n{color:var(--ink2)}
.multiples{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:12px}
.badge{display:inline-block;white-space:nowrap;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid var(--ring);color:var(--ink2)}
.badge.pass:before{content:"\\2713  ";color:var(--good)} .badge.fail:before{content:"\\2715  ";color:var(--bad)}
.badge.weak:before{content:"\\25CB  "}
.warn{border-left:3px solid var(--s4);padding:8px 12px;background:var(--surface);border-radius:6px;color:var(--ink2);margin:12px 0}
a{color:var(--s1)} code{font-size:12.5px}
@media (max-width:600px){td:first-child,th:first-child{min-width:150px} .tile .value{font-size:22px} h1{font-size:22px}}
details{margin:6px 0} summary{cursor:pointer;color:var(--ink2)}
pre.rules{white-space:pre-wrap;font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;margin:4px 0 10px;color:var(--ink)}
.state{display:inline-block;min-width:44px;font-weight:600} .state.buy{color:var(--good)} .state.sell{color:var(--bad)}
.toggle{float:right;font-size:12.5px;background:var(--surface);color:var(--ink2);border:1px solid var(--ring);border-radius:8px;padding:4px 10px;cursor:pointer}
"""

JS = r"""
(function(){
const NS='http://www.w3.org/2000/svg';
function el(tag,attrs,parent){const e=document.createElementNS(NS,tag);for(const k in attrs)e.setAttribute(k,attrs[k]);if(parent)parent.appendChild(e);return e;}
function niceTicks(lo,hi,n){const span=hi-lo||1;const step0=span/n;const mag=Math.pow(10,Math.floor(Math.log10(step0)));
 const step=[1,2,2.5,5,10].map(m=>m*mag).find(s=>span/s<=n)||10*mag;const out=[];for(let v=Math.ceil(lo/step)*step;v<=hi+1e-9;v+=step)out.push(+v.toFixed(10));return out;}
function logTicks(lo,hi){const out=[];const e0=Math.floor(Math.log10(lo)),e1=Math.ceil(Math.log10(hi));
 for(let e=e0;e<=e1;e++)for(const m of [1,2,5]){const v=m*Math.pow(10,e);if(v>=lo*0.98&&v<=hi*1.02)out.push(v);}return out;}
function fmtNum(v){const a=Math.abs(v);if(a>=1e6)return (v/1e6).toFixed(1)+'M';if(a>=1e5)return (v/1e3).toFixed(0)+'K';return v.toLocaleString('en-GB',{maximumFractionDigits:a<10?2:0});}
function fmtDate(t,span){const d=new Date(t);const mo=d.toLocaleString('en-GB',{month:'short',timeZone:'UTC'});
 const hm=String(d.getUTCHours()).padStart(2,'0')+':'+String(d.getUTCMinutes()).padStart(2,'0');
 return span>3*365*864e5? String(d.getUTCFullYear()) : (span>60*864e5? mo+' '+String(d.getUTCFullYear()).slice(2)
  : (span>3*864e5? d.getUTCDate()+' '+mo : (span>864e5? d.getUTCDate()+' '+mo+' '+hm : hm)));}
window.lineChart=function(host,cfg){
 const series=cfg.series.filter(s=>s.points&&s.points.length);
 const legend=document.createElement('div');legend.className='legend';
 series.forEach(s=>{const sp=document.createElement('span');const k=document.createElement('span');k.className='key';k.style.background=s.color;k.style.opacity=s.opacity||1;sp.appendChild(k);sp.appendChild(document.createTextNode(s.label));legend.appendChild(sp);});
 host.appendChild(legend);
 const wrap=document.createElement('div');wrap.className='chart';host.appendChild(wrap);
 const tip=document.createElement('div');tip.className='tip';wrap.appendChild(tip);
 let svg=null;
 function draw(){
  if(svg)svg.remove();
  const W=Math.max(280,wrap.clientWidth),H=cfg.height||260,m={l:52,r:14,t:10,b:26};
  svg=el('svg',{viewBox:`0 0 ${W} ${H}`,height:H,role:'img','aria-label':cfg.title||'line chart',tabindex:0});
  wrap.insertBefore(svg,tip);
  if(!series.length){const t=el('text',{x:W/2,y:H/2,'text-anchor':'middle',fill:'var(--muted)','font-size':13},svg);t.textContent='No data yet';return;}
  let xs=[],ys=[];series.forEach(s=>s.points.forEach(p=>{xs.push(p[0]);ys.push(p[1]);}));
  const x0=Math.min(...xs),x1=Math.max(...xs);let y0=Math.min(...ys),y1=Math.max(...ys);
  const log=!!cfg.log&&y0>0;
  if(!log){const pad=(y1-y0)*0.08||Math.abs(y1)*0.02||1;y0-=pad;y1+=pad;}else{y0*=0.95;y1*=1.05;}
  const X=t=>m.l+(x1===x0?0.5:(t-x0)/(x1-x0))*(W-m.l-m.r);
  const Y=v=>log? m.t+(Math.log(y1)-Math.log(v))/(Math.log(y1)-Math.log(y0))*(H-m.t-m.b) : m.t+(y1-v)/(y1-y0)*(H-m.t-m.b);
  const g=el('g',{},svg);
  const yt=log?logTicks(y0,y1):niceTicks(y0,y1,5);
  yt.forEach(v=>{el('line',{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),stroke:'var(--grid)','stroke-width':1},g);
   const t=el('text',{x:m.l-8,y:Y(v)+4,'text-anchor':'end',fill:'var(--muted)','font-size':11,style:'font-variant-numeric:tabular-nums'},g);t.textContent=fmtNum(v);});
  const span=x1-x0;const nx=Math.max(2,Math.min(7,Math.floor((W-m.l-m.r)/90)));
  for(let i=0;i<=nx;i++){const t=x0+span*i/nx;const tx=el('text',{x:X(t),y:H-6,'text-anchor':i===0?'start':(i===nx?'end':'middle'),fill:'var(--muted)','font-size':11},g);tx.textContent=fmtDate(t,span);}
  el('line',{x1:m.l,x2:W-m.r,y1:H-m.b,y2:H-m.b,stroke:'var(--axis)','stroke-width':1},g);
  if(cfg.baseline!=null&&cfg.baseline>y0&&cfg.baseline<y1)el('line',{x1:m.l,x2:W-m.r,y1:Y(cfg.baseline),y2:Y(cfg.baseline),stroke:'var(--axis)','stroke-width':1},g);
  series.forEach(s=>{const d=s.points.map((p,i)=>(i?'L':'M')+X(p[0]).toFixed(1)+' '+Y(p[1]).toFixed(1)).join('');
   el('path',{d,fill:'none',stroke:s.color,'stroke-opacity':s.opacity||1,'stroke-width':s.bench?1.5:2,'stroke-linejoin':'round','stroke-linecap':'round'},g);
   const last=s.points[s.points.length-1];el('circle',{cx:X(last[0]),cy:Y(last[1]),r:4,fill:s.color,stroke:'var(--surface)','stroke-width':2},g);});
  const times=[...new Set(xs)].sort((a,b)=>a-b);
  const cross=el('line',{y1:m.t,y2:H-m.b,stroke:'var(--axis)','stroke-width':1,visibility:'hidden'},svg);
  const dots=series.map(s=>el('circle',{r:4,fill:s.color,stroke:'var(--surface)','stroke-width':2,visibility:'hidden'},svg));
  function valueAt(s,t){let best=null;for(const p of s.points){if(p[0]<=t)best=p;else break;}return best;}
  let idx=times.length-1;
  function show(i){idx=Math.max(0,Math.min(times.length-1,i));const t=times[idx];const cx=X(t);
   cross.setAttribute('x1',cx);cross.setAttribute('x2',cx);cross.setAttribute('visibility','visible');
   tip.replaceChildren();const dd=document.createElement('div');dd.className='d';dd.textContent=new Date(t).toISOString().slice(0,cfg.intraday?16:10).replace('T',' ');tip.appendChild(dd);
   series.forEach((s,k)=>{const p=valueAt(s,t);if(!p){dots[k].setAttribute('visibility','hidden');return;}
    dots[k].setAttribute('cx',X(p[0]));dots[k].setAttribute('cy',Y(p[1]));dots[k].setAttribute('visibility','visible');
    const row=document.createElement('div');row.className='row';const key=document.createElement('span');key.className='key';key.style.background=s.color;
    const v=document.createElement('span');v.className='v';v.textContent=(cfg.prefix||'')+p[1].toLocaleString('en-GB',{maximumFractionDigits:cfg.decimals??2,minimumFractionDigits:cfg.decimals??2});
    const n=document.createElement('span');n.className='n';n.textContent=s.label;row.append(key,v,n);tip.appendChild(row);});
   tip.style.display='block';const tw=tip.offsetWidth;let left=cx+14;if(left+tw>W)left=cx-tw-14;tip.style.left=Math.max(0,left)+'px';tip.style.top='8px';}
  function hide(){cross.setAttribute('visibility','hidden');dots.forEach(d=>d.setAttribute('visibility','hidden'));tip.style.display='none';}
  const hit=el('rect',{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:'transparent'},svg);
  hit.addEventListener('pointermove',e=>{const r=svg.getBoundingClientRect();const px=(e.clientX-r.left)*W/r.width;
   const t=x0+(px-m.l)/(W-m.l-m.r)*span;let lo=0,hi=times.length-1;while(hi-lo>1){const mid=(lo+hi)>>1;if(times[mid]<t)lo=mid;else hi=mid;}
   show(Math.abs(times[lo]-t)<Math.abs(times[hi]-t)?lo:hi);});
  hit.addEventListener('pointerleave',hide);
  svg.addEventListener('focus',()=>show(idx));svg.addEventListener('blur',hide);
  svg.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){show(idx-1);e.preventDefault();}if(e.key==='ArrowRight'){show(idx+1);e.preventDefault();}});
 }
 let lastW=0,rt;const redraw=()=>{const w=wrap.clientWidth;if(Math.abs(w-lastW)<2)return;lastW=w;draw();};
 if(window.ResizeObserver){new ResizeObserver(()=>{clearTimeout(rt);rt=setTimeout(redraw,60);}).observe(wrap);}
 else{window.addEventListener('resize',()=>{clearTimeout(rt);rt=setTimeout(redraw,120);});}
 requestAnimationFrame(redraw);
};
window.toggleTheme=function(){const r=document.documentElement;const cur=r.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
 const nx=cur==='dark'?'light':'dark';r.setAttribute('data-theme',nx);try{localStorage.setItem('theme',nx);}catch(e){}};
try{const t=localStorage.getItem('theme');if(t)document.documentElement.setAttribute('data-theme',t);}catch(e){}
})();
"""


def _color(aid: str) -> str:
    return "var(--bench)" if aid in BENCH else f"var(--s{SLOTS.get(aid, 8)})"


def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _pct(x, d=1, signed=True) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{100 * x:+.{d}f}%" if signed else f"{100 * x:.{d}f}%"


def _cls(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)) or x == 0:
        return "num"
    return "num pos" if x > 0 else "num neg"


def _eur(x, d=2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    x = 0.0 if abs(x) < 0.5 * 10 ** -d else x
    return f"−€{abs(x):,.{d}f}" if x < 0 else f"€{x:,.{d}f}"


def _page(title: str, body: str, data: dict) -> str:
    payload = json.dumps(data, default=str).replace("</", "<\\/")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{_esc(title)}</title><style>{CSS}</style></head>"
            f"<body><main>{body}</main><script id='data' type='application/json'>{payload}</script>"
            f"<script>{JS}</script><script>{{const D=JSON.parse(document.getElementById('data').textContent);"
            f"(window.__init||function(){{}})(D);}}</script></body></html>")


# =========================================================================================== live
def load_state(state_dir: Path) -> dict:
    out = {"equity": pd.DataFrame(), "fills": pd.DataFrame(), "runs": [], "paper": {}, "ledgers": {}}
    if (state_dir / "equity.csv").exists():
        eq = pd.read_csv(state_dir / "equity.csv")
        eq["ts"] = pd.to_datetime(eq["ts"], utc=True, format="ISO8601")
        out["equity"] = eq
    if (state_dir / "fills.csv").exists():
        out["fills"] = pd.read_csv(state_dir / "fills.csv")
    if (state_dir / "runs.jsonl").exists():
        out["runs"] = [json.loads(line) for line in (state_dir / "runs.jsonl").read_text().splitlines() if line.strip()]
    if (state_dir / "paper.json").exists():
        out["paper"] = json.loads((state_dir / "paper.json").read_text())
    for p in sorted((state_dir / "ledgers").glob("*.json")) if (state_dir / "ledgers").exists() else []:
        out["ledgers"][p.stem] = json.loads(p.read_text())
    return out


def live_table(state: dict, settings: Settings) -> pd.DataFrame:
    eq = state["equity"]
    rows = []
    gates = _backtest_gates()
    bench_ret = {}
    if eq.empty:
        eq = pd.DataFrame(columns=["ts", "agent", "equity_eur"])
    for aid in ("bench_sp500", "bench_btc"):
        e = eq[eq["agent"] == aid]
        bench_ret[aid] = float(e["equity_eur"].iloc[-1] / 1000.0 - 1) if len(e) else float("nan")
    for aid, spec in settings.agents.items():
        if not spec.paper:
            continue
        e = eq[eq["agent"] == aid]
        led = state["ledgers"].get(aid, {})
        costs = led.get("costs", {})
        if len(e):
            cur = float(e["equity_eur"].iloc[-1])
            curve = e["equity_eur"].astype(float)
            mdd = float((curve / curve.cummax() - 1).min())
        else:
            cur, mdd = float("nan"), float("nan")
        ret = cur / 1000.0 - 1 if math.isfinite(cur) else float("nan")
        bench = "bench_btc" if spec.calendar == "CRYPTO" else "bench_sp500"
        n_fills = int((state["fills"]["agent"] == aid).sum()) if len(state["fills"]) else 0
        explicit = (costs.get("commissions", 0) + costs.get("spread_slippage", 0) + costs.get("fx", 0)
                    + costs.get("liquidation", 0))
        rows.append({
            "agent": aid, "name": spec.name, "equity": cur, "return": ret,
            "vs_bench": (ret - bench_ret[bench]) if aid not in BENCH else float("nan"), "max_dd": mdd,
            "trading_costs": explicit, "funding": costs.get("funding", 0.0), "inference": costs.get("inference", 0.0),
            "commissions": costs.get("commissions", 0.0), "spread": costs.get("spread_slippage", 0.0),
            "fx": costs.get("fx", 0.0), "fills": n_fills, "positions": len(led.get("positions", {})),
            "halted": led.get("halted", False), "halt_reason": led.get("halt_reason"),
            "gate": gates.get(aid, "benchmark" if aid in BENCH else "not backtested"),
        })
    return pd.DataFrame(rows)


def _backtest_gates() -> dict:
    from .config import ROOT
    p = ROOT / "reports" / "backtest" / "results.csv"
    if not p.exists():
        return {}
    t = pd.read_csv(p)
    if "gate" not in t:
        return {}
    t = t[(t["variant"] == "main") & (t["stress"] == 1.0)]
    return dict(zip(t["agent"], t["gate"].fillna("")))


def _gate_badge(g: str) -> str:
    g = str(g or "")
    cls = "pass" if g.startswith("PASS") else "fail" if g.startswith("FAIL") else "weak" if g.startswith("WEAK") else ""
    return f"<span class='badge {cls}'>{_esc(g or '–')}</span>"


def _time_exit_date(opened: str, max_days) -> str:
    if not max_days or not opened:
        return "–"
    try:
        return str(np.busday_offset(pd.Timestamp(opened).date(), int(max_days), roll="forward"))
    except Exception:
        return "–"


def _px(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    return f"{x:,.5g}" if math.isfinite(x) and x else "–"


def resting_rows(state: dict, settings: Settings) -> str:
    """One row per protective order resting on an open position."""
    rows = []
    for aid, led in state["ledgers"].items():
        name = SHORT.get(aid, aid)
        for iid, pr in (led.get("meta", {}).get("protect") or {}).items():
            pos = led.get("positions", {}).get(iid)
            if not pos or pos.get("qty", 0) <= 0:
                continue
            mark = pos.get("last_mark")
            stop, target = pr.get("stop"), pr.get("target")
            dist = (mark / stop - 1) if (mark and stop) else None
            spec = pr.get("spec", {})
            rows.append(
                f"<tr><td><span class='key' style='background:{_color(aid)}'></span>{_esc(name)}</td>"
                f"<td class='l'>{_esc(iid)}</td><td class='num'>{pos['qty']:,.6g}</td><td class='num'>{_px(pr['entry'])}</td>"
                f"<td class='num'>{_px(mark)}</td><td class='num'>{_px(stop)}</td>"
                f"<td class='num'>{_pct(dist, 1, False) if dist is not None else '–'}</td>"
                f"<td class='num'>{_px(target)}</td>"
                f"<td class='num'>{_esc(_time_exit_date(pr.get('opened'), spec.get('max_days')))}</td>"
                f"<td class='l small'>{_esc(spec.get('label', ''))}</td></tr>")
    return "".join(rows)


def _fmt_sig(k: str, v) -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if not math.isfinite(float(v)):
            return "–"
        return f"{float(v):+.1%}" if k.startswith(("roc", "funding")) else f"{float(v):,.5g}"
    return str(v)


def signal_rows(state: dict, settings: Settings) -> str:
    """The indicator readings and resulting state from each agent's last daily decision."""
    rows = []
    for aid, spec in settings.agents.items():
        led = state["ledgers"].get(aid)
        if not spec.paper or not led:
            continue
        meta = led.get("meta", {})
        sigs = meta.get("signals") or {}
        day = meta.get("signals_asof") or meta.get("last_decision") or "–"
        for iid, sg in sigs.items():
            st = str(sg.get("state", ""))
            cls = "buy" if st.startswith("BUY") else "sell" if st.startswith("SELL") else ""
            vals = " · ".join(f"{k} {_fmt_sig(k, v)}" for k, v in sg.items()
                              if k not in ("state", "buy_rule", "sell_rule") and v is not None)
            rows.append(f"<tr><td><span class='key' style='background:{_color(aid)}'></span>{_esc(SHORT.get(aid, aid))}"
                        f"<div class='small muted'>bars to {_esc(day)}</div></td><td class='l'>{_esc(iid)}</td>"
                        f"<td class='l'><span class='state {cls}'>{_esc(st)}</span></td><td class='l small'>{_esc(vals)}</td></tr>")
    return "".join(rows)


def rules_html(settings: Settings) -> str:
    out = []
    for aid, spec in settings.agents.items():
        if not spec.paper or not spec.rules:
            continue
        out.append(f"<details><summary><span class='key' style='background:{_color(aid)}'></span>{_esc(spec.name)}</summary>"
                   f"<pre class='rules'>{_esc(spec.rules)}</pre></details>")
    return "".join(out)


def write_dashboard(state_dir: Path, settings: Settings, docs_dir: Path, reports_dir: Path) -> None:
    state = load_state(state_dir)
    table = live_table(state, settings)
    paper = state["paper"]
    eq = state["equity"]
    last_run = state["runs"][-1] if state["runs"] else {}
    start = pd.Timestamp(paper["start"]) if paper.get("start") else None
    end = pd.Timestamp(paper["end"]) if paper.get("end") else None
    now = pd.Timestamp(last_run.get("ts")) if last_run.get("ts") else None
    days = (now - start).days if (start is not None and now is not None) else 0
    length = int(settings.paper.get("length_days", 31))

    # chart data: one point per run, value in EUR
    groups = []
    for title, bench, members in GROUPS:
        series = []
        for aid in [*members, bench]:
            e = eq[eq["agent"] == aid] if len(eq) else eq
            if not len(e):
                continue
            pts = [[int(t.timestamp() * 1000), round(float(v), 2)] for t, v in zip(e["ts"], e["equity_eur"])]
            series.append({"id": aid, "label": SHORT.get(aid, aid), "color": _color(aid), "points": pts,
                           "bench": aid in BENCH})
        groups.append({"title": title, "series": series})

    beating = table[(~table["agent"].isin(BENCH)) & (table["vs_bench"] > 0)] if len(table) else table
    n_agents = int((~table["agent"].isin(BENCH)).sum()) if len(table) else 0
    total_costs = float(table["trading_costs"].sum() + table["inference"].sum()) if len(table) else 0.0
    status = paper.get("status", "not started")
    errors = last_run.get("errors") or []

    rows_html = []
    for _, r in table.iterrows():
        halted = " <span class='badge fail'>halted</span>" if r["halted"] else ""
        rows_html.append(
            f"<tr><td><span class='key' style='background:{_color(r['agent'])}'></span>{_esc(r['name'])}{halted}</td>"
            f"<td class='num'>{_eur(r['equity'])}</td><td class='{_cls(r['return'])}'>{_pct(r['return'], 2)}</td>"
            f"<td class='{_cls(r['vs_bench'])}'>{_pct(r['vs_bench'], 2)}</td><td class='num'>{_pct(r['max_dd'], 1, False)}</td>"
            f"<td class='num'>{_eur(r['trading_costs'])}</td><td class='num'>{_eur(r['funding'])}</td>"
            f"<td class='num'>{r['fills']}</td><td class='l'>{_gate_badge(r['gate'])}</td></tr>")
    cost_rows = []
    for _, r in table.iterrows():
        cost_rows.append(f"<tr><td>{_esc(r['name'])}</td><td class='num'>{_eur(r['commissions'])}</td>"
                         f"<td class='num'>{_eur(r['spread'])}</td><td class='num'>{_eur(r['fx'])}</td>"
                         f"<td class='num'>{_eur(r['funding'])}</td>"
                         f"<td class='num'>{_pct(r['trading_costs'] / 1000.0, 2, False)}</td></tr>")
    notes_html = []
    for aid, a in (last_run.get("agents") or {}).items():
        notes = a.get("notes") or []
        extra = []
        if a.get("error"):
            extra.append(f"ERROR: {a['error']}")
        if a.get("llm_status"):
            extra.append(f"LLM: {a['llm_status']}")
        items = "".join(f"<li>{_esc(n)}</li>" for n in (extra + notes)[:12]) or "<li class='muted'>nothing to report</li>"
        name = settings.agents[aid].name if aid in settings.agents else aid
        notes_html.append(f"<details><summary>{_esc(name)} — market {'open' if a.get('market_open') else 'closed'}, "
                          f"{a.get('n_fills', 0)} fills, {a.get('n_rejects', 0)} rejects</summary><ul class='small'>{items}</ul></details>")
    fills = state["fills"].tail(40).iloc[::-1] if len(state["fills"]) else pd.DataFrame()
    resting = resting_rows(state, settings)
    signals = signal_rows(state, settings)
    fill_rows = "".join(
        f"<tr><td class='l'>{_esc(str(f['ts'])[:16].replace('T', ' '))}</td><td class='l'>{_esc(f['agent'])}</td>"
        f"<td class='l'>{_esc(f['inst'])}</td><td class='num'>{f['qty']:+,.4g}</td><td class='num'>{f['price']:,.4f}</td>"
        f"<td class='num'>{f['fees']:,.2f} {_esc(f['fee_ccy'])}</td><td class='num'>{_eur(f['value_eur'])}</td>"
        f"<td class='l small'>{_esc(f['reason'])}</td></tr>" for _, f in fills.iterrows())

    period = (f"{start:%d %b %Y} → {end:%d %b %Y}" if start is not None and end is not None else "not started yet")
    upd = f"{now:%d %b %Y %H:%M} UTC" if now is not None else "never"
    err_html = "".join(f"<div class='warn'>{_esc(e)}</div>" for e in errors)
    body = f"""
<button class='toggle' onclick='toggleTheme()'>Light / dark</button>
<h1>Trading agents — one-month paper run</h1>
<p class='sub'>Each agent trades its own simulated €1,000 account with fixed indicator rules: it buys and sells when its
indicators fire, and every entry carries a stop-loss and, where the strategy has one, a take-profit or time exit, checked
every hour while the market is open. Orders fill against live quotes with the full cost model (IBKR, Trading 212, Bitvavo
and Kraken fee schedules, the bid-ask spread, slippage, FX conversion and perp funding).
Period: <b>{_esc(period)}</b> · status: <b>{_esc(status)}</b> · last update: {_esc(upd)}.</p>
{err_html}
<div class='tiles'>
 <div class='tile'><div class='label'>Days elapsed</div><div class='value'>{min(days, length)} / {length}</div><div class='note'>One month cannot separate skill from luck</div></div>
 <div class='tile'><div class='label'>Agents ahead of their benchmark</div><div class='value'>{len(beating)} of {n_agents}</div><div class='note'>Net of all costs, since the start</div></div>
 <div class='tile'><div class='label'>Costs paid, all agents</div><div class='value'>{_eur(total_costs)}</div><div class='note'>Fees, spread, slippage, FX</div></div>
 <div class='tile'><div class='label'>EUR/USD used</div><div class='value'>{f"{last_run['fx_eurusd']:.4f}" if last_run.get('fx_eurusd') else '–'}</div><div class='note'>{_esc(last_run.get('fx_source') or '')}</div></div>
</div>
<h2>Equity</h2>
<div id='charts'></div>
<h2>Scoreboard</h2>
<div class='tablewrap'><table><thead><tr><th>Agent</th><th>Equity</th><th>Return</th><th>vs benchmark</th><th>Max drawdown</th>
<th>Trading costs</th><th>Funding paid</th><th>Fills</th><th class='l'>Backtest gate</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody></table></div>
<p class='small muted'>Benchmarks: EUR-funded agents vs the S&amp;P 500 UCITS ETF; crypto agents vs Bitcoin buy-and-hold. "Funding paid"
is negative when an agent received funding. Gate = verdict of the pre-registered multi-year backtest (reports/backtest).</p>
<h2>Stop-loss and take-profit orders resting now</h2>
<div class='tablewrap'><table><thead><tr><th>Agent</th><th class='l'>Instrument</th><th>Qty</th><th>Entry</th><th>Last price</th>
<th>Stop-loss</th><th>Room to stop</th><th>Take-profit</th><th>Time exit on</th><th class='l'>Rule</th></tr></thead>
<tbody>{resting or "<tr><td class='l muted' colspan='10'>No open positions with protective orders</td></tr>"}</tbody></table></div>
<p class='small muted'>Levels are set from the actual fill price; trailing stops only ever move up, once a day, from the last
completed daily bar. A level is treated as hit when the live price touches it at an hourly check, and the position is
then sold at the bid (so a fast move can fill beyond the level, as a real stop order would).</p>
<h2>Indicator signals at the last daily decision</h2>
<div class='tablewrap'><table><thead><tr><th>Agent</th><th class='l'>Instrument</th><th class='l'>State</th>
<th class='l'>Indicators</th></tr></thead>
<tbody>{signals or "<tr><td class='l muted' colspan='4'>No decisions yet</td></tr>"}</tbody></table></div>
<h2>Trading rules (pre-registered)</h2>
<div class='card'>{rules_html(settings) or "<p class='muted'>–</p>"}</div>
<h2>Where the costs went</h2>
<div class='tablewrap'><table><thead><tr><th>Agent</th><th>Commissions &amp; fees</th><th>Spread &amp; slippage</th><th>FX</th>
<th>Funding</th><th>Trading costs as % of €1,000</th></tr></thead><tbody>{''.join(cost_rows)}</tbody></table></div>
<h2>Latest run</h2>
<div class='card'>{''.join(notes_html) or "<p class='muted'>No runs yet.</p>"}</div>
<h2>Recent fills</h2>
<div class='tablewrap'><table><thead><tr><th class='l'>Time (UTC)</th><th class='l'>Agent</th><th class='l'>Instrument</th><th>Qty</th>
<th>Price</th><th>Fees</th><th>Value</th><th class='l'>Reason</th></tr></thead><tbody>{fill_rows or "<tr><td class='l muted' colspan='8'>No fills yet</td></tr>"}</tbody></table></div>
<p class='small muted' style='margin-top:28px'>Paper trading with simulated fills against (often 15-minute delayed) public quotes.
Simulated fills are not real executions. The LLM agent's headline labels come from a scheduled Claude task, not a paid API.
Not investment advice. <a href='index.html'>Trading bots</a> · <a href='backtest.html'>Backtest report</a></p>
"""
    page = _page("Trading agents dashboard", body, {"groups": groups}).replace(
        "(window.__init||function(){})(D);",
        "D.groups.forEach(g=>{const c=document.createElement('div');c.className='card';const h=document.createElement('h3');"
        "h.textContent=g.title;c.appendChild(h);document.getElementById('charts').appendChild(c);"
        "lineChart(c,{series:g.series,title:g.title,prefix:'€',decimals:2,baseline:1000,intraday:true});});")
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "classic.html").write_text(page, encoding="utf-8")
    (docs_dir / ".nojekyll").write_text("")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "STATUS.md").write_text(status_markdown(table, paper, last_run, upd), encoding="utf-8")


def status_markdown(table: pd.DataFrame, paper: dict, last_run: dict, upd: str) -> str:
    lines = ["# Paper run status", "", f"Last update: {upd} · status: **{paper.get('status', 'not started')}** · "
             f"period: {paper.get('start', '–')[:10]} → {paper.get('end', '–')[:10]}", "",
             "| Agent | Equity | Return | vs benchmark | Max DD | Trading costs | Funding | Fills | Backtest gate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['name']}{' (HALTED)' if r['halted'] else ''} | {_eur(r['equity'])} | {_pct(r['return'], 2)} | "
                     f"{_pct(r['vs_bench'], 2)} | {_pct(r['max_dd'], 1, False)} | {_eur(r['trading_costs'])} | "
                     f"{_eur(r['funding'])} | {r['fills']} | {r['gate']} |")
    if last_run.get("errors"):
        lines += ["", "## Errors in the last run", ""] + [f"- {e}" for e in last_run["errors"]]
    lines += ["", "Simulated fills against public quotes; not investment advice. Full dashboard: docs/classic.html (GitHub Pages)."]
    return "\n".join(lines) + "\n"


# =========================================================================================== backtest
def write_backtest_html(out_dir: Path, settings: Settings, docs_dir: Path | None = None) -> None:
    from .config import ROOT
    out_dir = Path(out_dir)
    res = pd.read_csv(out_dir / "results.csv")
    summary = json.loads((out_dir / "summary.json").read_text())
    details = json.loads((out_dir / "details.json").read_text()) if (out_dir / "details.json").exists() else {}
    curves = pd.read_csv(out_dir / "equity_curves.csv", index_col=0, parse_dates=True)
    main1 = res[(res["variant"] == "main") & (res["stress"] == 1.0)].set_index("agent")
    by = res[res["variant"] == "main"].pivot_table(index="agent", columns="stress", values="cagr")

    multiples = []
    for aid in main1.index:
        col = f"{aid}@x1"
        if col not in curves:
            continue
        a = curves[col].dropna()
        cal = settings.agents[aid].calendar if aid in settings.agents else "XETR"
        bench = "bench_btc" if cal == "CRYPTO" else "bench_sp500"
        series = [{"id": aid, "label": f"{SHORT.get(aid, aid)}, x1 costs", "color": _color(aid), "points": _pts(a)}]
        if f"{aid}@x2" in curves:
            series.append({"id": aid + "_x2", "label": "same, 2x costs", "color": _color(aid), "opacity": 0.4,
                           "points": _pts(curves[f"{aid}@x2"].dropna()), "bench": True})
        if aid not in BENCH and f"{bench}@x1" in curves:
            b = curves[f"{bench}@x1"].dropna()
            b = b[b.index >= a.index[0]]
            if len(b):
                series.append({"id": bench, "label": "benchmark (rebased)", "color": "var(--bench)",
                               "points": _pts(b / b.iloc[0] * a.loc[:b.index[0]].iloc[-1] if len(a.loc[:b.index[0]]) else b),
                               "bench": True})
        multiples.append({"id": aid, "title": main1.loc[aid, "name"], "series": series})

    rows = []
    for aid, r in main1.iterrows():
        cg = lambda s: _pct(by.loc[aid, s], 1) if s in by.columns and aid in by.index else "–"
        rows.append(
            f"<tr><td><span class='key' style='background:{_color(aid)}'></span>{_esc(r['name'])}"
            f"<div class='small muted'>{_esc(r['start'])} → {_esc(r['end'])}</div></td>"
            f"<td class='{_cls(by.loc[aid].get(0.0))}'>{cg(0.0)}</td>"
            f"<td class='{_cls(r['cagr'])}'>{cg(1.0)}</td><td class='{_cls(by.loc[aid].get(2.0))}'>{cg(2.0)}</td>"
            f"<td class='num'>{r['sharpe']:.2f} <span class='muted'>({_n(r.get('sharpe_ci_lo'))} to {_n(r.get('sharpe_ci_hi'))})</span></td>"
            f"<td class='num'>{_pct(r['max_drawdown'], 1, False)}</td><td class='num'>{_pct(r['cost_drag_per_year'], 2, False)}</td>"
            f"<td class='num'>{_n(r.get('turnover_per_year'), 1)}x</td><td class='{_cls(r.get('rel_excess_cagr'))}'>{_pct(r.get('rel_excess_cagr'), 1)}</td>"
            f"<td class='num'>{_pct(r.get('kill_switch_month_freq'), 0, False)}</td>"
            f"<td class='num'>{_n(r.get('dsr'))}</td><td class='l'>{_gate_badge(r.get('gate'))}</td></tr>")
    var = res[res["variant"] != "main"]
    var_rows = "".join(f"<tr><td>{_esc(r['agent'])}</td><td class='l'>{_esc(r['variant'])}</td><td class='{_cls(r.get('cagr'))}'>"
                       f"{_pct(r.get('cagr'))}</td><td class='num'>{_n(r.get('sharpe'))}</td><td class='num'>{_pct(r.get('max_drawdown'), 1, False)}</td></tr>"
                       for _, r in var.iterrows())
    notes = "".join(f"<li><b>{_esc(s.name)}</b>: {_esc(s.backtest_note)}</li>" for s in settings.agents.values() if s.backtest_note)
    dm = summary.get("data_meta", {})
    data_notes = []
    if dm.get("proxy"):
        data_notes.append("History extended with proxies before each UCITS ETF existed: " + ", ".join(
            f"{k} ← {v['symbol']} until {v['until']}" for k, v in dm["proxy"].items()))
    if dm.get("missing"):
        data_notes.append("Instruments with no data: " + ", ".join(dm["missing"]))
    fm = dm.get("funding", {})
    for k, v in fm.items():
        data_notes.append(f"Funding for {k}: source {v.get('chosen')} " + ", ".join(
            f"{s}: {i.get('events')} events from {str(i.get('first'))[:10]}"
            for s, i in v.get("sources", {}).items() if isinstance(i, dict)))
    cov = [(aid, d.get("coverage_summary")) for aid, d in details.items() if d.get("coverage_summary")]
    for aid, c in cov:
        data_notes.append(f"{aid}: on average {100 * c['avg_share_with_price_data']:.0f}% of index members had Yahoo price "
                          f"data (minimum {100 * c['min_share_with_price_data']:.0f}%). The missing members are mostly companies "
                          f"that were later delisted, acquired or renamed: a survivorship bias that flatters the result.")
    synth = "<div class='warn'><b>Synthetic data.</b> These numbers only test the code.</div>" if summary.get("synthetic_data") else ""
    body = f"""
<button class='toggle' onclick='toggleTheme()'>Light / dark</button>
<h1>Backtest report</h1>
<p class='sub'>Pre-registered rules, run on daily data through {_esc(summary.get('data_end'))} (commit <code>{_esc(summary.get('git'))}</code>).
Costs: x0 = no trading costs, x1 = modelled fees + spread + slippage + FX, x2 = those trading costs doubled (decisions are
identical at every level; perp funding is market data and is never scaled). {summary.get('n_trials')} distinct
configurations have been tested, which is what the Deflated Sharpe Ratio (DSR) corrects for.</p>
{synth}
<div class='tablewrap'><table><thead><tr><th>Agent and period</th><th>CAGR x0</th><th>CAGR x1</th><th>CAGR x2</th>
<th>Sharpe x1 (90% CI)</th><th>Max DD</th><th>Cost drag / yr</th><th>Turnover / yr</th><th>Excess vs benchmark</th>
<th>Kill switch, share of months</th><th>DSR</th>
<th class='l'>Gate</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class='small muted'>Gate: FAIL if the Sharpe ratio at 2x costs is not positive or the profit disappears without the best calendar
year; PASS needs DSR ≥ 0.95; otherwise WEAK. "Kill switch, share of months": how often a one-month paper run starting on a
random day of the backtest would have been halted by the agent's loss limit. The paper run trades every agent anyway.</p>
<h2>Equity curves (€1,000 start, log scale)</h2>
<div class='multiples' id='mult'></div>
<h2>Robustness variants</h2>
<p class='sub small'>Declared before the backtest. They count as trials in the DSR and are never used to pick the configuration that is paper traded.</p>
<div class='tablewrap'><table><thead><tr><th>Agent</th><th class='l'>Variant</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th></tr></thead>
<tbody>{var_rows or "<tr><td class='muted' colspan='5'>none</td></tr>"}</tbody></table></div>
<h2>Caveats</h2>
<ul class='small'>{notes}{''.join(f'<li>{_esc(n)}</li>' for n in data_notes)}</ul>
<p class='small muted'><a href='index.html'>Trading bots</a> · <a href='classic.html'>Research agents</a></p>
"""
    page = _page("Backtest report", body, {"multiples": multiples}).replace(
        "(window.__init||function(){})(D);",
        "const cards=D.multiples.map(m=>{const c=document.createElement('div');c.className='card';const h=document.createElement('h3');"
        "h.textContent=m.title;c.appendChild(h);document.getElementById('mult').appendChild(c);return c;});"
        "D.multiples.forEach((m,i)=>lineChart(cards[i],{series:m.series,title:m.title,log:true,prefix:'€',decimals:0,height:220}));")
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    docs = docs_dir or (ROOT / "docs")
    if not summary.get("synthetic_data"):
        docs.mkdir(parents=True, exist_ok=True)
        (docs / "backtest.html").write_text(page, encoding="utf-8")


def _pts(s: pd.Series, max_points: int = 600) -> list:
    s = s.dropna()
    if len(s) > max_points:
        step = int(math.ceil(len(s) / max_points))
        s = pd.concat([s.iloc[::step], s.iloc[-1:]])
        s = s[~s.index.duplicated(keep="last")]
    return [[int(pd.Timestamp(t).timestamp() * 1000), round(float(v), 2)] for t, v in s.items()]


def _n(x, d=2) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    return f"{x:.{d}f}" if math.isfinite(x) else "–"
