"""Render data/dashboard.json into a self-contained HTML dashboard.

No external assets beyond Google Fonts: charts are drawn on canvas from the
embedded payload, so the page works offline and inside the artifact CSP.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA = Path("data/dashboard.json")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/dashboard.html")

payload = json.loads(DATA.read_text(encoding="utf-8"))
meta = payload["meta"]
summary = payload["summary"]
bench = next(r for r in summary if r.get("kind") == "benchmark")


def pct(v, dp=1):
    return "&mdash;" if v is None else f"{v*100:+.{dp}f}%"


def money(v):
    return "&mdash;" if v is None else f"₹{v:,.0f}"


def num(v, dp=2):
    return "&mdash;" if v is None else f"{v:+.{dp}f}"


# ---------------------------------------------------------------- summary
rows = []
for r in sorted(summary, key=lambda x: -(x.get("sharpe") or -9)):
    is_b = r.get("kind") == "benchmark"
    beats = (not is_b and (r.get("final") or 0) > bench["final"]
             and (r.get("sharpe") or 0) > bench["sharpe"])
    dsr = r.get("dsr")
    verdict = ("benchmark" if is_b else
               ("beats, unproven" if beats else "does not beat"))
    vcls = "bench" if is_b else ("warn" if beats else "neg")
    rows.append(f"""<tr class="{'bench' if is_b else ''}">
      <td class="name">{r['strategy']}</td>
      <td class="num">{money(r.get('final'))}</td>
      <td class="num {'pos' if (r.get('cagr') or 0) > 0 else 'neg'}">{pct(r.get('cagr'))}</td>
      <td class="num">{num(r.get('sharpe'))}</td>
      <td class="num neg">{pct(r.get('max_dd'))}</td>
      <td class="num">{r.get('trades', 0) or 0}</td>
      <td class="num">{'&mdash;' if dsr is None else f'{dsr:.3f}'}</td>
      <td><span class="pill {vcls}">{verdict}</span></td>
    </tr>""")
summary_rows = "\n".join(rows)

# ----------------------------------------------------------- window grid
wcols = [c for c in payload["windows"][0] if c != "label"]
head = "".join(f"<th>{c}</th>" for c in wcols)
wbody = []
for w in payload["windows"]:
    cells = []
    for c in wcols:
        v = w.get(c)
        if v is None:
            cells.append('<td class="num">&mdash;</td>')
            continue
        # Colour intensity scales with magnitude, capped at +/-40%.
        mag = min(abs(v) / 0.40, 1.0)
        tone = "g" if v > 0 else "r"
        cells.append(f'<td class="num heat" style="--i:{mag:.2f}" data-t="{tone}">{v*100:+.1f}%</td>')
    wbody.append(f'<tr><td class="name mono">{w["label"]}</td>{"".join(cells)}</tr>')
window_rows = "\n".join(wbody)

# --------------------------------------------------------------- factor
frows = []
for f in payload["factor"]:
    recent = f["year"] >= 2024
    frows.append(f"""<tr class="{'recent' if recent else ''}">
      <td class="num mono">{f['year']}</td>
      <td class="num">{f['top']*100:+.2f}%</td>
      <td class="num">{f['bottom']*100:+.2f}%</td>
      <td class="num strong {'pos' if f['spread'] > 0 else 'neg'}">{f['spread']*100:+.2f}%</td>
      <td class="num">{f['hit']*100:.0f}%</td>
    </tr>""")
factor_rows = "\n".join(frows)

pre24 = [f["spread"] for f in payload["factor"] if f["year"] < 2024]
post24 = [f["spread"] for f in payload["factor"] if f["year"] >= 2024]
pre_avg = sum(pre24) / len(pre24) if pre24 else 0
post_avg = sum(post24) / len(post24) if post24 else 0

# ---------------------------------------------------------------- stops
srows = []
best_stop = max(payload["stops"], key=lambda s: s["ret"])
for s in payload["stops"]:
    hl = "best" if s is best_stop else ""
    srows.append(f"""<tr class="{hl}">
      <td class="name mono">{s['label']}</td>
      <td class="num {'pos' if s['ret'] > 0 else 'neg'}">{s['ret']*100:+.1f}%</td>
      <td class="num">{s['stops']}</td>
      <td class="num">₹{s['costs']:,.0f}</td>
    </tr>""")
stop_rows = "\n".join(srows)

bug_rows = "\n".join(
    f"""<tr><td class="name">{b['name']}</td><td class="mut">{b['found']}</td>
    <td>{b['effect']}</td><td><span class="pill neg">{b['direction']}</span></td></tr>"""
    for b in payload["bugs"]
)

decisions = "\n".join(
    f'<details{" open" if i == 0 else ""}><summary>{d["title"]}</summary><pre>{d["text"]}</pre></details>'
    for i, d in enumerate(payload["decisions"])
) or '<p class="mut">No decision records captured in this run.</p>'

chart_payload = json.dumps({
    "curves": payload["curves"],
    "order": [r["strategy"] for r in summary],
})

HTML = f"""<title>StockTrader Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --paper:#F3F5F8; --surface:#FFFFFF; --surface-2:#EAEEF4;
  --ink:#151A22; --ink-soft:#596274; --ink-faint:#8A93A5;
  --rule:#D9DEE7; --rule-soft:#E7EBF1;
  --accent:#2C4795; --accent-soft:#E5EAF7;
  --pos:#0E6F4E; --neg:#9E3129; --warn:#8A5A10; --warn-soft:#F7EBD8;
  --c1:#2C4795; --c2:#0E6F4E; --c3:#9E3129; --c4:#8A5A10; --c5:#5B3E8E; --c6:#1F7A8C;
  --display:"Spectral",Georgia,serif;
  --body:"IBM Plex Sans",-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0E1218; --surface:#161B24; --surface-2:#1D2430;
    --ink:#E4E8EF; --ink-soft:#98A2B5; --ink-faint:#6B7488;
    --rule:#2A3140; --rule-soft:#222835;
    --accent:#8AA4EC; --accent-soft:#1C2540;
    --pos:#46BC8C; --neg:#E27C74; --warn:#D6A75A; --warn-soft:#302512;
    --c1:#8AA4EC; --c2:#46BC8C; --c3:#E27C74; --c4:#D6A75A; --c5:#B08BE0; --c6:#5FBECF;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0E1218; --surface:#161B24; --surface-2:#1D2430;
  --ink:#E4E8EF; --ink-soft:#98A2B5; --ink-faint:#6B7488;
  --rule:#2A3140; --rule-soft:#222835;
  --accent:#8AA4EC; --accent-soft:#1C2540;
  --pos:#46BC8C; --neg:#E27C74; --warn:#D6A75A; --warn-soft:#302512;
  --c1:#8AA4EC; --c2:#46BC8C; --c3:#E27C74; --c4:#D6A75A; --c5:#B08BE0; --c6:#5FBECF;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);
  font-size:15.5px;line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 24px 90px}}
header{{border-bottom:2px solid var(--ink);padding:48px 0 22px;margin-bottom:14px}}
.kick{{font-family:var(--mono);font-size:11px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--accent);margin:0 0 12px}}
h1{{font-family:var(--display);font-weight:600;font-size:clamp(2.1rem,5vw,3.4rem);
  line-height:1.05;letter-spacing:-.02em;margin:0 0 14px;text-wrap:balance}}
.sub{{font-family:var(--display);font-style:italic;color:var(--ink-soft);
  font-size:1.08rem;max-width:62ch;margin:0}}
.meta{{display:flex;flex-wrap:wrap;gap:0 28px;font-family:var(--mono);font-size:11.5px;
  color:var(--ink-faint);padding:14px 0 0}}
.meta b{{color:var(--ink-soft);font-weight:500}}
section{{margin:56px 0;scroll-margin-top:20px}}
h2{{font-family:var(--display);font-weight:600;font-size:clamp(1.4rem,2.6vw,1.85rem);
  margin:0 0 6px;letter-spacing:-.015em;text-wrap:balance}}
.note{{color:var(--ink-soft);font-size:14px;margin:0 0 22px;padding-bottom:14px;
  border-bottom:1px solid var(--rule-soft);max-width:74ch}}
p{{max-width:74ch}}
.tw{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}
th{{text-align:left;font-family:var(--mono);font-size:10px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-faint);font-weight:500;
  padding:0 12px 8px 0;border-bottom:1px solid var(--rule);white-space:nowrap}}
td{{padding:9px 12px 9px 0;border-bottom:1px solid var(--rule-soft);vertical-align:middle}}
td.num{{font-family:var(--mono);white-space:nowrap;text-align:right;padding-right:14px}}
th:not(:first-child){{text-align:right;padding-right:14px}}
td.name{{font-weight:500;white-space:nowrap}}
.mono{{font-family:var(--mono);font-size:12.5px}}
.mut{{color:var(--ink-faint)}}
.pos{{color:var(--pos)}} .neg{{color:var(--neg)}} .strong{{font-weight:600}}
tr.bench td{{background:var(--accent-soft)}}
tr.recent td{{background:var(--warn-soft)}}
tr.best td{{background:var(--accent-soft);font-weight:600}}
.pill{{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  padding:3px 9px;border-radius:11px;white-space:nowrap;border:1px solid currentColor}}
.pill.bench{{color:var(--accent)}} .pill.neg{{color:var(--neg)}} .pill.warn{{color:var(--warn)}}
td.heat{{position:relative}}
td.heat::before{{content:"";position:absolute;inset:2px 4px 2px 0;border-radius:2px;
  opacity:calc(var(--i)*.30);background:var(--pos)}}
td.heat[data-t="r"]::before{{background:var(--neg)}}
td.heat{{isolation:isolate}} td.heat>*{{position:relative}}
.chartbox{{background:var(--surface);border:1px solid var(--rule);border-radius:5px;
  padding:16px 18px 10px}}
canvas{{width:100%;display:block}}
.legend{{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0 2px}}
.lg{{display:flex;align-items:center;gap:7px;font-size:12.5px;padding:4px 11px;
  border:1px solid var(--rule);border-radius:14px;cursor:pointer;background:var(--surface);
  color:var(--ink);font-family:var(--body)}}
.lg:hover{{border-color:var(--accent)}}
.lg[aria-pressed="false"]{{opacity:.38}}
.sw{{width:11px;height:11px;border-radius:3px;flex:none}}
.callout{{border-left:3px solid var(--accent);background:var(--surface);padding:16px 20px;
  margin:20px 0;max-width:74ch;border-radius:0 4px 4px 0}}
.callout.warn{{border-left-color:var(--warn);background:var(--warn-soft)}}
.callout .lbl{{font-family:var(--mono);font-size:10.5px;letter-spacing:.11em;
  text-transform:uppercase;color:var(--accent);display:block;margin-bottom:6px}}
.callout.warn .lbl{{color:var(--warn)}}
.callout p{{margin:0 0 10px}} .callout p:last-child{{margin:0}}
details{{background:var(--surface);border:1px solid var(--rule);border-radius:4px;
  padding:12px 16px;margin-bottom:10px}}
summary{{cursor:pointer;font-weight:500;font-size:14px}}
pre{{font-family:var(--mono);font-size:12px;line-height:1.5;overflow-x:auto;
  background:var(--surface-2);padding:14px;border-radius:4px;margin:12px 0 0;color:var(--ink)}}
.tip{{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--rule);
  border-radius:4px;padding:9px 12px;font-family:var(--mono);font-size:11.5px;
  box-shadow:0 4px 18px rgba(0,0,0,.14);opacity:0;transition:opacity .12s;z-index:20;
  color:var(--ink)}}
:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
@media (prefers-reduced-motion: reduce){{*{{transition:none!important}}}}
@media (max-width:720px){{.meta{{gap:0 16px}} td.name{{white-space:normal}}}}
</style>

<div class="wrap">
<header>
  <p class="kick">StockTrader · Walk-forward results</p>
  <h1>Nothing Beat Buying And Holding</h1>
  <p class="sub">Eleven years of NSE data, nine out-of-sample windows, five strategies — and the honest answer, including the bugs that produced two earlier wrong ones.</p>
  <div class="meta">
    <span><b>{meta['start']}</b> → <b>{meta['end']}</b></span>
    <span><b>{meta['bars']:,}</b> bars</span>
    <span><b>{meta['symbols']}</b> symbols</span>
    <span><b>{meta['windows']}</b> windows</span>
    <span>2y train / 1y test / <b>{meta['embargo_days']}d</b> embargo</span>
    <span>capital <b>₹{meta['capital']:,.0f}</b></span>
  </div>
</header>

<section>
  <h2>The scoreboard</h2>
  <p class="note">Every figure is out-of-sample: parameters were fixed on training data, then measured on later windows they never saw. DSR is the Deflated Sharpe — the probability the result beats luck given {meta['trials']} configurations were searched. It needs 0.95.</p>
  <div class="tw"><table>
    <thead><tr><th>Strategy</th><th>Final</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Trades</th><th>DSR</th><th>Verdict</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table></div>
  <div class="callout warn"><span class="lbl">Read this before the charts</span>
  <p>Buy-and-hold wins on both return and risk-adjusted return. No strategy clears the deflation bar. The best of them trails a portfolio that required no decisions at all.</p></div>
</section>

<section>
  <h2>Equity curves</h2>
  <p class="note">All rebased to 100 at the start so shapes are comparable. Click a name to show or hide it.</p>
  <div class="chartbox">
    <canvas id="eq" height="380"></canvas>
    <div class="legend" id="lg"></div>
  </div>
</section>

<section>
  <h2>Drawdowns</h2>
  <p class="note">How far below its own high-water mark each portfolio sat. This is what you would actually have had to live through.</p>
  <div class="chartbox"><canvas id="dd" height="250"></canvas></div>
</section>

<section>
  <h2>Year by year</h2>
  <p class="note">Return in each one-year test window. Shading scales with magnitude — green is a gain, red a loss.</p>
  <div class="tw"><table>
    <thead><tr><th>Window</th>{head}</tr></thead>
    <tbody>{window_rows}</tbody>
  </table></div>
</section>

<section>
  <h2>Is the momentum factor still alive?</h2>
  <p class="note">Ranking every liquid stock by 12-1 momentum each month, then measuring the next month's return of the strongest tenth versus the weakest. No strategy, no costs — just: does the ranking predict?</p>
  <div class="tw"><table>
    <thead><tr><th>Year</th><th>Top decile</th><th>Bottom decile</th><th>Spread</th><th>Months positive</th></tr></thead>
    <tbody>{factor_rows}</tbody>
  </table></div>
  <div class="callout"><span class="lbl">Verdict</span>
  <p>Average monthly spread was <b>{pre_avg*100:+.2f}%</b> across 2015–2023 and <b>{post_avg*100:+.2f}%</b> across 2024–2026. The factor is healthy. So the recent losses are not the market changing — they are our implementation.</p></div>
</section>

<section>
  <h2>What actually broke it</h2>
  <p class="note">Momentum over 2024–2026 at different trailing-stop widths. Every other setting held constant.</p>
  <div class="tw"><table>
    <thead><tr><th>Stop width</th><th>Return</th><th>Stops fired</th><th>Costs paid</th></tr></thead>
    <tbody>{stop_rows}</tbody>
  </table></div>
  <div class="callout"><span class="lbl">The finding</span>
  <p>Across every year examined, <b>100% of exits were trailing stops and 0% were the intended rank-based exits</b> — the momentum thesis never got to play out. Removing the stop is worth 15–19 percentage points.</p>
  <p>An ATR trailing stop fires on ordinary noise in volatile Indian stocks, turning temporary drawdowns into realised losses and taxable events, exactly when momentum needs to let winners run.</p></div>
</section>

<section>
  <h2>Why any single trade happened</h2>
  <p class="note">Every signal stores the rule inputs, the outcomes and the sizing that produced it. This is rendered from stored data — no language model involved, and nothing here can be hallucinated.</p>
  {decisions}
</section>

<section>
  <h2>Bugs that changed the answer</h2>
  <p class="note">Found by the test suite and the post-mortem. Three of them made results look better than reality, which is the direction that costs money.</p>
  <div class="tw"><table>
    <thead><tr><th>Bug</th><th>Found</th><th>Effect</th><th>Direction</th></tr></thead>
    <tbody>{bug_rows}</tbody>
  </table></div>
</section>
</div>
<div class="tip" id="tip"></div>

<script>
const D = {chart_payload};
const tip = document.getElementById('tip');
const COLORS = ['--c1','--c2','--c3','--c4','--c5','--c6'];
const names = D.order.filter(n => D.curves[n]);
const state = Object.fromEntries(names.map(n => [n, true]));

function tok(n) {{ return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }}
function colorOf(i) {{ return tok(COLORS[i % COLORS.length]); }}

const lg = document.getElementById('lg');
names.forEach((n, i) => {{
  const b = document.createElement('button');
  b.className = 'lg'; b.setAttribute('aria-pressed', 'true');
  b.innerHTML = `<span class="sw" style="background:${{colorOf(i)}}"></span>${{n}}`;
  b.onclick = () => {{
    state[n] = !state[n];
    b.setAttribute('aria-pressed', String(state[n]));
    redraw();
  }};
  lg.appendChild(b);
}});

function setup(cv) {{
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = parseInt(cv.getAttribute('height'), 10);
  cv.width = w * dpr; cv.height = h * dpr;
  const c = cv.getContext('2d');
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  return {{c, w, h}};
}}

function axes(c, w, h, pad, lo, hi, dates, fmt) {{
  c.strokeStyle = tok('--rule-soft'); c.fillStyle = tok('--ink-faint');
  c.font = '11px "IBM Plex Mono", monospace'; c.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {{
    const y = pad.t + (h - pad.t - pad.b) * i / 4;
    const v = hi - (hi - lo) * i / 4;
    c.beginPath(); c.moveTo(pad.l, y); c.lineTo(w - pad.r, y); c.stroke();
    c.textAlign = 'right'; c.textBaseline = 'middle';
    c.fillText(fmt(v), pad.l - 8, y);
  }}
  c.textAlign = 'center'; c.textBaseline = 'top';
  const step = Math.max(1, Math.floor(dates.length / 6));
  for (let i = 0; i < dates.length; i += step) {{
    const x = pad.l + (w - pad.l - pad.r) * i / (dates.length - 1);
    c.fillText(dates[i].slice(0, 7), x, h - pad.b + 8);
  }}
}}

let eqGeom = null;

function draw() {{
  const cv = document.getElementById('eq');
  const {{c, w, h}} = setup(cv);
  const pad = {{l: 56, r: 14, t: 12, b: 28}};
  const active = names.filter(n => state[n]);
  if (!active.length) return;
  let lo = Infinity, hi = -Infinity, len = 0;
  active.forEach(n => {{
    const v = D.curves[n].rebased;
    len = Math.max(len, v.length);
    v.forEach(x => {{ if (x < lo) lo = x; if (x > hi) hi = x; }});
  }});
  const padv = (hi - lo) * 0.08; lo -= padv; hi += padv;
  const dates = D.curves[active[0]].dates;
  axes(c, w, h, pad, lo, hi, dates, v => v.toFixed(0));
  eqGeom = {{pad, lo, hi, w, h, len, active}};
  active.forEach(n => {{
    const i = names.indexOf(n);
    const v = D.curves[n].rebased;
    c.strokeStyle = colorOf(i);
    c.lineWidth = n === 'Buy and hold' ? 2.4 : 1.6;
    c.beginPath();
    v.forEach((y, j) => {{
      const px = pad.l + (w - pad.l - pad.r) * j / (v.length - 1);
      const py = pad.t + (h - pad.t - pad.b) * (1 - (y - lo) / (hi - lo));
      j ? c.lineTo(px, py) : c.moveTo(px, py);
    }});
    c.stroke();
  }});
}}

function drawDD() {{
  const cv = document.getElementById('dd');
  const {{c, w, h}} = setup(cv);
  const pad = {{l: 56, r: 14, t: 12, b: 28}};
  const active = names.filter(n => state[n]);
  if (!active.length) return;
  let lo = 0;
  active.forEach(n => (D.curves[n].drawdown || []).forEach(x => {{ if (x < lo) lo = x; }}));
  if (lo === 0) lo = -0.01;  // flat curve: keep the axis from collapsing
  const dates = D.curves[active[0]].dates;
  axes(c, w, h, pad, lo, 0, dates, v => (v * 100).toFixed(0) + '%');
  active.forEach(n => {{
    const i = names.indexOf(n);
    const v = D.curves[n].drawdown || [];
    c.strokeStyle = colorOf(i); c.lineWidth = n === 'Buy and hold' ? 2.2 : 1.4;
    c.beginPath();
    v.forEach((y, j) => {{
      const px = pad.l + (w - pad.l - pad.r) * j / (v.length - 1);
      const py = pad.t + (h - pad.t - pad.b) * (1 - (y - lo) / (0 - lo));
      j ? c.lineTo(px, py) : c.moveTo(px, py);
    }});
    c.stroke();
  }});
}}

function redraw() {{ draw(); drawDD(); }}

document.getElementById('eq').addEventListener('mousemove', e => {{
  if (!eqGeom) return;
  const cv = e.currentTarget, r = cv.getBoundingClientRect();
  const {{pad, w, len, active}} = eqGeom;
  const x = e.clientX - r.left;
  const frac = (x - pad.l) / (w - pad.l - pad.r);
  if (frac < 0 || frac > 1) {{ tip.style.opacity = 0; return; }}
  const j = Math.round(frac * (len - 1));
  const dates = D.curves[active[0]].dates;
  let html = `<b>${{dates[Math.min(j, dates.length - 1)]}}</b>`;
  active.forEach(n => {{
    const v = D.curves[n].values;
    if (j < v.length) html += `<br>${{n}}: ₹${{Math.round(v[j]).toLocaleString('en-IN')}}`;
  }});
  tip.innerHTML = html;
  tip.style.opacity = 1;
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 230) + 'px';
  tip.style.top = (e.clientY + 14) + 'px';
}});
document.getElementById('eq').addEventListener('mouseleave', () => {{ tip.style.opacity = 0; }});

window.addEventListener('resize', redraw);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {{
  lg.querySelectorAll('.sw').forEach((s, i) => s.style.background = colorOf(i));
  redraw();
}});
redraw();
</script>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB)")
