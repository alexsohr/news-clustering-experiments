#!/usr/bin/env python
"""Build presentation/exec_pipeline_overview.html — executive one-pager v2.
Full-corpus 3D story map (vibrant per-story colors, dark scene, headline hovers),
detailed mouseover tooltips on every box, named algorithms/models in the gate card.
Fully offline (plotly.js inlined)."""
import json
from pathlib import Path

OUT = Path("presentation/exec_pipeline_overview.html")
data = json.loads(Path("presentation/cluster_map_data.json").read_text())

VIVID = ["#ff5e57", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#06b6d4", "#f43f5e",
         "#84cc16", "#fb923c", "#8b5cf6", "#14b8a6", "#eab308", "#ec4899", "#60a5fa",
         "#4ade80", "#fbbf24", "#c084fc", "#2dd4bf", "#fb7185", "#a3e635", "#f97316",
         "#7c3aed", "#0ea5e9", "#facc15"]

multi_i  = [i for i, c in enumerate(data["ci"]) if c >= 0]
single_i = [i for i, c in enumerate(data["ci"]) if c < 0]
take = lambda k, idxs: [data[k][i] for i in idxs]

traces = [
    {   # standalone items — vivid golden-angle hues, visually secondary
        "type": "scatter3d", "mode": "markers", "name": "standalone",
        "x": take("x", single_i), "y": take("y", single_i), "z": take("z", single_i),
        "marker": {"size": 2.6, "opacity": 0.5,
                   "color": [f"hsl({(k * 137.508) % 360:.0f},78%,62%)" for k in range(len(single_i))]},
        "hovertext": [f"{data['title'][i]}<br><i>standalone item</i>" for i in single_i],
        "hoverinfo": "text",
    },
    {   # clustered items — one vibrant color per story
        "type": "scatter3d", "mode": "markers", "name": "stories",
        "x": take("x", multi_i), "y": take("y", multi_i), "z": take("z", multi_i),
        "marker": {"size": 4.6, "opacity": 0.95,
                   "color": [VIVID[data["ci"][i] % len(VIVID)] for i in multi_i],
                   "line": {"width": 0.5, "color": "#0c1322"}},
        "hovertext": [f"{data['title'][i]}<br><b>story cluster · {data['n'][i]} items</b>" for i in multi_i],
        "hoverinfo": "text",
    },
]
AX = {"visible": False, "showbackground": False}
layout = {"autosize": True, "showlegend": False, "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
          "paper_bgcolor": "rgba(0,0,0,0)",
          "scene": {"bgcolor": "#0c1322", "xaxis": AX, "yaxis": AX, "zaxis": AX,
                    "camera": {"eye": {"x": 1.45, "y": 1.35, "z": 0.5}}},
          "hoverlabel": {"bgcolor": "#16243a", "bordercolor": "#3b82f6",
                         "font": {"color": "#ffffff", "size": 12}}}

import plotly  # noqa: E402
plotly_js = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text()

p6 = json.loads(sorted(Path("artifacts/v3/experiments").glob("*P6_final-ship*.json"))[-1].read_text())
n_stories = p6["pipeline_counts"]["final"]["stories"]

STAGES = [
    ("Client portfolio", "the companies we track", "20 clients (benchmark)",
     "Production starts from the firm's tracked-client list — each client gets its own story feed. "
     "The validation benchmark used 20 major names (JPMorgan, Apple, Goldman Sachs, …) over 2012–13. "
     "Adding a client is configuration, not retraining."),
    ("News search", "per-client search — items arrive company-tagged", "10,000 items / 2 yrs",
     "Per-client queries against our news providers; every result arrives already tagged with the client "
     "it concerns. Benchmark proxy: two years of Bloomberg + Reuters coverage, sampled to 10,000 tagged items."),
    ("Deduplicate", "URL canonicalization + MinHash fingerprints", "−1,265 wire copies",
     "Two layers. Exact: URLs are canonicalized (tracking parameters stripped, fragments dropped) and hashed. "
     "Near-duplicate: every item gets a 128-permutation MinHash fingerprint; locality-sensitive hashing finds "
     "re-published wire copies at ≥85% text overlap. One canonical copy is kept; duplicates are mapped to it, never lost."),
    ("Curate", "filings & templated PR noise set aside", "−864 non-editorial",
     "A template detector routes non-editorial wire out of clustering: regulatory filings (Form 8.3, NAV notices), "
     "“X to present at conference” press releases. They stay searchable but can't pollute stories — "
     "this guard removed an entire class of false merges."),
    ("Embed", "title + lede → 1024-dim semantic vector", "7,871 vectors",
     "Each item's title + opening text becomes a 1,024-dimension vector via OpenAI text-embedding-3-large. "
     "Distance between vectors ≈ difference in meaning — the foundation for clustering by event rather than keyword. "
     "Vectors are content-hash cached: each item is embedded exactly once (~$0.00002/item)."),
    ("Cluster", "three-gate assignment with an AI judge", f"{n_stories:,} stories",
     "Items stream chronologically. A calibrated fusion scorer (logistic regression over embedding similarity, "
     "MinHash overlap, wording and timing) routes each item: confident → join the story; uncertain → GPT judge "
     "with an event-level rubric; otherwise → new story. An HDBSCAN density sweep then proposes groups among "
     "the leftovers — every group is judge-verified. Details in the panel below."),
    ("Enrich", "LLM writes title, summary, topic, entities", "structured metadata",
     "GPT-4.1 generates each story's metadata through schema-enforced output: a headline (≤80 chars), "
     "a two-sentence factual summary, a topic tag, and the key entities — clean JSON for every downstream consumer."),
]
KPIS = [
    ("92%", "merge precision", "green",
     "Of all item pairs the system merged into one story, 91.7% were verified as genuinely the same event "
     "(543-pair human-validated benchmark). Precision is the metric clients feel — one wrong merge puts "
     "unrelated news in their feed."),
    ("0.87", "pair F1 · ship bar 0.85", "",
     "The balanced precision/recall score on the frozen benchmark. The pre-agreed ship bar was 0.85; "
     "the shipped configuration scores 0.870, up from 0.739 at the start of this iteration."),
    ("0", "long-range false merges", "",
     "A separate 199-pair guard set of look-alike items published weeks apart — the classic trap of recurring, "
     "templated news. The shipped system makes zero false merges on it."),
    ("~$0.0001", "cost per item", "",
     "Steady-state embedding + LLM cost per processed item; judge calls are spent only on uncertain pairs. "
     "At ~30,000 items/day this is roughly $31/day."),
]

hue0 = 215
stage_html = ""
for k, (name, desc, badge, tip) in enumerate(STAGES):
    hue = hue0 + k * 16
    stage_html += f"""
      <div class="stage" style="--h:{hue}">
        <div class="snum">{k + 1}</div>
        <div class="sname">{name}</div>
        <div class="sdesc">{desc}</div>
        <div class="sbadge">{badge}</div>
        <div class="tip">{tip}</div>
      </div>
      <div class="arrow">❯</div>"""
stage_html = stage_html.rsplit('<div class="arrow">', 1)[0]

kpi_html = ""
for value, label, cls, tip in KPIS:
    kpi_html += f"""
      <div class="kpi {cls}"><b>{value}</b><span>{label}</span>
        <div class="tip">{tip}</div></div>"""

PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>News Story Clustering — Solution Overview</title>
<style>
  :root { --ink:#14213b; --mut:#5a6a84; --line:#dfe6f1; --acc:#2563eb; --ok:#15803d; }
  * { box-sizing:border-box; margin:0; }
  body { font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif; color:var(--ink);
         background:linear-gradient(180deg,#f8fafd 0%,#edf2f9 100%); min-height:100vh; padding:26px 36px; }
  /* Stacking: top-level bands .head(70) > .pipe(50) > .lower(1); .lower's own context
     confines plotly's WebGL canvas below the tooltips. While a gate is hovered,
     .lower:has(.gate:hover) lifts the band to 55 so its upward tooltips clear .pipe.
     Hover z-indexes (.stage 30) and .tip's 60 are LOCAL to their band — the numbers
     are not comparable across bands. */
  .head { display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:18px; position:relative; z-index:70; }
  h1 { font-size:27px; font-weight:800; letter-spacing:-0.4px; }
  h1::after { content:""; display:block; width:64px; height:4px; border-radius:2px; margin-top:6px;
              background:linear-gradient(90deg,#2563eb,#a855f7); }
  .sub { color:var(--mut); font-size:14px; margin-top:8px; }
  .kpis { display:flex; gap:10px; }
  .kpi { position:relative; background:#fff; border:1px solid var(--line); border-left:3px solid var(--acc);
         border-radius:10px; padding:10px 18px; text-align:center;
         box-shadow:0 1px 3px rgba(20,33,59,.08); transition:transform .15s, box-shadow .15s; }
  .kpi:hover { transform:translateY(-2px); box-shadow:0 6px 18px rgba(20,33,59,.14); }
  .kpi b { display:block; font-size:22px; color:var(--acc); letter-spacing:-0.3px; }
  .kpi.green { border-left-color:var(--ok); } .kpi.green b { color:var(--ok); }
  .kpi span { font-size:11px; color:var(--mut); white-space:nowrap; }
  .pipe { display:flex; align-items:stretch; gap:7px; margin-bottom:18px; position:relative; z-index:50; }
  .stage { position:relative; flex:1; background:#fff; border:1px solid var(--line); border-radius:12px;
           padding:13px 13px 11px; box-shadow:0 1px 3px rgba(20,33,59,.08);
           transition:transform .15s, box-shadow .15s; }
  .stage::before { content:""; position:absolute; inset:0 0 auto 0; height:4px;
                   border-radius:12px 12px 0 0;
                   background:linear-gradient(90deg,hsl(var(--h),72%,52%),hsl(calc(var(--h) + 22),72%,60%)); }
  .stage:hover { transform:translateY(-3px); box-shadow:0 10px 26px rgba(20,33,59,.16); z-index:30; }
  .snum { width:23px; height:23px; border-radius:50%; color:#fff; font-size:12px; font-weight:700;
          display:flex; align-items:center; justify-content:center;
          background:hsl(var(--h),72%,50%); }
  .sname { font-weight:700; font-size:14px; margin:8px 0 3px; }
  .sdesc { font-size:11.5px; color:var(--mut); line-height:1.35; min-height:31px; }
  .sbadge { margin-top:7px; font-size:11px; font-weight:600; border-radius:6px; padding:3px 8px;
            display:inline-block; color:hsl(var(--h),72%,38%); background:hsl(var(--h),75%,95%); }
  .arrow { align-self:center; color:#b6c3d8; font-size:15px; }
  .tip { position:absolute; top:calc(100% + 10px); left:50%; transform:translate(-50%,-4px);
         width:312px; background:#16243a; color:#eef3fb; font-size:12px; font-weight:400;
         line-height:1.55; text-align:left; padding:12px 14px; border-radius:10px;
         box-shadow:0 14px 34px rgba(10,18,34,.4); opacity:0; visibility:hidden;
         transition:opacity .15s, transform .15s; pointer-events:none; z-index:60; white-space:normal; }
  .tip::before { content:""; position:absolute; top:-6px; left:50%; transform:translateX(-50%) rotate(45deg);
                 width:12px; height:12px; background:#16243a; }
  .stage:hover .tip, .kpi:hover .tip, .gate:hover .tip { opacity:1; visibility:visible; transform:translate(-50%,0); }
  .pipe .stage:first-child .tip { left:0; transform:translate(0,-4px); }
  .pipe .stage:first-child:hover .tip { transform:translate(0,0); }
  .pipe .stage:first-child .tip::before { left:36px; }
  .pipe .stage:last-child .tip { left:auto; right:0; transform:translate(0,-4px); }
  .pipe .stage:last-child:hover .tip { transform:translate(0,0); }
  .pipe .stage:last-child .tip::before { left:auto; right:36px; }
  .kpis .kpi .tip { width:280px; left:auto; right:0; transform:translate(0,-4px); }
  .kpis .kpi:hover .tip { transform:translate(0,0); }
  .kpis .kpi .tip::before { left:auto; right:30px; }
  .lower { display:flex; gap:16px; position:relative; z-index:1; }
  .lower:has(.gate:hover) { z-index:55; }   /* gate tips open upward over .pipe */
  .card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:16px 18px;
          box-shadow:0 1px 3px rgba(20,33,59,.08); }
  .gates { flex:0 0 37%; }
  .card h2 { font-size:15px; font-weight:700; margin-bottom:8px; }
  .gate { position:relative; display:flex; gap:10px; align-items:flex-start; padding:10px 0;
          border-top:1px solid var(--line); }
  .gate .tip { width:300px; top:auto; bottom:calc(100% + 10px); }
  .gate .tip::before { top:auto; bottom:-6px; }
  .gate:hover .tip { transform:translate(-50%,0); }
  .glabel { flex:0 0 132px; font-size:11.5px; font-weight:700; padding:4px 8px; border-radius:6px; text-align:center; }
  .g1 { background:#e6f6ec; color:var(--ok); } .g2 { background:#fff3df; color:#9a6700; }
  .g3 { background:#eef1f7; color:var(--mut); }
  .gate p { font-size:12.3px; line-height:1.5; }
  .gate p small { color:var(--mut); }
  .mono { font-family:ui-monospace,Menlo,monospace; font-size:11.3px; background:#f1f5fb;
          border-radius:4px; padding:0 4px; }
  .note { margin-top:10px; font-size:11.5px; color:var(--mut); line-height:1.5; }
  .mapcard { flex:1; display:flex; flex-direction:column; background:#0d1526; border-color:#22304a; }
  .mapcard h2 { color:#f2f6fd; display:flex; justify-content:space-between; align-items:center; }
  .hint { font-size:10.5px; font-weight:500; color:#9fb2cf; background:#16243a;
          border:1px solid #2a3a58; border-radius:99px; padding:3px 10px; }
  #map { flex:1; min-height:430px; }
  .cap { font-size:11.5px; color:#9fb2cf; margin-top:7px; line-height:1.45; }
  .cap b { color:#dbe6f6; }
  .foot { margin-top:14px; font-size:11.5px; color:var(--mut); text-align:center; }
</style></head>
<body>
  <div class="head">
    <div>
      <h1>News Story Clustering — how items become stories</h1>
      <div class="sub">From client-tagged news search results to deduplicated, enriched, clustered story feeds
        &nbsp;·&nbsp; hover any box for detail</div>
    </div>
    <div class="kpis">__KPIS__</div>
  </div>

  <div class="pipe">__STAGES__</div>

  <div class="lower">
    <div class="card gates">
      <h2>Inside stage 6 — how an item finds its story</h2>
      <div class="gate"><div class="glabel g1">CONFIDENT → JOIN</div>
        <p>A <span class="mono">logistic-regression</span> fusion scorer combines
           <span class="mono">text-embedding-3-large</span> cosine similarity with
           <span class="mono">MinHash</span> overlap, wording, length and timing signals.
           Calibrated so auto-joins are ≥99% precise.<br>
           <small>handles clear matches at zero LLM cost</small>
           <div class="tip">The scorer was trained and calibrated on the human-validated pair benchmark with
             grouped cross-validation (AUC 0.91 vs 0.75 for embedding similarity alone). Its decisive trick:
             given high embedding similarity, high <i>literal</i> wording overlap is evidence of templated
             look-alikes, not the same event — invisible to similarity-only systems.</div></p></div>
      <div class="gate"><div class="glabel g2">UNCERTAIN → JUDGE</div>
        <p>An AI judge — <span class="mono">GPT-4.1-mini</span>, escalating to
           <span class="mono">GPT-5.2</span> — decides borderline pairs with an event-level rubric:
           <i>same underlying event, not merely the same saga</i>.<br>
           <small>96.5% agreement with human-validated labels</small>
           <div class="tip">The rubric mirrors the one used to build the ground-truth labels: same primary
             entities + same underlying event = SAME; saga follow-ups reporting distinct developments = DIFFERENT.
             Every verdict is cached and auditable — same pair, same answer, forever.</div></p></div>
      <div class="gate"><div class="glabel g3">NO MATCH → NEW</div>
        <p>The item seeds a new story. <span class="mono">HDBSCAN</span> density clustering later proposes
           groups among the leftovers — each group must pass the same GPT judge before becoming a story.<br>
           <small>no merge without confidence or explicit approval</small>
           <div class="tip">HDBSCAN finds dense neighborhoods in embedding space without a preset cluster count.
             Its proposals are treated as suggestions, not decisions: a judge verifies group coherence
             (medoid-vs-outlier checks) and can peel members or reject the group outright.</div></p></div>
      <div class="note"><b>Why quality holds:</b> no merge happens without high model confidence or explicit
        judge approval — false merges are the costliest error for client-facing feeds.</div>
    </div>
    <div class="card mapcard">
      <h2>The result — every item from the 2-year benchmark, in semantic space
        <span class="hint">drag to rotate · scroll to zoom · hover any dot for its headline</span></h2>
      <div id="map"></div>
      <div class="cap">All <b>7,871 items</b> positioned by meaning. <b>Large bright dots</b> = items grouped into
        one of <b>439 detected stories</b> (one color per story); <b>small dots</b> = standalone items that
        correctly matched nothing. Tight same-color groups are the system doing its job.</div>
    </div>
  </div>

  <div class="foot">Validated on 2 years of Bloomberg &amp; Reuters coverage · 10,000 items · 20 tracked clients ·
    every uncertain merge reviewed by an AI judge · full audit trail per decision</div>

<script>__PLOTLY_JS__</script>
<script>
  Plotly.newPlot("map", __TRACES__, __LAYOUT__, {responsive:true, displayModeBar:false});
</script>
</body></html>"""

OUT.parent.mkdir(exist_ok=True)
page = (PAGE.replace("__STAGES__", stage_html)
            .replace("__KPIS__", kpi_html)
            .replace("__PLOTLY_JS__", plotly_js)
            .replace("__TRACES__", json.dumps(traces))
            .replace("__LAYOUT__", json.dumps(layout)))
OUT.write_text(page)
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB) | "
      f"{len(multi_i):,} clustered + {len(single_i):,} standalone points")
