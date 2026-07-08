"""Render a batch's result records as a self-contained HTML dashboard.

A static, offline HTML file (no server, no external requests) — the visual face
of an evaluation run. It embeds the records as a JSON blob and renders them with
inline CSS/JS: a summary band (resolve rate + outcome breakdown), a per-repo
table, and expandable per-instance rows showing the agent's plan, tool-call
trace, per-test pass/fail, and the diff it produced.

``render_dashboard`` is pure (records -> HTML string); the CLI writes it to disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from swe_agent.eval.analysis import OUTCOME_ORDER, Outcome, analyze, classify

# Label + semantic-color token for each outcome (colors defined in the CSS).
_OUTCOME_META: dict[str, dict[str, str]] = {
    Outcome.RESOLVED.value: {"label": "Resolved", "tone": "ok"},
    Outcome.REGRESSION.value: {"label": "Regression", "tone": "warn"},
    Outcome.INCOMPLETE_FIX.value: {"label": "Incomplete fix", "tone": "info"},
    Outcome.NO_PATCH.value: {"label": "No patch", "tone": "muted"},
    Outcome.PATCH_FAILED.value: {"label": "Patch didn't apply", "tone": "orange"},
    Outcome.EVAL_INCOMPLETE.value: {"label": "Eval incomplete", "tone": "violet"},
    Outcome.LLM_ERROR.value: {"label": "LLM error", "tone": "danger"},
    Outcome.RUN_ERROR.value: {"label": "Run error", "tone": "danger2"},
}


def build_data(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the JSON payload the dashboard's JS renders from."""
    report = analyze(records)
    enriched = [{**r, "_outcome": classify(r).value} for r in records]
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": {
            "total": report.total,
            "resolved": report.resolved,
            "resolve_rate": report.resolve_rate,
            "errored": (
                report.outcome_counts.get(Outcome.LLM_ERROR.value, 0)
                + report.outcome_counts.get(Outcome.RUN_ERROR.value, 0)
            ),
            "outcomes": [
                {
                    "value": o.value,
                    "label": _OUTCOME_META[o.value]["label"],
                    "tone": _OUTCOME_META[o.value]["tone"],
                    "count": report.outcome_counts.get(o.value, 0),
                }
                for o in OUTCOME_ORDER
            ],
            "per_repo": [
                {"repo": repo, "total": tot, "resolved": res}
                for repo, (tot, res) in sorted(report.per_repo.items())
            ],
        },
        "meta": _OUTCOME_META,
        "records": enriched,
    }


def render_dashboard(records: list[dict[str, Any]], *, standalone: bool = True) -> str:
    """Return the dashboard HTML. ``standalone`` wraps it as a full document."""
    body = _BODY.replace("/*__DATA__*/", json.dumps(build_data(records)))
    if not standalone:
        return body
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>SWE-bench Lite — Evaluation Dashboard</title>\n"
        "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
    )


_BODY = r"""
<style>
  :root {
    --bg:#f4f6f9; --surface:#ffffff; --surface-2:#eceff4; --border:#dde2ea;
    --text:#1a2030; --muted:#5c6675; --accent:#6d5cf0; --accent-soft:rgba(109,92,240,.12);
    --ok:#1f9d4d; --warn:#b57d00; --info:#2f6fe0; --muted-s:#78828f;
    --orange:#c5560f; --violet:#7d54d8; --danger:#d0342c; --danger2:#a13b22;
    --add-bg:rgba(31,157,77,.12); --add-fg:#137a3a;
    --del-bg:rgba(208,52,44,.12); --del-fg:#b02a23;
    --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    --radius:12px; --shadow:0 1px 2px rgba(20,28,45,.05),0 6px 20px rgba(20,28,45,.06);
  }
  @media (prefers-color-scheme:dark){
    :root{
      --bg:#0d1017; --surface:#151a22; --surface-2:#1c232e; --border:#28303c;
      --text:#e6e9ef; --muted:#95a0af; --accent:#8f82ff; --accent-soft:rgba(143,130,255,.16);
      --ok:#3fb950; --warn:#d6a125; --info:#5aa0ff; --muted-s:#8b949e;
      --orange:#e07b3a; --violet:#a684f2; --danger:#f6564c; --danger2:#d3663f;
      --add-bg:rgba(63,185,80,.14); --add-fg:#57c96a;
      --del-bg:rgba(246,86,76,.14); --del-fg:#f77a72;
      --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.28);
    }
  }
  :root[data-theme="light"]{
    --bg:#f4f6f9; --surface:#ffffff; --surface-2:#eceff4; --border:#dde2ea;
    --text:#1a2030; --muted:#5c6675; --accent:#6d5cf0; --accent-soft:rgba(109,92,240,.12);
    --ok:#1f9d4d; --warn:#b57d00; --info:#2f6fe0; --muted-s:#78828f;
    --orange:#c5560f; --violet:#7d54d8; --danger:#d0342c; --danger2:#a13b22;
    --add-bg:rgba(31,157,77,.12); --add-fg:#137a3a;
    --del-bg:rgba(208,52,44,.12); --del-fg:#b02a23;
    --shadow:0 1px 2px rgba(20,28,45,.05),0 6px 20px rgba(20,28,45,.06);
  }
  :root[data-theme="dark"]{
    --bg:#0d1017; --surface:#151a22; --surface-2:#1c232e; --border:#28303c;
    --text:#e6e9ef; --muted:#95a0af; --accent:#8f82ff; --accent-soft:rgba(143,130,255,.16);
    --ok:#3fb950; --warn:#d6a125; --info:#5aa0ff; --muted-s:#8b949e;
    --orange:#e07b3a; --violet:#a684f2; --danger:#f6564c; --danger2:#d3663f;
    --add-bg:rgba(63,185,80,.14); --add-fg:#57c96a;
    --del-bg:rgba(246,86,76,.14); --del-fg:#f77a72;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 8px 24px rgba(0,0,0,.28);
  }
  .tone-ok{--tone:var(--ok)} .tone-warn{--tone:var(--warn)} .tone-info{--tone:var(--info)}
  .tone-muted{--tone:var(--muted-s)} .tone-orange{--tone:var(--orange)}
  .tone-violet{--tone:var(--violet)} .tone-danger{--tone:var(--danger)}
  .tone-danger2{--tone:var(--danger2)}

  *{box-sizing:border-box}
  body{margin:0}
  .swe{
    background:var(--bg); color:var(--text); font-family:var(--sans);
    line-height:1.5; -webkit-font-smoothing:antialiased;
    padding:32px 24px 72px; min-height:100vh;
  }
  .wrap{max-width:1000px; margin:0 auto}
  .mono{font-family:var(--mono); font-variant-numeric:tabular-nums}

  header.top{display:flex; align-items:flex-start; justify-content:space-between; gap:16px; flex-wrap:wrap; margin-bottom:28px}
  .brand{display:flex; flex-direction:column; gap:6px}
  .eyebrow{font-family:var(--mono); font-size:11px; letter-spacing:.18em; text-transform:uppercase; color:var(--accent); font-weight:600}
  h1{font-size:26px; line-height:1.15; margin:0; font-weight:680; letter-spacing:-.01em; text-wrap:balance}
  .sub{color:var(--muted); font-size:13.5px; margin:0}
  .meta-line{font-family:var(--mono); font-size:11.5px; color:var(--muted); margin-top:2px}
  .theme-btn{
    font-family:var(--mono); font-size:12px; color:var(--muted); background:var(--surface);
    border:1px solid var(--border); border-radius:999px; padding:7px 13px; cursor:pointer; white-space:nowrap;
  }
  .theme-btn:hover{color:var(--text); border-color:var(--accent)}
  .theme-btn:focus-visible{outline:2px solid var(--accent); outline-offset:2px}

  /* summary band */
  .summary{display:grid; grid-template-columns:minmax(220px,1.1fr) 2fr; gap:16px; margin-bottom:16px}
  @media (max-width:720px){.summary{grid-template-columns:1fr}}
  .card{background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); box-shadow:var(--shadow)}
  .headline{padding:22px 24px; display:flex; flex-direction:column; justify-content:center}
  .headline .rate{font-family:var(--mono); font-size:52px; font-weight:680; letter-spacing:-.03em; line-height:1; color:var(--text)}
  .headline .rate small{font-size:22px; color:var(--muted); font-weight:600}
  .headline .cap{font-size:12.5px; color:var(--muted); margin-top:8px}
  .headline .cap b{color:var(--text); font-family:var(--mono)}

  .breakdown{padding:18px 22px}
  .breakdown h2{font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:0 0 14px; font-weight:600; font-family:var(--mono)}
  .bar-row{display:grid; grid-template-columns:120px 1fr 34px; align-items:center; gap:12px; margin-bottom:9px}
  .bar-row:last-child{margin-bottom:0}
  .bar-label{font-size:12.5px; color:var(--text); display:flex; align-items:center; gap:8px; min-width:0}
  .swatch{width:9px; height:9px; border-radius:2px; background:var(--tone); flex:none}
  .bar-track{height:8px; background:var(--surface-2); border-radius:999px; overflow:hidden}
  .bar-fill{height:100%; background:var(--tone); border-radius:999px; min-width:2px; transition:width .6s cubic-bezier(.2,.7,.2,1)}
  .bar-num{font-family:var(--mono); font-size:12.5px; color:var(--muted); text-align:right}

  .tiles{display:flex; gap:16px; margin-bottom:26px; flex-wrap:wrap}
  .tile{flex:1; min-width:130px; background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); padding:14px 16px; box-shadow:var(--shadow)}
  .tile .k{font-family:var(--mono); font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted)}
  .tile .v{font-family:var(--mono); font-size:24px; font-weight:660; margin-top:4px; letter-spacing:-.02em}

  /* per-repo */
  .repos{margin-bottom:26px}
  .section-h{font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:600; font-family:var(--mono); margin:0 0 12px}
  table.repo{width:100%; border-collapse:collapse; background:var(--surface); border:1px solid var(--border); border-radius:var(--radius); overflow:hidden; box-shadow:var(--shadow)}
  table.repo th,table.repo td{text-align:left; padding:10px 16px; font-size:13px; border-bottom:1px solid var(--border)}
  table.repo tr:last-child td{border-bottom:none}
  table.repo th{font-family:var(--mono); font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); font-weight:600}
  table.repo td.num,table.repo th.num{text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums}
  .repo-name{font-family:var(--mono); font-size:12.5px}

  /* instances */
  .filters{display:flex; gap:8px; flex-wrap:wrap; margin:0 0 14px}
  .chip{font-family:var(--mono); font-size:11.5px; padding:5px 11px; border-radius:999px; border:1px solid var(--border); background:var(--surface); color:var(--muted); cursor:pointer}
  .chip[aria-pressed="true"]{color:var(--text); border-color:var(--accent); background:var(--accent-soft)}
  .chip:focus-visible{outline:2px solid var(--accent); outline-offset:2px}

  .rows{display:flex; flex-direction:column; gap:10px}
  .inst{background:var(--surface); border:1px solid var(--border); border-left:3px solid var(--tone); border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden}
  .inst-head{display:grid; grid-template-columns:1fr auto; gap:12px 16px; align-items:center; padding:14px 18px; cursor:pointer; width:100%; background:none; border:none; text-align:left; color:inherit; font:inherit}
  .inst-head:hover{background:var(--surface-2)}
  .inst-head:focus-visible{outline:2px solid var(--accent); outline-offset:-2px}
  .id-col{display:flex; flex-direction:column; gap:3px; min-width:0}
  .inst-id{font-family:var(--mono); font-size:13.5px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
  .inst-repo{font-family:var(--mono); font-size:11.5px; color:var(--muted)}
  .right-col{display:flex; align-items:center; gap:14px; flex-wrap:wrap; justify-content:flex-end}
  .badge{font-family:var(--mono); font-size:11px; font-weight:600; padding:4px 10px; border-radius:999px; color:var(--tone); background:color-mix(in srgb,var(--tone) 14%,transparent); white-space:nowrap}
  .tests{display:flex; gap:12px; font-family:var(--mono); font-size:11.5px; color:var(--muted)}
  .tests b{color:var(--text)}
  .caret{color:var(--muted); transition:transform .18s ease; flex:none}
  .inst[open] .caret{transform:rotate(90deg)}

  .inst-body{padding:2px 18px 18px; display:none; border-top:1px solid var(--border)}
  .inst[open] .inst-body{display:block}
  .grid2{display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px}
  @media (max-width:640px){.grid2{grid-template-columns:1fr}}
  .panel h4{font-family:var(--mono); font-size:11px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin:0 0 8px; font-weight:600}
  .panel p{margin:0 0 6px; font-size:13px}
  .kv{font-size:12.5px; color:var(--muted)}
  .kv b{color:var(--text)}
  .files{display:flex; flex-wrap:wrap; gap:6px; margin-top:6px}
  .file{font-family:var(--mono); font-size:11px; background:var(--surface-2); border:1px solid var(--border); border-radius:6px; padding:2px 7px}
  .tp{display:flex; gap:16px; margin-top:4px}
  .tp .g{color:var(--ok); font-family:var(--mono); font-size:12.5px}
  .tp .r{color:var(--danger); font-family:var(--mono); font-size:12.5px}
  .testlist{margin:8px 0 0; padding:0; list-style:none; display:flex; flex-direction:column; gap:3px}
  .testlist li{font-family:var(--mono); font-size:11.5px; display:flex; gap:7px; align-items:baseline}
  .testlist li .m{flex:none; width:12px}
  .pass .m{color:var(--ok)} .fail .m{color:var(--danger)} .fail{color:var(--danger)}

  .trace{margin-top:16px}
  .steps{margin:8px 0 0; padding:0; list-style:none; display:flex; flex-wrap:wrap; gap:5px}
  .steps li{font-family:var(--mono); font-size:11px; padding:3px 8px; border-radius:6px; background:var(--surface-2); border:1px solid var(--border); color:var(--text); display:flex; gap:6px; align-items:center}
  .steps li .s{color:var(--muted)}
  .steps li.err{border-color:var(--danger); color:var(--danger)}
  .steps li .dot{width:5px; height:5px; border-radius:50%; background:var(--ok)}
  .steps li.err .dot{background:var(--danger)}

  .diffwrap{margin-top:16px}
  .diff{margin:8px 0 0; overflow-x:auto; background:var(--surface-2); border:1px solid var(--border); border-radius:10px}
  .diff pre{margin:0; padding:12px 14px; font-family:var(--mono); font-size:12px; line-height:1.55; white-space:pre}
  .dl{display:block}
  .dl.add{background:var(--add-bg); color:var(--add-fg)}
  .dl.del{background:var(--del-bg); color:var(--del-fg)}
  .dl.hunk{color:var(--accent)}
  .dl.meta{color:var(--muted)}
  .empty{color:var(--muted); font-size:12.5px; font-family:var(--mono); margin-top:8px}
  .err-note{margin-top:14px; font-family:var(--mono); font-size:11.5px; color:var(--danger); background:color-mix(in srgb,var(--danger) 8%,transparent); border:1px solid color-mix(in srgb,var(--danger) 30%,transparent); border-radius:8px; padding:9px 12px; overflow-x:auto}
  .foot{margin-top:40px; text-align:center; font-family:var(--mono); font-size:11px; color:var(--muted)}
  @media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>

<div class="swe">
  <div class="wrap">
    <header class="top">
      <div class="brand">
        <span class="eyebrow">SWE-bench Lite · agent evaluation</span>
        <h1>Resolution report</h1>
        <p class="sub">Autonomous coding agent, graded against hidden verifying tests.</p>
        <p class="meta-line" id="metaLine"></p>
      </div>
      <button class="theme-btn" id="themeBtn" type="button" aria-label="Toggle color theme">◐ Theme</button>
    </header>

    <div class="summary">
      <div class="card headline">
        <div class="rate" id="rate"></div>
        <div class="cap" id="rateCap"></div>
      </div>
      <div class="card breakdown">
        <h2>Outcome breakdown</h2>
        <div id="bars"></div>
      </div>
    </div>

    <div class="tiles" id="tiles"></div>

    <div class="repos" id="reposWrap" hidden>
      <h3 class="section-h">By repository</h3>
      <div style="overflow-x:auto"><table class="repo" id="repoTable"></table></div>
    </div>

    <h3 class="section-h">Instances</h3>
    <div class="filters" id="filters"></div>
    <div class="rows" id="rows"></div>

    <p class="foot">Generated by swe-agent · open this file in any browser · no data leaves your machine</p>
  </div>
</div>

<script>
const DATA = /*__DATA__*/;
const $ = (s,r=document)=>r.querySelector(s);
const el = (t,c,txt)=>{const e=document.createElement(t); if(c)e.className=c; if(txt!=null)e.textContent=txt; return e;};
const esc = s => String(s==null?"":s).replace(/[&<>]/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[m]));
const toneClass = t => "tone-"+t;

// ---- theme toggle (viewer theme wins by default; button flips explicitly) ----
$("#themeBtn").addEventListener("click",()=>{
  const cur = document.documentElement.getAttribute("data-theme")
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark":"light");
  document.documentElement.setAttribute("data-theme", cur==="dark"?"light":"dark");
});

// ---- summary ----
const S = DATA.summary;
$("#metaLine").textContent = `${DATA.records.length} record${DATA.records.length===1?"":"s"} · generated ${DATA.generated}`;
$("#rate").innerHTML = `${(S.resolve_rate*100).toFixed(1)}<small>%</small>`;
$("#rateCap").innerHTML = `<b>${S.resolved}</b> of <b>${S.total}</b> instance${S.total===1?"":"s"} resolved`;

const maxCount = Math.max(1, ...S.outcomes.map(o=>o.count));
const bars = $("#bars");
S.outcomes.filter(o=>o.count>0).forEach(o=>{
  const row = el("div","bar-row "+toneClass(o.tone));
  const label = el("div","bar-label");
  label.appendChild(el("span","swatch"));
  label.appendChild(el("span",null,o.label));
  const track = el("div","bar-track");
  const fill = el("div","bar-fill"); fill.style.width="2px";
  requestAnimationFrame(()=>{fill.style.width = (o.count/maxCount*100)+"%";});
  track.appendChild(fill);
  const num = el("div","bar-num", o.count);
  row.append(label,track,num);
  bars.appendChild(row);
});
if(!S.outcomes.some(o=>o.count>0)) bars.appendChild(el("div","empty","No graded instances yet."));

[["Instances",S.total],["Resolved",S.resolved],["Errored (infra / quota)",S.errored]].forEach(([k,v])=>{
  const t=el("div","tile"); t.append(el("div","k",k), el("div","v",String(v))); $("#tiles").appendChild(t);
});

// ---- per-repo ----
if(S.per_repo.length>1){
  $("#reposWrap").hidden=false;
  const tb=$("#repoTable");
  tb.innerHTML="<thead><tr><th>Repository</th><th class='num'>Instances</th><th class='num'>Resolved</th><th class='num'>Rate</th></tr></thead>";
  const body=el("tbody");
  S.per_repo.forEach(r=>{
    const tr=el("tr");
    tr.innerHTML=`<td class="repo-name">${esc(r.repo)}</td><td class="num">${r.total}</td><td class="num">${r.resolved}</td><td class="num">${(r.total?r.resolved/r.total*100:0).toFixed(0)}%</td>`;
    body.appendChild(tr);
  });
  tb.appendChild(body);
}

// ---- filters ----
let active="all";
const filterWrap=$("#filters");
const present=[...new Set(DATA.records.map(r=>r._outcome))];
const mk=(val,label,count)=>{
  const c=el("button","chip"); c.type="button"; c.textContent=`${label} · ${count}`;
  c.setAttribute("aria-pressed", val===active);
  c.addEventListener("click",()=>{active=val; renderRows(); [...filterWrap.children].forEach(x=>x.setAttribute("aria-pressed", x===c));});
  return c;
};
filterWrap.appendChild(mk("all","All",DATA.records.length));
DATA.summary.outcomes.filter(o=>present.includes(o.value)).forEach(o=>filterWrap.appendChild(mk(o.value,o.label,o.count)));

// ---- instance rows ----
function diffHtml(patch){
  if(!patch || !patch.trim()) return '<p class="empty">No diff produced.</p>';
  const lines = patch.split("\n").map(l=>{
    let cls="dl";
    if(l.startsWith("+++")||l.startsWith("---")||l.startsWith("diff ")||l.startsWith("index ")) cls="dl meta";
    else if(l.startsWith("@@")) cls="dl hunk";
    else if(l.startsWith("+")) cls="dl add";
    else if(l.startsWith("-")) cls="dl del";
    return `<span class="${cls}">${esc(l)||"&nbsp;"}</span>`;
  }).join("");
  return `<div class="diff"><pre>${lines}</pre></div>`;
}
function testList(obj, kind){
  const items=(obj&&obj[kind])||[];
  if(!items.length) return "";
  return `<ul class="testlist">`+items.slice(0,12).map(t=>
    `<li class="${kind==='failed'?'fail':'pass'}"><span class="m">${kind==='failed'?'✕':'✓'}</span><span>${esc(t)}</span></li>`
  ).join("")+(items.length>12?`<li class="kv">…and ${items.length-12} more</li>`:"")+`</ul>`;
}
function record(r){
  const meta=DATA.meta[r._outcome]||{label:r._outcome,tone:"muted"};
  const f2p=r.fail_to_pass||{passed:[],failed:[]}, p2p=r.pass_to_pass||{passed:[],failed:[]};
  const wrap=el("div","inst "+toneClass(meta.tone));

  const head=el("button","inst-head"); head.type="button"; head.setAttribute("aria-expanded","false");
  const idc=el("div","id-col");
  idc.append(el("div","inst-id",r.instance_id), el("div","inst-repo",r.repo||""));
  const right=el("div","right-col");
  const tests=el("div","tests");
  tests.innerHTML=`<span>F2P <b>${f2p.passed.length}</b>/${f2p.passed.length+f2p.failed.length}</span><span>P2P <b>${p2p.passed.length}</b>/${p2p.passed.length+p2p.failed.length}</span>`;
  right.append(tests, el("span","badge",meta.label));
  const caret=el("span","caret"); caret.textContent="▸";
  right.appendChild(caret);
  head.append(idc,right);

  const body=el("div","inst-body");
  const g=el("div","grid2");
  // plan / status panel
  const pl=el("div","panel");
  pl.appendChild(el("h4",null,"Diagnosis & plan"));
  if(r.plan){
    if(r.plan.root_cause){const p=el("p"); p.innerHTML=`<span class="kv"><b>Root cause.</b> ${esc(r.plan.root_cause)}</span>`; pl.appendChild(p);}
    if(r.plan.approach){const p=el("p"); p.innerHTML=`<span class="kv"><b>Approach.</b> ${esc(r.plan.approach)}</span>`; pl.appendChild(p);}
    if(r.plan.files_to_change&&r.plan.files_to_change.length){
      const fl=el("div","files"); r.plan.files_to_change.forEach(f=>fl.appendChild(el("span","file",f))); pl.appendChild(fl);
    }
  } else { pl.appendChild(el("p","empty","No plan was recorded.")); }
  const st=el("p","kv"); st.style.marginTop="10px";
  st.innerHTML=`status <b>${esc(r.status)}</b> · stop <b>${esc(r.stop_reason)}</b> · steps <b>${r.steps}</b> · patch ${r.patch_applied?"applied":"not applied"}`;
  pl.appendChild(st);
  if(r.error){const e=el("div","err-note",r.error); pl.appendChild(e);}

  // tests panel
  const tp=el("div","panel");
  tp.appendChild(el("h4",null,"Verifying tests"));
  const t1=el("div"); t1.innerHTML=`<div class="tp"><span class="g">FAIL→PASS ✓ ${f2p.passed.length}</span><span class="r">✕ ${f2p.failed.length}</span></div>`;
  const t2=el("div"); t2.style.marginTop="8px"; t2.innerHTML=`<div class="tp"><span class="g">PASS→PASS ✓ ${p2p.passed.length}</span><span class="r">✕ ${p2p.failed.length}</span></div>`;
  tp.append(t1);
  const fl1=testList(f2p,"failed"); if(fl1) tp.insertAdjacentHTML("beforeend",fl1);
  tp.append(t2);
  const pl2=testList(p2p,"failed"); if(pl2) tp.insertAdjacentHTML("beforeend",pl2);
  g.append(pl,tp);
  body.appendChild(g);

  // tool trace
  const tr=el("div","trace"); tr.appendChild(el("h4",null,"Tool trace"));
  const tc=r.tool_calls||[];
  if(tc.length){
    const ul=el("ul","steps");
    tc.forEach(c=>{const li=el("li",c.ok?"":"err"); li.innerHTML=`<span class="dot"></span><span class="s">${c.step}</span><span>${esc(c.name)}</span>`; ul.appendChild(li);});
    tr.appendChild(ul);
  } else { tr.appendChild(el("p","empty","No tool calls.")); }
  body.appendChild(tr);

  // diff
  const dw=el("div","diffwrap"); dw.appendChild(el("h4",null,"Patch")); dw.insertAdjacentHTML("beforeend",diffHtml(r.patch));
  body.appendChild(dw);

  head.addEventListener("click",()=>{
    const open=wrap.classList.toggle("open");
    wrap.toggleAttribute("open",open);
    head.setAttribute("aria-expanded",open);
    caret.textContent = open?"▾":"▸";
  });
  wrap.append(head,body);
  return wrap;
}
function renderRows(){
  const host=$("#rows"); host.innerHTML="";
  const list=DATA.records.filter(r=>active==="all"||r._outcome===active);
  if(!list.length){host.appendChild(el("p","empty","No instances match this filter.")); return;}
  list.forEach(r=>host.appendChild(record(r)));
}
renderRows();
</script>
"""
