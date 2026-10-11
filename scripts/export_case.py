"""从成功 attempt 的原始日志导出单题离线 HTML 审计页。"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from .analyze_common_errors import read_jsonl
except ImportError:  # 直接以脚本运行时没有包上下文。
    from analyze_common_errors import read_jsonl


def pubmed_url(evidence: dict[str, Any]) -> str | None:
    """只允许 PMID 对应的 PubMed HTTPS 链接，忽略日志里的任意 URL。"""
    value = str(evidence.get("document_id") or "")
    match = re.fullmatch(r"PMID:(\d{5,10})", value)
    return f"https://pubmed.ncbi.nlm.nih.gov/{match.group(1)}/" if match else None


def public_fields(item: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """只保留页面需要的字段，避免把请求头等调用元数据打包到 HTML。"""
    return {key: item.get(key) for key in fields if key in item} if isinstance(item, dict) else {}


def build_case(run_dir: Path, question_id: str, group: str, repeat_id: int,
               attempt_id: str | None = None) -> dict[str, Any]:
    warnings: list[str] = []
    results = read_jsonl(run_dir / "results.jsonl", warnings)
    events = read_jsonl(run_dir / "events.jsonl", warnings)
    if warnings:
        raise ValueError("结果或事件日志缺失、损坏：" + "；".join(warnings))
    matched = [r for r in results if str(r.get("question_id")) == question_id
               and r.get("group") == group and r.get("repeat_id") == repeat_id
               and r.get("variant", "natural") == "natural" and r.get("status", "ok") == "ok"
               and (attempt_id is None or r.get("attempt_id") == attempt_id)]
    if len(matched) != 1:
        raise ValueError(f"预期恰好一条成功结果，实际 {len(matched)} 条；可用 --attempt-id 消歧")
    result = matched[0]
    run_id, selected_attempt = result.get("run_id"), result.get("attempt_id")
    if not run_id or not selected_attempt:
        raise ValueError("成功结果缺少 run_id 或 attempt_id")
    own = [e for e in events if e.get("run_id") == run_id and e.get("attempt_id") == selected_attempt]
    turn_rows = [e for e in own if e.get("type") == "turn"]
    if not turn_rows:
        raise ValueError("成功 attempt 没有 turn 日志")
    evidence = {e["chunk_id"]: {"chunk_id": e["chunk_id"], "document_id": e.get("document_id"),
                                  "title": e.get("title"), "text": e.get("text"),
                                  "source": e.get("source"), "pubmed_url": pubmed_url(e)}
                for e in own if e.get("type") == "evidence" and e.get("chunk_id")}
    edges = [e for e in own if e.get("type") == "edge" and e.get("relation") == "cited"]
    raw_turn_ids = {e.get("turn_id") for e in turn_rows}
    role_order = {role: i for i, role in enumerate(result.get("roles") or [])}
    turns = []
    for row in turn_rows:
        tid = row.get("turn_id")
        payload = row.get("visible_payload") or {}
        claims = [{k: c.get(k) for k in ("claim_id", "text", "answer", "polarity", "conditions", "citation_ids")}
                  for c in row.get("claims") or []]
        claim_ids = {c.get("claim_id") for c in claims}
        turn_edges = [{"source_id": e.get("source_id"), "target_id": e.get("target_id")}
                      for e in edges if e.get("target_id") == tid or e.get("target_id") in claim_ids]
        cited = sorted({e["source_id"] for e in turn_edges if e.get("source_id")})
        messages = payload.get("messages") if isinstance(payload, dict) else []
        visible_evidence = payload.get("evidence") if isinstance(payload, dict) else []
        turns.append({"turn_id": tid, "role": row.get("role"), "round_id": row.get("round_id"),
                      "order": row.get("order"), "answer": row.get("answer"),
                      "probabilities": row.get("probabilities"),
                      "answer_probability": (row.get("probabilities") or {}).get(row.get("answer")),
                      "cited_ids": cited, "cited_edges": turn_edges,
                      "visible_evidence": [public_fields(e, ("chunk_id", "document_id", "title", "text", "source", "position"))
                                           for e in (visible_evidence or [])],
                      "visible_messages": [public_fields(m, ("id", "kind", "role", "answer", "claim", "opinion",
                                                                                 "polarity", "conditions", "citation_ids"))
                                           for m in (messages or [])],
                      "visible_message_ids": row.get("visible_messages") or [],
                      "claims": claims,
                      "reused_from": row.get("reused_from"),
                      "reused_attempt_id": row.get("reused_attempt_id")})
    turns.sort(key=lambda t: (t["round_id"], t["order"], role_order.get(t["role"], 99)))
    # 引用图只收事件里实际的 cited 边；暴露边仅作为可展开的文字。
    explicit_edges = [{"source_id": e.get("source_id"), "target_id": e.get("target_id")}
                      for e in edges if e.get("target_id") in raw_turn_ids
                      or any(e.get("target_id") == c.get("claim_id") for t in turns for c in t["claims"])]
    replay = [{"source_attempt_id": e.get("source_attempt_id"),
               "source_turn_id": e.get("source_id"), "target_turn_id": e.get("target_id")}
              for e in own if e.get("type") == "edge" and e.get("relation") == "replayed_from"]
    return {"question_id": question_id, "dataset": result.get("dataset"), "group": group,
            "repeat_id": repeat_id, "run_id": run_id, "attempt_id": selected_attempt,
            "gold": result.get("gold"), "final_answer": result.get("answer"),
            "roles": result.get("roles") or [], "turns": turns,
            "evidence": evidence, "explicit_edges": explicit_edges, "replay": replay,
            "supplemental_chunk_ids": result.get("supplemental_chunk_ids") or [],
            "note": "单题技术案例，仅展示日志中的外显行为，不能作为方法有效性或因果证据。"}


HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>单题协作审计</title><style>
:root{font-family:system-ui,-apple-system,"Microsoft YaHei",sans-serif;color:#17212b;background:#f5f7f8}
body{max-width:1200px;margin:0 auto;padding:24px}h1{margin:0 0 8px}h2{margin:28px 0 12px;font-size:1.2rem}
.note{background:#fff4d5;border-left:4px solid #b87300;padding:12px}.meta{color:#485766;word-break:break-all}
.grid{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(300px,.8fr);gap:18px}
.card{background:white;border:1px solid #dbe2e7;border-radius:8px;padding:16px;overflow:auto}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border-bottom:1px solid #e5eaee;padding:7px;text-align:left}th{background:#f0f4f6;position:sticky;top:0}
tr[data-turn]{cursor:pointer}tr[data-turn]:hover,tr.selected{background:#e8f3ff}.muted{color:#62717e}
button{font:inherit;border:1px solid #9db0bc;background:white;border-radius:5px;padding:5px 8px;cursor:pointer}
button:hover{background:#e8f3ff}a{color:#125d91}pre{white-space:pre-wrap;word-break:break-word;font:inherit;margin:6px 0}
.item{border-top:1px solid #e7edf0;padding:10px 0}.tag{font-size:.8rem;color:#43606e;background:#e9f1f4;border-radius:4px;padding:2px 5px}
details{margin:8px 0}.small{font-size:.83rem}.chart{width:100%;height:220px}.graph{min-height:130px}
@media(max-width:800px){.grid{grid-template-columns:1fr}body{padding:12px}}
</style></head><body>
<h1>单题协作审计</h1><p class="note" id="note"></p><p class="meta" id="meta"></p>
<h2>答案与自报概率</h2><div class="card"><svg class="chart" id="chart" role="img" aria-label="各专家及主持者答案自报概率变化"></svg>
<p class="muted small">纵轴为本次所选答案的自报概率；折线仅连接同一发言者的实际记录。点击表格行查看原文。</p>
<div style="max-height:340px;overflow:auto"><table><thead><tr><th>阶段</th><th>发言者</th><th>答案</th><th>该答案概率</th><th>完整概率</th></tr></thead><tbody id="turns"></tbody></table></div></div>
<div class="grid"><section><h2>发言详情</h2><div class="card" id="detail">请选择一条发言。</div></section>
<section><h2>显式引用关系</h2><div class="card"><svg class="graph" id="graph" role="img" aria-label="所选发言的显式引用边"></svg>
<p class="muted small">左侧是被引用的证据或消息，右侧是本次发言或其命题；每条箭头对应一条日志中的显式引用边。悬停可看完整原始 ID。箭头不表示证据真实支持或因果影响；消息暴露另列在发言详情。</p></div></section></div>
<h2>日志来源</h2><div class="card small" id="source"></div>
<script id="case-data" type="application/json">__CASE_JSON__</script>
<script>
const data=JSON.parse(document.getElementById('case-data').textContent);
const el=(name,txt,cls)=>{const x=document.createElement(name);if(txt!==undefined)x.textContent=String(txt);if(cls)x.className=cls;return x};
const add=(parent,name,txt,cls)=>{const x=el(name,txt,cls);parent.appendChild(x);return x};
const fmt=x=>typeof x==='number'?x.toFixed(3):'未知';
const stage=t=>t.role==='host'?(t.round_id===5?'主持者首次汇总':'主持者最终输出'):(t.round_id===0?'R0':t.round_id===6?'补检索复核':'R'+t.round_id);
const roleName=r=>r==='host'?'主持者':`专家${data.roles.indexOf(r)+1}`;
document.getElementById('note').textContent=data.note;
document.getElementById('meta').textContent=`${data.dataset} · 题号 ${data.question_id} · ${data.group} · repeat ${data.repeat_id} · 最终答案 ${data.final_answer} · gold ${data.gold??'未知'}`;
const tbody=document.getElementById('turns');
data.turns.forEach((t,i)=>{const tr=el('tr');tr.dataset.turn=i;[stage(t),roleName(t.role),t.answer,fmt(t.answer_probability),JSON.stringify(t.probabilities??{})].forEach(v=>add(tr,'td',v));tr.addEventListener('click',()=>select(i));tbody.appendChild(tr)});
const svgNS='http://www.w3.org/2000/svg';
const svg=(parent,name,attrs,text)=>{let x=document.createElementNS(svgNS,name);for(const [k,v] of Object.entries(attrs))x.setAttribute(k,String(v));if(text!==undefined)x.textContent=String(text);parent.appendChild(x);return x};
function drawChart(){const s=document.getElementById('chart');s.setAttribute('viewBox','0 0 900 220');svg(s,'line',{x1:44,y1:10,x2:44,y2:180,stroke:'#80909b'});svg(s,'line',{x1:44,y1:180,x2:885,y2:180,stroke:'#80909b'});[0,.5,1].forEach(v=>svg(s,'text',{x:5,y:184-v*170,'font-size':12,fill:'#62717e'},v));
const palette=['#0c68a4','#ce5a17','#6c51aa','#26845e','#263646'];const roles=[...data.roles,'host'];
roles.forEach((role,ri)=>{const pts=data.turns.map((t,i)=>({t,i})).filter(x=>x.t.role===role&&typeof x.t.answer_probability==='number').map(x=>({x:55+x.i*820/Math.max(1,data.turns.length-1),y:180-x.t.answer_probability*170,i:x.i}));if(pts.length>1)svg(s,'polyline',{points:pts.map(p=>`${p.x},${p.y}`).join(' '),fill:'none',stroke:palette[ri], 'stroke-width':2});pts.forEach(p=>{const c=svg(s,'circle',{cx:p.x,cy:p.y,r:5,fill:palette[ri],tabindex:0});c.addEventListener('click',()=>select(p.i))})});
roles.forEach((r,i)=>{const x=50+(i%3)*270,y=202+Math.floor(i/3)*15;svg(s,'text',{x,y,'font-size':11,fill:palette[i]},roleName(r))});s.setAttribute('viewBox','0 0 900 240')}
function section(parent,title){return add(parent,'h3',title)}
function evidenceItem(parent,e){const box=add(parent,'div',undefined,'item');add(box,'span',`${e.chunk_id??'未知片段'} · ${e.document_id??'未知文档'}`,'tag');if(e.title)add(box,'div',e.title);add(box,'pre',e.text??'无原文');const url=data.evidence[e.chunk_id]?.pubmed_url;if(url){const a=add(box,'a','PubMed 文献页');a.href=url;a.target='_blank';a.rel='noopener noreferrer'}}
const short=(value,n)=>{const text=String(value??'未知');return text.length>n?text.slice(0,n-1)+'…':text};
function sourceLabel(t,id){const evidence=data.evidence[id]||t.visible_evidence.find(e=>e.chunk_id===id);if(evidence)return `${evidence.document_id??'证据'} ${evidence.title??''}`.trim();const msg=t.visible_messages.find(m=>m.id===id);if(msg)return msg.claim?`消息命题：${msg.claim}`:`消息：${msg.role??'同伴'} 选 ${msg.answer??'未知'}`;const sourceTurn=data.turns.find(x=>x.turn_id===id);if(sourceTurn)return `${roleName(sourceTurn.role)} ${stage(sourceTurn)} 选 ${sourceTurn.answer}`;return id}
function targetLabel(t,id){if(id===t.turn_id)return `${roleName(t.role)} ${stage(t)}`;const claim=t.claims.find(c=>c.claim_id===id);return claim?`本次命题：${claim.text??'无文本'}`:id}
function drawGraph(t){const s=document.getElementById('graph');s.replaceChildren();const edges=t.cited_edges||[];const h=Math.max(130,55+edges.length*35);s.setAttribute('viewBox',`0 0 650 ${h}`);if(!edges.length){svg(s,'text',{x:20,y:50,fill:'#62717e'},'本次发言无显式引用边');return}edges.forEach((edge,i)=>{const y=35+i*35;const left=svg(s,'g',{});svg(left,'title',{},`原始来源 ID：${edge.source_id}`);svg(left,'rect',{x:10,y:y-14,width:255,height:28,rx:4,fill:'#eef2f4',stroke:'#bacbd4'});svg(left,'text',{x:18,y:y+4,'font-size':12},short(sourceLabel(t,edge.source_id),31));svg(s,'line',{x1:266,y1:y,x2:442,y2:y,stroke:'#416c87','stroke-width':1.5});svg(s,'path',{d:`M 442 ${y} l -7 -4 l 0 8 z`,fill:'#416c87'});const right=svg(s,'g',{});svg(right,'title',{},`原始目标 ID：${edge.target_id}`);svg(right,'rect',{x:445,y:y-14,width:195,height:28,rx:4,fill:'#e5f2fc',stroke:'#7aa8c7'});svg(right,'text',{x:452,y:y+4,'font-size':12},short(targetLabel(t,edge.target_id),23))})}
function select(i){const t=data.turns[i];document.querySelectorAll('tr[data-turn]').forEach(r=>r.classList.toggle('selected',Number(r.dataset.turn)===i));const d=document.getElementById('detail');d.replaceChildren();section(d,`${stage(t)} · ${roleName(t.role)}`);add(d,'p',`原始角色：${t.role}`,'muted small');add(d,'p',`答案 ${t.answer}；自报概率 ${fmt(t.answer_probability)}；完整概率 ${JSON.stringify(t.probabilities??{})}`);add(d,'p',`turn ID：${t.turn_id}`,'muted small');if(t.reused_from)add(d,'p',`R0 回放来源：${t.reused_from}；来源 attempt：${t.reused_attempt_id??'未知'}`,'muted small');
section(d,'外显命题');if(!t.claims.length)add(d,'p','无');t.claims.forEach(c=>{const box=add(d,'div',undefined,'item');add(box,'div',c.text??'');add(box,'div',`答案 ${c.answer??'未知'} · 极性 ${c.polarity??'未知'} · 条件 ${c.conditions??'无'} · 引用 ${(c.citation_ids||[]).join(', ')||'无'}`,'muted small')});
const ev=add(d,'details');add(ev,'summary',`当时可见证据（${t.visible_evidence.length}）`);t.visible_evidence.forEach(e=>evidenceItem(ev,e));
const msg=add(d,'details');add(msg,'summary',`当时可见消息（${t.visible_messages.length}）`);add(msg,'p',`原始消息 ID：${t.visible_message_ids.join(', ')||'无'}`,'muted small');t.visible_messages.forEach(m=>{const box=add(msg,'div',undefined,'item');add(box,'div',`${m.kind??'消息'} · ${m.role??''} · ${m.answer??''} · ${m.id??''}`,'tag');add(box,'pre',m.claim??m.opinion??JSON.stringify(m))});
const refs=add(d,'details');add(refs,'summary',`显式引用原始边（${t.cited_edges.length}）`);t.cited_edges.forEach(e=>add(refs,'p',`${e.source_id} → ${e.target_id}`,'small'));
drawGraph(t)}
drawChart();if(data.turns.length)select(0);
const src=document.getElementById('source');add(src,'p',`run_id：${data.run_id}；成功 attempt_id：${data.attempt_id}。本页仅由该 attempt 的事件构建。`);add(src,'p',`补入片段：${data.supplemental_chunk_ids.join(', ')||'无'}`);data.replay.forEach(r=>add(src,'p',`回放 ${r.source_turn_id}（来源 attempt ${r.source_attempt_id}）→ ${r.target_turn_id}`));
</script></body></html>"""


def render_html(case: dict[str, Any]) -> str:
    # script 原始文本中转义尖括号、& 及 JS 行分隔符，避免证据原文闭合 script。
    payload = json.dumps(case, ensure_ascii=False, separators=(",", ":"))
    payload = (payload.replace("&", "\\u0026").replace("<", "\\u003c")
               .replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    return HTML.replace("__CASE_JSON__", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--attempt-id")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = build_case(args.run_dir, args.question_id, args.group, args.repeat, args.attempt_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(case), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
