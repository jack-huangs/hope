"""Read-only provenance graph. Edges describe saved observations, never inferred medical causality."""
from __future__ import annotations

from collections import Counter
import json
import re


RELATIONS = {"cited": "明确引用", "exposed": "当时可见", "displayed_as": "展示为消息",
             "member_of": "观点收录到消息", "derived_from": "材料派生", "replayed_from": "复用发言",
             "part_of": "属于该发言", "question_input": "题目输入", "final_answer": "记录为最终答案"}


def response_data(row):
    response = row.get("response") or {}
    choices = response.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices else response.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except ValueError:
            return None
    return content if isinstance(content, dict) else None


def same_response(saved, accepted):
    """Match the adapter's documented numeric-string type conversion, without changing either log."""
    if saved == accepted:
        return True
    if not isinstance(saved, dict) or not isinstance(accepted, dict):
        return False
    copy = dict(saved)
    probs = saved.get("probabilities")
    if isinstance(probs, dict):
        copy["probabilities"] = {k: float(v) if isinstance(v, str) and
                                  re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", v)
                                  else v for k, v in probs.items()}
    return copy == accepted


def build_graph(case, events, calls, attempt="", warnings=None):
    warnings = list(warnings or [])
    # Unknown attempts form a separate scope; never borrow edges/answers from a known attempt.
    scopes = {}
    for event in events:
        key = str(event.get("attempt_id") or "unknown")
        scopes.setdefault(key, []).append(event)
    if not scopes:
        for entry, row in calls:
            scopes.setdefault(str(entry.get("attempt_id") or "unknown"), [])
    if not scopes:
        scopes["unknown"] = []
    attempts = []
    for i, (key, rows) in enumerate(scopes.items()):
        turns = [r for r in rows if r.get("type") == "turn"]
        attempts.append({"id": key, "label": "未标注执行记录" if key == "unknown" else f"第 {i+1} 次执行",
                         "turns": len(turns), "has_host": any(t.get("role") == "host" for t in turns)})
    if attempt and attempt not in scopes:
        raise ValueError("找不到该次执行的图谱")
    result_attempts = [str(r.get("attempt_id")) for r in case["results"] if r.get("attempt_id") in scopes]
    preferred = next((a["id"] for a in reversed(attempts) if a["has_host"]), attempts[-1]["id"])
    selected = attempt or (result_attempts[-1] if result_attempts else preferred)
    rows = scopes[selected]
    fallback = not any(r.get("type") == "turn" for r in rows)
    if fallback:
        rows = fallback_events([(e, r) for e, r in calls if str(e.get("attempt_id") or "unknown") == selected])
        scope_info = next(a for a in attempts if a["id"] == selected)
        scope_info["turns"] = sum(r.get("type") == "turn" for r in rows)
        scope_info["has_host"] = any(r.get("type") == "turn" and r.get("role") == "host" for r in rows)
        warnings.append("此执行没有可回放的 turn 事件；仅按保存的请求与返回展示，未验证其回答格式，也不推测观点来源。")
    if len(scopes) > 1:
        warnings.append("本 run_id 含多次执行；图中仅展示当前选择的一次，不跨执行拼接。")
    if selected == "unknown":
        warnings.append("日志未标注执行编号，无法区分同一记录中的重试；同名节点保留独立版本。")

    nodes, edges, aliases, edge_keys = {}, [], {}, set()
    question = case.get("question") or {}
    nodes["question"] = {"id": "question", "raw_id": case["meta"]["question_id"], "kind": "question",
                         "label": "题目 " + case["meta"]["question_id"], "text": question.get("stem", ""),
                         "data": question, "seq": 0}

    def put(raw_id, kind, data, event, label="", owner=None):
        raw_id = str(raw_id)
        # Snapshots are versioned in chronological order, including changing semantic clusters.
        previous = aliases.get(raw_id)
        if previous and nodes[previous]["kind"] == "unresolved":
            nid = previous
        else:
            nid = f"n{len(nodes)}"
        aliases[raw_id] = nid
        value = {"id": nid, "raw_id": raw_id, "kind": kind, "label": label,
                 "seq": len(nodes), "event_line": event.get("_line"), "call_line": event.get("_call_line"),
                 "text": data.get("text") or data.get("claim") or data.get("opinion") or "",
                 "role": data.get("role", ""), "round_id": data.get("round_id"),
                 "order": data.get("order", 0), "answer": data.get("answer"), "data": data}
        if owner:
            value["owner_id"] = owner
            value["round_id"] = nodes[owner].get("round_id")
            value["role"] = nodes[owner].get("role", "")
        nodes[nid] = value
        return nid

    def resolve(raw_id):
        raw_id = str(raw_id)
        if raw_id not in aliases:
            put(raw_id, "unresolved", {"text": "日志未提供此来源的内容或来源映射。"}, {}, "来源未解析")
        return aliases[raw_id]

    def connect(source, target, relation, event=None, origin="event"):
        key = (source, target, relation)
        if source == target or key in edge_keys:
            return
        edge_keys.add(key)
        edges.append({"source": source, "target": target, "relation": relation,
                      "label": RELATIONS.get(relation, relation), "origin": origin,
                      "event_line": (event or {}).get("_line")})

    for event in rows:
        typ = event.get("type")
        if typ == "evidence":
            put(event["chunk_id"], "evidence", event, event, event.get("title") or "文献片段")
        elif typ == "turn":
            data = {k: v for k, v in event.items() if k not in {"visible_payload", "raw_response", "claims"}}
            tid = put(event["turn_id"], "turn", data, event, event.get("role", "专家发言"))
            if not nodes[tid].get("call_line"):
                # Link only an unambiguous matching saved response; old calls often lack attempt IDs.
                matches = [e["line"] for e, r in calls if
                           (not e.get("attempt_id") or str(e["attempt_id"]) == selected)
                           and e.get("role") == event.get("role") and e.get("round") == event.get("round_id")
                           and same_response(response_data(r), event.get("raw_response"))]
                if len(matches) == 1:
                    nodes[tid]["call_line"] = matches[0]
            connect("question", tid, "question_input", origin="structure")
            for evidence in (event.get("visible_payload") or {}).get("evidence", []):
                if evidence.get("chunk_id") not in aliases:
                    put(evidence["chunk_id"], "evidence", evidence, event, evidence.get("title") or "文献片段")
            for claim in event.get("claims", []):
                cid = put(claim["claim_id"], "claim", claim, event, "具体观点", tid)
                connect(cid, tid, "part_of", origin="structure")
        elif typ == "claim":
            raw_id = event["claim_id"]
            # Embedded claims were already registered with their author; do not duplicate them.
            if raw_id not in aliases:
                put(raw_id, "claim", event, event, "具体观点（来源见日志）")
        elif typ == "view_node":
            vid = put(event["id"], "view", event, event, "展示的专家意见")
            if event.get("source_id"):
                connect(resolve(event["source_id"]), vid, "displayed_as", event)
        elif typ == "cluster":
            cid = put(event["id"], "cluster", event, event, "展示的观点")
            for member in event.get("member_claim_ids", []):
                connect(resolve(member), cid, "member_of", event)
        elif typ == "input_message":
            put(event["id"], "view", event, event, "请求中的消息（未记录来源映射）")
        elif typ == "edge":
            connect(resolve(event["source_id"]), resolve(event["target_id"]), event["relation"], event)
        elif typ == "injected_message":
            message = event.get("message") or event
            if message.get("id"):
                put(message["id"], "view", message, event, "干预插入的消息")

    turns = [n for n in nodes.values() if n["kind"] == "turn"]
    # A result is tied only by its saved attempt, or a sole scope. Gold never participates in graph edges.
    results = [r for r in case["results"] if str(r.get("attempt_id") or "unknown") == selected
               or (len(scopes) == 1 and not r.get("attempt_id"))]
    result = results[-1] if results else None
    hosts = [n for n in turns if n.get("role") == "host"]
    final_turn = hosts[-1] if hosts else (turns[-1] if fallback and turns else None)
    final_id = None
    if result and final_turn and result.get("answer") is not None and result["answer"] == final_turn.get("answer"):
        final_id = put("final", "answer", {"answer": result["answer"],
                       "text": str((question.get("options") or {}).get(result["answer"], "")),
                       "probabilities": result.get("probabilities", {}), "round_id": 8}, {}, "最终答案")
        connect(final_turn["id"], final_id, "final_answer", origin="result")
    if not final_id:
        warnings.append("当前执行未找到可对应的最终结果；仅展示已记录的过程。")
    for n in nodes.values():
        if n.get("owner_id"):
            n["call_line"] = nodes[n["owner_id"]].get("call_line")
    counts = Counter(n["kind"] for n in nodes.values())
    return {"batch": case["meta"]["batch"], "run_id": case["meta"]["run_id"], "attempt": selected,
            "attempts": attempts, "source_mode": "calls" if fallback else "events",
            "nodes": list(nodes.values()), "edges": edges, "counts": dict(counts),
            "final_node": final_id, "final_turn": final_turn["id"] if final_turn else None,
            "warnings": warnings, "gold": case["meta"].get("gold"),
            "note": "线条表示日志记录的引用、可见性或结构关系；不证明文献支持该观点，也不代表模型内部的真实思考。"}


def fallback_events(calls):
    """Only reconstruct explicit fields. Never guess a ref's author by matching text."""
    events = []
    for entry, row in calls:
        payload = row.get("request") or {}
        payload = payload.get("original_request") or payload
        if not payload.get("task"):
            payload = {"task": "direct", "role": "单模型", **payload}
        if payload.get("task") not in {"answer", "review", "host", "direct"}:
            continue
        response = response_data(row)
        if response is None and row.get("answer") is not None:
            response = {"answer": row["answer"], "opinion": "仅保存了答案，完整返回请查看模型对话。"}
        if response is None:
            continue
        common = {"_call_line": entry["line"]}
        tid = f"call:{entry['line']}"
        for e in payload.get("evidence", []):
            events.append({"type": "evidence", **e, **common})
        for m in payload.get("messages", []):
            if m.get("id"):
                events.append({"type": "input_message", **m, **common})
        claims = [{**c, "claim_id": f"{tid}:claim:{i}"} for i, c in enumerate(response.get("claims", [])) if isinstance(c, dict)]
        events.append({"type": "turn", "turn_id": tid, "role": payload.get("role", "单模型"),
                       "round_id": payload.get("round_id", 0), "claims": claims,
                       "answer": response.get("answer"), "probabilities": response.get("probabilities", {}),
                       "opinion": response.get("opinion", ""), "raw_response": response, **common})
        for ref in payload.get("evidence", []):
            events.append({"type": "edge", "source_id": ref["chunk_id"], "target_id": tid, "relation": "exposed", **common})
        for target, cites in [(tid, response.get("citations", []))] + [(c["claim_id"], c.get("citation_ids", [])) for c in claims]:
            for cite in cites:
                events.append({"type": "edge", "source_id": cite, "target_id": target, "relation": "cited", **common})
    return events
