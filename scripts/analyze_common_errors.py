"""只读审计自然共同错误；不做医学判断或额外语义聚类。"""
from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        warnings.append(f"缺少 {path.name}")
        return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except json.JSONDecodeError:
            warnings.append(f"{path.name} 第 {number} 行无法解析")
    return rows


def unique(values: list[str]) -> list[str]:
    return sorted(set(values))


def concentration(sets: list[set[str]]) -> dict[str, Any]:
    """分母是有可追溯显式引用的专家；空引用不是零集中度。"""
    known = [s for s in sets if s]
    counts = Counter(value for values in known for value in values)
    total = sum(counts.values())
    return {
        "experts_with_traceable_citations": len(known),
        "counts_by_source": dict(sorted(counts.items())),
        "top_source_share_of_citation_mentions": max(counts.values()) / total if total else None,
        "intersection_all_four": unique(list(set.intersection(*sets))) if len(known) == 4 else None,
        "pairwise_jaccard_mean": (
            sum(len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:]) / 6
            if len(known) == 4 else None),
    }


def analyze_result(result: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    run_id, attempt = result.get("run_id"), result.get("attempt_id")
    own = [e for e in events if e.get("run_id") == run_id and e.get("attempt_id") == attempt]
    indexed = list(enumerate(own))
    turns = {e["turn_id"]: e for _, e in indexed if e.get("type") == "turn" and e.get("turn_id")}
    turn_positions = {e["turn_id"]: i for i, e in indexed if e.get("type") == "turn" and e.get("turn_id")}
    evidence = {e["chunk_id"]: e for e in own if e.get("type") == "evidence" and e.get("chunk_id")}
    evidence_positions = {e["chunk_id"]: i for i, e in indexed
                          if e.get("type") == "evidence" and e.get("chunk_id")}
    views = defaultdict(list)
    clusters = defaultdict(list)
    claims = defaultdict(list)
    for i, e in indexed:
        if e.get("type") == "view_node" and e.get("id"):
            views[e["id"]].append((i, e))
        elif e.get("type") == "cluster" and e.get("id"):
            clusters[e["id"]].append((i, e))
        elif e.get("type") == "claim" and e.get("claim_id"):
            claims[e["claim_id"]].append((i, e))
    # 相同簇 ID 会随讨论增长；预建位置索引，只取目标发言之前的快照。
    version_positions = {id(versions): [i for i, _ in versions]
                         for table in (views, clusters, claims) for versions in table.values()}
    replay = {e["target_id"]: (e.get("source_attempt_id"), e.get("source_id"))
              for e in own if e.get("type") == "edge" and e.get("relation") == "replayed_from"}

    # 回放只允许沿日志中明确的 replayed_from 跨 attempt；同 run 的失败旧尝试不进入图。
    source_turns = {(e.get("attempt_id"), e.get("turn_id")): (i, e) for i, e in enumerate(events)
                    if e.get("type") == "turn" and e.get("turn_id")}
    replay_sources_present = all((source_attempt, source_id) in source_turns
                                 for source_attempt, source_id in replay.values())

    def as_of(versions: list[tuple[int, dict]], cutoff: int) -> tuple[int, dict] | None:
        index = bisect_left(version_positions[id(versions)], cutoff) - 1
        return versions[index] if index >= 0 else None

    trace_cache: dict[tuple[str, int], set[str]] = {}

    def trace(node: str, cutoff: int) -> set[str]:
        # 每走一条展示/簇/命题/消息边，cutoff 都严格变小；无时间回路，可按节点和时间缓存。
        key = (node, cutoff)
        if key in trace_cache:
            return trace_cache[key]
        found: set[str]
        if node in evidence:
            found = {node} if evidence_positions[node] < cutoff else set()
        elif node in views:
            view = as_of(views[node], cutoff)
            found = trace(view[1]["source_id"], view[0]) if view else set()
        elif node in clusters:
            cluster = as_of(clusters[node], cutoff)
            found = set().union(*(trace(x, cluster[0])
                                  for x in cluster[1].get("member_claim_ids", []))) if cluster else set()
        elif node in claims:
            claim = as_of(claims[node], cutoff)
            found = set().union(*(trace(x, claim[0])
                                  for x in claim[1].get("citation_ids", []))) if claim else set()
        elif node in turns:
            turn = turns[node]
            source_cutoff = turn_positions[node]
            if source_cutoff >= cutoff:
                found = set()
            else:
                cited = list(turn.get("citations") or [])
                cited += [x for c in turn.get("claims") or [] for x in c.get("citation_ids", [])]
                found = set().union(*(trace(x, source_cutoff) for x in cited))
                # 回放 R0 的证据节点在目标 attempt 通常已登记；仅在缺失时查明来源。
                if not found and node in replay:
                    source_attempt, source_id = replay[node]
                    source = source_turns.get((source_attempt, source_id))
                    found = {x for x in (source[1] if source else {}).get("citations", []) if x in evidence}
        else:
            found = set()
        trace_cache[key] = found
        return found

    roles = result.get("roles") or []
    use_review = bool(result.get("supplemental_chunk_ids"))
    final_round = 6 if use_review else 4
    final = [next((t for t in turns.values() if t.get("role") == role and t.get("round_id") == final_round), None)
             for role in roles]
    r0 = [next((t for t in turns.values() if t.get("role") == role and t.get("round_id") == 0), None)
          for role in roles]
    host_round = 7 if use_review else 5
    host = next((t for t in turns.values() if t.get("role") == "host" and t.get("round_id") == host_round), None)
    consistency = (host is not None and host.get("answer") == result.get("answer")
                   and result.get("expert_answers") == [t.get("answer") for t in final if t]
                   and result.get("r0_answers") == [t.get("answer") for t in r0 if t])
    complete = (len(roles) == 4 and all(final) and all(r0) and host is not None
                and consistency and replay_sources_present)
    gold, answer = result.get("gold"), result.get("answer")
    wrong = (answer != gold and all(t.get("answer") == answer for t in final)) if complete and gold is not None else None
    host_p = (host.get("probabilities") or {}).get(answer) if host else None
    high = (wrong and isinstance(host_p, (int, float)) and host_p >= .8) if wrong is not None else None
    cited_chunks = [trace(t["turn_id"], len(own) + 1) if t else set() for t in final]
    cited_docs = [{evidence[x].get("document_id") for x in values if evidence[x].get("document_id")}
                  for values in cited_chunks]
    transitions = [{"role": role, "r0_answer": before.get("answer") if before else None,
                    "final_answer": after.get("answer") if after else None,
                    "correct_to_wrong": (before.get("answer") == gold and after.get("answer") != gold)
                    if gold and before and after else None}
                   for role, before, after in zip(roles, r0, final)]

    # 可见性来自原始 turn.visible_messages；显式引用只取 citation，二者不互推。
    visibility = [{"role": t.get("role"), "turn_id": t.get("turn_id"),
                   "visible_message_ids": unique(t.get("visible_messages") or []),
                   "explicit_message_citation_ids": unique([x for x in t.get("citations", [])
                       if x not in evidence] + [x for c in t.get("claims") or []
                       for x in c.get("citation_ids", []) if x not in evidence])}
                  for t in final if t]
    semantic = None
    if result.get("group") in {"semantic", "semantic_retrieval"}:
        # 只数最终主持者输入里实际展示的簇；簇成员取本 attempt 的最后快照。
        shown = [m.get("id") for m in (host or {}).get("visible_payload", {}).get("messages", [])
                 if isinstance(m, dict) and m.get("kind") == "claim"]
        host_position = turn_positions.get(host.get("turn_id"), len(own)) if host else len(own)
        matched = [version for x in shown if x in clusters
                   for version in [as_of(clusters[x], host_position)] if version]
        semantic = {"shown_claim_units": len(shown) if host else None,
                    "matched_cluster_snapshots": len(matched) if host else None,
                    "shown_multi_member_units": sum(len(x[1].get("member_claim_ids", [])) > 1 for x in matched)
                    if host else None,
                    "shown_member_claims": sum(len(x[1].get("member_claim_ids", [])) for x in matched)
                    if host else None}
    return {
        "dataset": result.get("dataset"), "group": result.get("group"),
        "question_id": result.get("question_id"), "repeat_id": result.get("repeat_id"),
        "run_id": run_id, "attempt_id": attempt, "gold": gold, "host_answer": answer,
        "host_probability_for_answer": host_p, "complete_log": complete,
        "missing": [name for name, condition in (("four_roles", len(roles) == 4),
            ("r0_turns", all(r0)), ("final_expert_turns", all(final)), ("host_turn", host is not None),
            ("result_turn_consistency", consistency),
            ("replay_source_turns", replay_sources_present)) if not condition],
        "common_error_4_experts_plus_host": wrong,
        "high_confidence_common_error": high,
        "r0_to_final": transitions,
        "final_expert_cited_chunks": [unique(list(x)) for x in cited_chunks],
        "chunk_concentration": concentration(cited_chunks),
        "document_concentration": concentration(cited_docs),
        "visibility_and_explicit_citation": visibility,
        "semantic_cluster_aggregation": semantic,
        "case_index": {"results_jsonl_run_id": run_id, "events_attempt_id": attempt,
                       "r0_turn_ids": [x.get("turn_id") if x else None for x in r0],
                       "final_expert_turn_ids": [x.get("turn_id") if x else None for x in final],
                       "host_turn_id": host.get("turn_id") if host else None,
                       "replayed_from": [{"target_turn_id": k, "source_attempt_id": v[0],
                                          "source_turn_id": v[1],
                                          "source_event_index": source_turns.get((v[0], v[1]), (None,))[0]}
                                         for k, v in replay.items()]},
    }


def make_report(run_dir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    results = read_jsonl(run_dir / "results.jsonl", warnings)
    events = read_jsonl(run_dir / "events.jsonl", warnings)
    selected = [r for r in results if r.get("status", "ok") == "ok" and r.get("variant", "natural") == "natural"]
    keys = Counter((r.get("dataset"), r.get("question_id"), r.get("repeat_id"),
                    r.get("group"), r.get("variant", "natural")) for r in selected)
    if any(count > 1 for count in keys.values()):
        raise ValueError("重复成功结果键；请先核查 results.jsonl")
    rows = [analyze_result(r, events) for r in selected]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["group"])].append(row)
    summary = {}
    for key, bucket in sorted(groups.items()):
        complete = [r for r in bucket if r["complete_log"]]
        summary["/".join(key)] = {
            "successful_results": len(bucket), "complete_logs": len(complete),
            "common_errors": sum(r["common_error_4_experts_plus_host"] is True for r in complete),
            "high_confidence_common_errors": sum(r["high_confidence_common_error"] is True for r in complete),
            "r0_correct_with_known_final": sum(x["correct_to_wrong"] is not None and x["r0_answer"] == r["gold"]
                                                 for r in complete for x in r["r0_to_final"]),
            "r0_correct_to_final_wrong": sum(x["correct_to_wrong"] is True
                                              for r in complete for x in r["r0_to_final"]),
        }
    return {"scope": "success natural results, matching attempt only", "summary": summary,
            "cases": rows, "warnings": warnings,
            "interpretation": ["外显引用不等于完整依据；无可追溯引用时集中度为 null。",
                "消息暴露不等于因果影响；显式引用单独列出。",
                "语义等价簇不证明真实文献同源；普通组未聚类，聚合量为 null。",
                "只按数据集与组别描述烟测，不能作医学有效性或机制因果结论。"]}


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# 自然共同错误日志审计", "", "仅纳入成功的 natural 结果及其对应 attempt 的原始日志。", "",
             "| 数据集 / 组别 | 成功结果 | 完整日志 | 共同错 | 高置信共同错 | R0 正确→最终错 |", "|---|---:|---:|---:|---:|---:|"]
    for key, row in report["summary"].items():
        lines.append(f"| {key} | {row['successful_results']} | {row['complete_logs']} | {row['common_errors']} | {row['high_confidence_common_errors']} | {row['r0_correct_to_final_wrong']}/{row['r0_correct_with_known_final']} |")
    lines += ["", "## 案例索引", ""]
    for row in report["cases"]:
        if row["common_error_4_experts_plus_host"] or not row["complete_log"]:
            lines.append(f"- {row['dataset']}/{row['group']}/{row['question_id']}/repeat-{row['repeat_id']}："
                         f"共同错={row['common_error_4_experts_plus_host']}，高置信={row['high_confidence_common_error']}，"
                         f"attempt={row['attempt_id']}，run={row['run_id']}；缺失={','.join(row['missing']) or '无'}。")
    if not any(r["common_error_4_experts_plus_host"] or not r["complete_log"] for r in report["cases"]):
        lines.append("- 无共同错误或日志缺失案例。")
    lines += ["", "## 解释边界", ""] + [f"- {x}" for x in report["interpretation"]]
    if report["warnings"]:
        lines += ["", "读取提醒：" + "；".join(report["warnings"])]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = make_report(args.run_dir)
    output = args.output_dir or args.run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "common_errors.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "common_errors.md").write_text(render_markdown(report), encoding="utf-8")
    print(output / "common_errors.md")


if __name__ == "__main__":
    main()
