"""只读检查成功 attempt 的讨论顺序、概率与图引用；不输出医学原文。"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evidence_mdt.schema import validate_answer


def read_rows(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        warnings.append(f"缺少 {path.name}")
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"{path.name} 第 {line_number} 行未写完或不可解析，已忽略")
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def issue(issues: list[dict[str, str]], run_id: str, code: str) -> None:
    issues.append({"run_id": run_id, "code": code})


def r0_signature(turn: dict[str, Any]) -> str:
    claims = [{key: value for key, value in claim.items() if key != "claim_id"}
              for claim in turn.get("claims", [])]
    fields = {"role": turn.get("role"), "answer": turn.get("answer"),
              "probabilities": turn.get("probabilities"),
              "citations": turn.get("citations"), "claims": claims,
              "opinion": turn.get("opinion"),
              "visible_evidence": turn.get("visible_evidence")}
    return json.dumps(fields, sort_keys=True, ensure_ascii=False)


def audit(run_dir: Path) -> dict[str, Any]:
    warnings: list[str] = []
    results = read_rows(run_dir / "results.jsonl", warnings)
    events = read_rows(run_dir / "events.jsonl", warnings)
    successful = [row for row in results if row.get("status") == "ok" and row.get("attempt_id")]
    by_attempt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    nodes_by_attempt: dict[str, set[str]] = defaultdict(set)
    for event in events:
        attempt = event.get("attempt_id")
        if not isinstance(attempt, str):
            continue
        by_attempt[attempt].append(event)
        kind = event.get("type")
        node_id = (event.get("chunk_id") if kind == "evidence" else
                   event.get("turn_id") if kind == "turn" else
                   event.get("claim_id") if kind == "claim" else
                   event.get("id") if kind in {"cluster", "view_node", "injected_message"} else None)
        if isinstance(node_id, str):
            nodes_by_attempt[attempt].add(node_id)
    issues: list[dict[str, str]] = []
    r0_by_variant: dict[tuple[str, Any, str], dict[str, dict[str, str]]] = defaultdict(dict)
    checked_edges = 0
    checked_turns = 0
    for result in successful:
        run_id = result.get("run_id", "unknown")
        attempt = result["attempt_id"]
        run_events = by_attempt.get(attempt, [])
        turns = [e for e in run_events if e.get("type") == "turn" and e.get("run_id") == run_id]
        checked_turns += len(turns)
        roles = result.get("roles") or []
        r0 = [t for t in turns if t.get("round_id") == 0]
        if len(r0) != 4 or {t.get("role") for t in r0} != set(roles) or len(set(roles)) != 4:
            issue(issues, run_id, "r0_count_or_roles")
        if any(t.get("visible_messages") for t in r0):
            issue(issues, run_id, "r0_saw_peer_message")
        if bool(result.get("r0_reused")) != all(bool(t.get("reused_from")) for t in r0):
            issue(issues, run_id, "r0_reuse_flag")
        for round_id in range(1, 5):
            round_turns = [t for t in turns if t.get("round_id") == round_id]
            if len(round_turns) != 4 or sorted(t.get("order") for t in round_turns) != [0, 1, 2, 3]:
                issue(issues, run_id, f"round_{round_id}_count_or_order")
        first_host = [t for t in turns if t.get("round_id") == 5 and t.get("role") == "host"]
        if len(first_host) != 1:
            issue(issues, run_id, "first_host_count")
        reviews = [t for t in turns if t.get("round_id") == 6]
        final_hosts = [t for t in turns if t.get("round_id") == 7]
        if result.get("supplemental_chunk_ids"):
            if len(reviews) != 4 or len(final_hosts) != 1:
                issue(issues, run_id, "supplement_review_count")
        elif reviews or final_hosts:
            issue(issues, run_id, "unexpected_review")

        allowed_injections: dict[tuple[int, str], set[str]] = defaultdict(set)
        for event in run_events:
            if event.get("type") == "injection":
                allowed_injections[(event.get("round_id"), event.get("receiver"))].update(
                    m.get("id") for m in event.get("altered_raw_messages", []) if m.get("id"))
        for turn in turns:
            round_id = turn.get("round_id")
            if isinstance(round_id, int) and 1 <= round_id <= 4:
                expected = {t["turn_id"] for t in turns if t.get("round_id") == round_id - 1 or
                            (t.get("round_id") == round_id and isinstance(t.get("order"), int)
                             and isinstance(turn.get("order"), int) and t["order"] < turn["order"])}
                visible = set(turn.get("visible_messages") or [])
                injected = allowed_injections.get((round_id, turn.get("role")), set())
                if not expected <= visible or not visible <= expected | injected:
                    issue(issues, run_id, f"round_{round_id}_visibility")
            try:
                payload = turn.get("visible_payload") or {}
                message_ids = {m["id"] for m in payload.get("messages", []) if isinstance(m, dict) and "id" in m}
                validate_answer(turn.get("raw_response") or {}, result["options"],
                                set(turn.get("visible_evidence") or []), message_ids)
            except (ValueError, KeyError, TypeError):
                issue(issues, run_id, "invalid_turn_probability_or_citation")
        for edge in (e for e in run_events if e.get("type") == "edge" and e.get("run_id") == run_id):
            checked_edges += 1
            source, target = edge.get("source_id"), edge.get("target_id")
            source_attempt = edge.get("source_attempt_id")
            source_ok = (source in nodes_by_attempt.get(source_attempt, set()) if source_attempt
                         else source in nodes_by_attempt.get(attempt, set()))
            target_ok = target in nodes_by_attempt.get(attempt, set())
            if not source_ok or not target_ok:
                issue(issues, run_id, "unresolved_graph_endpoint")
        if len(r0) == 4:
            key = (result.get("question_id"), result.get("repeat_id"), result.get("variant", "natural"))
            r0_by_variant[key][result.get("group", "unknown")] = {
                t["role"]: r0_signature(t) for t in r0}

    for group_views in r0_by_variant.values():
        reference = group_views.get("ordinary")
        if reference is None:
            continue
        for group, signatures in group_views.items():
            if group != "ordinary" and signatures != reference:
                issue(issues, f"group:{group}", "paired_r0_content_differs")
    counts = dict(sorted(Counter(item["code"] for item in issues).items()))
    return {"note": "只审计已成功的 attempt；忽略进行中/失败 attempt 的不完整轮次。",
            "successful_runs": len(successful), "checked_turns": checked_turns,
            "checked_edges": checked_edges, "issue_count": len(issues),
            "issues_by_code": counts, "issues": issues[:100],
            "issues_truncated": max(0, len(issues) - 100), "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description="只读审计运行图与回放，不输出原始医学文本")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir)
    output = args.output or args.run_dir / "audit_run.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("successful_runs", "checked_turns", "checked_edges", "issue_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
