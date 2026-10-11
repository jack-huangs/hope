"""用小型原始日志验证审计口径，不调用模型。"""
import importlib.util
from pathlib import Path


SPEC = importlib.util.spec_from_file_location(
    "analyze_common_errors", Path(__file__).resolve().parents[1] / "scripts" / "analyze_common_errors.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(cited=True, probability=.79):
    run, good, bad = "q:repeat-1:natural:semantic", "success", "failed"
    result = {"run_id": run, "attempt_id": good, "question_id": "q", "dataset": "MedQA",
              "group": "semantic", "repeat_id": 1, "gold": "A", "answer": "B",
              "roles": ["e1", "e2", "e3", "e4"], "supplemental_chunk_ids": [],
              "r0_answers": ["A"] * 4, "expert_answers": ["B"] * 4}
    events = [{"run_id": run, "attempt_id": good, "type": "evidence", "chunk_id": "c1", "document_id": "d1"}]
    for i in range(4):
        role = f"e{i+1}"
        r0_id = f"{run}:{role}:R0"
        events.append({"run_id": run, "attempt_id": good, "type": "turn", "turn_id": r0_id,
                       "role": role, "round_id": 0, "answer": "A", "citations": ["c1"]})
        final_id = f"{run}:{role}:R4"
        if i:
            events.append({"run_id": run, "attempt_id": good, "type": "view_node",
                           "id": f"ref:{i-1}", "source_id": f"{run}:e{i}:R4"})
        events.append({"run_id": run, "attempt_id": good, "type": "turn", "turn_id": final_id,
                       "role": role, "round_id": 4, "answer": "B",
                       "citations": [f"ref:{i-1}"] if cited and i else (["c1"] if cited else []),
                       "visible_messages": [f"{run}:e1:R4"] if i else []})
    events.append({"run_id": run, "attempt_id": good, "type": "turn", "turn_id": f"{run}:host:first",
                   "role": "host", "round_id": 5, "answer": "B", "probabilities": {"A": 1-probability, "B": probability},
                   "visible_payload": {"messages": []}})
    # 相同逻辑 ID 的失败旧尝试故意注入另一条证据，绝不能改变成功结果。
    events += [{"run_id": run, "attempt_id": bad, "type": "evidence", "chunk_id": "old", "document_id": "old-doc"},
               {"run_id": run, "attempt_id": bad, "type": "turn", "turn_id": f"{run}:e1:R4",
                "role": "e1", "round_id": 4, "answer": "A", "citations": ["old"]}]
    return result, events


def test_failed_attempt_isolated_and_message_citation_traced():
    result, events = fixture()
    row = MODULE.analyze_result(result, events)
    assert row["complete_log"]
    assert row["common_error_4_experts_plus_host"] is True
    assert row["high_confidence_common_error"] is False
    assert all(x == ["c1"] for x in row["final_expert_cited_chunks"])
    assert "old-doc" not in row["document_concentration"]["counts_by_source"]
    assert sum(x["correct_to_wrong"] for x in row["r0_to_final"]) == 4
    assert row["visibility_and_explicit_citation"][1]["explicit_message_citation_ids"] == ["ref:0"]


def test_high_confidence_is_separate_threshold():
    result, events = fixture(probability=.8)
    assert MODULE.analyze_result(result, events)["high_confidence_common_error"] is True


def test_no_citation_is_unknown_not_zero_overlap():
    result, events = fixture(cited=False)
    row = MODULE.analyze_result(result, events)
    assert row["chunk_concentration"]["top_source_share_of_citation_mentions"] is None
    assert row["chunk_concentration"]["intersection_all_four"] is None
    assert row["chunk_concentration"]["pairwise_jaccard_mean"] is None


def test_replay_edge_keeps_source_attempt_identity():
    result, events = fixture()
    edge = {"run_id": result["run_id"], "attempt_id": "success", "type": "edge",
            "relation": "replayed_from", "target_id": f"{result['run_id']}:e1:R0",
            "source_id": "source:e1:R0", "source_attempt_id": "source-success"}
    events.append(edge)
    row = MODULE.analyze_result(result, events)
    assert row["case_index"]["replayed_from"] == [{"target_turn_id": edge["target_id"],
        "source_attempt_id": "source-success", "source_turn_id": "source:e1:R0",
        "source_event_index": None}]


def test_later_cluster_snapshot_does_not_rewrite_earlier_citation():
    result, events = fixture()
    run, attempt = result["run_id"], result["attempt_id"]
    turn_id = f"{run}:e1:R4"
    position = next(i for i, e in enumerate(events) if e.get("turn_id") == turn_id and e.get("attempt_id") == attempt)
    events[position:position] = [
        {"run_id": run, "attempt_id": attempt, "type": "claim", "claim_id": "claim-early", "citation_ids": ["c1"]},
        {"run_id": run, "attempt_id": attempt, "type": "cluster", "id": "cluster", "member_claim_ids": ["claim-early"]},
    ]
    turn = next(e for e in events if e.get("turn_id") == turn_id and e.get("attempt_id") == attempt)
    turn["citations"] = ["cluster"]
    events += [
        {"run_id": run, "attempt_id": attempt, "type": "evidence", "chunk_id": "future", "document_id": "future-doc"},
        {"run_id": run, "attempt_id": attempt, "type": "claim", "claim_id": "claim-future", "citation_ids": ["future"]},
        {"run_id": run, "attempt_id": attempt, "type": "cluster", "id": "cluster",
         "member_claim_ids": ["claim-early", "claim-future"]},
    ]
    row = MODULE.analyze_result(result, events)
    assert row["final_expert_cited_chunks"][0] == ["c1"]
    assert "future-doc" not in row["document_concentration"]["counts_by_source"]


def test_nested_old_claim_cannot_see_later_inner_cluster_members():
    result, events = fixture()
    run, attempt = result["run_id"], result["attempt_id"]
    turn_id = f"{run}:e1:R4"
    position = next(i for i, e in enumerate(events) if e.get("turn_id") == turn_id and e.get("attempt_id") == attempt)
    early = [
        {"run_id": run, "attempt_id": attempt, "type": "claim", "claim_id": "base", "citation_ids": ["c1"]},
        {"run_id": run, "attempt_id": attempt, "type": "cluster", "id": "inner", "member_claim_ids": ["base"]},
        {"run_id": run, "attempt_id": attempt, "type": "claim", "claim_id": "old", "citation_ids": ["inner"]},
        {"run_id": run, "attempt_id": attempt, "type": "cluster", "id": "outer", "member_claim_ids": ["old"]},
    ]
    events[position:position] = early
    turn = next(e for e in events if e.get("turn_id") == turn_id and e.get("attempt_id") == attempt)
    turn["citations"] = ["outer"]
    position = events.index(turn)
    events[position + 1:position + 1] = [
        {"run_id": run, "attempt_id": attempt, "type": "evidence", "chunk_id": "future", "document_id": "future-doc"},
        {"run_id": run, "attempt_id": attempt, "type": "claim", "claim_id": "future-claim", "citation_ids": ["future"]},
        {"run_id": run, "attempt_id": attempt, "type": "cluster", "id": "inner",
         "member_claim_ids": ["base", "future-claim"]},
    ]
    row = MODULE.analyze_result(result, events)
    assert row["final_expert_cited_chunks"][0] == ["c1"]
    assert "future-doc" not in row["document_concentration"]["counts_by_source"]


def test_missing_gold_is_unknown_and_empty_evidence_is_valid():
    result, events = fixture(cited=False)
    result["gold"] = None
    events = [e for e in events if e.get("type") != "evidence"]
    row = MODULE.analyze_result(result, events)
    assert row["complete_log"] is True
    assert row["common_error_4_experts_plus_host"] is None
    assert row["high_confidence_common_error"] is None


def test_result_turn_answer_mismatch_reports_incomplete():
    result, events = fixture()
    result["expert_answers"][0] = "A"
    row = MODULE.analyze_result(result, events)
    assert row["complete_log"] is False
    assert "result_turn_consistency" in row["missing"]
    assert row["common_error_4_experts_plus_host"] is None
