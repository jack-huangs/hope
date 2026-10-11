import json

import pytest

from evidence_mdt.models import MockModel, ModelResult
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import Runner
from evidence_mdt.schema import AgentTurn, Claim, Evidence, Question


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def question():
    return Question("q-one", "synthetic", "Choose the supported symbol",
                    {"A": "alpha", "B": "beta"}, "A")


def evidence():
    return [Evidence("e-one", "d-one", "alpha study finding")]


def turn(turn_id, role, claim):
    return AgentTurn(turn_id, role, 1, 0, ["e-one"], [], {}, "A",
                     {"A": 0.8, "B": 0.2}, ["e-one"], [claim],
                     "Free form audit text", {})


def test_same_structure_and_strict_semantic_merge(tmp_path):
    runner = Runner(MockModel(), Retriever(evidence(), MockDense()), tmp_path)
    a = Claim("a", "alpha helps", "A", ("e-one",), polarity="support")
    b = Claim("b", "alpha helps", "A", ("e-one",), polarity="support")
    conditional = Claim("c", "alpha helps", "A", ("e-one",), polarity="support", conditions="older adults")
    contrary = Claim("d", "alpha helps", "A", ("e-one",), polarity="contradict")
    turns = [turn("t1", "cardiologist", a), turn("t2", "epidemiologist", b),
             turn("t3", "pharmacologist", conditional), turn("t4", "internist", contrary)]
    plain = runner.message_view("q-one:repeat-1:natural:ordinary", turns, False)
    merged = runner.message_view("q-one:repeat-1:natural:semantic", turns, True)
    assert len([m for m in plain if m["kind"] == "opinion"]) == 4
    assert len([m for m in merged if m["kind"] == "opinion"]) == 4
    assert len([m for m in plain if m["kind"] == "claim"]) == 4
    claims = [m for m in merged if m["kind"] == "claim"]
    assert len(claims) == 3
    assert any(set(m["member_claim_ids"]) == {"a", "b"} for m in claims)
    assert any(m["conditions"] == "older adults" for m in claims)
    assert any(m["polarity"] == "contradict" for m in claims)
    assert len({m["id"] for m in plain}) == len(plain)


def test_semantic_conditions_use_relation_not_literal_string(tmp_path):
    class ConditionModel(MockModel):
        def complete(self, payload):
            if payload["task"] == "semantic_batch":
                relations = []
                for pair in payload["pairs"]:
                    conditions = {pair["a"]["conditions"], pair["b"]["conditions"]}
                    relations.append("equivalent" if conditions == {"adults", "成人"}
                                     else "condition_different")
                return ModelResult({"relations": relations}, {"fixture": True}, {"total_tokens": 0})
            return super().complete(payload)

    runner = Runner(ConditionModel(), Retriever(evidence(), MockDense()), tmp_path)
    claims = [Claim("a", "Treatment helps adults", "A", conditions="adults"),
              Claim("b", "治疗对成人有益", "A", conditions="成人"),
              Claim("c", "Treatment helps older adults", "A", conditions="older adults")]
    view = runner.message_view("q-one:repeat-1:natural:semantic",
                               [turn(f"t{i}", "expert", claim) for i, claim in enumerate(claims)], True)
    units = [item for item in view if item["kind"] == "claim"]
    assert len(units) == 2
    assert any(set(item["member_claim_ids"]) == {"a", "b"} for item in units)
    assert any(item["member_claim_ids"] == ["c"] for item in units)
    calls = records(tmp_path / "calls.jsonl")
    assert any(call["request"]["task"] == "semantic_batch" and
               any({pair["a"]["conditions"], pair["b"]["conditions"]} == {"adults", "成人"}
                   for pair in call["request"]["pairs"]) for call in calls)


def test_semantic_cache_shared_within_repeat_but_isolated_across_repeats(tmp_path):
    runner = Runner(MockModel(), Retriever(evidence(), MockDense()), tmp_path)
    turns = [turn("t1", "doctor one", Claim("c1", "first reason", "A")),
             turn("t2", "doctor two", Claim("c2", "second reason", "A"))]
    runner.current_attempt_id = "first-attempt"
    runner.message_view("q-one:repeat-1:natural:semantic", turns, True)
    runner.current_attempt_id = "second-group-attempt"
    runner.message_view("q-one:repeat-1:natural:semantic_retrieval", turns, True)
    runner.current_attempt_id = "retry-attempt"
    runner.message_view("q-one:repeat-1:natural:semantic", turns, True)
    runner.current_attempt_id = "independent-repeat-attempt"
    runner.message_view("q-one:repeat-2:natural:semantic", turns, True)
    calls = records(tmp_path / "calls.jsonl")
    assert [c["request"]["task"] for c in calls] == ["semantic_batch", "semantic_batch"]
    events = records(tmp_path / "events.jsonl")
    shared = [e for e in events if e.get("type") == "semantic_cache_hit"
              and e.get("cache_kind") == "shared_prior_decision"]
    assert len(shared) == 2
    assert shared[0]["source_run_id"] == "q-one:repeat-1:natural:semantic"
    assert shared[0]["source_attempt_id"] == "first-attempt"
    assert shared[0]["source_call_index"] == 1
    assert shared[0]["source_pair_index"] == 0
    assert len(shared[0]["pair_sha256"]) == 64
    assert shared[1]["attempt_id"] == "retry-attempt"


def test_cluster_snapshots_record_a_to_ab_to_a_and_new_attempt_nodes(tmp_path):
    runner = Runner(MockModel(), Retriever(evidence(), MockDense()), tmp_path)
    claim_a = Claim("a", "same claim", "A")
    claim_b = Claim("b", "same claim", "A")
    runner.current_attempt_id = "attempt-1"
    runner.message_view("q:repeat-1:natural:semantic", [turn("t1", "expert", claim_a)], True)
    runner.message_view("q:repeat-1:natural:semantic", [turn("t1", "expert", claim_a),
                                                         turn("t2", "expert", claim_b)], True)
    runner.message_view("q:repeat-1:natural:semantic", [turn("t1", "expert", claim_a)], True)
    clusters = [e for e in records(tmp_path / "events.jsonl") if e.get("type") == "cluster"]
    assert [e["member_claim_ids"] for e in clusters] == [["a"], ["a", "b"], ["a"]]
    runner.current_attempt_id = "attempt-2"
    runner.message_view("q:repeat-1:natural:semantic", [turn("t1", "expert", claim_a)], True)
    view_nodes = [e for e in records(tmp_path / "events.jsonl")
                  if e.get("type") == "view_node" and e.get("source_id") == "t1"]
    assert len(view_nodes) == 2


def test_injection_enters_raw_messages_before_semantic_view_and_labels_hidden(tmp_path):
    q = question()
    ev = evidence()
    runner = Runner(MockModel(), Retriever(ev, MockDense()), tmp_path)
    injection = {"receiver": 0, "speaker": 1, "wrong_option": "B", "round_id": 1,
                 "text": "Repeated erroneous assertion", "repeat_message": True}
    runner.run_group(q, 1, "semantic", ev, variant="C:repeated_message",
                     message_injection=injection)
    calls = records(tmp_path / "calls.jsonl")
    events = records(tmp_path / "events.jsonl")
    exposed = [x for x in events if x.get("type") == "injection"]
    assert len(exposed) == 1
    event = exposed[0]
    assert len(event["altered_raw_messages"]) == 2
    assert len([m for m in event["displayed_messages"] if m["kind"] == "opinion"])
    duplicate = [m for m in event["displayed_messages"]
                 if m["kind"] == "claim" and m["claim"] == "Repeated erroneous assertion"]
    assert len(duplicate) == 1
    assert len(duplicate[0]["member_claim_ids"]) == 2
    for call in calls:
        payload = json.dumps(call["request"], ensure_ascii=False)
        assert "C:repeated_message" not in payload
        assert '":semantic"' not in payload
        assert "derived_from" not in payload
        assert ":repeated" not in payload


def test_replayed_r0_has_full_graph_and_repeats_are_independent(tmp_path):
    q = question()
    ev = evidence()
    runner = Runner(MockModel(), Retriever(ev, MockDense()), tmp_path)
    first, r0 = runner.run_group(q, 1, "ordinary", ev)
    runner.run_group(q, 1, "semantic", ev, r0_source=r0)
    events = records(tmp_path / "events.jsonl")
    replay_id = "q-one:repeat-1:natural:semantic"
    replay = [e for e in events if e.get("run_id") == replay_id and e.get("round_id") == 0
              and e.get("type") == "turn"]
    assert len(replay) == 4
    assert all(x["reused_from"] for x in replay)
    assert sum(e.get("type") == "edge" and e.get("relation") == "replayed_from"
               and e.get("run_id") == replay_id for e in events) == 4
    assert all(any(e.get("type") == "edge" and e.get("target_id") == r["turn_id"]
                   and e.get("relation") == "exposed" for e in events) for r in replay)
    second, r0_second = runner.run_group(q, 2, "ordinary", ev)
    assert not second["r0_reused"]
    assert {t.turn_id for t in r0}.isdisjoint(t.turn_id for t in r0_second)
    assert first["options"] == q.options
    with pytest.raises(ValueError, match="same question"):
        runner.run_group(q, 2, "semantic", ev, r0_source=r0)
    with (tmp_path / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"run_id": first["run_id"], "type": "turn", "round_id": 0,
                                 "attempt_id": "failed-later-attempt"}) + "\n")
    assert [t.turn_id for t in runner.load_r0(first["run_id"])] == [t.turn_id for t in r0]
