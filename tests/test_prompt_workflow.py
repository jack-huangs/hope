import json

import pytest

from evidence_mdt.models import MockModel, ModelResult, OpenAICompatibleModel
from evidence_mdt.prompts import PROMPT_VERSION, response_system
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import Runner, opaque_id
from evidence_mdt.schema import Evidence, Question


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def setup_runner(tmp_path, model=None):
    evidence = [Evidence("e", "d", "Initial finding", derived_from="hidden-parent"),
                Evidence("new", "new-doc", "Distinct additional treatment condition")]
    q = Question("q", "synthetic", "Choose", {"A": "alpha", "B": "beta"}, "A")
    return Runner(model or MockModel(), Retriever(evidence, MockDense()), tmp_path), q, evidence


def test_recruitment_profiles_shared_across_groups_and_restart_but_not_old_logs(tmp_path):
    runner, q, ev = setup_runner(tmp_path)
    tmp_path.joinpath("events.jsonl").write_text(json.dumps({
        "type": "recruitment", "question_id": q.question_id, "repeat_id": 1,
        "roles": ["legacy one", "legacy two", "legacy three", "legacy four"]}) + "\n")
    experts = runner.experts_for(q, 1)
    _, r0 = runner.run_group(q, 1, "ordinary", ev[:1])
    result, _ = runner.run_group(q, 1, "semantic", ev[:1], r0_source=r0)
    assert result["experts"] == list(experts)
    resumed, _, _ = setup_runner(tmp_path)
    assert resumed.experts_for(q, 1) == experts
    recruits = [c for c in records(runner.call_log) if c["request"]["task"] == "recruit"]
    assert len(recruits) == 1
    assert recruits[0]["request"]["prompt_version"] == PROMPT_VERSION
    resumed.experts_for(q, 2)
    assert len([c for c in records(runner.call_log) if c["request"]["task"] == "recruit"]) == 2
    r0[0].visible_payload["focus"] = "different responsibility"
    with pytest.raises(ValueError, match="matching expert responsibilities"):
        resumed.run_group(q, 1, "semantic", ev[:1], r0_source=r0)
    r0[0].visible_payload.pop("prompt_version")
    with pytest.raises(ValueError, match="current prompt version"):
        resumed.run_group(q, 1, "semantic", ev[:1], r0_source=r0)


def test_recruiter_rejects_missing_responsibilities(tmp_path):
    class LegacyRecruiter(MockModel):
        def complete(self, payload):
            return ModelResult({"roles": ["one", "two", "three", "four"]}, {}, {})

    runner, q, _ = setup_runner(tmp_path, LegacyRecruiter())
    with pytest.raises(ValueError, match="expertise, focus"):
        runner.experts_for(q, 1)


def test_self_previous_is_real_separate_and_citable_without_hidden_metadata(tmp_path):
    class SelfCiting(MockModel):
        def complete(self, payload):
            result = super().complete(payload)
            if payload.get("self_previous"):
                result.content["citations"] = [payload["self_previous"]["id"]]
                result.content["claims"][0]["citation_ids"] = result.content["citations"]
            return result

    runner, q, ev = setup_runner(tmp_path, SelfCiting())
    runner.run_group(q, 1, "semantic", ev[:1])
    events = records(runner.event_log)
    turns = [e for e in events if e.get("type") == "turn"]
    for current in [t for t in turns if 1 <= t["round_id"] <= 4]:
        payload = current["visible_payload"]
        own = payload["self_previous"]
        previous = next(t for t in turns if t["role"] == current["role"]
                        and t["round_id"] == current["round_id"] - 1)
        assert own["id"] == opaque_id(previous["turn_id"])
        assert own["answer"] == previous["answer"]
        assert own["probabilities"] == previous["probabilities"]
        assert own["opinion"] == previous["opinion"]
        assert own["citations"] == previous["citations"]
        assert [c["text"] for c in own["claims"]] == [c["text"] for c in previous["claims"]]
        for actual, source in zip(own["claims"], previous["claims"]):
            assert actual["id"] == opaque_id(source["claim_id"])
            assert actual["citation_ids"] == source["citation_ids"]
        assert not any(m.get('id') == own['id'] for m in payload['messages'])
        assert all(m.get('round_id') < previous['round_id']
                   for m in payload['messages'] if m.get('kind')=='opinion' and m.get('role')==current['role'])
        serialized = json.dumps(payload)
        for secret in ("sources", "member_claim_ids", "derived_from", "hidden-parent", ":semantic"):
            assert secret not in serialized
        assert any(e.get("type") == "view_node" and e.get("id") == own["id"] for e in events)
        assert any(e.get("relation") == "exposed" and e.get("source_id") == previous["turn_id"]
                   and e.get("target_id") == current["turn_id"] for e in events)


def test_experts_receive_all_past_turns_without_latest_self_duplicate_or_future(tmp_path):
    runner,q,ev=setup_runner(tmp_path,MockModel())
    runner.run_group(q,1,'ordinary',ev[:1])
    turns=[e for e in records(runner.event_log) if e.get('type')=='turn']
    past=[]
    for current in turns:
        if 1 <= current['round_id'] <= 4:
            payload=current['visible_payload'];own=payload['self_previous']
            messages=[m for m in payload['messages'] if m['kind']=='opinion']
            assert len(messages)==len(past)-1
            assert {m['id'] for m in messages}|{own['id']}=={opaque_id(t['turn_id']) for t in past}
            assert len({m['id'] for m in messages}|{own['id']})==len(past)
            for m in messages:
                t=next(t for t in past if opaque_id(t['turn_id'])==m['id'])
                assert (m['role'],m['round_id'],m['answer'],m['probabilities'],m['opinion'])==(
                    t['role'],t['round_id'],t['answer'],t['probabilities'],t['opinion'])
            if current['round_id']==4:
                assert any(m['round_id']==0 for m in messages)
                assert any(m['role']==current['role'] and m['round_id']==0 for m in messages)
        elif current['round_id']==0:
            assert current['visible_payload']['messages']==[]
        past.append(current)


@pytest.mark.parametrize("group", ["ordinary", "semantic"])
def test_history_claim_ids_remain_citable_without_allowlist_or_repair(tmp_path, group):
    class HistoryCiting(MockModel):
        def complete(self, payload):
            assert payload["task"] != "repair", "Historical citations must not trigger repair"
            result = super().complete(payload)
            if payload.get("self_previous"):
                # 本人最新命题的编号在 self_previous 中，不需要另建允许清单。
                cite = payload["self_previous"]["claims"][0]["id"]
                result.content["citations"] = [cite]
                result.content["claims"][0]["citation_ids"] = [cite]
            elif payload["task"] == "host":
                # 主持者沿历史发言保留的引文，直接引用更早的命题编号。
                latest = next(m for m in payload["messages"]
                              if m.get("kind") == "opinion" and m.get("round_id") == 4)
                cite = latest["citations"][0]
                result.content["citations"] = [cite]
                result.content["claims"][0]["citation_ids"] = [cite]
            return result

    runner, q, _ = setup_runner(tmp_path, HistoryCiting())
    runner.run_group(q, 1, group, [Evidence("e", "d", "Initial finding")])
    turns = [e for e in records(runner.event_log) if e.get("type") == "turn"]
    for current in (t for t in turns if 1 <= t["round_id"] <= 4):
        previous = next(t for t in turns if t["role"] == current["role"]
                        and t["round_id"] == current["round_id"] - 1)
        assert current["citations"] == [opaque_id(previous["claims"][0]["claim_id"])]
        own = current["visible_payload"]["self_previous"]
        assert own["claims"][0]["citation_ids"] == previous["claims"][0]["citation_ids"]
    host = next(t for t in turns if t["role"] == "host")
    assert host["citations"] and host["claims"][0]["citation_ids"] == host["citations"]
    assert not any(c["request"]["task"] == "repair" for c in records(runner.call_log))
    from scripts.audit_run import audit
    assert audit(tmp_path)["issue_count"] == 0


def test_review_separates_added_material_and_keeps_expert_identity(tmp_path):
    class NewCiting(MockModel):
        def complete(self, payload):
            result = super().complete(payload)
            if payload.get("additional_evidence"):
                result.content["citations"] = [payload["additional_evidence"][0]["chunk_id"]]
                result.content["claims"][0]["citation_ids"] = result.content["citations"]
            return result

    runner, q, ev = setup_runner(tmp_path, NewCiting())
    runner._trigger = lambda *args: {"triggered": True}
    runner.run_group(q, 1, "semantic_retrieval", ev[:1])
    requests = [c["request"] for c in records(runner.call_log)]
    reviews = [r for r in requests if r["task"] == "review"]
    assert len(reviews) == 4
    for review in reviews:
        assert [e["chunk_id"] for e in review["evidence"]] == ["e"]
        assert [e["chunk_id"] for e in review["additional_evidence"]] == ["new"]
        assert review["self_previous"]["round_id"] == 4
        assert all(m.get("role") != review["role"] for m in review["messages"])
        assert review["expertise"] and review["focus"]
    final = [r for r in requests if r["task"] == "host"][-1]
    assert [e["chunk_id"] for e in final["additional_evidence"]] == ["new"]
    events = records(runner.event_log)
    assert all(t["citations"] == ["new"] for t in events
               if t.get("type") == "turn" and t.get("round_id", 0) >= 6)
    assert not any(r["task"] == "repair" for r in requests)


@pytest.mark.parametrize("task,round_id,expected", [
    ("answer", 0, "independent initial assessment"),
    ("answer", 2, "discussion round 2"),
    ("review", 6, "Additional evidence is supplied in additional_evidence"),
    ("host", 5, "clinical moderator making the final decision"),
    ("recruit", 0, "Assign assessment responsibilities"),
])
def test_actual_wire_system_uses_stage_identity_and_protocol(monkeypatch, task, round_id, expected):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

    monkeypatch.setenv("FIXTURE_KEY", "offline")
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response())
    model = OpenAICompatibleModel("fixture", "https://fixture.invalid", "FIXTURE_KEY")
    payload = {"task": task, "round_id": round_id, "role": "radiologist",
               "expertise": "Imaging interpretation", "focus": "Assess supplied imaging findings",
               "question": {"options": {"A": "alpha", "B": "beta"}},
               "evidence": [{"chunk_id": "e"}], "additional_evidence": [{"chunk_id": "new"}],
               "messages": [{"id": "peer"}], "self_previous": {"id": "own"}}
    wire = model.complete(payload).wire_request
    system = wire["messages"][0]["content"]
    assert expected in system
    assert json.loads(wire["messages"][1]["content"]) == payload
    if task not in {"recruit"}:
        assert "Use your medical knowledge, the case" not in system
        assert "Not every inference requires a citation" not in system
        assert "When citing a source, ensure it supports" not in system
        assert "Use the IDs attached to the supplied evidence and discussion history when citing them." in system
        assert "Do not invent IDs." in system
        assert not any(text in system for text in (
            "visible IDs", "Historical citations in self_previous are omitted",
            '"e"', '"new"', '"peer"', '"own"'))
        assert "Use only the provided" not in system
        if task != "host":
            assert "Your expertise is Imaging interpretation" in system
            assert "Your responsibility in this case is Assess supplied imaging findings" in system
    if task == "recruit":
        assert '"experts"' in system
        assert "Do not solve the question" in system


def test_repair_preserves_original_stage_and_numeric_guardrails():
    original = {"task": "review", "role": "pathologist", "round_id": 6,
                "expertise": "Histology", "focus": "Assess supplied tissue findings",
                "question": {"options": {"C": "one", "D": "two"}},
                "additional_evidence": [{"chunk_id": "new"}]}
    system = response_system({"task": "repair", "original_request": original})
    assert "You are the pathologist" in system
    assert "Additional evidence is supplied in additional_evidence" in system
    assert "every original numeric probability exactly" in system
    assert '"C"' in system and '"new"' not in system
    assert "Do not invent IDs." in system
