import json

import pytest

from evidence_mdt.models import MockModel
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import Runner, opaque_id
from evidence_mdt.schema import Evidence, Question


@pytest.mark.parametrize("group", ["ordinary", "semantic", "semantic_retrieval"])
def test_host_and_experts_receive_full_history_without_future_messages(tmp_path, group):
    evidence = [Evidence("e", "d", "Initial finding"),
                Evidence("new", "new-doc", "Distinct additional treatment condition")]
    q = Question("q", "synthetic", "Choose", {"A": "alpha", "B": "beta"}, "A")
    runner = Runner(MockModel(), Retriever(evidence, MockDense()), tmp_path)
    runner._trigger = lambda *args: {"triggered": True}
    runner.run_group(q, 1, group, evidence[:1])
    events = [json.loads(line) for line in runner.event_log.read_text(encoding="utf-8").splitlines()]
    turns = [e for e in events if e.get("type") == "turn"]
    hosts = [t for t in turns if t["role"] == "host"]
    assert len(hosts) == (2 if group == "semantic_retrieval" else 1)
    for host in hosts:
        expected = [t for t in turns if t["role"] != "host" and t["round_id"] < host["round_id"]]
        messages = host["visible_payload"]["messages"]
        opinions = {m["id"]: m for m in messages if m["kind"] == "opinion"}
        assert len(opinions) == (24 if host["round_id"] == 7 else 20)
        assert set(host["visible_messages"]) == {t["turn_id"] for t in expected}
        for turn in expected:
            opinion = opinions[opaque_id(turn["turn_id"])]
            for key in ("role", "round_id", "order", "answer", "probabilities", "opinion", "citations"):
                assert opinion[key] == turn[key]
            attributed = [c for c in messages if c["kind"] == "claim" and
                          any(a["role"] == turn["role"] and a["round_id"] == turn["round_id"]
                              for a in c["attributions"])]
            assert attributed
        assert all("sources" not in m and "member_claim_ids" not in m for m in messages)
    for turn in turns:
        if turn["role"] != "host" and 1 <= turn["round_id"] <= 4:
            rounds = {m["round_id"] for m in turn["visible_payload"]["messages"] if m["kind"] == "opinion"}
            assert set(range(turn['round_id'])) <= rounds
            assert rounds <= set(range(turn['round_id']+1))
            prior=[t for t in turns if t['round_id']<turn['round_id'] or
                   (t['round_id']==turn['round_id'] and t['order']<turn['order'])]
            assert set(turn['visible_messages'])=={t['turn_id'] for t in prior}
    if group == "semantic_retrieval":
        calls = [json.loads(line) for line in runner.call_log.read_text(encoding="utf-8").splitlines()]
        query = next(c["request"] for c in calls if c["request"]["task"] == "queries")
        assert {m["round_id"] for m in query["latest_view"] if m["kind"] == "opinion"} == {4}
