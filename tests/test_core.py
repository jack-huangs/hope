import json
from pathlib import Path

import pytest

from evidence_mdt.data import normalize_question
from evidence_mdt.interventions import b_evidence, c_evidence
from evidence_mdt.metrics import brier
from evidence_mdt.models import MockModel
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import GROUPS, Runner, stable_order
from evidence_mdt.schema import Budget, Claim, Evidence, Question, validate_answer


def fixture_question():
    return Question("fixture-1", "synthetic", "Which symbolic option matches token alpha?",
                    {"A": "alpha", "B": "beta", "C": "gamma"}, "A")


def fixture_evidence():
    return [Evidence("chunk-1", "doc-1", "alpha appears in the token list"),
            Evidence("chunk-2", "doc-2", "beta appears in another token list")]


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_gold_and_pubmed_long_answer_hidden():
    q = normalize_question({"QUESTION": "Synthetic question", "CONTEXTS": ["Background"],
                            "final_decision": "yes", "LONG_ANSWER": "SECRET ANSWER",
                            "reasoning_free_pred": "SECRET PRED"}, "PubMedQA", 0)
    public = json.dumps(q.public())
    assert q.gold == "yes"
    assert "SECRET" not in public
    assert "final_decision" not in public
    bigbio = normalize_question({"question": "Q", "options": [{"key": "A", "value": "one"},
                                                              {"key": "B", "value": "two"}],
                                 "answer_idx": "A"}, "MedQA", 1)
    assert bigbio.options == {"A": "one", "B": "two"}


def test_answer_validation_rejects_missing_labels_and_invisible_citation():
    data = {"answer": "A", "probabilities": {"A": 0.7, "B": 0.3},
            "citations": ["hidden"], "claims": [], "opinion": "test"}
    with pytest.raises(ValueError, match="citation"):
        validate_answer(data, {"A": "a", "B": "b"}, {"visible"}, set())
    data["citations"] = []
    data["probabilities"] = {"A": 1.0}
    with pytest.raises(ValueError, match="every option"):
        validate_answer(data, {"A": "a", "B": "b"}, set(), set())


def test_retrieval_id_fusion_and_b_duplicates():
    evidence = fixture_evidence()
    retriever = Retriever(evidence, MockDense())
    got = retriever.search("alpha", limit=6)
    assert len(got) == 2
    copied = b_evidence(got, "literal5", got[0].chunk_id)
    assert len(copied) == len(got) + 4
    assert len({x.chunk_id for x in copied}) == len(copied)
    assert all(x.derived_from == got[0].chunk_id for x in copied[len(got):])
    with pytest.raises(ValueError, match="paraphrases"):
        b_evidence(got, "paraphrase3", got[0].chunk_id)


def test_mock_three_group_e2e_and_visibility(tmp_path):
    q = fixture_question()
    ev = fixture_evidence()
    runner = Runner(MockModel(), Retriever(ev, MockDense()), tmp_path)
    r0 = None
    for group in GROUPS:
        record, r0_new = runner.run_group(q, 1, group, ev, r0_source=r0)
        if r0 is None:
            r0 = r0_new
        assert record["group"] == group
        assert set(record["probabilities"]) == set(q.options)
    calls = read_jsonl(tmp_path / "calls.jsonl")
    events = read_jsonl(tmp_path / "events.jsonl")
    results = read_jsonl(tmp_path / "results.jsonl")
    assert len(results) == 3
    assert results[0]["r0_reused"] is False
    assert all(r["r0_reused"] for r in results[1:])
    assert not any("gold" in c["request"]["question"] for c in calls if "question" in c["request"])
    for group in GROUPS:
        group_id = f"{q.question_id}:repeat-1:natural:{group}"
        round_one = [x for x in events if x.get("type") == "turn" and x.get("run_id") == group_id
                     and x.get("round_id") == 1]
        assert len(round_one) == 4
        assert [x["role"] for x in round_one] == [
            ("diagnostic specialist", "clinical epidemiologist", "pharmacology specialist", "general medicine specialist")[i]
            for i in stable_order(q.question_id, 1, 1)]
        for t in round_one:
            visible = t["visible_messages"]
            assert not any(x.endswith(":R1") and x.split(":R1")[0].endswith(t["role"]) for x in visible)
            assert all(not x.endswith(":R2") for x in visible)
    assert brier({"A": 1, "B": 0}, "A") == 0


def test_budget_stops_without_final_result(tmp_path):
    q = fixture_question()
    ev = fixture_evidence()
    runner = Runner(MockModel(), Retriever(ev, MockDense()), tmp_path, Budget(max_calls=2))
    with pytest.raises(RuntimeError, match="budget"):
        runner.run_group(q, 1, "ordinary", ev)
    assert not (tmp_path / "results.jsonl").exists()


def test_c_evidence_is_new_material_before_r0(tmp_path):
    q = fixture_question()
    ev = c_evidence(fixture_evidence(), "symbolic injected text", q.question_id)
    runner = Runner(MockModel(), Retriever(fixture_evidence(), MockDense()), tmp_path)
    runner.run_group(q, 1, "ordinary", ev, variant="C:evidence_error")
    r0 = [x for x in read_jsonl(tmp_path / "events.jsonl") if x.get("type") == "turn" and x.get("round_id") == 0]
    assert len(r0) == 4
    assert all(len(x["visible_evidence"]) == 3 for x in r0)
    assert all("derived_from" not in e for x in r0 for e in x["visible_payload"]["evidence"])
