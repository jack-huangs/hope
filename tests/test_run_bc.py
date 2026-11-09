"""配对B/C入口的真实Runner离线集成测试。"""
import json

import pytest

from evidence_mdt.models import MockModel
from evidence_mdt.run_bc import NoRetrieval, run_bc
from evidence_mdt import run_batch
from evidence_mdt.retrieval import MockDense
from preparation import prepare
from preparation.batch import prepare_batch
from test_preparation_batch import fetch as fetch_ten_plus
from test_preparation_unified import Model, Tokenizer, fetch, source


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepared(tmp_path, count=2):
    path, _ = source(tmp_path, count)
    folder = tmp_path / "prepared"
    assert prepare_batch(path, folder, Model(), MockDense(), modes=("b", "c"),
                         tokenizer=Tokenizer(), fetcher=fetch, retries=0)["status"] == "complete"
    return folder


def test_bc_parallel_exactly_five_conditions_one_baseline_and_zero_supplement(tmp_path):
    prep = prepared(tmp_path)
    out = tmp_path / "run"
    result = run_bc(prep, out, MockModel(), workers=2)
    assert result["status"] == "complete" and result["completed"] == result["planned"] == 10
    records = rows(out / "results.jsonl")
    assert {r["variant"] for r in records} == {
        "B:base", "B:literal11", "B:paraphrase11", "C:message_error", "C:repeated_message"}
    assert len({r["run_id"] for r in records}) == 10
    assert {r["group"] for r in records} == {"ordinary"}
    assert sum(r["variant"] == "B:base" for r in records) == 2
    assert all(not r["supplemental_chunk_ids"] for r in records)
    for record in records:
        assert record["r0_reused"] == record["variant"].startswith("C:")
    calls = [r for p in (out / "questions").glob("*/calls.jsonl") for r in rows(p)]
    assert len(calls) == 196  # 每题招募1+独立初答/讨论/主持63+C复用R0后34。
    assert all(r["request"]["task"] in {"recruit", "answer", "host"} for r in calls)
    for call in calls:
        if call["request"]["task"] in {"answer", "host"}:
            expected = 13 if any(f":B:{c}:" in call["run_id"] for c in ("literal11", "paraphrase11")) else 3
            assert len(call["request"]["evidence"]) == expected
    events = [r for p in (out / "questions").glob("*/events.jsonl") for r in rows(p)]
    assert sum(e["type"] == "injected_message" for e in events) == 6
    assert not any(e["type"] in {"supplement", "semantic_relation"} for e in events)
    before = {str(p): p.read_bytes() for p in (out / "questions").rglob("*.jsonl")}
    again = run_bc(prep, out, MockModel(), workers=1)
    assert again["status"] == "complete" and again["calls"] == 196
    assert before == {str(p): p.read_bytes() for p in (out / "questions").rglob("*.jsonl")}


def test_resume_missing_c_condition_keeps_baseline_and_other_results(tmp_path):
    prep = prepared(tmp_path, 1)
    out = tmp_path / "run"
    run_bc(prep, out, MockModel())
    path = out / "questions/q0001/results.jsonl"
    old = rows(path)
    assert old[-1]["variant"] == "C:repeated_message"
    path.write_text("\n".join(json.dumps(r) for r in old[:-1]) + "\n", encoding="utf-8")
    result = run_bc(prep, out, MockModel())
    assert result["completed"] == 5 and result["calls"] == 98 + 17
    assert rows(path)[:-1] == old[:-1]
    assert rows(path)[-1]["r0_reused"]


def test_global_budget_stops_without_losing_assignments(tmp_path):
    prep = prepared(tmp_path, 1)
    out = tmp_path / "run"
    result = run_bc(prep, out, MockModel(), max_calls=2, attempts=1)
    assert result["status"] == "incomplete" and result["calls"] == 2
    assert len(rows(out / "assignments.jsonl")) == 5
    resumed = run_bc(prep, out, MockModel(), max_calls=200)
    assert resumed["status"] == "complete" and resumed["completed"] == 5


def test_ordinary_retrieval_guard_is_hard_failure():
    with pytest.raises(RuntimeError, match="disabled"):
        NoRetrieval([]).search("unexpected runtime query")


def test_transient_team_failure_has_bounded_retry_and_keeps_failed_attempt(tmp_path):
    class TransientModel(MockModel):
        def __init__(self):
            self.state = {"failed": False}
        def complete(self, payload):
            if payload["task"] == "host" and not self.state["failed"]:
                self.state["failed"] = True
                raise TimeoutError("fixture network interruption")
            return super().complete(payload)
    prep = prepared(tmp_path, 1)
    out = tmp_path / "run"
    result = run_bc(prep, out, TransientModel(), workers=1, attempts=2)
    assert result["status"] == "complete" and result["completed"] == 5
    assert result["calls"] == 119
    failures = rows(out / "questions/q0001/failures.jsonl")
    assert len(failures) == 1 and failures[0]["variant"] == "B:base"
    assert len(rows(out / "questions/q0001/results.jsonl")) == 5


def test_ten_question_bcd_cli_uses_frozen_material_and_one_base_per_question(tmp_path, monkeypatch):
    path, questions = source(tmp_path, 10)
    shared, prep, out = (tmp_path / name for name in ("shared", "prepared", "run"))
    options = dict(tokenizer=Tokenizer(), fetcher=fetch_ten_plus, retries=0)
    assert prepare(path, shared, Model(), MockDense(), **options)["status"] == "complete"
    assert prepare_batch(path, prep, Model(), source_preparation=shared, **options)["status"] == "complete"
    assert not (prep / "shared").exists()

    model = MockModel()
    model.api_key_env = "OFFLINE_BCD_FIXTURE"
    monkeypatch.setenv(model.api_key_env, "fixture")
    monkeypatch.setattr(run_batch, "create_model", lambda *args: model)

    def frozen_child(command, *, env):
        assert command[2:4] == ["-m", "evidence_mdt.run_batch"]
        assert (out / "software_snapshot/src/evidence_mdt/run_batch.py").is_file()
        assert command[-1] == "--frozen-worker"
        return run_batch.main(command[4:])

    monkeypatch.setattr(run_batch.subprocess, "call", frozen_child)
    assert run_batch.main(["--preparation", str(prep), "--output", str(out)]) == 0
    result = json.loads((out / "summary.json").read_bytes())
    assert result["completed"] == result["planned"] == len(questions) * 6 == 60
    assert result["by_condition"] == {variant: 10 for variant in (
        "B:base", "B:literal11", "B:paraphrase11", "C:message_error", "C:repeated_message", "D:evidence_error")}
    records = rows(out / "results.jsonl")
    assert all(r["r0_reused"] == r["variant"].startswith("C:") for r in records)
    assert all(len(r["d_evidence_audit"]) == 5 for r in records if r["variant"] == "D:evidence_error")
    calls = rows(out / "questions/q0001/calls.jsonl")
    for call in calls:
        if call["request"]["task"] in {"answer", "host"}:
            sizes = {"base": 10, "literal11": 20, "paraphrase11": 20, "message_error": 10,
                     "repeated_message": 10, "evidence_error": 15}
            assert len(call["request"]["evidence"]) == sizes[call["run_id"].split(":")[-2]]
    report = json.loads((out / "report.json").read_bytes())
    assert report["d_target_probability_change"]["MedXpertQA/ordinary/D:evidence_error"]["n_valid_pairs"] == 10
    again = run_batch.run_batch(prep, out, model, workers=1)
    assert again["completed"] == 60 and again["calls"] == result["calls"]


def test_fail_fast_stops_on_invalid_response_without_repair_or_next_question(tmp_path):
    class InvalidResponse(MockModel):
        def complete(self, payload):
            result = super().complete(payload)
            if payload["task"] == "answer":
                result.content.pop("claims")
            return result

    prep, out = prepared(tmp_path, 2), tmp_path / "run"
    result = run_batch.run_batch(prep, out, InvalidResponse(), modes=("b", "c"), fail_fast=True)
    assert result["status"] == "stopped" and result["completed"] == 0 and result["calls"] == 2
    calls = rows(out / "questions/q0001/calls.jsonl")
    assert [call["request"]["task"] for call in calls] == ["recruit", "answer"]
    assert not (out / "questions/q0002/calls.jsonl").exists()
