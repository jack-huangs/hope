import importlib.util
import json
from pathlib import Path

import pytest

from evidence_mdt.data import write_jsonl
from evidence_mdt.models import ModelResult
from evidence_mdt.schema import Evidence, Question


SPEC = importlib.util.spec_from_file_location(
    "prepare_question_corpus", Path(__file__).resolve().parents[1] / "scripts/prepare_question_corpus.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CountingModel:
    name = "mock"

    def __init__(self):
        self.payloads = []

    def complete(self, payload):
        self.payloads.append(payload)
        return ModelResult({"queries": ["term one", "term two"]}, {"model": "fixture"},
                           {"total_tokens": 7}, payload)


def questions(tmp_path):
    path = tmp_path / "input.jsonl"
    write_jsonl(path, [Question("111", "PubMedQA", "Secret study?", {"yes": "yes", "no": "no"},
                                "yes", "Visible background")])
    return path


def test_resume_after_fetch_failure_without_new_model_call(tmp_path, monkeypatch):
    model = CountingModel()
    monkeypatch.setattr(module, "create_model", lambda backend: model)
    attempts = []

    def fetch(query, max_records):
        attempts.append(query)
        if query == "term two" and attempts.count(query) == 1:
            raise TimeoutError("fixture timeout")
        evidence = [Evidence("PMID:111:abstract", "PMID:111", "target conclusion", content_hash="target"),
                    Evidence("PMID:222:abstract", "PMID:222", "independent abstract", content_hash="safe")]
        return evidence, {"query": query, "ids": ["111", "222"],
                          "raw_xml_sha256": module.digest(b"<root/>"),
                          "article_metadata": [{"pmid": "111"}, {"pmid": "222"}], "_raw_xml": "<root/>"}

    monkeypatch.setattr(module, "pubmed_snapshot", fetch)
    source = questions(tmp_path)
    output = tmp_path / "snapshot"
    first = module.prepare(source, output, 8, 3, 100, "mock", pause_seconds=0)
    assert first["status"] == "incomplete"
    assert first["records"][0]["snapshots"][1]["status"] == "fetch_failed"
    assert len(model.payloads) == 1
    assert "gold" not in json.dumps(model.payloads)
    assert "target conclusion" not in (output / "corpus.jsonl").read_text(encoding="utf-8")

    second = module.prepare(source, output, 8, 3, 100, "mock", pause_seconds=0)
    assert second["status"] == "complete"
    assert len(model.payloads) == 1
    assert attempts == ["term one", "term two", "term two"]
    assert len(second["chunk_sources"]["PMID:222:abstract"]) == 2
    assert second["records"][0]["snapshots"][0]["excluded_target_pmids"] == ["PMID:111:abstract"]
    assert second["model_calls"] == 1 and second["known_tokens"] == 7
    assert (output / "q0001.query1.raw.xml").read_text(encoding="utf-8") == "<root/>"
    assert module.prepare(source, output, 8, 3, 100, "mock", pause_seconds=0) == second
    assert len(attempts) == 3


def test_budget_stop_and_config_guard(tmp_path, monkeypatch):
    model = CountingModel()
    monkeypatch.setattr(module, "create_model", lambda backend: model)
    monkeypatch.setattr(module, "pubmed_snapshot", lambda query, max_records: ([], {"query": query, "ids": []}))
    source = questions(tmp_path)
    output = tmp_path / "snapshot"
    stopped = module.prepare(source, output, 8, 0, 100, "mock", pause_seconds=0)
    assert stopped["status"] == "budget_stopped" and stopped["model_calls"] == 0
    done = module.prepare(source, output, 8, 1, 100, "mock", pause_seconds=0)
    assert done["status"] == "complete"
    with pytest.raises(ValueError, match="different input/code/model"):
        module.prepare(source, output, 7, 1, 100, "mock", pause_seconds=0)


def test_failed_model_response_counts_known_usage(tmp_path, monkeypatch):
    class FailedModel(CountingModel):
        def complete(self, payload):
            error = ValueError("bad response")
            error.usage = {"total_tokens": 9}
            error.raw = {"usage": error.usage}
            raise error

    monkeypatch.setattr(module, "create_model", lambda backend: FailedModel())
    result = module.prepare(questions(tmp_path), tmp_path / "snapshot", 8, 1, 100, "mock", pause_seconds=0)
    assert result["status"] == "incomplete"
    assert result["model_calls"] == 1 and result["known_tokens"] == 9
    assert result["records"][0]["status"] == "query_failed"
