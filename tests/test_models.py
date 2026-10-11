import json

import pytest

from evidence_mdt.models import Model, ModelResponseError, ModelResult, OpenAICompatibleModel
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import Runner
from evidence_mdt.schema import Budget, Evidence, Question, validate_answer


class BadJsonModel(Model):
    name = "bad-json"

    def complete(self, payload):
        raise ModelResponseError("invalid JSON", {"usage": {"total_tokens": 13}},
                                 {"model": self.name, "messages": []})


def test_bad_json_still_counts_call_and_tokens(tmp_path):
    runner = Runner(BadJsonModel(), Retriever([Evidence("e", "d", "text")], MockDense()),
                    tmp_path, Budget(max_calls=1, max_tokens=100))
    with pytest.raises(ModelResponseError):
        runner.invoke("run", {"task": "answer"})
    assert runner.budget.calls == 1
    assert runner.budget.tokens == 13
    item = json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8"))
    assert item["usage"]["total_tokens"] == 13


def test_nonfinite_probability_and_bad_claim_citation_rejected():
    answer = {"answer": "A", "probabilities": {"A": float("nan"), "B": 0.5},
              "citations": [], "claims": [], "opinion": "text"}
    with pytest.raises(ValueError, match="probability"):
        validate_answer(answer, {"A": "a", "B": "b"}, set(), set())
    answer["probabilities"] = {"A": 0.5, "B": 0.5}
    answer["claims"] = [{"text": "assertion", "answer": "A", "citation_ids": "e"}]
    with pytest.raises(ValueError, match="claim citation"):
        validate_answer(answer, {"A": "a", "B": "b"}, {"e"}, set())


def test_provider_request_parameters_and_actual_revision(monkeypatch):
    observed = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"model": "resolved-revision", "usage": {"total_tokens": 9},
                               "choices": [{"message": {"content": '{"relation":"unknown"}'}}]}).encode()

    def fake_open(request, timeout):
        observed.update(json.loads(request.data))
        return Response()

    monkeypatch.setenv("TEST_API_KEY", "fixture-only")
    monkeypatch.setattr("urllib.request.urlopen", fake_open)
    model = OpenAICompatibleModel("requested-model", "https://fixture.invalid", "TEST_API_KEY",
                                  answer_temperature=0.2, max_tokens=123, disable_thinking=True)
    result = model.complete({"task": "semantic", "a": "a", "b": "b"})
    assert observed["max_tokens"] == 123
    assert observed["temperature"] == 0
    assert observed["thinking"] == {"type": "disabled"}
    assert result.raw["model"] == "resolved-revision"
    assert result.usage["total_tokens"] == 9


class FormatRepairModel(Model):
    name = "format-fixture"

    def __init__(self, bad_json=False):
        self.bad_json = bad_json
        self.payloads = []

    def complete(self, payload):
        self.payloads.append(payload)
        if payload["task"] == "repair":
            content = {"answer": "D", "probabilities": {"C": 0.2, "D": 0.8},
                       "citations": ["e"], "claims": [{"text": "fixture reason", "answer": "D",
                       "citation_ids": ["e"], "polarity": "support", "conditions": "adults"}],
                       "opinion": "fixture"}
            return ModelResult(content, {"model": self.name}, {"total_tokens": 7})
        if self.bad_json:
            raise ModelResponseError("invalid JSON", {"choices": [{"message": {"content": "{"}}],
                                                       "usage": {"total_tokens": 11}}, {})
        content = {"answer": "Nitrofurantoin", "probabilities": {"C": 0.2, "D": 0.8},
                   "citations": ["e"], "claims": [{"text": "fixture reason",
                   "answer": "Nitrofurantoin", "citation_ids": ["e"],
                   "polarity": "support", "conditions": ["adults"]}], "opinion": "fixture"}
        return ModelResult(content, {"model": self.name}, {"total_tokens": 11})


@pytest.mark.parametrize("bad_json", [False, True])
def test_single_repair_call_keeps_original_response_and_usage(tmp_path, bad_json):
    model = FormatRepairModel(bad_json)
    runner = Runner(model, Retriever([Evidence("e", "d", "fixture")], MockDense()), tmp_path)
    question = Question("q", "Fixture", "Choose", {"C": "other", "D": "Nitrofurantoin"}, "C")
    turn = runner._turn("run", question, [Evidence("e", "d", "fixture")], [],
                        "expert", 0, 0, "turn")
    assert turn.answer == "D"
    assert turn.probabilities == {"C": 0.2, "D": 0.8}
    assert runner.budget.calls == 2
    assert runner.budget.tokens == 18
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 2
    assert calls[0]["response"]
    assert calls[1]["request"]["task"] == "repair"
    assert calls[1]["request"]["validation_error"]
    assert "gold" not in json.dumps(calls[1]["request"])
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(e.get("type") == "response_validation_failed" for e in events)
