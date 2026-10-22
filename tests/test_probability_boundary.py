"""真实失败只提取概率向量；其余输入均为合成离线夹具，不使用医学答案。"""
import copy
import itertools

import pytest

from evidence_mdt.metrics import _valid_probabilities
from evidence_mdt.models import Model, ModelResult
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import Runner
from evidence_mdt.schema import Question, probability_sum_within_tolerance, validate_answer


REAL_VECTORS = [
    dict(zip("ABCDEFGHIJ", [0.03, 0.01, 0.01, 0.01, 0.01, 0.85, 0.02, 0.02, 0.02, 0.01])),
    dict(zip("ABCDEFGHIJ", [0.02, 0.01, 0.01, 0.01, 0.01, 0.86, 0.02, 0.02, 0.02, 0.01])),
]


def response(probs):
    return {"answer": "A", "probabilities": probs, "citations": [],
            "claims": [], "opinion": "synthetic fixture"}


class VectorModel(Model):
    name = "offline-probability-fixture"

    def __init__(self, value, repaired=None):
        self.value = value
        self.repaired = repaired
        self.tasks = []

    def complete(self, payload):
        self.tasks.append(payload["task"])
        value = self.repaired if payload["task"] == "repair" else self.value
        assert value is not None, "Unexpected paid repair path"
        return ModelResult(copy.deepcopy(value), {"fixture": True}, {"total_tokens": 0})


def runner_turn(tmp_path, model, labels):
    runner = Runner(model, Retriever([], MockDense()), tmp_path)
    question = Question("synthetic", "fixture", "Choose a symbol",
                        {label: label for label in labels})
    return runner._turn("fixture-run", question, [], [], "expert", 0, 0, "fixture-turn")


@pytest.mark.parametrize("probs", REAL_VECTORS)
def test_real_boundary_vectors_pass_without_repair_or_numeric_change(tmp_path, probs):
    original = response(copy.deepcopy(probs))
    model = VectorModel(original)
    turn = runner_turn(tmp_path, model, probs)
    assert model.tasks == ["answer"]
    assert turn.raw_response == original
    assert turn.probabilities == probs
    assert _valid_probabilities({**original, "options": list(probs), "gold": "A"})


@pytest.mark.parametrize("values", [[0.85, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
                                   [0.87, 0.03, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]])
def test_inclusive_boundaries_ignore_order(values):
    # Rotate/reverse all positions, plus unique permutations of a shorter vector.
    for n in range(len(values)):
        rotated = values[n:] + values[:n]
        assert probability_sum_within_tolerance(rotated)
        assert probability_sum_within_tolerance(reversed(rotated))
    for perm in itertools.permutations([values[0], 0.03, round(sum(values[2:]), 2)]):
        assert probability_sum_within_tolerance(perm)


@pytest.mark.parametrize("probs", [
    {"A": 0.989, "B": 0}, {"A": 1, "B": 0.011},
    {"A": 0.989999999999, "B": 0}, {"A": 1, "B": 0.010000000001},
    {"A": 0, "B": 0}, {"A": -0.01, "B": 1},
    {"A": True, "B": 0}, {"A": float("nan"), "B": 0},
    {"A": float("inf"), "B": 0}, {"A": 1}, {},
])
def test_invalid_vectors_rejected_by_all_paths_without_repair(tmp_path, probs):
    original = response(probs)
    with pytest.raises(ValueError):
        validate_answer(original, {"A": "a", "B": "b"}, set(), set())
    assert not _valid_probabilities({**original, "options": ["A", "B"], "gold": "A"})
    model = VectorModel(original, response({"A": 1, "B": 0}))
    with pytest.raises(ValueError):
        runner_turn(tmp_path, model, ["A", "B"])
    assert model.tasks == ["answer"]


def test_empty_sum_is_not_valid():
    assert not probability_sum_within_tolerance([])


def test_valid_boundary_can_repair_structure_but_not_probabilities(tmp_path):
    original = response({"A": 0.99, "B": 0})
    original["claims"] = [{"text": "fixture", "answer": "A", "conditions": []}]
    repaired = copy.deepcopy(original)
    repaired["claims"][0]["conditions"] = ""
    model = VectorModel(original, repaired)
    turn = runner_turn(tmp_path, model, ["A", "B"])
    assert model.tasks == ["answer", "repair"]
    assert turn.probabilities == original["probabilities"]
    repaired["probabilities"] = {"A": 1, "B": 0}
    with pytest.raises(ValueError, match="repair changed"):
        runner_turn(tmp_path / "changed", VectorModel(original, repaired), ["A", "B"])
