"""B 先标注全部冻结证据，再按支持关系确定随机抽样池；全部离线。"""
from copy import deepcopy
import random

import pytest

from evidence_mdt.schema import Evidence, Question
from preparation.materials import prepare_material


def fixture():
    question = Question("selection-fixture", "MedXpertQA", "Choose an option.",
                        {"A": "alpha", "B": "beta"}, "A")
    evidence = [Evidence(f"e{i}", f"d{i}", f"Original passage {i}.") for i in range(10)]
    return question, evidence


def annotations(base, supported):
    return [{"chunk_id": e.chunk_id, "target_answer": supported.get(e.chunk_id),
             "reason": f"Support assessment for {e.chunk_id}."} for e in base]


def generator(rows, calls):
    def call(stage, instruction, data):
        calls.append((stage, instruction, data))
        if stage == "b_annotation":
            return {"annotations": deepcopy(rows)}
        assert stage == "b"
        # Extra output cannot override the previously saved support annotation.
        return {"paraphrases": [f"Whole passage rewording number {i}." for i in range(10)],
                "target_answer": "invalid"}
    return call


def test_b_uniform_supported_pool_includes_any_supported_option_and_ignores_model_order():
    question, base = fixture()
    rows = annotations(base, {"e1": "A", "e8": "B"})
    seen = set()
    before = deepcopy(base)
    for seed in range(40):
        calls = []
        spec = prepare_material(question, base, "b", seed, generator(list(reversed(rows)), calls))
        expected = random.Random(f"{seed}:{question.question_id}:b").choice(["e1", "e8"])
        assert spec["b_target_id"] == expected
        assert spec["b_target_candidate_ids"] == ["e1", "e8"]
        assert spec["target_answer"] == {"e1": "A", "e8": "B"}[expected]
        assert spec["b_target_selection"] == "supported"
        assert spec["b_evidence_annotations"] == rows
        assert [c[0] for c in calls] == ["b_annotation", "b"]
        assert calls[0][2]["question"] == question.public()
        assert "gold" not in calls[0][2]["question"]
        assert [e["chunk_id"] for e in calls[0][2]["evidence"]] == [e.chunk_id for e in base]
        assert set(calls[1][2]) == {"original"}
        assert calls[1][2]["original"]["chunk_id"] == expected
        seen.add(expected)
    assert seen == {"e1", "e8"}
    assert base == before


def test_b_all_null_falls_back_to_all_frozen_evidence_and_keeps_null():
    question, base = fixture()
    rows = annotations(base, {})
    for seed in range(20):
        spec = prepare_material(question, base, "b", seed, generator(rows, []))
        pool = [e.chunk_id for e in base]
        expected = random.Random(f"{seed}:{question.question_id}:b").choice(pool)
        assert spec["b_target_id"] == expected
        assert spec["b_target_candidate_ids"] == pool
        assert spec["b_target_selection"] == "all_null_fallback"
        assert spec["target_answer"] is None
        assert spec["b_target_reason"] == f"Support assessment for {expected}."
        assert len(spec["b_paraphrases"]) == 10


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unknown", "option", "absent", "reason"])
def test_invalid_annotations_never_select_or_generate_paraphrases(corruption):
    question, base = fixture()
    rows = annotations(base, {"e1": "A"})
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif corruption == "unknown":
        rows[-1]["chunk_id"] = "outside-frozen-snapshot"
    elif corruption == "option":
        rows[-1]["target_answer"] = "Z"
    elif corruption == "absent":
        rows[-1].pop("target_answer")
    else:
        rows[-1]["reason"] = " "
    calls, saved = [], []
    with pytest.raises(ValueError):
        prepare_material(question, base, "b", 123, generator(rows, calls),
                         checkpoint=lambda spec: saved.append(deepcopy(spec)))
    assert [c[0] for c in calls] == ["b_annotation"]
    assert not saved


def test_annotation_checkpoint_and_target_survive_paraphrase_failure_and_resume():
    question, base = fixture()
    rows = annotations(base, {"e1": "B", "e8": "A"})
    saved = []
    def failing(stage, instruction, data):
        if stage == "b_annotation":
            return {"annotations": rows}
        raise TimeoutError("temporary paraphrase failure")
    with pytest.raises(TimeoutError):
        prepare_material(question, base, "b", 123, failing,
                         checkpoint=lambda spec: saved.append(deepcopy(spec)))
    assert "b_evidence_annotations" in saved[0] and "b_target_id" not in saved[0]
    assert saved[-1]["b_target_id"] in {"e1", "e8"}
    for checkpoint in saved:  # Resume both before and after assignment was saved.
        calls = []
        spec = prepare_material(question, base, "b", 123, generator([], calls), previous=checkpoint)
        assert [c[0] for c in calls] == ["b"]
        assert spec["b_target_id"] == saved[-1]["b_target_id"]
        assert spec["target_answer"] == saved[-1]["target_answer"]
        assert spec["b_evidence_annotations"] == rows


def test_single_supported_passage_is_always_selected_and_saved_mismatch_is_rejected():
    question, base = fixture()
    rows = annotations(base, {"e9": "B"})
    spec = prepare_material(question, base, "b", 123, generator(rows, []))
    assert spec["b_target_id"] == "e9" and spec["target_answer"] == "B"
    spec["b_target_id"] = "e0"
    calls = []
    with pytest.raises(ValueError, match="Saved B selection differs"):
        prepare_material(question, base, "b", 123, generator(rows, calls), previous=spec)
    assert not calls
