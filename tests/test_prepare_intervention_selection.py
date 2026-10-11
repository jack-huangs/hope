import json

import pytest

from evidence_mdt.data import load_questions, write_jsonl
from evidence_mdt.interventions import subset25
from scripts.prepare_intervention_selection import prepare


def fixture(tmp_path):
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    for filename, dataset, count in (("MedQA.jsonl", "MedQA", 30),
                                     ("MedXpertQA.jsonl", "MedXpertQA", 20),
                                     ("PubMedQA.jsonl", "PubMedQA", 20)):
        write_jsonl(pilot / filename, [
            {"question_id": f"{dataset}-{i}", "dataset": dataset, "stem": f"Question {i}",
             "options": {"A": "alpha", "B": "beta"}, "gold": "A", "context": ""}
            for i in range(count)
        ])
    return pilot


def test_two_branches_match_cli_subset_without_gold(tmp_path):
    pilot = fixture(tmp_path)
    output = tmp_path / "selected"
    records = prepare(pilot, output)
    assert set(records) == {"medqa_branch", "medxpertqa_branch"}
    for name, clinical_file in (("medqa_branch", "MedQA.jsonl"),
                                ("medxpertqa_branch", "MedXpertQA.jsonl")):
        saved = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        expected = subset25(load_questions(pilot / clinical_file)
                            + load_questions(pilot / "PubMedQA.jsonl"))
        assert [(x["question_id"], x["dataset"]) for x in saved["selected"]] == [
            (q.question_id, q.dataset) for q in expected]
        assert len(saved["selected"]) == 25
        assert saved["branch_selected"] is False
        assert all(len(x["public_question_sha256"]) == 64 for x in saved["selected"])
        assert "gold" not in json.dumps(saved).lower()
        assert "answer" not in json.dumps(saved).lower()
    pqa_a = [x["question_id"] for x in records["medqa_branch"]["selected"]
             if x["dataset"] == "PubMedQA"]
    pqa_b = [x["question_id"] for x in records["medxpertqa_branch"]["selected"]
             if x["dataset"] == "PubMedQA"]
    assert pqa_a == pqa_b
    original = {p.name: p.read_bytes() for p in output.glob("*.json")}
    assert prepare(pilot, output) == records
    assert {p.name: p.read_bytes() for p in output.glob("*.json")} == original


def test_changed_source_cannot_overwrite_frozen_selection(tmp_path):
    pilot = fixture(tmp_path)
    output = tmp_path / "selected"
    prepare(pilot, output)
    path = pilot / "MedQA.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["stem"] = "Changed public question"
    write_jsonl(path, rows)
    before = {p.name: p.read_bytes() for p in output.glob("*.json")}
    with pytest.raises(ValueError, match="differs"):
        prepare(pilot, output)
    assert {p.name: p.read_bytes() for p in output.glob("*.json")} == before
