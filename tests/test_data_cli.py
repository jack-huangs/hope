import json
from pathlib import Path

import pytest

from evidence_mdt.cli import args_parser, main
from evidence_mdt.data import local_text_corpus, normalize_question
from evidence_mdt.schema import Evidence, Question
from evidence_mdt.data import write_jsonl


def test_run_repeats_default_is_one():
    args = args_parser().parse_args([
        "run", "--questions", "questions.jsonl", "--corpus", "corpus.jsonl",
        "--output", "run", "--model-backend", "mock", "--retrieval-backend", "mock",
    ])
    assert args.repeats == 1


def test_same_named_local_documents_have_distinct_chunk_ids(tmp_path):
    for dirname in ("a", "b"):
        folder = tmp_path / dirname
        folder.mkdir()
        (folder / "note.txt").write_text("example evidence", encoding="utf-8")
    chunks = local_text_corpus(tmp_path)
    assert len({e.chunk_id for e in chunks}) == 2


def test_pubmedqa_background_kept_but_answer_excluded():
    question = normalize_question({"_source_id": "123", "QUESTION": "Does it work?",
                                   "CONTEXTS": ["Background one", "Background two"],
                                   "LONG_ANSWER": "LEAK", "final_decision": "yes",
                                   "reasoning_free_pred": "LEAK"}, "PubMedQA", 0)
    assert "Background one\nBackground two" == question.context
    assert "LEAK" not in json.dumps(question.public())


def test_ordinary_assignment_manifest_and_resume(tmp_path):
    questions = tmp_path / "questions.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "run"
    write_jsonl(questions, [Question("fixture", "MedQA", "alpha?", {"A": "alpha", "B": "beta"}, "A")])
    write_jsonl(corpus, [Evidence("c1", "doc", "alpha evidence", "local")])
    argv = ["run", "--questions", str(questions), "--corpus", str(corpus),
            "--output", str(output), "--model-backend", "mock", "--retrieval-backend", "mock",
            "--groups", "ordinary", "--repeats", "2"]
    main(argv)
    assigned = [json.loads(x) for x in (output / "assignments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(assigned) == 2
    assert all(a["group"] == "ordinary" for a in assigned)
    first_results = (output / "results.jsonl").read_text(encoding="utf-8")
    main(argv)
    assert (output / "results.jsonl").read_text(encoding="utf-8") == first_results
    results = [json.loads(x) for x in first_results.splitlines()]
    assert all(r["fingerprint"] == assigned[0]["fingerprint"] for r in results)


def test_budget_failure_keeps_full_assignment_denominator(tmp_path):
    questions = tmp_path / "questions.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "run"
    write_jsonl(questions, [Question(f"q{i}", "MedQA", "alpha?", {"A": "a", "B": "b"}, "A")
                            for i in range(2)])
    write_jsonl(corpus, [Evidence("c1", "doc", "alpha evidence", "local")])
    with pytest.raises(RuntimeError, match="budget"):
        main(["run", "--questions", str(questions), "--corpus", str(corpus),
              "--output", str(output), "--model-backend", "mock", "--retrieval-backend", "mock",
              "--groups", "ordinary", "--repeats", "1", "--max-calls", "1"])
    assigned = [json.loads(x) for x in (output / "assignments.jsonl").read_text(encoding="utf-8").splitlines()]
    failures = [json.loads(x) for x in (output / "failures.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(assigned) == 2
    assert len(failures) == 1
    assert failures[0]["run_id"] == assigned[0]["run_id"]
