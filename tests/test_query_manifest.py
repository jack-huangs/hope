import hashlib
import json

import pytest

from evidence_mdt.data import write_jsonl
from evidence_mdt.cli import main
from evidence_mdt.query_manifest import (initial_queries, public_question_sha256,
                                         validate_medcpt_query_lengths)
from evidence_mdt.schema import Question


def fixture(tmp_path):
    questions = [Question("q1", "MedQA", "Long public stem", {"A": "alpha", "B": "beta"}, "A"),
                 Question("q2", "PubMedQA", "Other public stem", {"yes": "yes", "no": "no"}, "yes")]
    rows = [{"question_id": q.question_id, "dataset": q.dataset,
             "public_question_sha256": public_question_sha256(q),
             "retrieval_query": f"short query {index}"} for index, q in enumerate(questions)]
    path = tmp_path / "queries.jsonl"
    write_jsonl(path, rows)
    return questions, rows, path


def test_public_hash_and_short_query_lookup(tmp_path):
    questions, rows, path = fixture(tmp_path)
    assert initial_queries(questions, path) == {"q1": "short query 0", "q2": "short query 1"}
    assert initial_queries(questions, None)["q1"] == "Long public stem "
    altered_gold = Question("q1", "MedQA", "Long public stem", {"A": "alpha", "B": "beta"}, "B")
    assert public_question_sha256(altered_gold) == rows[0]["public_question_sha256"]
    changed_stem = Question("q1", "MedQA", "Changed stem", {"A": "alpha", "B": "beta"}, "A")
    assert public_question_sha256(changed_stem) != rows[0]["public_question_sha256"]


def test_full_manifest_can_serve_medqa_prescreen_subset(tmp_path):
    questions, _rows, path = fixture(tmp_path)
    assert initial_queries(questions[:1], path) == {"q1": "short query 0"}


def test_cli_rejects_medcpt_model_revision_mismatch_before_calls(tmp_path, monkeypatch):
    questions, _rows, query_path = fixture(tmp_path)
    questions_path = tmp_path / "questions.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    manifest_path = tmp_path / "index.json"
    write_jsonl(questions_path, questions)
    write_jsonl(corpus_path, [{"chunk_id": "e", "document_id": "d", "text": "evidence", "source": "local"}])
    manifest_path.write_text(json.dumps({"sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                                         "backend": "medcpt", "model_revisions": {"query": "old"}}),
                             encoding="utf-8")

    class FakeDense:
        revisions = {"query": "new"}
        qtok = staticmethod(lambda _query, truncation: {"input_ids": list(range(64))})

    monkeypatch.setattr("evidence_mdt.cli.dense_backend", lambda _name: FakeDense())
    with pytest.raises(ValueError, match="revisions"):
        main(["run", "--questions", str(questions_path), "--corpus", str(corpus_path),
              "--output", str(tmp_path / "run"), "--model-backend", "mock",
              "--retrieval-backend", "medcpt", "--index-manifest", str(manifest_path),
              "--query-manifest", str(query_path)])
    assert not (tmp_path / "run" / "calls.jsonl").exists()


def test_explicit_medcpt_query_limit_includes_special_tokens():
    seen = []

    def tokenizer(query, truncation):
        seen.append(truncation)
        return {"input_ids": list(range(64 if query == "fits" else 65))}

    with pytest.raises(ValueError, match="q-bad.*65"):
        validate_medcpt_query_lengths({"q-ok": "fits", "q-bad": "too long"}, tokenizer)
    assert seen == [False, False]


@pytest.mark.parametrize("change,expected", [
    ("hidden", "four public fields"),
    ("duplicate", "exactly once"),
    ("missing", "exactly once"),
    ("wrong_hash", "mismatch"),
    ("wrong_dataset", "mismatch"),
    ("empty", "Empty"),
])
def test_manifest_rejects_hidden_or_unmatched_data(tmp_path, change, expected):
    questions, rows, path = fixture(tmp_path)
    if change == "hidden":
        rows[0]["gold"] = "A"
    elif change == "duplicate":
        rows.append(dict(rows[0]))
    elif change == "missing":
        rows.pop()
    elif change == "wrong_hash":
        rows[0]["public_question_sha256"] = "wrong"
    elif change == "wrong_dataset":
        rows[0]["dataset"] = "Other"
    elif change == "empty":
        rows[0]["retrieval_query"] = " "
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match=expected):
        initial_queries(questions, path)
