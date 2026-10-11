import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "audit_initial_retrieval", Path(__file__).resolve().parents[1] / "scripts/audit_initial_retrieval.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def fixture_snapshot(tmp_path):
    folder = tmp_path / "snapshot"
    folder.mkdir()
    question = {"question_id": "123", "dataset": "PubMedQA", "stem": "alpha treatment?",
                "options": {"yes": "yes", "no": "no"}, "context": "public background"}
    (folder / "questions.jsonl").write_text(json.dumps(question) + "\n", encoding="utf-8")
    corpus = [{"chunk_id": "PMID:456:abstract", "document_id": "PMID:456", "text": "alpha treatment evidence",
               "source": "https://pubmed.ncbi.nlm.nih.gov/456/", "position": "abstract", "retrieved_at": "",
               "content_hash": "abc", "derived_from": None, "title": "Alpha treatment"}]
    corpus_path = folder / "corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(e) + "\n" for e in corpus), encoding="utf-8")
    manifest = {"status": "complete", "question_ids": ["123"], "corpus_sha256": module.sha(corpus_path.read_bytes()),
                "forbidden_pubmedqa_pmids": ["123"],
                "chunk_sources": {"PMID:456:abstract": [{"question_id": "123", "query_index": 0, "query": "alpha"}]}}
    (folder / "preparation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_mock_audit_has_full_trace_without_gold(tmp_path):
    snapshot = fixture_snapshot(tmp_path)
    output = tmp_path / "audit"
    summary = module.audit(snapshot, output, "mock")
    assert summary["questions"] == 1 and summary["total_citations"] == 1
    row = module.read_jsonl(output / "retrieval_trace.jsonl")[0]
    assert row["top6"][0]["evidence"]["chunk_id"] == "PMID:456:abstract"
    assert row["top6"][0]["from_this_question"] is True
    assert row["top6"][0]["scores"]["reranked"] is not None
    assert all(key in row["trace"] for key in ("bm25_top50", "dense_top50", "rrf_top30", "reranked"))
    assert "gold" not in json.dumps(row)
    assert module.audit(snapshot, output, "mock") == summary


def test_rejects_answer_field_and_mismatched_corpus(tmp_path):
    snapshot = fixture_snapshot(tmp_path)
    output = tmp_path / "audit"
    question_file = snapshot / "questions.jsonl"
    question = json.loads(question_file.read_text(encoding="utf-8"))
    question["gold"] = "yes"
    question_file.write_text(json.dumps(question), encoding="utf-8")
    with pytest.raises(ValueError, match="answer fields"):
        module.audit(snapshot, output, "mock")
    del question["gold"]
    question_file.write_text(json.dumps(question), encoding="utf-8")
    (snapshot / "corpus.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Corpus hash"):
        module.audit(snapshot, output, "mock")


def test_frozen_short_query_manifest(tmp_path):
    snapshot = fixture_snapshot(tmp_path)
    question = json.loads((snapshot / "questions.jsonl").read_text(encoding="utf-8"))
    manifest_file = tmp_path / "queries.jsonl"
    item = {"question_id": "123", "dataset": "PubMedQA",
            "public_question_sha256": module.sha(json.dumps(question, sort_keys=True,
                                         ensure_ascii=False, separators=(",", ":")).encode()),
            "retrieval_query": "alpha ; treatment"}
    manifest_file.write_text(json.dumps(item) + "\n", encoding="utf-8")
    result = module.audit(snapshot, tmp_path / "short_audit", "mock", query_manifest=manifest_file)
    assert result["questions"] == 1
    trace = module.read_jsonl(tmp_path / "short_audit" / "retrieval_trace.jsonl")[0]
    assert trace["retrieval_query"] == "alpha ; treatment"
    item["public_question_sha256"] = "bad"
    manifest_file.write_text(json.dumps(item) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        module.audit(snapshot, tmp_path / "other_audit", "mock", query_manifest=manifest_file)
