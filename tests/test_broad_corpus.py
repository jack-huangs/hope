import importlib.util
import json
import sys
from pathlib import Path

from evidence_mdt.models import ModelResult


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("prepare_broad_corpus", SCRIPTS / "prepare_broad_corpus.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class FixtureModel:
    name = "mock"

    def __init__(self):
        self.payloads = []

    def complete(self, payload):
        self.payloads.append(payload)
        content = {"queries": ["cranial nerve AND palsy", "diplopia AND ptosis"]}
        return ModelResult(content, {"model": "fixture", "choices": [{"message": {"content": json.dumps(content)}}]},
                           {"total_tokens": 11}, payload)


def fixture_source(tmp_path):
    source = tmp_path / "parent"
    source.mkdir()
    questions = [{"question_id": "123", "dataset": "PubMedQA", "stem": "visible stem",
                  "options": {"yes": "yes", "no": "no"}, "context": "LEAK_CONTEXT"},
                 {"question_id": "m2", "dataset": "MedQA", "stem": "control", "options": {"A": "a", "B": "b"},
                  "context": ""}]
    (source / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
    evidence = {"chunk_id": "PMID:456:abstract", "document_id": "PMID:456", "text": "old abstract",
                "source": "https://pubmed.ncbi.nlm.nih.gov/456/", "position": "abstract", "retrieved_at": "",
                "content_hash": module.base.sha(b"old abstract"), "derived_from": None, "title": "old"}
    (source / "corpus.jsonl").write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    records = [{"question_id": "123", "dataset": "PubMedQA",
                "public_question_sha256": module.base.sha(module.base.canonical(questions[0])),
                "snapshots": [{"status": "empty", "kept_chunk_ids": []}]},
               {"question_id": "m2", "dataset": "MedQA",
                "public_question_sha256": module.base.sha(module.base.canonical(questions[1])),
                "snapshots": [{"status": "ok", "kept_chunk_ids": ["c1", "c2", "c3"]}]}]
    parent = {"status": "complete", "config": {"sort": "relevance"}, "records": records,
              "question_ids": ["123", "m2"], "forbidden_pubmedqa_pmids": ["123"],
              "source_model_calls": 2, "source_known_tokens": 20, "corpus_chunks": 1,
              "corpus_sha256": module.base.sha((source / "corpus.jsonl").read_bytes()),
              "chunk_sources": {"PMID:456:abstract": [{"question_id": "m2", "query_index": 0, "query": "old"}]}}
    (source / "preparation_manifest.json").write_text(json.dumps(parent), encoding="utf-8")
    return source


XML = (b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><Abstract>"
       b"<AbstractText>target conclusion</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle>"
       b"<PubmedArticle><MedlineCitation><PMID>789</PMID><Article><Abstract>"
       b"<AbstractText>new abstract</AbstractText></Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>")


def test_stem_only_budget_and_resume(tmp_path, monkeypatch):
    source = fixture_source(tmp_path)
    model = FixtureModel()
    monkeypatch.setattr(module, "create_model", lambda backend: model)
    fail = True
    calls = []

    def request(url, filename, timeout):
        nonlocal fail
        calls.append(url)
        if "esearch" in url and "diplopia" in url and fail:
            raise TimeoutError("fixture")
        raw = b'{"esearchresult":{"idlist":["123","789"]}}' if "esearch" in url else XML
        filename.write_bytes(raw)
        return raw

    monkeypatch.setattr(module.base, "request_once", request)
    output = tmp_path / "broad"
    first = module.prepare(source, output, backend="mock", max_model_calls=1,
                           retries=0, pause=0, backoff=0)
    assert first["status"] == "incomplete" and first["new_model_calls"] == 1
    assert first["selected_questions"][0]["question_id"] == "123"
    assert first["records"][1]["status"] == "not_triggered"
    assert "LEAK_CONTEXT" not in json.dumps(model.payloads)
    assert set(model.payloads[0]["question"]) == {"question_id", "dataset", "stem"}
    assert "target conclusion" not in (output / "corpus.jsonl").read_text(encoding="utf-8")
    fail = False
    done = module.prepare(source, output, backend="mock", max_model_calls=1,
                          retries=0, pause=0, backoff=0)
    assert done["status"] == "complete" and done["new_model_calls"] == 1
    assert len(model.payloads) == 1
    assert done["corpus_chunks"] == 2
    assert done["records"][0]["snapshots"][0]["excluded_target_pmids"] == ["PMID:123:abstract"]


def test_zero_budget_keeps_fixed_denominator(tmp_path, monkeypatch):
    source = fixture_source(tmp_path)
    model = FixtureModel()
    monkeypatch.setattr(module, "create_model", lambda backend: model)
    result = module.prepare(source, tmp_path / "broad", backend="mock", max_model_calls=0,
                            retries=0, pause=0, backoff=0)
    assert result["status"] == "budget_stopped" and result["selected_questions"][0]["question_id"] == "123"
    assert result["new_model_calls"] == 0 and not model.payloads
