import importlib.util
import json
import runpy
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest


SPEC = importlib.util.spec_from_file_location(
    "prepare_relevance_corpus", Path(__file__).resolve().parents[1] / "scripts/prepare_relevance_corpus.py")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def source_fixture(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    public = {"question_id": "123", "dataset": "PubMedQA", "stem": "Visible question",
              "options": {"yes": "yes", "no": "no"}, "context": "Visible background"}
    (source / "questions.jsonl").write_text(json.dumps(public) + "\n", encoding="utf-8")
    row = {"question_id": "123", "dataset": "PubMedQA", "public_question_sha256": module.sha(module.canonical(public)),
           "status": "ready", "queries": ["fixed alpha", "fixed beta"],
           "model_calls": [{"raw_response": {"choices": [{"message": {"content":
                           json.dumps({"queries": ["fixed alpha", "fixed beta"]})}}]}}],
           "snapshots": [{"query": "fixed alpha"}, {"query": "fixed beta"}]}
    parent = {"status": "complete", "question_ids": ["123"], "records": [row],
              "forbidden_pubmedqa_pmids": ["123"], "model_calls": 1, "known_tokens": 17}
    (source / "preparation_manifest.json").write_text(json.dumps(parent), encoding="utf-8")
    return source


XML = (b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article>"
       b"<ArticleTitle>Target</ArticleTitle><Abstract><AbstractText>target conclusion</AbstractText></Abstract>"
       b"</Article></MedlineCitation></PubmedArticle><PubmedArticle><MedlineCitation><PMID>456</PMID>"
       b"<Article><ArticleTitle>Safe</ArticleTitle><Abstract><AbstractText>independent abstract</AbstractText>"
       b"</Abstract></Article></MedlineCitation></PubmedArticle></PubmedArticleSet>")


def test_fixed_queries_retries_raw_and_resume(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    output = tmp_path / "relevance"
    calls = []
    fail_beta = True

    def fake_request(url, filename, timeout):
        nonlocal fail_beta
        calls.append(url)
        assert parse_qs(urlparse(url).query).get("sort") == ["relevance"] if "esearch" in url else True
        if "esearch" in url:
            query = parse_qs(urlparse(url).query)["term"][0]
            if query == "fixed alpha" and sum("fixed+alpha" in x for x in calls) == 1:
                raw = b'{"esearchresult":{}}'
            elif query == "fixed beta" and fail_beta:
                raw = b'{"esearchresult":{}}'
            else:
                raw = b'{"esearchresult":{"idlist":["123","456"]}}'
        else:
            raw = XML
        filename.write_bytes(raw)
        return raw

    monkeypatch.setattr(module, "request_once", fake_request)
    first = module.prepare(source, output, retries=1, pause=0, backoff=0)
    assert first["status"] == "incomplete" and first["new_model_calls"] == 0
    alpha = first["records"][0]["snapshots"][0]
    assert [x["status"] for x in alpha["attempts"][:2]] == ["failed", "ok"]
    assert (output / alpha["attempts"][0]["raw_file"]).read_bytes() == b'{"esearchresult":{}}'
    assert alpha["excluded_target_pmids"] == ["PMID:123:abstract"]
    beta = first["records"][0]["snapshots"][1]
    old_beta_files = {x["raw_file"]: (output / x["raw_file"]).read_bytes() for x in beta["attempts"]}
    assert sorted(x["attempt"] for x in beta["attempts"]) == [1, 2]
    assert "target conclusion" not in (output / "corpus.jsonl").read_text(encoding="utf-8")
    n = len(calls)
    fail_beta = False
    done = module.prepare(source, output, retries=1, pause=0, backoff=0)
    assert done["status"] == "complete"
    assert len(calls) == n + 2  # beta ESearch + EFetch; alpha stays cached
    assert [x["attempt"] for x in done["records"][0]["snapshots"][1]["attempts"]
            if x["phase"] == "esearch"] == [1, 2, 3]
    assert all((output / name).read_bytes() == raw for name, raw in old_beta_files.items())
    assert done["corpus_chunks"] == 1
    assert len(done["chunk_sources"]["PMID:456:abstract"]) == 2
    assert done["config"]["source_manifest_sha256"] == module.sha((source / "preparation_manifest.json").read_bytes())
    assert (output / "q0001.query1.efetch.attempt1.raw").read_bytes() == XML
    audited = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_corpus_snapshot.py"))["audit"](output)
    assert audited["issues"] == []
    assert module.prepare(source, output, retries=1, pause=0, backoff=0) == done
    assert len(calls) == n + 2


def test_refuse_changed_parent_or_output(tmp_path, monkeypatch):
    source = source_fixture(tmp_path)
    output = tmp_path / "relevance"
    def empty_search(url, filename, timeout):
        raw = b'{"esearchresult":{"idlist":[]}}'
        filename.write_bytes(raw)
        return raw

    monkeypatch.setattr(module, "request_once", empty_search)
    assert module.prepare(source, output, retries=0, pause=0, backoff=0)["status"] == "complete"
    parent_file = source / "preparation_manifest.json"
    parent = json.loads(parent_file.read_text(encoding="utf-8"))
    parent["known_tokens"] = 18
    parent_file.write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(ValueError, match="different configuration"):
        module.prepare(source, output, retries=0, pause=0, backoff=0)
    with pytest.raises(ValueError, match="separate directory"):
        module.prepare(source, source, retries=0, pause=0, backoff=0)
