import hashlib
import json
import re

import pytest

from scripts.chunk_corpus import chunk_rows, prepare, spans, token_count


class WordOffsetTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [match.span() for match in matches]
        return result


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture_source(tmp_path):
    source = tmp_path / "parent"
    source.mkdir()
    short = "Short abstract."
    long = ("Sentence one has several words. Sentence two has additional words. " * 18)
    rows = [
        {"chunk_id": "p1", "document_id": "PMID:11", "text": short,
         "source": "https://pubmed.ncbi.nlm.nih.gov/11/", "position": "abstract",
         "retrieved_at": "time", "content_hash": sha(short.encode()), "derived_from": None,
         "title": "Short title"},
        {"chunk_id": "p2", "document_id": "PMID:12", "text": long,
         "source": "https://pubmed.ncbi.nlm.nih.gov/12/", "position": "abstract",
         "retrieved_at": "time", "content_hash": sha(long.encode()), "derived_from": None,
         "title": "Long title"},
    ]
    corpus = source / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (source / "questions.jsonl").write_text(json.dumps({"question_id": "q", "dataset": "Fixture",
        "stem": "public", "options": {"A": "a"}, "context": ""}) + "\n", encoding="utf-8")
    manifest = {"status": "complete", "corpus_sha256": sha(corpus.read_bytes()),
                "question_ids": ["q"], "forbidden_pubmedqa_pmids": ["999"],
                "chunk_sources": {"p1": [{"question_id": "q", "query_index": 0}],
                                  "p2": [{"question_id": "q", "query_index": 1}]}}
    (source / "preparation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source, rows


def test_exact_reconstruction_metadata_and_replay_guard(tmp_path):
    source, parents = fixture_source(tmp_path)
    output = tmp_path / "out"
    tokenizer = WordOffsetTokenizer()
    result = prepare(source, output, tokenizer, "test-revision", limit=18)
    assert result["status"] == "complete"
    assert result["question_ids"] == ["q"]
    assert result["forbidden_pubmedqa_pmids"] == ["999"]
    assert (output / "questions.jsonl").read_bytes() == (source / "questions.jsonl").read_bytes()
    children = [json.loads(line) for line in (output / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    assert children[0] == parents[0]
    long_children = [row for row in children if row["document_id"] == "PMID:12"]
    assert len(long_children) > 1
    assert "".join(row["text"] for row in long_children) == parents[1]["text"]
    assert all(row["title"] == parents[1]["title"] and row["source"] == parents[1]["source"]
               and row["derived_from"] is None for row in long_children)
    assert all(token_count(tokenizer, row["title"], row["text"]) <= 18 for row in children)
    assert all(row["chunk_id"] in result["chunk_sources"] for row in children)
    assert len(result["parent_to_children"]["p2"]) == len(long_children)
    assert prepare(source, output, tokenizer, "test-revision", limit=18) == result
    with pytest.raises(ValueError, match="different"):
        prepare(source, output, tokenizer, "other-revision", limit=18)


def test_long_single_sentence_uses_offset_or_character_boundary():
    tokenizer = WordOffsetTokenizer()
    text = "word " * 90 + "超长连续字符串" * 20
    pieces = spans("Title", text, tokenizer, limit=15)
    assert len(pieces) > 2
    assert "".join(text[a:b] for a, b in pieces) == text
    assert all(token_count(tokenizer, "Title", text[a:b]) <= 15 for a, b in pieces)


def test_parent_hash_and_forbidden_pmid_checked(tmp_path):
    source, rows = fixture_source(tmp_path)
    rows[0]["content_hash"] = "tampered"
    with pytest.raises(ValueError, match="Parent content hash"):
        chunk_rows(rows, WordOffsetTokenizer(), limit=18)
    rows[0]["content_hash"] = sha(rows[0]["text"].encode())
    manifest_path = source / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["forbidden_pubmedqa_pmids"] = ["12"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Forbidden"):
        prepare(source, tmp_path / "out", WordOffsetTokenizer(), "test-revision", limit=18)
