import importlib
import json
import sys
from pathlib import Path
import pytest
from evidence_mdt.schema import Question

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
query = importlib.import_module("prepare_question_corpus")
broad = importlib.import_module("prepare_broad_corpus")
reuse = importlib.import_module("reuse_query_snapshot")


def test_query_stem_excludes_terminal_choices_and_hidden_background():
    q = Question("Text-1", "MedXpertQA", "Case mentions Answer Choices casually.\nAnswer Choices: (A) Candidate",
                 {"A": "Candidate", "B": "Other"}, "A", "background")
    payload = query.query_payload(q)
    assert payload["question"] == {"question_id": "Text-1", "dataset": "MedXpertQA",
                                   "stem": "Case mentions Answer Choices casually."}
    assert broad.model_payload(q.public())["question"] == payload["question"]
    assert q.stem.endswith("Candidate")


def test_reuse_hash_guard_and_byte_preservation(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "new"
    source.mkdir()
    output.mkdir()
    (source / "e.jsonl").write_bytes(b"fixture\n")
    (source / "s.json").write_text('{}')
    row = {"question_id": "111", "dataset": "PubMedQA", "public_question_sha256": "hash",
           "status": "ready", "queries": ["term"], "snapshots": [{"status": "empty",
           "evidence_file": "e.jsonl", "snapshot_file": "s.json"}]}
    (source / "preparation_manifest.json").write_text(json.dumps({"status": "complete", "records": [row]}))
    manifest = {"records": [dict(row)]}
    reuse.reuse_rows(output, manifest, source)
    assert (output / "e.jsonl").read_bytes() == b"fixture\n"
    assert manifest["reuse_provenance"]["files_sha256"]["e.jsonl"]
    manifest["records"][0]["public_question_sha256"] = "changed"
    with pytest.raises(ValueError, match="public question differs"):
        reuse.reuse_rows(output, manifest, source)


