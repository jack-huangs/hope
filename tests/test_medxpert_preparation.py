import importlib
import json
import sys
from pathlib import Path
import pytest
from types import SimpleNamespace
from evidence_mdt.schema import Question
from evidence_mdt.models import ModelResult

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


@pytest.mark.parametrize("endpoint,raw", [("efetch.fcgi", b"<PubmedArticleSet>"),
                                         ("esearch.fcgi", b'{"esearchresult":')])
def test_transport_rejects_truncated_http200(tmp_path, monkeypatch, endpoint, raw):
    transport = importlib.import_module("resume_medxpert_transport_oct12")
    path = tmp_path / "raw"
    def fake_run(args, **kwargs):
        path.write_bytes(raw)
        return SimpleNamespace(returncode=0, stdout="200", stderr="")
    monkeypatch.setattr(transport.subprocess, "run", fake_run)
    with pytest.raises((ValueError, transport.ET.ParseError)):
        transport.curl_request("https://fixture/" + endpoint, path, 30)
    assert path.read_bytes() == raw  # 失败原始体也保留，不能伪称零召回。


def test_completed_query_manifest_is_byte_immutable_on_resume(tmp_path, monkeypatch):
    preparation = importlib.import_module("prepare_medxpert_oct12")
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    source = tmp_path / "data/pilot_corpus_oct09"
    source.mkdir(parents=True)
    (source / "preparation_manifest.json").write_text(json.dumps({"records": [], "config": {"model": {}}}))
    calls = []
    class Model:
        name = "fixture"
        def complete(self, payload):
            calls.append(payload)
            content = {"queries": ["heart AND murmur", "infant AND shunt"]}
            raw = {"choices": [{"message": {"content": json.dumps(content)}}]}
            return ModelResult(content, raw, {"total_tokens": 5}, {})
    monkeypatch.setattr(preparation, "load_env", lambda path: None)
    monkeypatch.setattr(preparation, "create_model", lambda backend: Model())
    questions = [Question("Text-1", "MedXpertQA", "case", {"A": "a", "B": "b"}, "A")]
    output = tmp_path / "new"
    preparation.prepare_queries(questions, output)
    before = (output / "preparation_manifest.json").read_bytes()
    preparation.prepare_queries(questions, output)
    assert (output / "preparation_manifest.json").read_bytes() == before
    assert len(calls) == 1


def test_format_repair_preserves_words_and_query_pairing():
    repair = importlib.import_module("repair_broad_query_format_oct12")
    original = ["stable angina beta-blocker dose adjustment", "chronic coronary disease antianginal therapy"]
    valid = ['"stable angina" AND "beta-blocker" AND "dose adjustment"',
             '"chronic coronary disease" AND "antianginal therapy"']
    assert repair.validate_repair(original, valid) == valid
    with pytest.raises(ValueError, match="original query words"):
        repair.validate_repair(original, [valid[0] + " metoprolol", valid[1]])
    with pytest.raises(ValueError, match="original query words"):
        repair.validate_repair(original, [valid[1], valid[0]])


def test_failed_response_hash_sidecar_only_fills_failed_missing_hash(tmp_path):
    finish = importlib.import_module("finish_medxpert_oct12")
    (tmp_path / "failed.raw").write_bytes(b"HTTP400 body")
    (tmp_path / "success.raw").write_bytes(b"success")
    manifest = {"records": [{"snapshots": [{"attempts": [
        {"status": "failed", "raw_file": "failed.raw"},
        {"status": "ok", "raw_file": "success.raw", "raw_sha256": "original"}]}]}]}
    raw = json.dumps(manifest).encode()
    (tmp_path / "preparation_manifest.json").write_bytes(raw)
    finish.record_failed_response_hashes(tmp_path)
    sidecar = json.loads((tmp_path / "failed_response_hashes.json").read_bytes())
    assert sidecar["manifest_sha256"] == finish.base.sha(raw)
    assert sidecar["files"] == {"failed.raw": finish.base.sha(b"HTTP400 body")}
    assert (tmp_path / "preparation_manifest.json").read_bytes() == raw
    (tmp_path / "failed.raw").write_bytes(b"changed")
    with pytest.raises(ValueError, match="sidecar differs"):
        finish.record_failed_response_hashes(tmp_path)
