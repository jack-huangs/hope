"""通用批次流程的离线集成测试；不调用真实模型、PubMed 或 MedCPT。"""
from dataclasses import asdict
import json

import pytest

from evidence_mdt.models import ModelResult
from evidence_mdt.retrieval import MockDense
from evidence_mdt.schema import Question
from preparation import prepare
from preparation.__main__ import main
from preparation.audit import verify_batch, verify_preparation
from preparation.batch import prepare_batch
from preparation.selection import select_questions
from preparation.state import ConfigurationError, digest, jsonl, save
from d_material_fixture import d_passages


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


class Model:
    name = "batch-offline"

    def __init__(self, fail_b=False):
        self.calls = []
        self.fail_b = fail_b

    def complete(self, payload):
        self.calls.append(payload)
        instruction = payload["instruction"]
        if "PubMed queries" in instruction:
            qid = payload["question"]["question_id"]
            value = {"queries": [f"explicit stem {qid}", f"stem concept {qid}"]}
        elif "annotations" in instruction:
            value = {"annotations": [{"chunk_id": e["chunk_id"], "target_answer": None,
                                      "reason": "No direct option support."} for e in payload["evidence"]]}
        elif "paraphrases" in instruction:
            if self.fail_b:
                raise TimeoutError("temporary B failure")
            value = {"paraphrases": [f"Rephrased whole passage number {i}." for i in range(10)]}
        elif "c_message_text" in instruction:
            value = {"c_message_text": "Incorrect peer rationale."}
        else:
            value = {"d_evidence_passages": d_passages()}
        return ModelResult(value, {}, {"total_tokens": 7}, payload)


def fetch(query, cache, max_records, timeout):
    rows = [{"chunk_id": f"e{i}", "document_id": f"d{i}", "text": f"Explicit stem evidence {i}.",
             "source": "local", "content_hash": digest(f"Explicit stem evidence {i}.".encode())}
            for i in range(12)]
    return {"status": "ok", "ids": [str(i) for i in range(12)], "evidence": rows}


def questions_file(tmp_path, n=2):
    path = tmp_path / "input.jsonl"
    jsonl(path, [asdict(Question(f"fixture-{i}", "MedXpertQA", f"Explicit stem {i}",
                               {"A": "alpha", "B": "beta"}, "A")) for i in range(n)])
    return path


def test_batch_20_questions_share_once_and_resume_without_calls(tmp_path):
    path = questions_file(tmp_path, 20)
    output, model = tmp_path / "batch", Model()
    searches = []
    def tracked(*args):
        searches.append(args[0])
        return fetch(*args)
    options = dict(retries=0, tokenizer=Tokenizer(), fetcher=tracked)
    result = prepare_batch(path, output, model, MockDense(), **options)
    assert result["status"] == "complete"
    assert len(searches) == 40  # 两条初始查询/题，只准备一份共用检索材料。
    assert len(model.calls) == 100  # 20短查询 + 20B逐条标注 + 20B改写 + 20C + 20D。
    assert result["stages"]["b"]["verification"]["unannotated"] == 20
    assert result["stages"]["b"]["verification"]["all_null_fallback"] == 20
    assert all(n == 10 for n in result["shared_verification"]["initial_evidence_counts"])
    for mode in ("b", "c", "d"):
        for name in ("corpus.jsonl", "initial_queries.jsonl", "initial_top10.jsonl"):
            assert (output / "shared" / name).read_bytes() == (output / mode / name).read_bytes()
    prepare_batch(path, output, model, MockDense(), **options)
    assert len(searches) == 40 and len(model.calls) == 100
    assert not list(output.rglob("*.py"))
    assert (output / "preparation_report.md").exists()
    assert sum(v["known_tokens"] for v in result["saved_model_responses"].values()) == 700
    assert verify_batch(output)["status"] == "passed"
    assert main(["verify", "--preparation", str(output)]) == 0


def test_batch_failed_b_does_not_block_cd_and_resumes_only_b(tmp_path):
    path, output = questions_file(tmp_path), tmp_path / "batch"
    model = Model(fail_b=True)
    options = dict(retries=0, tokenizer=Tokenizer(), fetcher=fetch)
    result = prepare_batch(path, output, model, MockDense(), **options)
    assert result["status"] == "incomplete"
    assert result["stages"]["c"]["status"] == result["stages"]["d"]["status"] == "complete"
    assert not (output / "b" / "intervention_spec.jsonl").exists()
    before = len(model.calls)
    model.fail_b = False
    result = prepare_batch(path, output, model, MockDense(), **options)
    assert result["status"] == "complete"
    assert len(model.calls) - before == 2


def test_batch_external_source_only_generates_materials(tmp_path):
    path = questions_file(tmp_path)
    source = tmp_path / "source"
    prepare(path, source, Model(), MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    model = Model()
    def forbidden(*args):
        raise AssertionError("Must not fetch")
    result = prepare_batch(path, tmp_path / "batch", model, source_preparation=source,
                           fetcher=forbidden, retries=0)
    assert result["status"] == "complete" and len(model.calls) == 8
    assert "shared" not in result["saved_model_responses"]
    with pytest.raises(ConfigurationError, match="configuration changed"):
        prepare_batch(path, tmp_path / "batch", model, modes=("b",), source_preparation=source)


def test_offline_audit_rejects_corruption_and_old_policy(tmp_path):
    path = questions_file(tmp_path)
    source = tmp_path / "source"
    prepare(path, source, Model(), MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert verify_preparation(source)["status"] == "passed"
    manifest = source / "preparation_manifest.json"
    state = json.loads(manifest.read_bytes())
    state["config"]["initial_evidence_limit"] = 6
    save(manifest, state)
    with pytest.raises(ConfigurationError, match="policy differs"):
        verify_preparation(source)
    state["config"]["initial_evidence_limit"] = 10
    save(manifest, state)
    (source / "corpus.jsonl").write_text("corrupt")
    with pytest.raises(ConfigurationError, match="hash mismatch"):
        verify_preparation(source)


def test_selection_raw_exclusions_reproducible_and_no_overwrite(tmp_path):
    path = questions_file(tmp_path, 8)
    raw = tmp_path / "official.raw"
    raw.write_bytes(path.read_bytes())
    prior = tmp_path / "prior"
    jsonl(prior / "questions.jsonl", [json.loads(path.read_text().splitlines()[0])])
    output = tmp_path / "selected"
    result = select_questions(raw, output, 3, 123, [prior])
    assert "fixture-0" not in result["selected_ids"] and result["eligible_count"] == 7
    assert select_questions(raw, output, 3, 123, [prior]) == result
    with pytest.raises(ConfigurationError, match="Selection output differs"):
        select_questions(raw, output, 3, 124, [prior])
    with pytest.raises(ConfigurationError, match="Not enough"):
        select_questions(raw, tmp_path / "overflow", 9, 123)
    assert main(["sample", "--source", str(raw), "--output", str(tmp_path / "cli"),
                 "--count", "2", "--seed", "123"]) == 0


def test_verify_cli_does_not_need_api_or_write_inside_frozen_folder(tmp_path, monkeypatch):
    path, output = questions_file(tmp_path), tmp_path / "source"
    prepare(path, output, Model(), MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert main(["verify", "--preparation", str(output), "--report", str(tmp_path / "audit.json")]) == 0
    with pytest.raises(SystemExit):
        main(["verify", "--preparation", str(output), "--report", str(output / "audit.json")])


def test_batch_changed_questions_rejected_before_new_calls(tmp_path):
    path, output, model = questions_file(tmp_path), tmp_path / "batch", Model()
    prepare_batch(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    questions_file(tmp_path, 3)
    before = len(model.calls)
    with pytest.raises(ConfigurationError, match="configuration changed"):
        prepare_batch(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert len(model.calls) == before


@pytest.mark.parametrize("modes,expected_calls,limit", [
    (["b", "c", "d"], 10, 12000), (["d"], 4, 6000), (["c"], 4, 3000)])
def test_batch_cli_dispatch_with_offline_dependencies(tmp_path, monkeypatch, modes, expected_calls, limit):
    import preparation.__main__ as cli
    path = questions_file(tmp_path)
    manifest = tmp_path / "local_models.json"
    manifest.write_text("{}")
    model = Model()
    model.api_key_env = "OFFLINE_BATCH_KEY"
    dense = MockDense()
    dense.atok = Tokenizer()
    monkeypatch.setenv("OFFLINE_BATCH_KEY", "fixture")
    monkeypatch.setattr(cli, "create_model", lambda *args: model)
    monkeypatch.setattr(cli, "MedCPTDense", lambda *args: dense)
    # prepare 的默认fetcher在定义时绑定，因此通过批量入口注入fixture。
    from preparation.batch import prepare_batch as original
    import preparation.batch as batch
    def offline_batch(*args, **kwargs):
        return original(*args, **kwargs, fetcher=fetch)
    monkeypatch.setattr(batch, "prepare_batch", offline_batch)
    assert main(["batch", "--questions", str(path), "--output", str(tmp_path / "batch"),
                 "--model", "deepseek", "--model-manifest", str(manifest), "--retries", "0",
                 "--modes", *modes]) == 0
    assert len(model.calls) == expected_calls
    assert model.max_tokens == limit


def test_batch_fatal_auth_error_stops_following_modes(tmp_path):
    path, output = questions_file(tmp_path), tmp_path / "batch"
    model = Model()
    complete = model.complete
    def fail(payload):
        if "paraphrases" in payload["instruction"]:
            raise ConfigurationError("Invalid authentication")
        return complete(payload)
    model.complete = fail
    with pytest.raises(ConfigurationError, match="authentication"):
        prepare_batch(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert json.loads((output / "batch_manifest.json").read_bytes())["status"] == "stopped"
    assert not (output / "c").exists() and not (output / "d").exists()


def test_batch_fail_fast_stops_at_first_generation_error(tmp_path):
    model = Model(fail_b=True)
    path, output = questions_file(tmp_path, 2), tmp_path / "batch"
    with pytest.raises(TimeoutError, match="temporary B failure"):
        prepare_batch(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch, fail_fast=True)
    manifest = json.loads((output / "batch_manifest.json").read_bytes())
    assert manifest["status"] == "stopped" and manifest["failed_stage"] == "b"
    assert sum("paraphrases" in call["instruction"] for call in model.calls) == 1
    assert not (output / "c").exists() and not (output / "d").exists()


def test_audit_checks_evidence_content_even_if_artifact_hash_updated(tmp_path):
    path, output = questions_file(tmp_path), tmp_path / "source"
    prepare(path, output, Model(), MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    frozen = output / "initial_top10.jsonl"
    rows = [json.loads(line) for line in frozen.read_text().splitlines()]
    rows[0]["evidence"][0]["text"] = "Altered evidence text"
    jsonl(frozen, rows)
    manifest = output / "preparation_manifest.json"
    state = json.loads(manifest.read_bytes())
    state["artifacts"][frozen.name] = digest(frozen.read_bytes())
    save(manifest, state)
    with pytest.raises(ConfigurationError, match="differs from corpus"):
        verify_preparation(output)


@pytest.mark.parametrize("field", ["b_target_id", "b_target_candidate_ids", "b_evidence_annotations"])
def test_audit_checks_b_selection_even_if_artifact_hash_updated(tmp_path, field):
    path, output = questions_file(tmp_path, 1), tmp_path / "prepared"
    prepare(path, output, Model(), MockDense(), interventions="b",
            tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    specs_path = output / "intervention_spec.jsonl"
    specs = [json.loads(line) for line in specs_path.read_text().splitlines()]
    spec = specs[0]
    if field == "b_target_id":
        spec[field] = next(cid for cid in spec["b_target_candidate_ids"] if cid != spec[field])
    else:
        spec[field].pop()
    jsonl(specs_path, specs)
    manifest = output / "preparation_manifest.json"
    state = json.loads(manifest.read_bytes())
    state["artifacts"][specs_path.name] = digest(specs_path.read_bytes())
    save(manifest, state)
    with pytest.raises(ConfigurationError, match="Invalid B selection"):
        verify_preparation(output)


def test_audit_old_b_without_exposure_configuration_retains_legacy_three_presentations(tmp_path):
    path, output = questions_file(tmp_path, 1), tmp_path / "prepared"
    prepare(path, output, Model(), MockDense(), interventions="b",
            tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    specs_path = output / "intervention_spec.jsonl"
    specs = [json.loads(line) for line in specs_path.read_text().splitlines()]
    specs[0]["b_paraphrases"] = specs[0]["b_paraphrases"][:2]
    jsonl(specs_path, specs)
    manifest = output / "preparation_manifest.json"
    state = json.loads(manifest.read_bytes())
    state["config"].pop("b_conditions")
    state["fingerprint"] = digest(state["config"])
    state["artifacts"][specs_path.name] = digest(specs_path.read_bytes())
    save(manifest, state)
    before = specs_path.read_bytes()
    assert verify_preparation(output)["status"] == "passed"
    assert specs_path.read_bytes() == before
