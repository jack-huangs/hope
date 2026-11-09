"""仅使用合成题和离线fixtures，不访问网络或真实模型。"""
import json
from dataclasses import asdict
from pathlib import Path
import urllib.error

import pytest

from evidence_mdt.models import ModelResult
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.intervention_preflight import prepare_interventions
from evidence_mdt.interventions import B_CONDITIONS
from evidence_mdt.query_manifest import initial_queries
from evidence_mdt.schema import Question
from preparation import prepare, recover_b_paraphrases
from preparation.state import ConfigurationError, digest, jsonl, Cache
from d_material_fixture import d_passages


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split()))),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class Model:
    name = "offline-fixture"
    def __init__(self):
        self.calls = []
    def complete(self, payload):
        self.calls.append(payload)
        instruction = payload["instruction"]
        if "PubMed queries" in instruction:
            value = {"queries": ["explicit stem", "stem concept"]}
        elif "annotations" in instruction:
            value = {"annotations": [{"chunk_id": e["chunk_id"], "target_answer": "A",
                                      "reason": "Direct fixture support."} for e in payload["evidence"]]}
        elif "paraphrases" in instruction:
            value = {"paraphrases": [f"Rephrased whole passage number {i}." for i in range(10)]}
        elif "c_message_text" in instruction:
            value = {"c_message_text": "The wrong option follows from this false rationale."}
        else:
            value = {"d_evidence_passages": d_passages()}
        return ModelResult(value, {"fixture": value}, {}, payload)


def fetch(query, cache, max_records, timeout):
    rows = [{"chunk_id": f"PMID:{i}:abstract", "document_id": f"PMID:{i}",
             "text": f"Explicit stem evidence {i}.", "title": "Fixture",
             "source": f"https://pubmed.ncbi.nlm.nih.gov/{i}/",
             "content_hash": digest(f"Explicit stem evidence {i}.".encode())} for i in range(1, 4)]
    return {"status": "ok", "ids": ["1", "2", "3"], "evidence": rows}


def source(tmp_path, count=2):
    questions = [Question(f"fixture-{i}", "MedXpertQA", f"Explicit stem {i}", {"A": "alpha", "B": "beta"}, "A") for i in range(count)]
    path = tmp_path / "input.jsonl"
    jsonl(path, [asdict(q) for q in questions])
    return path, questions


@pytest.mark.parametrize("mode", ["none", "b", "c", "d"])
def test_complete_and_cli_preflight(tmp_path, mode):
    path, questions = source(tmp_path)
    model, dense = Model(), MockDense()
    output = tmp_path / mode
    state = prepare(path, output, model, dense, interventions=mode, tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert state["status"] == "complete"
    assert all(not r["broad_triggered"] for r in state["records"])
    query = initial_queries(questions, output / "initial_queries.jsonl")
    assert set(query.values()) == {"explicit stem ; stem concept"}
    if mode != "none":
        corpus = [__import__("evidence_mdt.schema", fromlist=["Evidence"]).Evidence(**r)
                  for r in [json.loads(x) for x in (output / "corpus.jsonl").read_text().splitlines()]]
        prepare_interventions(questions, Retriever(corpus, MockDense()), mode,
                              output / "intervention_spec.jsonl", 1, B_CONDITIONS, query)
    before = len(model.calls)
    prepare(path, output, model, MockDense(), interventions=mode, tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert len(model.calls) == before
    for payload in model.calls:
        if "PubMed queries" in payload["instruction"]:
            assert set(payload["question"]) == {"question_id", "stem"}


def test_three_short_queries_are_searched_and_frozen(tmp_path):
    path, questions = source(tmp_path, 1)
    terms = ["explicit stem", "stem concept", "third finding"]
    model = Model()
    model.complete = lambda payload: ModelResult({"queries": terms}, {}, {}, payload)
    searches = []

    def fetch_fixture(query, *args):
        searches.append(query)
        return fetch(query, *args)

    output = tmp_path / "prepared"
    state = prepare(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch_fixture, fail_fast=True)
    assert state["status"] == "complete"
    assert searches == terms
    assert initial_queries(questions, output / "initial_queries.jsonl") == {
        questions[0].question_id: " ; ".join(terms)
    }


def test_b_recovery_preserves_original_order_and_requests_only_missing_versions(tmp_path):
    original = {"text": "Original text."}
    initial = [f"First response {i}." for i in range(8)]
    extra = ["New rewrite 1.", "New rewrite 2."]
    calls = []

    def request(payload):
        calls.append(payload)
        if len(calls) == 1:
            parts = initial + [original["text"], initial[0]]
        else:
            assert payload["existing_paraphrases"] == initial
            assert "exactly 2" in payload["instruction"]
            parts = extra
        return {"content": {"paraphrases": parts}, "raw": {"response": len(calls)}}

    result = recover_b_paraphrases(tmp_path, "q", {"original": original, "instruction": "initial"}, request)
    assert result["content"]["paraphrases"] == initial + extra
    assert len(result["paraphrase_sources"]) == 10


def test_failed_question_continues_and_resumes(tmp_path):
    path, _ = source(tmp_path)
    model = Model()
    original = model.complete
    def failure(payload):
        if payload["question"]["question_id"] == "fixture-0":
            raise TimeoutError("fixture transient error")
        return original(payload)
    model.complete = failure
    output = tmp_path / "prepared"
    state = prepare(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert state["status"] == "incomplete"
    assert state["records"][1]["status"] == "retrieved"
    assert not (output / "initial_top10.jsonl").exists()
    assert not (output / "initial_queries.jsonl").exists()
    model.complete = original
    state = prepare(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert state["status"] == "complete"


def test_empty_is_not_failure_and_broad_is_bounded(tmp_path):
    path, _ = source(tmp_path, 1)
    calls = []
    def empty(query, *args):
        calls.append(query)
        return {"status": "empty", "ids": [], "evidence": []}
    state = prepare(path, tmp_path / "out", Model(), MockDense(), tokenizer=Tokenizer(), fetcher=empty, retries=0)
    assert state["status"] == "complete"
    assert state["records"][0]["broad_triggered"]
    assert all(s["status"] == "empty" for s in state["records"][0]["snapshots"])
    assert len(calls) <= 4


def test_auth_stops_and_corrupt_cache_rejected(tmp_path):
    cache = Cache(tmp_path / "cache", 0)
    with pytest.raises(ConfigurationError):
        cache.get("auth", {}, lambda: (_ for _ in ()).throw(urllib.error.HTTPError("fixture", 401, "auth", {}, None)))
    cache.get("value", {}, lambda: {"ok": 1})
    artifact = cache.root / (digest("value") + ".json")
    row = json.loads(artifact.read_bytes())
    row["value"]["ok"] = 2
    artifact.write_text(json.dumps(row))
    with pytest.raises(ConfigurationError, match="hash mismatch"):
        cache.get("value", {}, lambda: None)


def test_invalid_modes_rejected(tmp_path):
    path, _ = source(tmp_path)
    for mode in ("all", "both", "bc"):
        with pytest.raises(ConfigurationError):
            prepare(path, tmp_path / mode, Model(), MockDense(), interventions=mode)


def test_invalid_generation_is_not_success_cached(tmp_path):
    path, _ = source(tmp_path, 1)
    model = Model()
    original = model.complete
    model.complete = lambda payload: ModelResult({"queries": []}, {"invalid": True}, {}, payload)
    output = tmp_path / "out"
    assert prepare(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)["status"] == "incomplete"
    model.complete = original
    assert prepare(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)["status"] == "complete"
    assert "invalid" in (output / "cache" / "model_calls.jsonl").read_text()


def test_pubmed_missing_ids_keeps_raw_and_no_success_cache(tmp_path, monkeypatch):
    from preparation.corpus import fetch_pubmed
    import preparation.corpus as corpus
    import io
    monkeypatch.setattr(corpus.time, "sleep", lambda _: None)
    calls = []
    def urlopen(url, timeout):
        calls.append(url)
        return io.BytesIO(b'{"esearchresult":{"idlist":["1"]}}' if "esearch" in url else b"<PubmedArticleSet/>")
    monkeypatch.setattr(corpus.urllib.request, "urlopen", urlopen)
    with pytest.raises(ValueError, match="omitted"):
        fetch_pubmed("fixture", Cache(tmp_path / "cache", 0))
    assert len(calls) == 2
    assert len(list((tmp_path / "cache" / "raw").glob("*.json"))) == 2


def test_real_token_lengths_do_not_truncate():
    from preparation.corpus import check_lengths
    class Real:
        qtok = staticmethod(lambda query, **kw: {"input_ids": list(range(65))})
        atok = rtok = qtok
    with pytest.raises(ValueError, match="64 tokens"):
        check_lengths(Real(), "fixture", [])


def test_resume_after_artifact_write_before_manifest_commit(tmp_path, monkeypatch):
    import preparation
    path, _ = source(tmp_path, 1)
    output = tmp_path / "out"
    original = preparation.save
    def interrupt(path, value):
        if path.name == "preparation_manifest.json" and value.get("status") == "complete":
            raise SystemExit("fixture crash after artifact write")
        original(path, value)
    monkeypatch.setattr(preparation, "save", interrupt)
    model = Model()
    with pytest.raises(SystemExit):
        prepare(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert (output / "initial_queries.jsonl").exists()
    monkeypatch.setattr(preparation, "save", original)
    before = len(model.calls)
    state = prepare(path, output, model, MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert state["status"] == "complete"
    assert len(model.calls) == before


def test_pubmed_query_rejection_is_local(tmp_path, monkeypatch):
    import preparation.corpus as corpus
    import io
    monkeypatch.setattr(corpus.time, "sleep", lambda _: None)
    def rejected(url, timeout):
        raise urllib.error.HTTPError(url, 400, "bad query", {}, io.BytesIO(b"query rejected"))
    monkeypatch.setattr(corpus.urllib.request, "urlopen", rejected)
    with pytest.raises(ValueError, match="PubMed query request rejected"):
        corpus.fetch_pubmed("fixture", Cache(tmp_path / "cache", 0))


def test_reuse_freezes_top10_and_only_generates_materials(tmp_path):
    path, _ = source(tmp_path, 1)
    original = tmp_path / "original"
    prepare(path, original, Model(), MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    model = Model()
    def forbidden(*args):
        raise AssertionError("Reused source must not search")
    output = tmp_path / "reused"
    state = prepare(path, output, model, None, interventions="c", source_preparation=original,
                    fetcher=forbidden, retries=0)
    assert state["status"] == "complete"
    assert len(model.calls) == 1
    assert "c_message_text" in model.calls[0]["instruction"]
    for name in ("initial_top10.jsonl", "initial_queries.jsonl", "corpus.jsonl"):
        assert (original / name).read_bytes() == (output / name).read_bytes()
    (original / "corpus.jsonl").write_text("corrupt")
    with pytest.raises(ConfigurationError, match="Source artifact hash mismatch"):
        prepare(path, tmp_path / "corrupt-source", Model(), None, interventions="d", source_preparation=original)


def test_reuse_rejects_question_changes(tmp_path):
    path, questions = source(tmp_path, 1)
    original = tmp_path / "original"
    prepare(path, original, Model(), MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    changed = asdict(questions[0])
    changed["stem"] = "Changed explicit stem"
    jsonl(path, [changed])
    with pytest.raises(ConfigurationError, match="question set"):
        prepare(path, tmp_path / "changed", Model(), None, interventions="c", source_preparation=original)


def test_fatal_reuse_never_exports_executable_queries(tmp_path):
    path, _ = source(tmp_path, 1)
    original = tmp_path / "original"
    prepare(path, original, Model(), MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    model = Model()
    def unauthorized(payload):
        raise urllib.error.HTTPError("fixture", 401, "auth", {}, None)
    model.complete = unauthorized
    output = tmp_path / "failed-reuse"
    with pytest.raises(ConfigurationError):
        prepare(path, output, model, None, interventions="c", source_preparation=original, retries=0)
    assert not (output / "initial_queries.jsonl").exists()
    assert not (output / "intervention_spec.jsonl").exists()
    assert json.loads((output / "preparation_manifest.json").read_bytes())["status"] == "stopped"


def test_null_b_target_prepares_runs_and_keeps_all_outcome_trials(tmp_path):
    from evidence_mdt.cli import main

    class NoDirectSupport(Model):
        def complete(self, payload):
            result = super().complete(payload)
            if "annotations" in payload["instruction"]:
                for row in result.content["annotations"]:
                    row.update(target_answer=None, reason="Background only; no option directly supported.")
            return result

    path, questions = source(tmp_path, 2)
    output = tmp_path / "prepared"
    state = prepare(path, output, NoDirectSupport(), MockDense(), interventions="b",
                    tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert state["status"] == "complete"
    assert all(row["spec"]["target_answer"] is None and row["spec"]["b_target_reason"]
               for row in state["records"])
    # Even a legacy fallback must not override an explicit null annotation.
    specs = [row["spec"] | {"target_option": "B"} for row in state["records"]]
    jsonl(output / "intervention_spec.jsonl", specs)
    run = tmp_path / "run"
    main(["run", "--questions", str(path), "--corpus", str(output / "corpus.jsonl"),
          "--output", str(run), "--model-backend", "mock", "--retrieval-backend", "mock",
          "--query-manifest", str(output / "initial_queries.jsonl"),
          "--interventions", "b", "--intervention-spec", str(output / "intervention_spec.jsonl"),
          "--groups", "ordinary"])
    results = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    assert len(results) == len(questions) * 3
    assert all(row["b_target_answer"] is None for row in results)
    main(["evaluate", "--results", str(run / "results.jsonl"), "--output", str(run / "report.json")])
    report = json.loads((run / "report.json").read_bytes())
    for metric in report["b_target_probability_change"].values():
        assert metric["n_valid_pairs"] == 0
        assert metric["n_excluded_no_target_annotation"] == 2
        assert metric["mean_delta_target_probability"] is None
    assert sum(row["n_completed"] for row in report["all_variants"].values()) == 6


@pytest.mark.parametrize("annotation", [{}, {"target_answer": "Z"}, {"target_answer": None}])
def test_b_requires_explicit_valid_annotation_and_null_reason(tmp_path, annotation):
    class BadAnnotation(Model):
        def complete(self, payload):
            result = super().complete(payload)
            if "annotations" in payload["instruction"]:
                row = result.content["annotations"][0]
                row.pop("target_answer", None)
                row.pop("reason", None)
                row.update(annotation)
            return result

    path, _ = source(tmp_path, 1)
    model = BadAnnotation()
    state = prepare(path, tmp_path / "out", model, MockDense(), interventions="b",
                    tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert state["status"] == "incomplete"
    assert state["records"][0]["status"] == "failed"
    assert "b_target_id" not in state["records"][0].get("spec", {})
    assert not any("paraphrases" in p["instruction"] for p in model.calls)
    valid_model = Model()
    resumed = prepare(path, tmp_path / "out", valid_model, MockDense(), interventions="b",
                      tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert resumed["status"] == "complete"
    assert len(valid_model.calls) == 2  # Invalid annotations were not committed as successful cache.


def test_ten_initial_passages_reach_preparation_reuse_and_expert_requests(tmp_path):
    from evidence_mdt.cli import main

    path, _ = source(tmp_path, 1)

    def rich_fetch(*args):
        evidence = [{"chunk_id": f"e{i}", "document_id": f"d{i}",
                     "text": f"Explicit stem evidence {i}.", "source": "local",
                     "content_hash": digest(f"Explicit stem evidence {i}.".encode())}
                    for i in range(12)]
        return {"status": "ok", "ids": [str(i) for i in range(12)], "evidence": evidence}

    output = tmp_path / "prepared"
    model = Model()
    state = prepare(path, output, model, MockDense(), interventions="b",
                    tokenizer=Tokenizer(), fetcher=rich_fetch, retries=0)
    assert state["status"] == "complete"
    assert state["config"]["initial_evidence_limit"] == 10
    assert len(state["records"][0]["top10"]) == 10
    annotation_calls = [p for p in model.calls if "annotations" in p["instruction"]]
    assert len(annotation_calls) == 1 and len(annotation_calls[0]["evidence"]) == 10
    assert "gold" not in annotation_calls[0]["question"]
    assert len(state["records"][0]["spec"]["b_evidence_annotations"]) == 10
    assert state["config"]["b_conditions"] == ["base", "literal11", "paraphrase11"]
    assert len(state["records"][0]["spec"]["b_paraphrases"]) == 10
    reused = prepare(path, tmp_path / "reused", Model(), None, interventions="c",
                     source_preparation=output, retries=0)
    assert len(reused["records"][0]["top10"]) == 10
    run = tmp_path / "run"
    main(["run", "--questions", str(path), "--corpus", str(output / "corpus.jsonl"),
          "--output", str(run), "--model-backend", "mock", "--retrieval-backend", "mock",
          "--query-manifest", str(output / "initial_queries.jsonl"),
          "--interventions", "b", "--intervention-spec", str(output / "intervention_spec.jsonl"),
          "--groups", "ordinary"])
    events = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    initial = [e for e in events if e.get("type") == "turn" and e["round_id"] == 0]
    assert len(initial) == 12
    for turn in initial:
        expected = 10 if ":B:base:" in turn["run_id"] else 20
        assert len(turn["visible_payload"]["evidence"]) == expected


def test_reuse_requires_explicit_current_evidence_policy(tmp_path):
    path, _ = source(tmp_path, 1)
    output = tmp_path / "old"
    prepare(path, output, Model(), MockDense(), tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    manifest_path = output / "preparation_manifest.json"
    state = json.loads(manifest_path.read_bytes())
    state["config"].pop("initial_evidence_limit")
    manifest_path.write_text(json.dumps(state))
    with pytest.raises(ConfigurationError, match="initial evidence limit differs"):
        prepare(path, tmp_path / "new", Model(), None, interventions="c", source_preparation=output)


@pytest.mark.parametrize("count", [2, 9, 11])
def test_b_wrong_paraphrase_count_not_cached_and_resume_keeps_selected_target(tmp_path, count):
    class WrongCount(Model):
        def complete(self, payload):
            result = super().complete(payload)
            if "paraphrases" in payload["instruction"]:
                result.content["paraphrases"] = [f"Whole passage wording {i}." for i in range(count)]
            return result

    path, _ = source(tmp_path, 1)
    output = tmp_path / "prepared"
    state = prepare(path, output, WrongCount(), MockDense(), interventions="b",
                    tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert state["status"] == "incomplete"
    saved = state["records"][0]["spec"]
    assert "b_target_id" in saved and "b_paraphrases" not in saved
    model = Model()
    state = prepare(path, output, model, MockDense(), interventions="b",
                    tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert state["status"] == "complete" and len(model.calls) == 1
    assert state["records"][0]["spec"]["b_target_id"] == saved["b_target_id"]
    assert len(state["records"][0]["spec"]["b_paraphrases"]) == 10
