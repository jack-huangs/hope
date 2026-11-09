import hashlib
import json

import pytest

from evidence_mdt.cli import main
from evidence_mdt.data import write_jsonl
from evidence_mdt.intervention_preflight import prepare_interventions
from evidence_mdt.interventions import B_CONDITIONS, B_EXTENDED_CONDITIONS
from evidence_mdt.models import MockModel
from evidence_mdt.query_manifest import initial_queries
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.schema import Evidence, Question
from d_material_fixture import d_passages


def fixture(tmp_path, clinical_count=3):
    questions = [Question(f"clinical-{i}", "MedXpertQA", "shared question",
                          {"A": "alpha", "B": "beta"}, "A") for i in range(clinical_count)]
    corpus = [Evidence("chunk-one", "doc-one", "alpha material", "local")]
    retriever = Retriever(corpus, MockDense())
    spec = [{"question_id": q.question_id, "b_target_id": "chunk-one",
             "target_answer": "B", "b_paraphrases": [f"paraphrase {i}" for i in range(10)],
             "d_evidence_passages": d_passages(), "d_wrong_option": "B",
             "c_message_text": "Peer favors {wrong_label}"} for q in questions]
    spec_path = tmp_path / "spec.jsonl"
    write_jsonl(spec_path, spec)
    return questions, corpus, retriever, spec, spec_path


@pytest.mark.parametrize("count", [1, 7, 31])
@pytest.mark.parametrize("mode", ["b", "c", "d"])
def test_all_supplied_medxpertqa_questions_without_sampling(tmp_path, count, mode):
    questions, _corpus, retriever, _spec, spec_path = fixture(tmp_path, count)
    chosen, specs, bases, traces, injections, manifest = prepare_interventions(
        questions, retriever, mode, spec_path, 2, B_CONDITIONS,
        initial_queries(questions, None))
    assert len(chosen) == len(specs) == len(bases) == len(traces) == count
    assert manifest["question_count"] == count
    assert manifest["b_conditions"] == (["base", "literal11", "paraphrase11"] if mode == "b" else [])
    assert manifest["c_conditions"] == (["message_error", "repeated_message"] if mode == "c" else [])
    assert manifest["d_conditions"] == (["evidence_error"] if mode == "d" else [])
    assert len(injections) == (count * 2 * 2 if mode == "c" else 0)
    assert len(manifest["d_assignments"]) == (count if mode == "d" else 0)
    assert manifest["spec_sha256"] == hashlib.sha256(spec_path.read_bytes()).hexdigest()
    assert manifest['message_protocol'] == 'public_history_at_round_start_v2'
    assert all('receiver' not in c and c['visibility'] == 'public' and c['insertion_point'] == 'round_start'
               for c in manifest['c_assignments'])
    assert all(d["wrong_option"] == "B" and d["insertion_point"] == "before_r0"
               for d in manifest["d_assignments"])
    assert all(d["evidence_count"] == len(d["passages"]) == 5 for d in manifest["d_assignments"])


@pytest.mark.parametrize("change,expected", [
    ("missing_row", "exactly one row"),
    ("duplicate_row", "exactly one row"),
    ("missing_target", "top-10"),
    ("bad_answer", "option key"),
    ("empty_paraphrase", "nonempty"),
    ("short_paraphrases", "10 nonempty"),
    ("duplicate_last_paraphrase", "distinct"),
    ("bad_wrong_option", "non-gold"),
    ("d_gold", "non-gold"), ("d_unknown", "non-gold"),
    ("d_legacy_single", "Legacy single-passage"),
    ("legacy_c_evidence", "Legacy c_evidence_text"),
])
def test_invalid_spec_rejected_before_model_call(tmp_path, monkeypatch, change, expected):
    questions, corpus, _retriever, spec, spec_path = fixture(tmp_path)
    if change == "missing_row":
        spec.pop()
    elif change == "duplicate_row":
        spec.append(dict(spec[0]))
    elif change == "missing_target":
        spec[0]["b_target_id"] = "unknown"
    elif change == "bad_answer":
        spec[0]["target_answer"] = "Z"
    elif change == "empty_paraphrase":
        spec[0]["b_paraphrases"][0] = ""
    elif change == "short_paraphrases":
        spec[0]["b_paraphrases"] = spec[0]["b_paraphrases"][:2]
    elif change == "duplicate_last_paraphrase":
        spec[0]["b_paraphrases"][-1] = spec[0]["b_paraphrases"][0]
    elif change == "bad_wrong_option":
        spec[0]["c_wrong_option"] = "A"
    if change == "d_gold":
        spec[0]["d_wrong_option"] = "A"
    elif change == "d_unknown":
        spec[0]["d_wrong_option"] = "Z"
    elif change == "d_legacy_single":
        spec[0].update(d_evidence_text="old single passage", d_error_explanation="old audit")
        spec[0].pop("d_evidence_passages")
    elif change == "legacy_c_evidence":
        spec[0]["c_evidence_text"] = "legacy perturbation"
    write_jsonl(spec_path, spec)
    questions_path = tmp_path / "questions.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    write_jsonl(questions_path, questions)
    write_jsonl(corpus_path, corpus)
    called = []

    def forbidden(_self, _payload):
        called.append(True)
        raise AssertionError("model called before preflight")

    monkeypatch.setattr(MockModel, "complete", forbidden)
    mode = "d" if change.startswith("d_") else "c" if change == "bad_wrong_option" else "b"
    with pytest.raises(ValueError, match=expected):
        main(["run", "--questions", str(questions_path), "--corpus", str(corpus_path),
              "--output", str(tmp_path / "run"), "--model-backend", "mock",
              "--retrieval-backend", "mock", "--interventions", mode,
              "--intervention-spec", str(spec_path), "--groups", "ordinary"])
    assert called == []
    assert not (tmp_path / "run" / "calls.jsonl").exists()


def test_eleven_presentation_conditions_require_ten_paraphrases_even_with_legacy_extension(tmp_path):
    questions, _corpus, retriever, spec, spec_path = fixture(tmp_path)
    full = [dict(row) for row in spec]
    for row in spec:
        row["b_paraphrases"] = row["b_paraphrases"][:9]
    write_jsonl(spec_path, spec)
    with pytest.raises(ValueError, match="10 nonempty"):
        prepare_interventions(questions, retriever, "b", spec_path, 1,
                              B_CONDITIONS + B_EXTENDED_CONDITIONS,
                              initial_queries(questions, None))
    write_jsonl(spec_path, full)
    _chosen, _specs, _bases, _traces, _injections, manifest = prepare_interventions(
        questions, retriever, "b", spec_path, 1, B_CONDITIONS + B_EXTENDED_CONDITIONS,
        initial_queries(questions, None))
    assert manifest["b_conditions"] == ["base", "literal11", "paraphrase11", "literal5", "paraphrase5"]


def test_preflight_and_main_run_use_same_short_query(tmp_path):
    questions, corpus, retriever, _spec, spec_path = fixture(tmp_path)
    lookup = {q.question_id: "short alpha query" for q in questions}
    _chosen, _specs, _bases, traces, _injections, _manifest = prepare_interventions(
        questions, retriever, "b", spec_path, 1, B_CONDITIONS, lookup)
    assert all(trace["query"] == "short alpha query" for trace in traces.values())

    questions_path = tmp_path / "questions.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    query_path = tmp_path / "queries.jsonl"
    write_jsonl(questions_path, questions)
    write_jsonl(corpus_path, corpus)
    from evidence_mdt.query_manifest import public_question_sha256
    write_jsonl(query_path, [{"question_id": q.question_id, "dataset": q.dataset,
                              "public_question_sha256": public_question_sha256(q),
                              "retrieval_query": lookup[q.question_id]} for q in questions])
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="budget"):
        main(["run", "--questions", str(questions_path), "--corpus", str(corpus_path),
              "--output", str(output), "--model-backend", "mock", "--retrieval-backend", "mock",
              "--interventions", "b", "--intervention-spec", str(spec_path),
              "--query-manifest", str(query_path), "--groups", "ordinary",
              "--repeats", "1", "--max-calls", "1"])
    snapshots = [json.loads(line) for line in (output / "retrieval_snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    assert snapshots[0]["trace"]["query"] == "short alpha query"
    assignments = [json.loads(line) for line in (output / "assignments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(assignments) == len(questions) * 3
    assert {a["variant"] for a in assignments if a["variant"].startswith("B:")} == {
        "B:base", "B:literal11", "B:paraphrase11"}
    manifest = json.loads((output / "intervention_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["selected"]) == len(questions)
