import hashlib
import json

import pytest

from evidence_mdt.cli import main
from evidence_mdt.data import write_jsonl
from evidence_mdt.intervention_preflight import prepare_interventions
from evidence_mdt.interventions import B_CONDITIONS, B_EXTENDED_CONDITIONS, subset25
from evidence_mdt.models import MockModel
from evidence_mdt.query_manifest import initial_queries
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.schema import Evidence, Question


def fixture(tmp_path, clinical_count=30):
    questions = ([Question(f"clinical-{i}", "MedQA", "shared question", {"A": "alpha", "B": "beta"}, "A")
                  for i in range(clinical_count)]
                 + [Question(f"pqa-{i}", "PubMedQA", "shared question", {"A": "alpha", "B": "beta"}, "A")
                    for i in range(20)])
    corpus = [Evidence("chunk-one", "doc-one", "alpha material", "local")]
    retriever = Retriever(corpus, MockDense())
    selected = subset25(questions)
    spec = [{"question_id": q.question_id, "b_target_id": "chunk-one",
             "target_answer": "B", "b_paraphrases": ["paraphrase one", "paraphrase two"],
             "c_evidence_text": "Investigator perturbation, truth unverified",
             "c_message_text": "Peer favors {wrong_label}"} for q in selected]
    spec_path = tmp_path / "spec.jsonl"
    write_jsonl(spec_path, spec)
    return questions, corpus, retriever, spec, spec_path


def test_stratified_25_selection_and_three_default_b_conditions(tmp_path):
    questions, _corpus, retriever, _spec, spec_path = fixture(tmp_path)
    chosen, specs, bases, traces, injections, manifest = prepare_interventions(
        questions, retriever, "both", spec_path, 2, B_CONDITIONS,
        initial_queries(questions, None))
    assert len(chosen) == len(specs) == len(bases) == len(traces) == 25
    assert sum(q.dataset == "MedQA" for q in subset25(questions)) == 15
    assert sum(q.dataset == "PubMedQA" for q in subset25(questions)) == 10
    assert manifest["b_conditions"] == ["base", "literal3", "paraphrase3"]
    assert len(injections) == 25 * 2 * 2
    assert manifest["spec_sha256"] == hashlib.sha256(spec_path.read_bytes()).hexdigest()
    for qid, repeat, _name in injections:
        assert qid in chosen and repeat in {1, 2}
    replacement = ([Question(f"reserve-{i}", "MedXpertQA", "shared question", {"A": "alpha", "B": "beta"}, "A")
                    for i in range(20)] + questions[30:])
    assert sum(q.dataset == "MedXpertQA" for q in subset25(replacement)) == 15
    assert {q.question_id for q in subset25(questions) if q.dataset == "PubMedQA"} == {
        q.question_id for q in subset25(replacement) if q.dataset == "PubMedQA"}


@pytest.mark.parametrize("change,expected", [
    ("missing_row", "exactly one row"),
    ("duplicate_row", "exactly one row"),
    ("missing_target", "top-6"),
    ("bad_answer", "option key"),
    ("empty_paraphrase", "nonempty"),
    ("bad_wrong_option", "non-gold"),
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
    elif change == "bad_wrong_option":
        spec[0]["c_wrong_option"] = "A"
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
    with pytest.raises(ValueError, match=expected):
        main(["run", "--questions", str(questions_path), "--corpus", str(corpus_path),
              "--output", str(tmp_path / "run"), "--model-backend", "mock",
              "--retrieval-backend", "mock", "--interventions", "both",
              "--intervention-spec", str(spec_path), "--groups", "ordinary"])
    assert called == []
    assert not (tmp_path / "run" / "calls.jsonl").exists()


def test_extended_condition_requires_four_paraphrases(tmp_path):
    questions, _corpus, retriever, spec, spec_path = fixture(tmp_path)
    with pytest.raises(ValueError, match="4 nonempty"):
        prepare_interventions(questions, retriever, "b", spec_path, 1,
                              B_CONDITIONS + B_EXTENDED_CONDITIONS,
                              initial_queries(questions, None))
    for row in spec:
        row["b_paraphrases"] += ["paraphrase three", "paraphrase four"]
    write_jsonl(spec_path, spec)
    _chosen, _specs, _bases, _traces, _injections, manifest = prepare_interventions(
        questions, retriever, "b", spec_path, 1, B_CONDITIONS + B_EXTENDED_CONDITIONS,
        initial_queries(questions, None))
    assert manifest["b_conditions"] == ["base", "literal3", "paraphrase3", "literal5", "paraphrase5"]


def test_preflight_and_main_run_use_same_short_query(tmp_path):
    questions, corpus, retriever, _spec, spec_path = fixture(tmp_path)
    lookup = {q.question_id: "short alpha query" for q in questions}
    _chosen, _specs, _bases, traces, _injections, _manifest = prepare_interventions(
        questions, retriever, "both", spec_path, 1, B_CONDITIONS, lookup)
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
              "--interventions", "both", "--intervention-spec", str(spec_path),
              "--query-manifest", str(query_path), "--groups", "ordinary",
              "--repeats", "1", "--max-calls", "1"])
    snapshots = [json.loads(line) for line in (output / "retrieval_snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    assert snapshots[0]["trace"]["query"] == "short alpha query"
    assignments = [json.loads(line) for line in (output / "assignments.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(assignments) == 50 + 25 * 6
    assert {a["variant"] for a in assignments if a["variant"].startswith("B:")} == {
        "B:base", "B:literal3", "B:paraphrase3"}
    manifest = json.loads((output / "intervention_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["selected"]) == 25
