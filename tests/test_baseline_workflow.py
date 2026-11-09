"""离线验证全题基线、C/D干预及续跑分配；不调用真实模型。"""
import json
import uuid
from dataclasses import replace

import pytest

from evidence_mdt.cli import main
from evidence_mdt.data import write_jsonl
from evidence_mdt.metrics import baseline_records, b_target_probability_change, d_target_probability_change
from evidence_mdt.interventions import d_evidence, intervention_modes
from evidence_mdt.runner import Runner, append_jsonl
from evidence_mdt.schema import AgentTurn, Evidence, Question
from d_material_fixture import d_passages


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("mode", ["both", "all"])
def test_removed_intervention_modes_are_rejected(mode):
    # 命令行和直接调用都不能再启用组合模式。
    with pytest.raises(ValueError, match="Unknown intervention mode"):
        intervention_modes(mode)
    with pytest.raises(SystemExit) as exc:
        main(["run", "--questions", "unused.jsonl", "--corpus", "unused.jsonl",
              "--output", "unused", "--interventions", mode])
    assert exc.value.code == 2


def scheduling_only(self, question, repeat_id, group, evidence, variant="B:base",
                    r0_source=None, message_injection=None):
    """替换耗时讨论，仅记录调度器传入的条件与R0来源；真正讨论另有端到端测试。"""
    run_id = f"{question.question_id}:repeat-{repeat_id}:{variant}:{group}"
    attempt = uuid.uuid4().hex
    self.current_attempt_id = attempt
    r0 = [AgentTurn(f"{run_id}:expert-{i}:R0", f"expert-{i}", 0, i, [], [], {},
                    "A", {"A": .7, "B": .3}, [], [], "fixture", {}, attempt_id=attempt)
          for i in range(4)]
    if r0_source:
        r0 = [replace(t, reused_from=source.turn_id) for t, source in zip(r0, r0_source)]
    for t in r0:
        self._event({"run_id": run_id, "type": "turn", **t.record()})
    result = {"run_id": run_id, "question_id": question.question_id, "dataset": question.dataset,
              "repeat_id": repeat_id, "group": group, "variant": variant,
              "status": "ok", "fingerprint": self.fingerprint, "attempt_id": attempt,
              "r0_source_ids": [t.turn_id for t in r0_source or []],
              "b_target_answer": self.b_target_answer, "d_target_answer": self.d_target_answer, "answer": "A", "gold": "A",
              "options": question.options, "probabilities": {"A": .7, "B": .3}}
    append_jsonl(self.result_log, result)
    return result, r0


@pytest.mark.parametrize("mode,extended,multiplier", [
    ("none", False, 1), ("b", False, 3), ("c", False, 3), ("d", False, 2),
    ("b", True, 5),
])
def test_cli_all_questions_baseline_once_and_interventions_resume(tmp_path, monkeypatch, mode, extended, multiplier):
    monkeypatch.setattr(Runner, "run_group", scheduling_only)
    count = 4
    expected = count * 3 * multiplier
    questions = [Question(f"q{i}", "MedXpertQA", "alpha?",
                          {"A": "alpha", "B": "beta"}, "A") for i in range(count)]
    selected = {q.question_id for q in questions}
    write_jsonl(tmp_path / "questions.jsonl", questions)
    write_jsonl(tmp_path / "corpus.jsonl", [Evidence("e", "doc", "alpha study finding")])
    write_jsonl(tmp_path / "spec.jsonl", [
        {"question_id": qid, "b_target_id": "e", "target_answer": "B",
         "b_paraphrases": [f"Distinct paraphrase {i}" for i in range(10)],
         "d_evidence_passages": d_passages(), "d_wrong_option": "B",
         "c_message_text": "Peer favors {wrong_label}"} for qid in sorted(selected)])
    output = tmp_path / "run"
    argv = ["run", "--questions", str(tmp_path / "questions.jsonl"),
            "--corpus", str(tmp_path / "corpus.jsonl"), "--output", str(output),
            "--model-backend", "mock", "--retrieval-backend", "mock", "--interventions", mode]
    if mode != "none":
        argv += ["--intervention-spec", str(tmp_path / "spec.jsonl")]
    if extended:
        argv += ["--b-extended"]
    main(argv)
    assigned, results = rows(output / "assignments.jsonl"), rows(output / "results.jsonl")
    assert len(assigned) == len(results) == expected
    assert len({r["run_id"] for r in results}) == expected
    assert not any(r["variant"] == "natural" for r in assigned + results)
    assert len([r for r in results if r["variant"] == "B:base"]) == count * 3
    assert {r["question_id"] for r in results if r["variant"] != "B:base"} == (selected if mode != "none" else set())
    for r in results:
        sources = r["r0_source_ids"]
        if r["variant"] in {"C:message_error", "C:repeated_message"}:
            assert len(sources) == 4
            assert all(f"{r['question_id']}:repeat-1:B:base:" in s for s in sources)
        if r["variant"] == "D:evidence_error" and r["group"] == "ordinary":
            assert not sources
    snapshot = (output / "results.jsonl").read_bytes()
    main(argv)
    assert (output / "results.jsonl").read_bytes() == snapshot
    # 中断后只补回缺失的一组，已完成结果逐字保留。
    missing = results.pop()
    write_jsonl(output / "results.jsonl", results)
    main(argv)
    resumed = rows(output / "results.jsonl")
    assert resumed[:-1] == results
    assert resumed[-1]["run_id"] == missing["run_id"]
    assert len(resumed) == expected


def test_evaluate_baseline_prefers_new_label_and_does_not_count_targetless_base(tmp_path):
    records = []
    for group in ("ordinary", "semantic", "semantic_retrieval"):
        common = {"dataset": "MedXpertQA", "question_id": "q", "repeat_id": 1, "group": group,
                  "options": {"A": "one", "B": "two"}, "gold": "A", "answer": "A",
                  "probabilities": {"A": .8, "B": .2}, "status": "ok"}
        records += [{**common, "variant": "natural", "run_id": "old-" + group},
                    {**common, "variant": "B:base", "run_id": "base-" + group}]
    # 一个仅有历史 natural 的题仍可读取，但不会覆盖新基线。
    records.append({**records[0], "question_id": "legacy", "run_id": "legacy"})
    write_jsonl(tmp_path / "results.jsonl", records)
    write_jsonl(tmp_path / "assignments.jsonl", records)
    original = (tmp_path / "results.jsonl").read_bytes()
    main(["evaluate", "--results", str(tmp_path / "results.jsonl"), "--output", str(tmp_path / "report.json")])
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "natural" not in report
    assert report["baseline"]["MedXpertQA/ordinary/B:base"]["n_completed"] == 2
    assert report["baseline"]["MedXpertQA/semantic/B:base"]["n_completed"] == 1
    assert report["baseline"]["MedXpertQA/ordinary/B:base"]["n_assigned"] == 2
    assert report["paired"]["semantic"]["MedXpertQA"]["paired_trials"] == 1
    assert report["b_target_probability_change"] == {}
    assert (tmp_path / "results.jsonl").read_bytes() == original
    assert len(baseline_records(records)) == 4
    targetless = {**records[1], "question_id": "unpaired"}
    intervention = {**records[1], "variant": "B:literal3", "b_target_answer": "B",
                    "probabilities": {"A": .6, "B": .4}}
    change = b_target_probability_change([records[1], targetless, intervention])
    assert change["MedXpertQA/ordinary/B:literal3"]["n_valid_pairs"] == 1
    assert change["MedXpertQA/ordinary/B:literal3"]["mean_delta_target_probability"] == pytest.approx(.2)
    assert b_target_probability_change([targetless]) == {}
    # 历史natural可以读作普通基线，但不能悄悄充当B干预目标概率对照。
    write_jsonl(tmp_path / "results.jsonl", [records[0], intervention])
    (tmp_path / "assignments.jsonl").unlink()
    main(["evaluate", "--results", str(tmp_path / "results.jsonl"), "--output", str(tmp_path / "report.json")])
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    delta = report["b_target_probability_change"]["MedXpertQA/ordinary/B:literal3"]
    assert delta["n_valid_pairs"] == 0
    assert delta["n_missing_or_invalid"] == 1


@pytest.mark.parametrize("dataset", ["MedQA", "PubMedQA", "Other"])
def test_new_cli_rejects_other_datasets_before_model_creation(tmp_path, monkeypatch, dataset):
    write_jsonl(tmp_path / "questions.jsonl", [Question("q", dataset, "choose", {"A": "one", "B": "two"}, "A")])

    def forbidden(*args):
        raise AssertionError("Model creation must not precede dataset validation")

    monkeypatch.setattr("evidence_mdt.cli.create_model", forbidden)
    with pytest.raises(ValueError, match="MedXpertQA only"):
        main(["run", "--questions", str(tmp_path / "questions.jsonl"), "--corpus", "unused.jsonl",
              "--output", str(tmp_path / "run"), "--model-backend", "deepseek", "--retrieval-backend", "mock"])
    assert not (tmp_path / "run" / "calls.jsonl").exists()


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_question_limit_is_rejected(tmp_path, limit):
    write_jsonl(tmp_path / "questions.jsonl", [Question("q", "MedXpertQA", "choose", {"A": "one", "B": "two"}, "A")])
    with pytest.raises(ValueError, match="must be positive"):
        main(["run", "--questions", str(tmp_path / "questions.jsonl"), "--corpus", "unused.jsonl",
              "--output", str(tmp_path / "run"), "--model-backend", "mock", "--retrieval-backend", "mock",
              "--max-questions", str(limit)])


def test_legacy_single_d_material_retains_stable_ids_without_altering_original():
    text = "same text already available"
    base = [Evidence("e", "doc", text)]
    first = d_evidence(base, text, "q")
    assert base == first[:-1] and len(base) == 1
    assert first[-1].chunk_id != base[0].chunk_id
    # 同样注入ID已经出现时仍生成独立片段，且不丢失任何原片段。
    again = d_evidence(first, text, "q")
    assert again[:-1] == first
    assert len({e.chunk_id for e in again}) == 3
    assert d_evidence(base, text, "another-q")[-1].chunk_id != first[-1].chunk_id


def test_real_mock_cli_d_independent_r0_target_audit_and_no_metadata_in_model_request(tmp_path):
    write_jsonl(tmp_path / "questions.jsonl", [Question("q", "MedXpertQA Text", "choose", {"A": "one", "B": "two"}, "A")])
    write_jsonl(tmp_path / "corpus.jsonl", [Evidence(f"e{i}", f"doc{i}", f"initial finding {i}") for i in range(10)])
    write_jsonl(tmp_path / "spec.jsonl", [{"question_id": "q", "d_wrong_option": "B",
                                          "d_evidence_passages": d_passages()}])
    output = tmp_path / "run"
    main(["run", "--questions", str(tmp_path / "questions.jsonl"), "--corpus", str(tmp_path / "corpus.jsonl"),
          "--output", str(output), "--model-backend", "mock", "--retrieval-backend", "mock",
          "--groups", "ordinary", "--interventions", "d", "--intervention-spec", str(tmp_path / "spec.jsonl")])
    results = rows(output / "results.jsonl")
    assert [r["variant"] for r in results] == ["B:base", "D:evidence_error"]
    assert all(r["d_target_answer"] == "B" for r in results)
    assert results[-1]["d_error_explanation"].count("HIDDEN_AUDIT_ERROR_DETAIL") == 5
    audit = results[-1]["d_evidence_audit"]
    assert len(audit) == len({row["chunk_id"] for row in audit}) == 5
    manifest = json.loads((output / "intervention_manifest.json").read_text(encoding="utf-8"))
    assert manifest["d_evidence_count"] == manifest["d_assignments"][0]["evidence_count"] == 5
    assert manifest["d_assignments"][0]["passages"] == audit
    assert not results[-1]["r0_reused"]
    turns = [e for e in rows(output / "events.jsonl") if e.get("type") == "turn" and e.get("round_id") == 0]
    base_r0 = [t for t in turns if ":B:base:" in t["run_id"]]
    d_r0 = [t for t in turns if ":D:evidence_error:" in t["run_id"]]
    assert len(d_r0) == len(base_r0) == 4
    assert {t["turn_id"] for t in d_r0}.isdisjoint(t["turn_id"] for t in base_r0)
    assert all(len(t["visible_payload"]["evidence"]) == 10 for t in base_r0)
    assert all(len(t["visible_payload"]["evidence"]) == 15 for t in d_r0)
    expected_texts = {row["text"].replace("{wrong_text}", "two") for row in d_passages()}
    for turn in d_r0:
        visible = {e["chunk_id"]: e["text"] for e in turn["visible_payload"]["evidence"]}
        assert {visible[row["chunk_id"]] for row in audit} == expected_texts
    for call in rows(output / "calls.jsonl"):
        request = json.dumps(call["request"])
        assert "HIDDEN_AUDIT_ERROR_DETAIL" not in request
        assert "HIDDEN_FALSE_CLAIM" not in request and "d_evidence_audit" not in request
        assert "d_target_answer" not in request and "d_wrong_option" not in request
        assert "d_error_explanation" not in request and '"gold"' not in request
        assert "D:evidence_error" not in request
        if ":D:evidence_error:" in call["run_id"] and call["request"]["task"] in {"answer", "host"}:
            assert len(call["request"]["evidence"]) == 15
    main(["evaluate", "--results", str(output / "results.jsonl"), "--output", str(output / "report.json")])
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["d_target_probability_change"]["MedXpertQA Text/ordinary/D:evidence_error"]["n_valid_pairs"] == 1
    assert "without sum-to-one checking" in report["probability_note"]
    bad_target = {**results[-1], "d_target_answer": "A"}
    assert d_target_probability_change([results[0], bad_target])["MedXpertQA Text/ordinary/D:evidence_error"]["n_valid_pairs"] == 0
