import pytest

from evidence_mdt.metrics import (b_target_probability_change, brier, cost_summary,
                                  paired_cluster_bootstrap, summarize)


def row(question="q1", repeat=1, group="ordinary", dataset="MedQA", variant="natural",
        probs=None, answer="A", gold="A"):
    return {"dataset": dataset, "question_id": question, "repeat_id": repeat,
            "group": group, "variant": variant, "gold": gold, "answer": answer,
            "options": {"A": "one", "B": "two"},
            "probabilities": probs if probs is not None else {"A": 0.8, "B": 0.2},
            "r0_answers": ["A", "A", "B", "A"],
            "expert_answers": ["B", "A", "B", "B"]}


def test_brier_requires_complete_finite_probability_vector():
    assert brier({"A": 0.8, "B": 0.2}, "A") == pytest.approx(0.08)
    for probs in ({"A": 1.0, "B": float("nan")}, {"A": 1.1, "B": -0.1},
                  {"A": 0.3}, {"A": True, "B": 0}):
        with pytest.raises(ValueError):
            brier(probs, "A")


def test_summary_shows_assigned_denominator_and_invalid_probabilities():
    good = row()
    bad = row("q2", probs={"A": 1.0}, answer="B")
    good["expert_answers"] = ["B"] * 4
    good["answer"] = "B"
    good["probabilities"] = {"A": 0.1, "B": 0.9}
    assignments = [row("q1"), row("q2"), row("q3")]
    result = summarize([good, bad], assignments)["MedQA/ordinary/natural"]
    assert result["n_assigned"] == 3
    assert result["n_completed"] == 2
    assert result["n_failed_or_pending"] == 1
    assert result["n_valid_probability"] == 1
    assert result["brier_denominator"] == 1
    assert result["accuracy_assigned"] == 0
    assert result["high_confidence_joint_error_count"] == 1
    assert result["high_confidence_joint_error_rate_assigned"] == pytest.approx(1 / 3)
    assert result["r0_correct_expert_count"] == 6
    assert result["r0_correct_to_final_wrong_count"] == 5
    assert len(result["probability_diagnostics"]["risk_coverage"]) == 4
    assert summarize([good])["MedQA/ordinary/natural"]["accuracy_assigned"] is None


def test_bootstrap_requires_same_repeat_and_separates_datasets():
    records = [row("q1", 1, "ordinary"), row("q1", 2, "semantic", probs={"A": 0.1, "B": 0.9}),
               row("q2", 1, "ordinary"), row("q2", 1, "semantic", probs={"A": 0.7, "B": 0.3}),
               row("q2", 1, "ordinary", dataset="PubMedQA"),
               row("q2", 1, "semantic", dataset="PubMedQA", probs={"A": 0.6, "B": 0.4})]
    report = paired_cluster_bootstrap(records, "ordinary", "semantic", iterations=50)
    assert report["MedQA"]["paired_trials"] == 1
    assert report["PubMedQA"]["paired_trials"] == 1
    assert report["MedQA"]["brier_difference_b_minus_a"] == pytest.approx(0.10)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_cluster_bootstrap(records + [records[0]], "ordinary", "semantic")


def test_b_target_probability_uses_same_run_and_target():
    base = row(variant="B:base")
    treated = row(variant="B:literal3", probs={"A": 0.6, "B": 0.4})
    treated["b_target_answer"] = "B"
    result = b_target_probability_change([base, treated])
    assert result["MedQA/ordinary/B:literal3"]["mean_delta_target_probability"] == pytest.approx(0.2)
    assert b_target_probability_change([row(), treated])["MedQA/ordinary/B:literal3"]["mean_delta_target_probability"] is None
    assert cost_summary([{"run_id": "one", "usage": {"total_tokens": 10}, "elapsed_seconds": 1.2},
                         {"run_id": "one", "error": "timeout"}])["one"] == {
        "calls": 2, "failed_calls": 1, "tokens_recorded": 10,
        "calls_without_usage": 1, "elapsed_seconds_recorded": 1.2}


def test_duplicate_results_rejected_and_failures_separated():
    completed = row()
    assigned = [row("q1"), row("q2"), row("q3")]
    assigned[1]["run_id"] = "failed-run"
    report = summarize([completed], assigned, [{"run_id": "failed-run"}])["MedQA/ordinary/natural"]
    assert report["n_failed"] == 1
    assert report["n_pending"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([completed, completed])
