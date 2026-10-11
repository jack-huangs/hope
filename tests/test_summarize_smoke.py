import json

from scripts.summarize_smoke import make_report, render_markdown


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_partial_report_counts_attempts_usage_and_hides_raw_content(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_backend": "deepseek",
                                                     "model_name": "fixture-model",
                                                     "fingerprint": "fixed"}), encoding="utf-8")
    assignments = [{"run_id": f"q:repeat-1:natural:{group}", "dataset": "MedQA",
                    "question_id": "q", "repeat_id": 1, "variant": "natural", "group": group}
                   for group in ("ordinary", "semantic", "semantic_retrieval")]
    write_lines(tmp_path / "assignments.jsonl", assignments)
    write_lines(tmp_path / "results.jsonl", [{**assignments[0], "status": "ok", "gold": "A",
        "answer": "A", "options": {"A": "alpha", "B": "beta"},
        "probabilities": {"A": 0.8, "B": 0.2}, "expert_answers": ["A"] * 4,
        "r0_answers": ["A"] * 4, "trigger": {"triggered": False}}])
    write_lines(tmp_path / "failures.jsonl", [{**assignments[0], "error": "invalid answer label"},
                                                   {**assignments[1], "error": "API budget exhausted"}])
    write_lines(tmp_path / "events.jsonl", [
        {"run_id": assignments[0]["run_id"], "type": "response_validation_failed",
         "response": "PRIVATE_BAD_RESPONSE"},
        {"run_id": assignments[0]["run_id"], "type": "response_repair_checked"},
        {"run_id": assignments[1]["run_id"], "type": "supplement", "chunk_ids": ["secret-id"]},
    ])
    write_lines(tmp_path / "calls.jsonl", [
        {"run_id": assignments[0]["run_id"], "request": {"task": "answer", "question": "PRIVATE_QUESTION"},
         "response": "PRIVATE_RAW", "status": "ok", "usage": {"total_tokens": 100,
         "prompt_cache_hit_tokens": 20, "prompt_cache_miss_tokens": 50, "completion_tokens": 30},
         "elapsed_seconds": 2.0},
        {"run_id": assignments[0]["run_id"], "request": {"task": "repair", "original_response": "PRIVATE_RAW"},
         "response": "PRIVATE_REPAIR", "status": "ok", "usage": {"total_tokens": 40,
         "prompt_cache_hit_tokens": 10, "prompt_cache_miss_tokens": 20, "completion_tokens": 10},
         "elapsed_seconds": 1.0},
    ])
    report = make_report(tmp_path)
    assert report["status"] == {"assigned": 3, "completed": 1, "currently_failed": 1,
                                "pending": 1, "failure_attempts_recorded": 2,
                                "completed_after_prior_failure": 1}
    assert report["usage"]["known_total_tokens"] == 140
    assert report["usage"]["known_cache_hit_input_tokens"] == 30
    assert report["usage"]["known_cache_miss_input_tokens"] == 70
    assert report["usage"]["known_output_tokens"] == 40
    assert report["format"]["issues_observed"] == 1
    assert report["format"]["repair_calls"] == 1
    assert report["supplement"]["new_chunks_in_attempt_events"] == 1
    assert report["supplement"]["criteria_met_completed_runs"] == 0
    assert report["supplement"]["eligible_triggered_completed_runs"] == 0
    assert report["natural_by_dataset_group"]["MedQA/ordinary/natural"]["brier_denominator"] == 1
    output = json.dumps(report, ensure_ascii=False) + render_markdown(report)
    assert all(secret not in output for secret in
               ("PRIVATE_QUESTION", "PRIVATE_RAW", "PRIVATE_REPAIR", "PRIVATE_BAD_RESPONSE", "secret-id"))
    assert "下一步" in output
    assert "执行增量成本" in output


def test_supplement_criteria_separate_from_actual_retrieval(tmp_path):
    assignments = [{"run_id": group, "dataset": "MedQA", "question_id": "q", "repeat_id": 1,
                    "variant": "natural", "group": group}
                   for group in ("ordinary", "semantic", "semantic_retrieval")]
    write_lines(tmp_path / "assignments.jsonl", assignments)
    write_lines(tmp_path / "results.jsonl", [{**a, "status": "ok", "trigger": {"triggered": True},
        "supplemental_chunk_ids": []} for a in assignments])
    write_lines(tmp_path / "events.jsonl", [{"run_id": "semantic_retrieval", "type": "supplement",
                                               "chunk_ids": []}])
    report = make_report(tmp_path)
    assert report["supplement"]["criteria_met_completed_runs"] == 3
    assert report["supplement"]["eligible_triggered_completed_runs"] == 1
    assert report["supplement"]["supplement_attempt_events"] == 1
