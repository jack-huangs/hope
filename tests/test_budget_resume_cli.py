import json

import pytest

from evidence_mdt.cli import main
from evidence_mdt.data import write_jsonl
from evidence_mdt.schema import Evidence, Question


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_new_science_budget_increase_keeps_completed_record_and_resumes(tmp_path):
    questions = tmp_path / "questions.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "run"
    write_jsonl(questions, [Question("fixture", "Fixture", "alpha?", {"A": "alpha", "B": "beta"}, "A")])
    write_jsonl(corpus, [Evidence("e", "doc", "alpha evidence", "local")])
    argv = ["run", "--questions", str(questions), "--corpus", str(corpus), "--output", str(output),
            "--model-backend", "mock", "--retrieval-backend", "mock", "--groups", "ordinary",
            "--repeats", "2", "--max-calls", "1000", "--max-tokens", "100000"]
    main(argv)
    result_path = output / "results.jsonl"
    original = rows(result_path)
    assert len(original) == 2
    original_calls = len(rows(output / "calls.jsonl"))

    config_path = output / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    first = original[0]
    write_jsonl(result_path, [first])

    resumed = argv.copy()
    resumed[resumed.index("--max-calls") + 1] = "2000"
    resumed[resumed.index("--max-tokens") + 1] = "200000"
    main(resumed)
    finished = rows(result_path)
    assert len(finished) == 2
    assert finished[0] == first
    assert all(r["fingerprint"] == config["fingerprint"] for r in finished)
    assert len(rows(output / "calls.jsonl")) > original_calls
    assert json.loads(config_path.read_text(encoding="utf-8")) == config
    assert not (output / "execution_migrations.jsonl").exists()
    limits = rows(output / "budget_limits.jsonl")
    assert limits[-1]["old_max_tokens"] == 100000
    assert limits[-1]["new_max_tokens"] == 200000
    assert limits[-1]["consumed_calls_at_start"] == original_calls
    main(resumed)
    assert len(rows(result_path)) == 2
    assert not (output / "execution_migrations.jsonl").exists()


def test_legacy_resume_rejects_scientific_change(tmp_path):
    questions = tmp_path / "questions.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "run"
    write_jsonl(questions, [Question("fixture", "Fixture", "alpha?", {"A": "alpha", "B": "beta"}, "A")])
    write_jsonl(corpus, [Evidence("e", "doc", "alpha evidence", "local")])
    argv = ["run", "--questions", str(questions), "--corpus", str(corpus), "--output", str(output),
            "--model-backend", "mock", "--retrieval-backend", "mock", "--groups", "ordinary",
            "--repeats", "1"]
    main(argv)
    with pytest.raises(ValueError, match="scientific"):
        main(argv + ["--repeats", "2"])
