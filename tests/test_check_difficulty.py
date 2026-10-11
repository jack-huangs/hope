import json
import hashlib
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_difficulty import decide  # noqa: E402


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")


def base_rows():
    assigned, results = [], []
    for q in range(30):
        for repeat in (1, 2):
            base = {"dataset": "MedQA", "group": "ordinary", "variant": "natural",
                    "question_id": str(q), "repeat_id": repeat,
                    "run_id": f"{q}:repeat-{repeat}:natural:ordinary", "fingerprint": "fixed"}
            assigned.append(base)
            results.append({**base, "status": "ok", "gold": "A", "answer": "A",
                            "options": {"A": "a", "B": "b", "C": "c", "D": "d"}})
    return assigned, results


def setup_source(tmp_path):
    questions = [{"question_id": str(q), "dataset": "MedQA", "gold": "A",
                  "options": {"A": "a", "B": "b", "C": "c", "D": "d"}} for q in range(30)]
    path = tmp_path / "questions.jsonl"
    write_rows(path, questions)
    (tmp_path / "config.json").write_text(json.dumps({"questions": str(path),
        "questions_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fingerprint": "fixed"}), encoding="utf-8")


def test_pending_until_all_sixty_successes(tmp_path):
    setup_source(tmp_path)
    assigned, results = base_rows()
    write_rows(tmp_path / "assignments.jsonl", assigned)
    write_rows(tmp_path / "results.jsonl", results[:-1])
    report = decide(tmp_path)
    assert report["status"] == "pending"
    assert report["switch_to_medxpertqa_text"] is None


def test_threshold_and_duplicate_guard(tmp_path):
    setup_source(tmp_path)
    assigned, results = base_rows()
    for row in results[54:]:
        row["answer"] = "B"
    write_rows(tmp_path / "assignments.jsonl", assigned)
    write_rows(tmp_path / "results.jsonl", results)
    report = decide(tmp_path)
    assert report["correct_of_60"] == 54
    assert report["switch_to_medxpertqa_text"] is True
    write_rows(tmp_path / "results.jsonl", results + [results[0]])
    assert decide(tmp_path)["status"] == "invalid"


def test_below_threshold_and_source_mismatch(tmp_path):
    setup_source(tmp_path)
    assigned, results = base_rows()
    for row in results[53:]:
        row["answer"] = "B"
    write_rows(tmp_path / "assignments.jsonl", assigned)
    write_rows(tmp_path / "results.jsonl", results)
    report = decide(tmp_path)
    assert report["correct_of_60"] == 53
    assert report["switch_to_medxpertqa_text"] is False
    results[0]["gold"] = "B"
    write_rows(tmp_path / "results.jsonl", results)
    assert decide(tmp_path)["status"] == "invalid"


def test_hash_and_fingerprint_and_bad_log_are_invalid(tmp_path):
    setup_source(tmp_path)
    assigned, results = base_rows()
    write_rows(tmp_path / "assignments.jsonl", assigned)
    write_rows(tmp_path / "results.jsonl", results)
    assigned[0]["fingerprint"] = "other"
    write_rows(tmp_path / "assignments.jsonl", assigned)
    assert decide(tmp_path)["status"] == "invalid"
    assigned[0]["fingerprint"] = "fixed"
    write_rows(tmp_path / "assignments.jsonl", assigned)
    path = tmp_path / "questions.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert decide(tmp_path)["status"] == "invalid"
    setup_source(tmp_path)
    with (tmp_path / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{bad json\n")
    assert decide(tmp_path)["status"] == "invalid"
    results[0]["gold"] = "A"
    results[0]["question_id"] = "outside"
    write_rows(tmp_path / "results.jsonl", results)
    assert decide(tmp_path)["status"] == "invalid"
