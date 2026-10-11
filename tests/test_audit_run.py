import json

from evidence_mdt.models import MockModel
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import Runner
from evidence_mdt.schema import Evidence, Question
from scripts.audit_run import audit


def test_audit_detects_visibility_probability_and_graph_error(tmp_path):
    question = Question("fixture", "Fixture", "Choose symbol", {"A": "alpha", "B": "beta"}, "A")
    evidence = [Evidence("chunk", "doc", "alpha evidence")]
    runner = Runner(MockModel(), Retriever(evidence, MockDense()), tmp_path)
    _, r0 = runner.run_group(question, 1, "ordinary", evidence)
    runner.run_group(question, 1, "semantic", evidence, r0_source=r0)
    clean = audit(tmp_path)
    assert clean["successful_runs"] == 2
    assert clean["issue_count"] == 0

    path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ordinary = "fixture:repeat-1:natural:ordinary"
    first = next(e for e in events if e.get("type") == "turn" and e.get("run_id") == ordinary
                 and e.get("round_id") == 1 and e.get("order") == 0)
    future = next(e for e in events if e.get("type") == "turn" and e.get("run_id") == ordinary
                  and e.get("round_id") == 1 and e.get("order") == 1)
    first["visible_messages"].append(future["turn_id"])
    first["raw_response"]["probabilities"] = {"A": 0.9, "B": 0.9}
    edge = next(e for e in events if e.get("type") == "edge" and e.get("run_id") == ordinary)
    edge["source_id"] = "missing-node"
    path.write_text("".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
    broken = audit(tmp_path)
    assert broken["issues_by_code"]["round_1_visibility"] >= 1
    assert broken["issues_by_code"]["invalid_turn_probability_or_citation"] >= 1
    assert broken["issues_by_code"]["unresolved_graph_endpoint"] >= 1
