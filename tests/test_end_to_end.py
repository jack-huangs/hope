"""合成题仅检验软件流程；绝不视为医学结果。"""
import json

from evidence_mdt.interventions import B_CONDITIONS, b_evidence, c_evidence, c_message
from evidence_mdt.models import MockModel
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import GROUPS, Runner
from evidence_mdt.schema import Budget, Evidence, Question


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_all_variants_two_repeats_with_mock_and_auditable_graph(tmp_path):
    question = Question("synthetic-1", "SoftwareFixture", "Choose the symbolic answer",
                        {"A": "alpha", "B": "beta", "C": "gamma"}, "A")
    base = [Evidence("chunk-alpha", "doc-alpha", "symbol alpha appears"),
            Evidence("chunk-beta", "doc-beta", "symbol beta appears")]
    runner = Runner(MockModel(), Retriever(base, MockDense()), tmp_path, Budget(max_calls=100000))
    all_records = []
    variants = [("natural", base, None)]
    for condition in B_CONDITIONS:
        exposed = b_evidence(base, condition, "chunk-alpha",
                             [f"paraphrase {i} of alpha" for i in range(4)])
        variants.append(("B:" + condition, exposed, None))
        assert len(exposed) == len(base) + (0 if condition == "base" else
                                            (2 if condition.endswith("3") else 4))
    variants.append(("C:evidence_error", c_evidence(base, "synthetic erroneous assertion", question.question_id), None))
    for name in ("message_error", "repeated_message"):
        variants.append(("C:" + name, base, name))

    for repeat in (1, 2):
        natural_r0 = None
        for variant, exposed, injection_name in variants:
            first_r0 = natural_r0 if injection_name else None
            for group in GROUPS:
                injection = (c_message(question, repeat, "synthetic peer favors {wrong_label}",
                                       repeat_message=injection_name == "repeated_message")
                             if injection_name else None)
                result, r0 = runner.run_group(question, repeat, group, exposed, variant,
                                               first_r0, injection)
                all_records.append(result)
                if first_r0 is None:
                    first_r0 = r0
            if variant == "natural":
                natural_r0 = first_r0
    assert len(all_records) == 42
    assert len({r["run_id"] for r in all_records}) == 42
    assert all(r["r0_reused"] for r in all_records if r["group"] != "ordinary")
    assert all(r["r0_reused"] for r in all_records if r["variant"] in
               {"C:message_error", "C:repeated_message"})
    assert all(not r["r0_reused"] for r in all_records if r["variant"] == "C:evidence_error"
               and r["group"] == "ordinary")

    events = _read(tmp_path / "events.jsonl")
    calls = _read(tmp_path / "calls.jsonl")
    ids = {e["turn_id"] for e in events if e.get("type") == "turn"}
    ids |= {e["claim_id"] for e in events if e.get("type") == "claim"}
    ids |= {e["id"] for e in events if e.get("type") in {"cluster", "view_node"}}
    ids |= {e["id"] for e in events if e.get("type") == "injected_message"}
    ids |= {e.chunk_id for e in base}
    ids |= {c["chunk_id"] for call in calls for c in call["request"].get("evidence", [])}
    assert all(e["source_id"] in ids and e["target_id"] in ids
               for e in events if e.get("type") == "edge")
    assert any(e.get("type") == "injection" for e in events)
    assert all("gold" not in call["request"].get("question", {}) for call in calls)
    assert all("derived_from" not in json.dumps(call["request"]) for call in calls)
    assert all("C:message_error" not in json.dumps(call["request"]) for call in calls)
