"""改写以不同虚构来源呈现，真实谱系只保留在审计日志中。"""
import hashlib
import json
from dataclasses import asdict
from urllib.parse import urlparse

import pytest

from evidence_mdt.interventions import b_evidence
from evidence_mdt.models import MockModel
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import Runner
from evidence_mdt.schema import Evidence, Question


def original_evidence():
    return Evidence(
        "PMID:12345678:abstract", "PMID:12345678", "A finding with limited certainty.",
        source="https://pubmed.ncbi.nlm.nih.gov/12345678/",
        title="Original publication title", position="abstract chars[40:80]",
        retrieved_at="2026-10-28T00:00:00+00:00",
    )


@pytest.mark.parametrize("count", [3, 5, 11])
def test_paraphrase_sources_are_distinct_reproducible_and_auditable(count):
    target = original_evidence()
    before = asdict(target)
    parts = [f"Rewording {i} of the limited-certainty finding." for i in range(count - 1)]
    condition = f"paraphrase{count}"
    got = b_evidence([target], condition, target.chunk_id, parts)
    assert got == b_evidence([target], condition, target.chunk_id, parts)
    assert len(got) == count and got[0] is target and asdict(target) == before
    for key in ("chunk_id", "document_id", "source", "title"):
        assert len({getattr(e, key) for e in got}) == count
    for e, text in zip(got[1:], parts):
        assert e.text == text and e.derived_from == target.chunk_id
        assert e.content_hash == hashlib.sha256(text.encode()).hexdigest()
        assert urlparse(e.source).hostname.endswith(".example")
        assert e.position == "abstract" and not e.retrieved_at
        for key in ("chunk_id", "document_id", "source", "title", "position", "retrieved_at"):
            assert before[key] not in json.dumps({k: v for k, v in asdict(e).items()
                                               if k not in {"derived_from", "text"}})
    literal = b_evidence([target], f"literal{count}", target.chunk_id)
    assert all(e.text == target.text for e in literal)
    assert all(e.document_id == target.document_id and e.source == target.source
               and e.title == target.title for e in literal)


def test_experts_and_host_see_fictional_identity_but_audit_keeps_real_lineage(tmp_path):
    target = original_evidence()
    evidence = b_evidence([target], "paraphrase11", target.chunk_id,
                          [f"The limited-certainty finding, wording {i}." for i in range(10)])
    copies = {e.chunk_id: e for e in evidence[1:]}
    question = Question("identity-fixture", "synthetic", "Choose", {"A": "alpha", "B": "beta"}, "A")
    runner = Runner(MockModel(), Retriever([target], MockDense()), tmp_path)
    runner.run_group(question, 1, "ordinary", evidence, variant="B:paraphrase11")
    calls = [json.loads(l) for l in runner.call_log.read_text(encoding="utf-8").splitlines()]
    requests = [x["request"] for x in calls if x["request"]["task"] in {"answer", "host"}]
    assert len(requests) == 21 and sum(p["task"] == "host" for p in requests) == 1
    for p in requests:
        visible = {e["chunk_id"]: e for e in p["evidence"]}
        assert len(visible) == 11
        for cid, expected in copies.items():
            assert visible[cid] == {k: v for k, v in asdict(expected).items() if k != "derived_from"}
        assert "derived_from" not in json.dumps(p)
    events = [json.loads(l) for l in runner.event_log.read_text(encoding="utf-8").splitlines()]
    audited = {e["chunk_id"]: e for e in events if e["type"] == "evidence"}
    assert all(audited[cid]["derived_from"] == target.chunk_id for cid in copies)
    assert all(any(e.get("type") == "edge" and e.get("relation") == "derived_from"
                   and e["source_id"] == target.chunk_id and e["target_id"] == cid
                   for e in events) for cid in copies)
