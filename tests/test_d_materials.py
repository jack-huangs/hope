"""D五段不同错误证据的离线生成、冻结、暴露及历史读取测试。"""
from copy import deepcopy
import json
import random

import pytest

from evidence_mdt.cli import main as run_cli
from evidence_mdt.interventions import D_EVIDENCE_ANGLES, d_evidence
from evidence_mdt.retrieval import MockDense
from evidence_mdt.schema import Evidence, Question
from preparation import prepare
from preparation.audit import verify_preparation
from preparation.materials import prepare_material
from preparation.state import digest, jsonl, save
from d_material_fixture import d_passages
from test_preparation_unified import Model, Tokenizer, fetch, source


def test_five_d_passages_append_independent_ids_without_changing_original_or_losing_items():
    base = [Evidence(f"e{i}", f"d{i}", f"Original passage {i}.") for i in range(10)]
    before = deepcopy(base)
    texts = [row["text"] for row in d_passages()]
    evidence = d_evidence(base, texts, "q")
    assert evidence == d_evidence(base, texts, "q")
    assert base == before == evidence[:10]
    assert len(evidence) == len({e.chunk_id for e in evidence}) == 15
    assert len({e.document_id for e in evidence}) == 15
    assert [e.text for e in evidence[10:]] == texts
    assert all(e.content_hash == digest(e.text.encode()) for e in evidence[10:])
    again = d_evidence(evidence, texts, "q")
    assert again[:15] == evidence and len({e.chunk_id for e in again}) == 20
    other = d_evidence(base, texts, "another-q")
    assert {e.chunk_id for e in other[10:]}.isdisjoint({e.chunk_id for e in evidence[10:]})


def test_d_generation_uses_one_common_seeded_wrong_option_and_saves_before_generation():
    question = Question("d-fixture", "MedXpertQA", "Choose", {"A": "one", "B": "two", "C": "three"}, "A")
    selected = set()
    for seed in range(20):
        calls, saved = [], []
        def call(stage, instruction, data):
            calls.append((stage, data))
            assert saved[-1]["d_wrong_option"] == data["wrong_option"]
            return {"d_evidence_passages": list(reversed(d_passages()))}
        spec = prepare_material(question, [], "d", seed, call,
                                checkpoint=lambda row: saved.append(deepcopy(row)))
        expected = random.Random(f"{seed}:{question.question_id}:d").choice(["B", "C"])
        assert spec["d_wrong_option"] == expected
        assert calls == [("d", {"question": question.public(), "gold": "A", "wrong_option": expected})]
        assert [row["angle"] for row in spec["d_evidence_passages"]] == list(reversed(D_EVIDENCE_ANGLES))
        assert len({row["false_claim"] for row in spec["d_evidence_passages"]}) == 5
        selected.add(expected)
    assert selected == {"B", "C"}


def test_d_timeout_resume_keeps_wrong_option_and_does_not_expand_one_passage():
    question = Question("d-fixture", "MedXpertQA", "Choose", {"A": "one", "B": "two"}, "A")
    saved = []
    def timeout(*args):
        raise TimeoutError("temporary model failure")
    with pytest.raises(TimeoutError):
        prepare_material(question, [], "d", 123, timeout, checkpoint=lambda row: saved.append(deepcopy(row)))
    assert saved == [{"question_id": question.question_id, "d_wrong_option": "B"}]
    spec = prepare_material(question, [], "d", 123, lambda *args: {"d_evidence_passages": d_passages()},
                            previous=saved[0])
    assert spec["d_wrong_option"] == "B" and len(spec["d_evidence_passages"]) == 5


@pytest.mark.parametrize("output_kind", ["four", "six", "copy", "same_claim", "missing_audit"])
def test_d_generation_is_cached_and_injected_without_passage_checks(tmp_path, output_kind):
    class UncheckedD(Model):
        def complete(self, payload):
            result = super().complete(payload)
            if "d_evidence_passages" in payload["instruction"]:
                passages = result.content["d_evidence_passages"]
                if output_kind == "four":
                    passages.pop()
                elif output_kind == "six":
                    passages.append(deepcopy(passages[0]))
                elif output_kind == "copy":
                    passages[-1]["text"] = passages[0]["text"]
                elif output_kind == "same_claim":
                    passages[-1]["angle"] = passages[0]["angle"]
                    passages[-1]["false_claim"] = passages[0]["false_claim"]
                else:
                    passages[:] = [{"text": row["text"]} for row in passages]
                    passages[-1]["text"] = ""
                self.generated = deepcopy(passages)
            return result
    path, _ = source(tmp_path, 1)
    output = tmp_path / "prepared"
    model = UncheckedD()
    state = prepare(path, output, model, MockDense(), interventions="d",
                    tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert state["status"] == "complete"
    saved = state["records"][0]["spec"]
    assert saved["d_evidence_passages"] == model.generated
    assert verify_preparation(output)["d_evidence_count"] == len(model.generated)
    before = len(model.calls)
    prepare(path, output, model, MockDense(), interventions="d",
            tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    assert len(model.calls) == before
    run = tmp_path / "run"
    run_cli(["run", "--questions", str(path), "--corpus", str(output / "corpus.jsonl"),
             "--output", str(run), "--model-backend", "mock", "--retrieval-backend", "mock",
             "--query-manifest", str(output / "initial_queries.jsonl"),
             "--interventions", "d", "--intervention-spec", str(output / "intervention_spec.jsonl"),
             "--groups", "ordinary"])
    manifest = json.loads((run / "intervention_manifest.json").read_bytes())
    injected = manifest["d_assignments"][0]["passages"]
    assert len(injected) == len({row["chunk_id"] for row in injected}) == len(model.generated)
    results = [json.loads(line) for line in (run / "results.jsonl").read_bytes().splitlines()]
    assert results[-1]["variant"] == "D:evidence_error" and results[-1]["d_evidence_audit"] == injected


def test_old_single_d_material_is_only_verified_as_legacy_without_rewriting(tmp_path):
    path, _ = source(tmp_path, 1)
    output = tmp_path / "prepared"
    prepare(path, output, Model(), MockDense(), interventions="d",
            tokenizer=Tokenizer(), fetcher=fetch, retries=0)
    spec_path = output / "intervention_spec.jsonl"
    jsonl(spec_path, [{"question_id": "fixture-0", "d_wrong_option": "B",
                       "d_evidence_text": "Old single false passage.", "d_error_explanation": "Old audit."}])
    manifest_path = output / "preparation_manifest.json"
    state = json.loads(manifest_path.read_bytes())
    state["config"].pop("d_evidence_count")
    state["config"].pop("d_evidence_angles")
    state["fingerprint"] = digest(state["config"])
    state["artifacts"][spec_path.name] = digest(spec_path.read_bytes())
    save(manifest_path, state)
    before = spec_path.read_bytes()
    assert verify_preparation(output)["d_evidence_count"] == 1
    assert spec_path.read_bytes() == before
