"""在首个模型调用前固定并检查 B/C 素材。医学内容仍须研究者人工核对。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .data import read_json_records
from .interventions import b_evidence, c_evidence, c_message, subset25
from .retrieval import Retriever
from .schema import Evidence, Question


def prepare_interventions(questions: list[Question], retriever: Retriever,
                          mode: str, spec_file: Path, repeats: int,
                          b_conditions: tuple[str, ...], query_lookup: dict[str, str],
                          seed: int = 20261008
                          ) -> tuple[set[str], dict[str, dict[str, Any]],
                                     dict[str, list[Evidence]], dict[str, dict[str, Any]],
                                     dict[tuple[str, int, str], dict[str, Any]], dict[str, Any]]:
    selected = subset25(questions, seed)
    selected_ids = {q.question_id for q in selected}
    if len(selected_ids) != 25:
        raise ValueError("B/C selected question IDs must be unique")
    rows = read_json_records(spec_file)
    ids = [row.get("question_id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != selected_ids:
        raise ValueError("Intervention spec must have exactly one row for each selected question")
    specs = {row["question_id"]: row for row in rows}
    bases: dict[str, list[Evidence]] = {}
    traces: dict[str, dict[str, Any]] = {}
    injections: dict[tuple[str, int, str], dict[str, Any]] = {}
    for question in selected:
        excluded = set()
        if question.dataset.lower().startswith("pubmedqa"):
            excluded = {e.chunk_id for e in retriever.evidence if
                        e.document_id in {question.question_id, f"PMID:{question.question_id}"}
                        or (question.context and e.text.strip() in question.context)}
        base = retriever.search(query_lookup[question.question_id], limit=6, exclude=excluded)
        bases[question.question_id] = base
        traces[question.question_id] = dict(retriever.last_trace)
        spec = specs[question.question_id]
        if mode in {"b", "both"}:
            target_id = spec.get("b_target_id")
            if target_id not in {e.chunk_id for e in base}:
                raise ValueError(f"B target not in initial top-6 snapshot: {question.question_id}")
            if spec.get("target_answer") not in question.options:
                raise ValueError(f"B target_answer must be an option key: {question.question_id}")
            paraphrases = spec.get("b_paraphrases")
            needed = 4 if "paraphrase5" in b_conditions else 2
            if (not isinstance(paraphrases, list) or len(paraphrases) < needed
                    or any(not isinstance(text, str) or not text.strip() for text in paraphrases[:needed])):
                raise ValueError(f"B requires {needed} nonempty investigator paraphrases: {question.question_id}")
            original = next(e.text for e in base if e.chunk_id == target_id)
            normalized = [" ".join(text.casefold().split()) for text in paraphrases[:needed]]
            if len(set(normalized)) != needed or " ".join(original.casefold().split()) in normalized:
                raise ValueError(f"B paraphrases must be distinct from each other and original: {question.question_id}")
            for condition in b_conditions:
                b_evidence(base, condition, target_id, paraphrases)
        if mode in {"c", "both"}:
            perturbation = spec.get("c_evidence_text")
            if not isinstance(perturbation, str):
                raise ValueError(f"C evidence text required: {question.question_id}")
            c_evidence(base, perturbation, question.question_id)
            template = spec.get("c_message_text")
            if not isinstance(template, str):
                raise ValueError(f"C message template required: {question.question_id}")
            for repeat_id in range(1, repeats + 1):
                injection = c_message(question, repeat_id, template, False,
                                      spec.get("c_wrong_option"))
                injections[(question.question_id, repeat_id, "message_error")] = injection
                injections[(question.question_id, repeat_id, "repeated_message")] = {
                    **injection, "repeat_message": True}
    manifest = {
        "seed": seed, "selected": [{"question_id": q.question_id, "dataset": q.dataset} for q in selected],
        "clinical_count": 15, "pubmedqa_count": 10,
        "b_conditions": list(b_conditions) if mode in {"b", "both"} else [],
        "c_conditions": ["evidence_error", "message_error", "repeated_message"] if mode in {"c", "both"} else [],
        "spec_sha256": hashlib.sha256(spec_file.read_bytes()).hexdigest(),
        "initial_snapshots": {qid: trace.get("returned_chunk_ids", []) for qid, trace in traces.items()},
        "c_assignments": [{"question_id": qid, "repeat_id": repeat_id, "condition": name,
                           "receiver": item["receiver"], "speaker": item["speaker"],
                           "wrong_option": item["wrong_option"], "round_id": item["round_id"]}
                          for (qid, repeat_id, name), item in sorted(injections.items())],
        "audit_note": "Human-authored perturbations are not verified medical truth; hidden labels stay evaluator-only.",
    }
    return selected_ids, specs, bases, traces, injections, manifest
