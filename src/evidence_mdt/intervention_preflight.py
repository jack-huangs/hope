"""在首个模型调用前固定并检查 B/C/D 素材。医学内容仍须研究者人工核对。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .data import read_json_records
from .interventions import (b_evidence, b_exposure_count, d_evidence, c_message, intervention_modes,
                            validate_experiment_questions)
from .retrieval import INITIAL_EVIDENCE_LIMIT, Retriever
from .schema import Evidence, Question


def prepare_interventions(questions: list[Question], retriever: Retriever,
                          mode: str, spec_file: Path, repeats: int,
                          b_conditions: tuple[str, ...], query_lookup: dict[str, str],
                          seed: int = 20261008, *, allow_legacy_d: bool = False
                          ) -> tuple[set[str], dict[str, dict[str, Any]],
                                     dict[str, list[Evidence]], dict[str, dict[str, Any]],
                                     dict[tuple[str, int, str], dict[str, Any]], dict[str, Any]]:
    validate_experiment_questions(questions)
    enabled = intervention_modes(mode)
    selected = questions  # 明确传入的全部题；不在实验入口内重新抽样。
    selected_ids = {q.question_id for q in selected}
    rows = read_json_records(spec_file)
    ids = [row.get("question_id") for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != selected_ids:
        raise ValueError("Intervention spec must have exactly one row for each selected question")
    specs = {row["question_id"]: row for row in rows}
    bases: dict[str, list[Evidence]] = {}
    traces: dict[str, dict[str, Any]] = {}
    injections: dict[tuple[str, int, str], dict[str, Any]] = {}
    d_assignments = []
    for question in selected:
        if "c_evidence_text" in specs[question.question_id]:
            raise ValueError("Legacy c_evidence_text is no longer executable; explicitly prepare d_evidence_passages and d_wrong_option for D")
        base = retriever.search(query_lookup[question.question_id], limit=INITIAL_EVIDENCE_LIMIT)
        bases[question.question_id] = base
        traces[question.question_id] = dict(retriever.last_trace)
        spec = specs[question.question_id]
        if "b" in enabled:
            target_id = spec.get("b_target_id")
            if target_id not in {e.chunk_id for e in base}:
                raise ValueError(f"B target not in initial top-{INITIAL_EVIDENCE_LIMIT} snapshot: {question.question_id}")
            if "target_answer" not in spec or (spec["target_answer"] is not None and (
                    not isinstance(spec["target_answer"], str) or spec["target_answer"] not in question.options)):
                raise ValueError(f"B target_answer must be an option key or explicit null: {question.question_id}")
            if spec["target_answer"] is None and (
                    not isinstance(spec.get("b_target_reason"), str) or not spec["b_target_reason"].strip()):
                raise ValueError(f"B null target_answer requires b_target_reason: {question.question_id}")
            paraphrases = spec.get("b_paraphrases")
            needed = max((b_exposure_count(condition) - 1 for condition in b_conditions
                          if condition.startswith("paraphrase")), default=0)
            if (not isinstance(paraphrases, list) or len(paraphrases) < needed
                    or any(not isinstance(text, str) or not text.strip() for text in paraphrases[:needed])):
                raise ValueError(f"B requires {needed} nonempty investigator paraphrases: {question.question_id}")
            original = next(e.text for e in base if e.chunk_id == target_id)
            normalized = [" ".join(text.casefold().split()) for text in paraphrases[:needed]]
            if len(set(normalized)) != needed or " ".join(original.casefold().split()) in normalized:
                raise ValueError(f"B paraphrases must be distinct from each other and original: {question.question_id}")
            for condition in b_conditions:
                b_evidence(base, condition, target_id, paraphrases)
        if "d" in enabled:
            wrong = spec.get("d_wrong_option")
            if question.gold not in question.options or wrong not in question.options or wrong == question.gold:
                raise ValueError(f"D d_wrong_option must be a non-gold option with known gold: {question.question_id}")
            legacy = allow_legacy_d and "d_evidence_passages" not in spec
            if legacy:
                passages = [{"angle": "legacy_single", "text": spec["d_evidence_text"],
                             "error_explanation": spec.get("d_error_explanation", "")}]
            else:
                if "d_evidence_text" in spec or "d_error_explanation" in spec:
                    raise ValueError("Legacy single-passage D spec is not executable; prepare five d_evidence_passages")
                passages = spec["d_evidence_passages"]
            passages = [dict(row, text=row["text"].replace("{wrong_label}", wrong).replace(
                "{wrong_text}", question.options[wrong])) for row in passages]
            spec["d_evidence_passages"] = passages
            texts = [row["text"] for row in passages]
            injected = d_evidence(base, texts, question.question_id)[len(base):]
            d_assignments.append({"question_id": question.question_id, "wrong_option": wrong,
                                  "evidence_count": len(injected),
                                  "passages": [{"chunk_id": evidence.chunk_id,
                                                "document_id": evidence.document_id,
                                                "content_hash": evidence.content_hash,
                                                **{key: row.get(key, "") for key in ("angle", "false_claim", "error_explanation")}}
                                               for row, evidence in zip(passages, injected)],
                                  "insertion_point": "before_r0"})
        if "c" in enabled:
            template = spec.get("c_message_text")
            if not isinstance(template, str):
                raise ValueError(f"C message template required: {question.question_id}")
            for repeat_id in range(1, repeats + 1):
                injection = c_message(question, repeat_id, template, False,
                                      spec.get("c_wrong_option"))
                injections[(question.question_id, repeat_id, "message_error")] = injection
                injections[(question.question_id, repeat_id, "repeated_message")] = c_message(
                    question, repeat_id, template, True, spec.get("c_wrong_option"))
    d_counts = {row["evidence_count"] for row in d_assignments} or {0}
    manifest = {
        "selection": "all_supplied_medxpertqa_questions_no_sampling", "selected": [{"question_id": q.question_id, "dataset": q.dataset} for q in selected],
        "question_count": len(selected),
        "b_conditions": list(b_conditions) if "b" in enabled else [],
        "c_conditions": ["message_error", "repeated_message"] if "c" in enabled else [],
        "d_conditions": ["evidence_error"] if "d" in enabled else [],
        "d_assignments": d_assignments,
        "d_evidence_count": next(iter(d_counts)) if len(d_counts) == 1 else None,
        "spec_sha256": hashlib.sha256(spec_file.read_bytes()).hexdigest(),
        "initial_snapshots": {qid: trace.get("returned_chunk_ids", []) for qid, trace in traces.items()},
        "c_assignments": [{"question_id": qid, "repeat_id": repeat_id, "condition": name,
                           "speaker": item["speaker"], "visibility": "public",
                           "insertion_point": "round_start",
                           "second_speaker": item.get("second_speaker"),
                           "wrong_option": item["wrong_option"], "round_id": item["round_id"]}
                          for (qid, repeat_id, name), item in sorted(injections.items())],
        "message_protocol": "public_history_at_round_start_v2",
        "repeated_message_protocol": "two_distinct_speakers_public_same_error_v2",
        "audit_note": "Human-authored perturbations are not verified medical truth; hidden labels stay evaluator-only.",
    }
    return selected_ids, specs, bases, traces, injections, manifest
