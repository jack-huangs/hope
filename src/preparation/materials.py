"""固定干预材料；只有此阶段允许读取金标。"""
from dataclasses import asdict
import random

from evidence_mdt.interventions import B_PARAPHRASE_COUNT
from . import prompts


B_TARGET_SELECTION_POLICY = "supported-options-uniform-v1"


def validate_b_annotations(value, evidence_ids, options):
    """要求逐条覆盖冻结证据，并按冻结顺序返回，避免模型输出顺序影响抽样。"""
    rows = value.get("annotations") if isinstance(value, dict) else None
    if (not isinstance(rows, list) or len(rows) != len(evidence_ids) or
            len(set(evidence_ids)) != len(evidence_ids)):
        raise ValueError("B annotations must cover every frozen evidence ID exactly once")
    by_id = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("chunk_id"), str) or
                row["chunk_id"] not in evidence_ids or row["chunk_id"] in by_id):
            raise ValueError("B annotations contain duplicate or unknown evidence IDs")
        if "target_answer" not in row or (row["target_answer"] is not None and (
                not isinstance(row["target_answer"], str) or row["target_answer"] not in options)):
            raise ValueError("B target_answer must be an option key or explicit null")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise ValueError("B evidence annotation requires a reason")
        by_id[row["chunk_id"]] = {key: row[key] for key in ("chunk_id", "target_answer", "reason")}
    return [by_id[cid] for cid in evidence_ids]


def b_selection(annotations, seed, question_id):
    supported = [row for row in annotations if row["target_answer"] is not None]
    pool = supported or annotations
    target = random.Random(f"{seed}:{question_id}:b").choice(pool)
    return {"b_target_id": target["chunk_id"], "target_answer": target["target_answer"],
            "b_target_reason": target["reason"],
            "b_target_candidate_ids": [row["chunk_id"] for row in pool],
            "b_target_selection": "supported" if supported else "all_null_fallback"}


def validate_b_paraphrases(value, original_text):
    parts = value.get("paraphrases") if isinstance(value, dict) else None
    norm = lambda text: " ".join(text.casefold().split())
    if (not isinstance(parts, list) or len(parts) != B_PARAPHRASE_COUNT or
            any(not isinstance(x, str) or not x.strip() for x in parts) or
            len({norm(x) for x in parts} | {norm(original_text)}) != B_PARAPHRASE_COUNT + 1):
        raise ValueError(f"B requires exactly {B_PARAPHRASE_COUNT} distinct paraphrases")
    return parts


def prepare_material(question, base, mode, seed, call, previous=None, checkpoint=None):
    spec = dict(previous or {"question_id": question.question_id})
    if mode == "none":
        return spec
    rng = random.Random(f"{seed}:{question.question_id}:{mode}")
    if mode == "b":
        if not base:
            raise ValueError("B requires nonempty frozen initial evidence")
        evidence_ids = [e.chunk_id for e in base]
        if "b_evidence_annotations" not in spec:
            if "b_target_id" in spec:
                raise ValueError("B target lacks full evidence annotations; use a new preparation")
            value = call("b_annotation", prompts.B_ANNOTATE,
                         {"question": question.public(), "evidence": [asdict(e) for e in base]})
        else:
            value = {"annotations": spec["b_evidence_annotations"]}
        annotations = validate_b_annotations(value, evidence_ids, question.options)
        spec["b_evidence_annotations"] = annotations
        if checkpoint:
            checkpoint(spec)
        selection = b_selection(annotations, seed, question.question_id)
        if any(key in spec and spec[key] != value for key, value in selection.items()):
            raise ValueError("Saved B selection differs from annotated candidate pool and seed")
        spec.update(selection)
        # 在生成前保存随机分配；重试绝不另选片段。
        if checkpoint:
            checkpoint(spec)
        target = next(e for e in base if e.chunk_id == spec["b_target_id"])
        value = call("b", prompts.B, {"original": asdict(target)})
        spec["b_paraphrases"] = validate_b_paraphrases(value, target.text)
    else:
        if question.gold not in question.options:
            raise ValueError("C/D require known valid gold")
        wrong = spec.get(f"{mode}_wrong_option") or rng.choice(
            sorted(key for key in question.options if key != question.gold))
        if wrong not in question.options or wrong == question.gold:
            raise ValueError("C/D wrong_option must be a non-gold option")
        spec[f"{mode}_wrong_option"] = wrong
        if checkpoint:
            checkpoint(spec)
        value = call(mode, prompts.C if mode == "c" else prompts.D,
                     {"question": question.public(), "gold": question.gold, "wrong_option": wrong})
        if mode == "d":
            spec["d_evidence_passages"] = value["d_evidence_passages"]
        else:
            if not isinstance(value.get("c_message_text"), str) or not value["c_message_text"].strip():
                raise ValueError("Missing material field: c_message_text")
            spec["c_message_text"] = value["c_message_text"]
    return spec
