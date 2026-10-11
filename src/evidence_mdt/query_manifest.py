"""固定初始检索短查询；只依据公开题面校验，不能携带金标字段。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .data import read_json_records
from .schema import Question


def public_question_sha256(question: Question) -> str:
    canonical = json.dumps(question.public(), sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def initial_queries(questions: list[Question], manifest: Path | None) -> dict[str, str]:
    if manifest is None:
        return {q.question_id: q.stem + " " + q.context for q in questions}
    rows = read_json_records(manifest)
    allowed = {"question_id", "dataset", "public_question_sha256", "retrieval_query"}
    if any(set(row) != allowed for row in rows):
        raise ValueError("Query manifest must have exactly the four public fields")
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("Question IDs must be unique for query manifest")
    by_id = {row["question_id"]: row for row in rows}
    if len(by_id) != len(rows) or not {q.question_id for q in questions} <= set(by_id):
        raise ValueError("Query manifest must contain every selected question exactly once")
    out = {}
    for question in questions:
        row = by_id[question.question_id]
        if (row["dataset"] != question.dataset
                or row["public_question_sha256"] != public_question_sha256(question)):
            raise ValueError(f"Query manifest public question mismatch: {question.question_id}")
        query = row["retrieval_query"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Empty retrieval query: {question.question_id}")
        out[question.question_id] = query.strip()
    return out


def validate_medcpt_query_lengths(lookup: dict[str, str], tokenizer, limit: int = 64) -> None:
    """显式短查询不允许被 MedCPT 悄悄截断；长度包含特殊 token。"""
    for question_id, query in lookup.items():
        encoded = tokenizer(query, truncation=False)
        ids = encoded["input_ids"]
        if len(ids) > limit:
            raise ValueError(f"MedCPT query exceeds {limit} tokens: {question_id} ({len(ids)})")
