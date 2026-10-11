from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any


@dataclass(frozen=True)
class Evidence:
    """可核验的证据片段；derived_from 仅作审计，不能发给被测模型。"""
    chunk_id: str
    document_id: str
    text: str
    source: str = "unknown"
    position: str = ""
    retrieved_at: str = ""
    content_hash: str = ""
    derived_from: str | None = None
    title: str = ""


@dataclass(frozen=True)
class Question:
    """题目与评价标签分开存；public() 永不包含金标准答案。"""
    question_id: str
    dataset: str
    stem: str
    options: dict[str, str]
    gold: str | None = None
    context: str = ""

    def public(self) -> dict[str, Any]:
        return {"question_id": self.question_id, "dataset": self.dataset,
                "stem": self.stem, "options": self.options, "context": self.context}


@dataclass(frozen=True)
class Claim:
    """专家外显命题；答案、极性和适用条件参与语义判等。"""
    claim_id: str
    text: str
    answer: str
    citation_ids: tuple[str, ...] = ()
    semantic_cluster: str | None = None
    polarity: str = "support"
    conditions: str = ""


@dataclass(frozen=True)
class DependencyEdge:
    """日志中的观察关系；exposed 只表示可见，不能解释成因果影响。"""
    source_id: str
    target_id: str
    relation: str
    round_id: int
    observed: bool = True
    observed_at: float = field(default_factory=time.time)


@dataclass
class AgentTurn:
    """一次专家或主持者发言，保存当时可见材料以供回放。"""
    turn_id: str
    role: str
    round_id: int
    order: int
    visible_evidence: list[str]
    visible_messages: list[str]
    visible_payload: dict[str, Any]
    answer: str
    probabilities: dict[str, float]
    citations: list[str]
    claims: list[Claim]
    opinion: str
    raw_response: dict[str, Any]
    reused_from: str | None = None
    attempt_id: str | None = None
    reused_attempt_id: str | None = None

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Budget:
    """跨运行累计 API 次数和已报告 token；失败调用也消耗次数。"""
    max_calls: int = 500
    max_tokens: int = 500000
    calls: int = 0
    tokens: int = 0

    def before_call(self) -> None:
        if self.calls >= self.max_calls or self.tokens >= self.max_tokens:
            raise RuntimeError("API budget exhausted")

    def after_call(self, usage: dict[str, Any]) -> None:
        self.calls += 1
        self.tokens += int(usage.get("total_tokens") or 0)


def validate_answer(data: dict[str, Any], options: dict[str, str], evidence_ids: set[str],
                    message_ids: set[str]) -> dict[str, Any]:
    """在入图前检查完整概率、合法选项和真实可见的引用。"""
    labels = set(options)
    if data.get("answer") not in labels:
        raise ValueError("invalid answer label")
    probs = data.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != labels:
        raise ValueError("probabilities must include every option exactly once")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1 for v in probs.values()):
        raise ValueError("invalid probability value")
    if abs(sum(probs.values()) - 1) > 0.01:
        raise ValueError("probabilities must sum to one")
    cites = data.get("citations", [])
    if not isinstance(cites, list) or not all(isinstance(x, str) and (x in evidence_ids or x in message_ids) for x in cites):
        raise ValueError("citation not visible")
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("claims must be a list")
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str) or not claim["text"].strip():
            raise ValueError("invalid claim")
        if claim.get("answer") not in labels:
            raise ValueError("claim answer invalid")
        claim_cites = claim.get("citation_ids", [])
        if not isinstance(claim_cites, list) or not all(isinstance(c, str) and (c in evidence_ids or c in message_ids) for c in claim_cites):
            raise ValueError("claim citation not visible")
        if claim.get("polarity", "support") not in {"support", "contradict", "limit", "uncertain"}:
            raise ValueError("invalid claim polarity")
        if not isinstance(claim.get("conditions", ""), str):
            raise ValueError("claim conditions must be text")
    if not isinstance(data.get("opinion"), str):
        raise ValueError("opinion is required")
    return data
