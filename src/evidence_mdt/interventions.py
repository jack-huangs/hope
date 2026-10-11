from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from typing import Any

from .schema import Evidence, Question


B_CONDITIONS = ("base", "literal3", "paraphrase3")
B_EXTENDED_CONDITIONS = ("literal5", "paraphrase5")
C_CONDITIONS = ("evidence_error", "message_error", "repeated_message")


def subset25(questions: list[Question], seed: int = 20261008) -> list[Question]:
    clinical = [q for q in questions if not q.dataset.lower().startswith("pubmedqa")]
    pqa = [q for q in questions if q.dataset.lower().startswith("pubmedqa")]
    if len(clinical) < 15 or len(pqa) < 10:
        raise ValueError("B/C requires at least 15 clinical and 10 PubMedQA questions")
    # 两个分层独立随机流，临床题集切换不改变 PubMedQA 抽样。
    return (random.Random(f"{seed}:clinical").sample(clinical, 15)
            + random.Random(f"{seed}:pubmedqa").sample(pqa, 10))


def b_evidence(base: list[Evidence], condition: str, target_id: str,
               paraphrases: list[str] | None = None) -> list[Evidence]:
    """按总暴露 3/5 次复制目标；原始片段已算一次，副本保留真实谱系供审计。"""
    if condition not in B_CONDITIONS + B_EXTENDED_CONDITIONS:
        raise ValueError("Unknown B condition")
    target = next((e for e in base if e.chunk_id == target_id), None)
    if target is None:
        raise ValueError("Target evidence not in initial snapshot")
    if condition == "base":
        return list(base)
    count = 3 if condition.endswith("3") else 5
    texts = [target.text] * (count - 1)
    if condition.startswith("paraphrase"):
        if paraphrases is None or len(paraphrases) < count - 1:
            raise ValueError("Investigator-supplied paraphrases required")
        texts = paraphrases[:count - 1]
    out = list(base)
    for i, text in enumerate(texts, 1):
        # Distinct IDs preserve experimental duplicate exposure after retrieval ID fusion.
        clone_id = "chunk:" + hashlib.sha256(f"{target.chunk_id}:{condition}:{i}".encode()).hexdigest()[:16]
        doc_id = ("document:" + hashlib.sha256(clone_id.encode()).hexdigest()[:12]) if condition.startswith("paraphrase") else target.document_id
        out.append(replace(target, chunk_id=clone_id, document_id=doc_id,
                           text=text, content_hash=hashlib.sha256(text.encode()).hexdigest(),
                           derived_from=target.chunk_id))
    return out


def c_evidence(base: list[Evidence], perturbation: str, question_id: str) -> list[Evidence]:
    if not perturbation.strip():
        raise ValueError("Investigator-supplied perturbation required")
    digest = hashlib.sha256(perturbation.encode()).hexdigest()
    return list(base) + [Evidence(f"chunk:{digest[:16]}", f"document:{digest[16:28]}",
                                  perturbation, "local-document", "", "", digest)]


def c_message(question: Question, repeat_id: int, text: str,
              repeat_message: bool = False, wrong_option: str | None = None) -> dict[str, Any]:
    """先固定接收者和错选项，再把选项填入研究者提供的消息模板。"""
    if not text.strip():
        raise ValueError("Investigator-supplied erroneous message required")
    if question.gold is None:
        raise ValueError("C message condition requires known gold label for wrong-option selection")
    wrong = [x for x in question.options if x != question.gold]
    rng = random.Random(f"{question.question_id}:{repeat_id}:C")
    receiver = rng.randrange(4)
    speaker = rng.choice([x for x in range(4) if x != receiver])
    selected = wrong_option or rng.choice(wrong)
    if selected not in wrong:
        raise ValueError("C wrong_option must be a non-gold option")
    if wrong_option is None and "{wrong_label}" not in text and "{wrong_text}" not in text:
        raise ValueError("C message text must use {wrong_label} or {wrong_text} when option is sampled")
    displayed = text.replace("{wrong_label}", selected).replace("{wrong_text}", question.options[selected])
    return {"receiver": receiver, "speaker": speaker, "wrong_option": selected,
            "round_id": rng.randint(1, 4), "text": displayed,
            "text_template": text, "repeat_message": repeat_message}
