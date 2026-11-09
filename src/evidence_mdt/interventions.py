#生成 B、C、D 实验所需的干预材料和配置
from __future__ import annotations

import hashlib
import random
from dataclasses import replace
from typing import Any

from .schema import Evidence, Question


B_PRESENTATIONS = 11
B_PARAPHRASE_COUNT = B_PRESENTATIONS - 1
B_CONDITIONS = ("base", f"literal{B_PRESENTATIONS}", f"paraphrase{B_PRESENTATIONS}")
# 旧日期脚本与冻结材料仍可按原条件读取，不改变新实验的默认条件。
B_LEGACY_CONDITIONS = ("base", "literal3", "paraphrase3")
B_EXTENDED_CONDITIONS = ("literal5", "paraphrase5")
C_CONDITIONS = ("message_error", "repeated_message")
D_CONDITIONS = ("evidence_error",)
D_EVIDENCE_ANGLES = ("mechanism", "clinical_pattern", "diagnostic_interpretation",
                     "differential_comparison", "course_or_response")
D_EVIDENCE_COUNT = len(D_EVIDENCE_ANGLES)


def intervention_modes(mode: str) -> frozenset[str]:
    modes = {"none": (), "b": ("b",), "c": ("c",), "d": ("d",)}
    if mode not in modes:
        raise ValueError("Unknown intervention mode")
    return frozenset(modes[mode])


def validate_experiment_questions(questions: list[Question]) -> None:
    """仅限制新实验入口；通用数据加载和历史日志读取仍支持其他题集。"""
    if not questions:
        raise ValueError("Experiment requires a nonempty MedXpertQA question set")
    if any(q.dataset.casefold().replace(" ", "").replace("_", "")
           not in {"medxpertqa", "medxpertqatext"} for q in questions):
        raise ValueError("New experiments accept MedXpertQA only")
    if len({q.question_id for q in questions}) != len(questions):
        raise ValueError("Experiment question IDs must be unique")


def subset25(questions: list[Question], seed: int = 20261008) -> list[Question]:
    """仅供历史抽样工具读取旧设计；新CLI/preflight从不调用此函数。"""
    clinical = [q for q in questions if not q.dataset.lower().startswith("pubmedqa")]
    pqa = [q for q in questions if q.dataset.lower().startswith("pubmedqa")]
    if len(clinical) < 15 or len(pqa) < 10:
        raise ValueError("B/C requires at least 15 clinical and 10 PubMedQA questions")
    return (random.Random(f"{seed}:clinical").sample(clinical, 15)
            + random.Random(f"{seed}:pubmedqa").sample(pqa, 10))



def b_exposure_count(condition: str) -> int:
    if condition not in B_CONDITIONS + B_LEGACY_CONDITIONS + B_EXTENDED_CONDITIONS:
        raise ValueError("Unknown B condition")
    return 1 if condition == "base" else int(condition.removeprefix("literal").removeprefix("paraphrase"))


def b_evidence(base: list[Evidence], condition: str, target_id: str,
               paraphrases: list[str] | None = None) -> list[Evidence]:
    """默认总暴露11次：原始片段一次，加10份副本；保留真实谱系供审计。"""
    count = b_exposure_count(condition)
    target = next((e for e in base if e.chunk_id == target_id), None)
    if target is None:
        raise ValueError("Target evidence not in initial snapshot")
    if condition == "base":
        return list(base)
    texts = [target.text] * (count - 1)
    if condition.startswith("paraphrase"):
        if paraphrases is None or len(paraphrases) < count - 1:
            raise ValueError("Investigator-supplied paraphrases required")
        texts = paraphrases[:count - 1]
    out = list(base)
    for i, text in enumerate(texts, 1):
        # Distinct IDs preserve experimental duplicate exposure after retrieval ID fusion.
        clone_id = "chunk:" + hashlib.sha256(f"{target.chunk_id}:{condition}:{i}".encode()).hexdigest()[:16]
        metadata = {}
        if condition.startswith("paraphrase"):
            # 改写以独立的虚构文献身份呈现；真实来源只通过 derived_from 留在审计侧。
            # 使用保留的 .example 域名，不冒用真实论文链接，也不发起下载。
            identity = hashlib.sha256(clone_id.encode()).hexdigest()
            metadata = {
                "document_id": "document:" + identity[:12],
                "source": f"https://clinical-reports.example/articles/{identity[:24]}",
                "title": f"Clinical observations and evidence review: {identity[24:36]}",
                "position": "abstract",
                "retrieved_at": "",
            }
        out.append(replace(target, chunk_id=clone_id,
                           text=text, content_hash=hashlib.sha256(text.encode()).hexdigest(),
                           derived_from=target.chunk_id, **metadata))
    return out


def d_evidence(base: list[Evidence], perturbation: list[str] | str, question_id: str) -> list[Evidence]:
    """按给定顺序追加证据，并兼容历史单段字符串。"""
    texts = [perturbation] if isinstance(perturbation, str) else perturbation
    existing = {e.chunk_id for e in base}
    out = list(base)
    for text in texts:
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        salt = 0
        while True:
            digest = hashlib.sha256(f"{question_id}\0{text}\0{salt}".encode()).hexdigest()
            chunk_id = f"chunk:{digest[:16]}"
            if chunk_id not in existing:
                break
            salt += 1
        existing.add(chunk_id)
        out.append(Evidence(chunk_id, f"document:{digest[16:28]}",
                            text, "local-document", "", "", content_hash))
    return out


def c_message(question: Question, repeat_id: int, text: str,
              repeat_message: bool = False, wrong_option: str | None = None) -> dict[str, Any]:
    """固定公开消息的说话者、错选项和轮次；重复条件用第二位不同专家。"""
    if not text.strip():
        raise ValueError("Investigator-supplied erroneous message required")
    if question.gold is None:
        raise ValueError("C message condition requires known gold label for wrong-option selection")
    wrong = [x for x in question.options if x != question.gold]
    rng = random.Random(f"{question.question_id}:{repeat_id}:C")
    speaker = rng.randrange(4)
    selected = wrong_option or rng.choice(wrong)
    if selected not in wrong:
        raise ValueError("C wrong_option must be a non-gold option")
    if wrong_option is None and "{wrong_label}" not in text and "{wrong_text}" not in text:
        raise ValueError("C message text must use {wrong_label} or {wrong_text} when option is sampled")
    displayed = text.replace("{wrong_label}", selected).replace("{wrong_text}", question.options[selected])
    injection = {"speaker": speaker, "wrong_option": selected,
            "round_id": rng.randint(1, 4), "text": displayed,
            "text_template": text, "repeat_message": repeat_message}
    if repeat_message:
        # 最后才抽第二位专家，保持首位说话者、错选项和轮次与单消息条件配对。
        injection["second_speaker"] = rng.choice([i for i in range(4) if i != speaker])
    return injection
