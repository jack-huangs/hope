from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import Model, ModelResponseError
from .retrieval import Retriever
from .schema import (AgentTurn, Budget, Claim, DependencyEdge, Evidence, Question,
                     probability_sum_within_tolerance, validate_answer)


GROUPS = ("ordinary", "semantic", "semantic_retrieval")
ROLES = ("diagnostic specialist", "clinical epidemiologist", "pharmacology specialist", "general medicine specialist")


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def stable_order(question_id: str, repeat_id: int, round_id: int) -> list[int]:
    key = f"{question_id}:{repeat_id}:{round_id}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
    order = list(range(4))
    random.Random(seed).shuffle(order)
    return order


def opaque_id(value: str) -> str:
    """模型只见不含实验组与干预名称的稳定引用 ID。"""
    return "ref:" + hashlib.sha256(value.encode()).hexdigest()[:20]


def target_pubmed_ids(question: Question, evidence: list[Evidence]) -> set[str]:
    """排除题目对应研究的 PMID 摘要及同一文献的其他片段。"""
    if not question.dataset.lower().startswith("pubmedqa"):
        return set()
    pmid = re.search(r"\d{5,10}", question.question_id)
    target = pmid.group() if pmid else question.question_id
    out = set()
    for item in evidence:
        provenance = " ".join((item.document_id, item.chunk_id, item.source))
        if (re.search(rf"(?<!\d){re.escape(target)}(?!\d)", provenance)
                or (question.context and item.text.strip() in question.context)):
            out.add(item.chunk_id)
    return out


def public_view(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """隐藏簇成员及原始消息 ID，仅保留专家需要的决策内容。"""
    return [{k: v for k, v in message.items() if k not in {"sources", "member_claim_ids"}}
            for message in messages]


class Runner:
    def __init__(self, model: Model, retriever: Retriever, output: Path,
                 budget: Budget | None = None):
        self.model = model
        self.retriever = retriever
        self.output = output
        self.budget = budget or Budget()
        self.call_log = output / "calls.jsonl"
        self.event_log = output / "events.jsonl"
        self.result_log = output / "results.jsonl"
        self.semantic_cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self.semantic_cache_events: set[tuple[str, str | None, tuple[str, ...]]] = set()
        self.role_cache: dict[tuple[str, int], tuple[str, ...]] = {}
        self.cluster_snapshots: dict[tuple[str | None, str, str], tuple[str, ...]] = {}
        self.opinion_nodes: set[tuple[str | None, str, str]] = set()
        self.evidence_nodes: set[tuple[str | None, str]] = set()
        self.last_model_revision = model.name
        self.fingerprint: str | None = None
        self.b_target_answer: str | None = None
        self.current_attempt_id: str | None = None

    def _event(self, item: dict[str, Any]) -> None:
        """图节点按 attempt 分区；失败重跑即使逻辑 ID 相同也不会混在一起。"""
        item.setdefault("attempt_id", None if item.get("type") == "recruitment" else self.current_attempt_id)
        append_jsonl(self.event_log, item)

    def _register_evidence(self, run_id: str, evidence: list[Evidence]) -> None:
        """保留原文和已知生成谱系；派生关系只放审计图。"""
        for item in evidence:
            key = (self.current_attempt_id, item.chunk_id)
            if key in self.evidence_nodes:
                continue
            self.evidence_nodes.add(key)
            self._event({"run_id": run_id, "type": "evidence", **asdict(item)})
            if item.derived_from:
                self._event({"run_id": run_id, "type": "edge",
                             **asdict(DependencyEdge(item.derived_from, item.chunk_id,
                                                     "derived_from", -1))})

    def roles_for(self, question: Question, repeat_id: int) -> tuple[str, ...]:
        key = (question.question_id, repeat_id)
        if key in self.role_cache:
            return self.role_cache[key]
        if self.event_log.exists():
            for line in self.event_log.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item.get("type") == "recruitment" and item.get("question_id") == key[0] and item.get("repeat_id") == key[1]:
                    self.role_cache[key] = tuple(item["roles"])
                    return self.role_cache[key]
        data = self.invoke(f"{question.question_id}:repeat-{repeat_id}:recruit",
                           {"task": "recruit", "question": question.public()})
        roles = data.get("roles")
        if not isinstance(roles, list) or len(roles) != 4 or not all(isinstance(r, str) and r.strip() for r in roles) or len(set(roles)) != 4:
            raise ValueError("Recruiter must return four distinct roles")
        self.role_cache[key] = tuple(roles)
        self._event( {"type": "recruitment", "question_id": key[0],
                                     "repeat_id": key[1], "roles": roles})
        return self.role_cache[key]

    def invoke(self, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.budget.before_call()
        started = time.time()
        try:
            result = self.model.complete(payload)
        except ModelResponseError as exc:
            self.budget.after_call(exc.usage)
            append_jsonl(self.call_log, {"run_id": run_id, "model": self.model.name,
                                         "request": payload, "wire_request": exc.wire_request,
                                         "response": exc.raw, "usage": exc.usage,
                                         "model_revision": exc.raw.get("model", self.model.name),
                                         "status": "failed", "error": str(exc), "started_at": started,
                                         "elapsed_seconds": time.time() - started})
            raise
        except Exception as exc:
            # 网络失败也已经消耗一次尝试；服务端未给 usage 时 token 只能记为未知。
            self.budget.after_call({})
            append_jsonl(self.call_log, {"run_id": run_id, "model": self.model.name,
                                         "request": payload, "error": repr(exc),
                                         "usage": {}, "status": "failed",
                                         "started_at": started, "elapsed_seconds": time.time() - started})
            raise
        self.budget.after_call(result.usage)
        self.last_model_revision = result.raw.get("model", self.model.name)
        append_jsonl(self.call_log, {"run_id": run_id, "model": self.model.name,
                                     "request": payload, "wire_request": result.wire_request,
                                     "response": result.raw,
                                     "usage": result.usage, "model_revision": result.raw.get("model", self.model.name),
                                     "status": "ok", "started_at": started,
                                     "elapsed_seconds": time.time() - started})
        if self.budget.tokens > self.budget.max_tokens:
            raise RuntimeError("API token budget exhausted after response")
        return result.content

    def _semantic_relation(self, run_id: str, a: Claim | str, b: Claim | str) -> str:
        left = self._semantic_fields(a)
        right = self._semantic_fields(b)
        if left == right:
            key = self._relation_key(run_id, left, right)
            event_key = (run_id, self.current_attempt_id, key)
            if event_key not in self.semantic_cache_events:
                self.semantic_cache_events.add(event_key)
                self._event({"run_id": run_id, "type": "semantic_cache_hit",
                             "cache_kind": "exact_fields", "relation": "equivalent",
                             "pair_sha256": self._pair_sha256(key),
                             "source_run_id": None, "source_attempt_id": None,
                             "source_call_index": None, "source_pair_index": None})
            return "equivalent"
        key = self._relation_key(run_id, left, right)
        if key not in self.semantic_cache:
            data = self.invoke(run_id, {"task": "semantic", "a": left, "b": right})
            relation = data.get("relation", "unknown")
            if relation not in {"equivalent", "complementary", "contradictory", "condition_different", "unknown"}:
                raise ValueError("invalid semantic relation")
            self.semantic_cache[key] = {"relation": relation, "source_run_id": run_id,
                                        "source_attempt_id": self.current_attempt_id,
                                        "source_call_index": self.budget.calls,
                                        "source_pair_index": 0}
        entry = self.semantic_cache[key]
        if (entry["source_run_id"], entry["source_attempt_id"]) != (run_id, self.current_attempt_id):
            event_key = (run_id, self.current_attempt_id, key)
            if event_key not in self.semantic_cache_events:
                self.semantic_cache_events.add(event_key)
                self._event({"run_id": run_id, "type": "semantic_cache_hit",
                             "cache_kind": "shared_prior_decision",
                             "pair_sha256": self._pair_sha256(key), **entry})
        return entry["relation"]

    @staticmethod
    def _semantic_fields(claim: Claim | str) -> dict[str, str]:
        if isinstance(claim, Claim):
            return {"text": claim.text, "answer": claim.answer,
                    "polarity": claim.polarity, "conditions": claim.conditions}
        return {"text": claim}

    @staticmethod
    def _relation_key(run_id: str, left: dict[str, str], right: dict[str, str]) -> tuple[str, ...]:
        question_id, suffix = run_id.split(":repeat-", 1)
        paired_id = f"{question_id}:repeat-{suffix.split(':', 1)[0]}"
        return (paired_id,
                *sorted((json.dumps(left, sort_keys=True), json.dumps(right, sort_keys=True))))

    @staticmethod
    def _pair_sha256(key: tuple[str, ...]) -> str:
        """稳定标识命题对，便于从批量调用定位来源且不在事件重复原文。"""
        return hashlib.sha256(json.dumps(key, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _prefetch_semantic(self, run_id: str, claim: Claim, others: list[Claim]) -> None:
        """只判当前命题与已有簇需要的比较；保留逐对判定供审计。"""
        pending: dict[tuple[str, ...], dict[str, Any]] = {}
        for other in others:
            left, right = self._semantic_fields(claim), self._semantic_fields(other)
            if left == right:
                continue
            key = self._relation_key(run_id, left, right)
            if key not in self.semantic_cache:
                pending[key] = {"a": left, "b": right}
        items = list(pending.items())
        allowed = {"equivalent", "complementary", "contradictory", "condition_different", "unknown"}
        for start in range(0, len(items), 12):
            batch = items[start:start + 12]
            data = self.invoke(run_id, {"task": "semantic_batch",
                                        "pairs": [pair for _, pair in batch]})
            relations = data.get("relations")
            if (not isinstance(relations, list) or len(relations) != len(batch)
                    or any(relation not in allowed for relation in relations)):
                raise ValueError("invalid semantic batch relations")
            for pair_index, ((key, _), relation) in enumerate(zip(batch, relations)):
                self.semantic_cache[key] = {"relation": relation, "source_run_id": run_id,
                                            "source_attempt_id": self.current_attempt_id,
                                            "source_call_index": self.budget.calls,
                                            "source_pair_index": pair_index}

    def message_view(self, run_id: str, turns: list[AgentTurn | dict[str, Any]], semantic: bool) -> list[dict[str, Any]]:
        """两组同样呈现角色答案和命题；只在语义组折叠严格等价的命题。"""
        opinions: list[dict[str, Any]] = []
        pairs: list[tuple[str, Claim]] = []
        for turn in turns:
            if isinstance(turn, AgentTurn):
                mid, role, answer, claims = turn.turn_id, turn.role, turn.answer, turn.claims
            else:
                mid, role, answer = turn["id"], turn["role"], turn["answer"]
                claims = [Claim(f"{mid}:claim-{i}", c["text"], c["answer"],
                                tuple(c.get("citation_ids", [])), polarity=c.get("polarity", "support"),
                                conditions=c.get("conditions", "")) for i, c in enumerate(turn.get("claims", []))]
            public_mid = opaque_id(mid)
            opinions.append({"id": public_mid, "kind": "opinion", "role": role, "answer": answer,
                             "sources": [mid]})
            opinion_key = (self.current_attempt_id, run_id, public_mid)
            if opinion_key not in self.opinion_nodes:
                self.opinion_nodes.add(opinion_key)
                self._event( {"run_id": run_id, "type": "view_node",
                                             "id": public_mid, "kind": "opinion", "source_id": mid})
                self._event( {"run_id": run_id, "type": "edge",
                                             **asdict(DependencyEdge(mid, public_mid, "displayed_as", -1))})
            pairs.extend((mid, c) for c in claims)
        units: list[dict[str, Any]] = []
        for mid, claim in pairs:
            found = None
            if semantic:
                candidates = [unit for unit in units if unit["answer"] == claim.answer
                              and unit["polarity"] == claim.polarity]
                self._prefetch_semantic(run_id, claim, [unit["_claims"][0] for unit in candidates])
                for unit in candidates:
                    if self._semantic_relation(run_id, claim, unit["_claims"][0]) != "equivalent":
                        continue
                    self._prefetch_semantic(run_id, claim, unit["_claims"][1:])
                    if all(self._semantic_relation(run_id, claim, old) == "equivalent"
                           for old in unit["_claims"][1:]):
                        found = unit
                        break
            if found is None:
                fingerprint = (f"{run_id}:{claim.answer}:{claim.polarity}:{claim.conditions}:"
                               f"{claim.text.strip().lower()}") if semantic else claim.claim_id
                cluster_id = opaque_id(fingerprint)
                found = {"id": cluster_id, "kind": "claim", "claim": claim.text,
                         "answer": claim.answer, "polarity": claim.polarity,
                         "conditions": claim.conditions, "sources": [], "member_claim_ids": [],
                         "citation_ids": list(claim.citation_ids), "_claims": []}
                units.append(found)
            found["sources"].append(mid)
            found["member_claim_ids"].append(claim.claim_id)
            found["_claims"].append(claim)
            found["citation_ids"] = sorted(set(found["citation_ids"]) | set(claim.citation_ids))
        for unit in units:
            members = tuple(unit["member_claim_ids"])
            snapshot_key = (self.current_attempt_id, run_id, unit["id"])
            if self.cluster_snapshots.get(snapshot_key) != members:
                self.cluster_snapshots[snapshot_key] = members
                self._event( {"run_id": run_id, "type": "cluster",
                                             **{k: v for k, v in unit.items() if k != "_claims"}})
                for claim_id in unit["member_claim_ids"]:
                    self._event( {"run_id": run_id, "type": "edge",
                                                 **asdict(DependencyEdge(claim_id, unit["id"], "member_of", -1))})
        return opinions + [{k: v for k, v in unit.items() if k != "_claims"} for unit in units]

    def _turn(self, run_id: str, question: Question, evidence: list[Evidence],
              messages: list[dict[str, Any]], role: str, round_id: int, order: int,
              turn_id: str, task: str = "answer") -> AgentTurn:
        # Generation lineage is evaluator-only metadata, never model-visible.
        public_evidence = [{k: v for k, v in asdict(e).items() if k != "derived_from"}
                           for e in evidence]
        public_messages = public_view(messages)
        payload = {"task": task, "question": question.public(), "role": role,
                   "round_id": round_id, "evidence": public_evidence, "messages": public_messages}
        evidence_ids = {e.chunk_id for e in evidence}
        message_ids = {m["id"] for m in public_messages}
        raw_message_ids = sorted({source for m in messages for source in m.get("sources", [m["id"]])})

        def check_response(value: dict[str, Any]) -> None:
            validate_answer(value, question.options, evidence_ids, message_ids)
            if role == "host" and (not isinstance(value.get("unexplained_contradiction"), bool)
                                   or not isinstance(value.get("missing_distinction"), bool)):
                raise ValueError("host must report contradiction and missing distinction flags")

        repair_error: ValueError | None = None
        original: dict[str, Any] | None = None
        try:
            data = self.invoke(run_id, payload)
        except ModelResponseError as exc:
            repair_error, original = exc, exc.raw
        else:
            try:
                check_response(data)
            except ValueError as exc:
                repair_error, original = exc, data
        if repair_error is not None:
            # 最多一次付费格式修复；原始坏响应和错误仍留在调用／事件日志中。
            self._event({"run_id": run_id, "type": "response_validation_failed",
                         "task": task, "turn_id": turn_id, "error": str(repair_error),
                         "response": original})
            original_probs = None
            original_answer = None
            if not isinstance(repair_error, ModelResponseError):
                original_probs = original.get("probabilities") if isinstance(original, dict) else None
                if (not isinstance(original_probs, dict) or set(original_probs) != set(question.options)
                        or any(not isinstance(p, (int, float)) or isinstance(p, bool)
                               or not math.isfinite(p) or not 0 <= p <= 1
                               for p in original_probs.values())
                        or not probability_sum_within_tolerance(original_probs.values())):
                    raise repair_error
                stated = original.get("answer")
                if stated in question.options:
                    original_answer = stated
                else:
                    matches = [key for key, value in question.options.items() if value == stated]
                    if len(matches) != 1:
                        raise repair_error
                    original_answer = matches[0]
            data = self.invoke(run_id, {"task": "repair", "original_task": task,
                                        "original_request": payload,
                                        "original_response": original,
                                        "validation_error": str(repair_error)})
            check_response(data)
            if original_probs is not None and (data["probabilities"] != original_probs
                                               or data["answer"] != original_answer):
                raise ValueError("repair changed original probabilities or selected option")
            self._event({"run_id": run_id, "type": "response_repair_checked",
                         "task": task, "turn_id": turn_id,
                         "probabilities_verified_unchanged": original_probs is not None,
                         "original_json_parseable": not isinstance(repair_error, ModelResponseError)})
        claims = [Claim(f"{turn_id}:claim-{i}", c["text"], c["answer"],
                        tuple(c.get("citation_ids", [])), polarity=c.get("polarity", "support"),
                        conditions=c.get("conditions", "")) for i, c in enumerate(data["claims"])]
        turn = AgentTurn(turn_id, role, round_id, order, sorted(evidence_ids),
                         raw_message_ids, payload, data["answer"],
                         data["probabilities"], data["citations"], claims,
                         data["opinion"], data, attempt_id=self.current_attempt_id)
        self._record_turn(run_id, turn, evidence)
        return turn

    def _record_turn(self, run_id: str, turn: AgentTurn, evidence: list[Evidence]) -> None:
        """记录可回放节点与边；暴露和引用是观察事实，并非内部因果。"""
        self._event( {"run_id": run_id, "type": "turn",
                                     "attempt_id": self.current_attempt_id, **turn.record()})
        turn_id, round_id = turn.turn_id, turn.round_id
        for e in evidence:
            self._event( {"run_id": run_id, "type": "edge",
                                         **asdict(DependencyEdge(e.chunk_id, turn_id, "exposed", round_id))})
        for source_id in turn.visible_messages:
            self._event( {"run_id": run_id, "type": "edge",
                                         **asdict(DependencyEdge(source_id, turn_id, "exposed", round_id))})
        for cid in turn.citations:
            self._event( {"run_id": run_id, "type": "edge",
                                         **asdict(DependencyEdge(cid, turn_id, "cited", round_id))})
        for claim in turn.claims:
            self._event( {"run_id": run_id, "type": "claim", **asdict(claim)})
            for cid in claim.citation_ids:
                self._event( {"run_id": run_id, "type": "edge",
                                             **asdict(DependencyEdge(cid, claim.claim_id, "cited", round_id))})

    def _replay_r0(self, run_id: str, source: list[AgentTurn]) -> list[AgentTurn]:
        out = []
        for turn in source:
            clone_id = f"{run_id}:{turn.role}:R0"
            cloned_claims = [Claim(f"{clone_id}:claim-{i}", c.text, c.answer,
                                   c.citation_ids, c.semantic_cluster, c.polarity, c.conditions)
                             for i, c in enumerate(turn.claims)]
            clone = AgentTurn(**{**turn.__dict__, "turn_id": clone_id,
                                 "claims": cloned_claims, "reused_from": turn.turn_id,
                                 "reused_attempt_id": turn.attempt_id,
                                 "attempt_id": self.current_attempt_id})
            replay_evidence = [e for e in self.retriever.evidence if e.chunk_id in clone.visible_evidence]
            # 干预副本可能不在检索索引，仍需按可见 ID 恢复暴露边。
            missing = set(clone.visible_evidence) - {e.chunk_id for e in replay_evidence}
            replay_evidence += [Evidence(cid, "replayed", "") for cid in missing]
            self._record_turn(run_id, clone, replay_evidence)
            self._event( {"run_id": run_id, "type": "edge",
                          "source_attempt_id": turn.attempt_id,
                          **asdict(DependencyEdge(turn.turn_id, clone_id, "replayed_from", 0))})
            out.append(clone)
        return out

    def load_r0(self, run_id: str) -> list[AgentTurn]:
        if not self.event_log.exists():
            return []
        completed_attempt = None
        if self.result_log.exists():
            for line in self.result_log.read_text(encoding="utf-8").splitlines():
                result = json.loads(line)
                if result.get("run_id") == run_id and result.get("status", "ok") == "ok":
                    completed_attempt = result.get("attempt_id")
        if completed_attempt is None:
            return []
        found = []
        for line in self.event_log.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if (item.get("run_id") == run_id and item.get("type") == "turn"
                    and item.get("round_id") == 0 and item.get("attempt_id") == completed_attempt):
                item.pop("run_id")
                item.pop("type")
                item["attempt_id"] = completed_attempt
                item["claims"] = [Claim(**{**c, "citation_ids": tuple(c.get("citation_ids", []))})
                                  for c in item["claims"]]
                found.append(AgentTurn(**item))
        return found if len(found) == 4 else []

    def _trigger(self, run_id: str, latest: list[AgentTurn], host: AgentTurn,
                 semantic: bool) -> dict[str, Any]:
        claims = [c for t in latest for c in t.claims]
        view = [x for x in self.message_view(run_id, latest, True) if x["kind"] == "claim"] if semantic else []
        normalized = [" ".join(c.text.lower().split()) for c in claims]
        distinct = len(view) if semantic else len(set(normalized))
        duplicate_fraction = (len(claims) - distinct) / max(len(claims), 1)
        support = {x["id"] for x in view if x["answer"] == host.answer and x["polarity"] == "support"} if semantic else {
            " ".join(c.text.lower().split()) for c in claims if c.answer == host.answer and c.polarity == "support"}
        unexplained_contra = bool(host.raw_response.get("unexplained_contradiction", False))
        # Missing distinction is an explicit host output field; absent is not inferred.
        missing_distinction = bool(host.raw_response.get("missing_distinction", False))
        return {"duplicate_fraction": duplicate_fraction, "distinct_support": len(support),
                "unexplained_contra": unexplained_contra,
                "missing_distinction": missing_distinction,
                "triggered": (duplicate_fraction >= 0.5 and len(support) < 2)
                             or unexplained_contra or missing_distinction}

    def run_group(self, question: Question, repeat_id: int, group: str,
                  evidence: list[Evidence], variant: str = "natural",
                  r0_source: list[AgentTurn] | None = None,
                  message_injection: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[AgentTurn]]:
        if group not in GROUPS:
            raise ValueError("Only the three planned methods are allowed")
        if variant.startswith("B:") and self.b_target_answer is not None and self.b_target_answer not in question.options:
            raise ValueError("B target answer must be a question option")
        roles = self.roles_for(question, repeat_id)
        if r0_source is not None and (len(r0_source) != 4
                                      or [turn.role for turn in r0_source] != list(roles)
                                      or any(turn.round_id != 0 or not turn.turn_id.startswith(
                                          f"{question.question_id}:repeat-{repeat_id}:") for turn in r0_source)):
            raise ValueError("R0 replay must come from the same question and independent repeat")
        run_id = f"{question.question_id}:repeat-{repeat_id}:{variant}:{group}"
        self.current_attempt_id = uuid.uuid4().hex
        self._event( {"run_id": run_id, "type": "attempt_start",
                                     "attempt_id": self.current_attempt_id})
        self._register_evidence(run_id, evidence)
        if r0_source is None:
            r0 = [self._turn(run_id, question, evidence, [], role, 0, i,
                             f"{run_id}:{role}:R0") for i, role in enumerate(roles)]
        else:
            r0 = self._replay_r0(run_id, r0_source)
        turns = list(r0)
        semantic = group != "ordinary"
        for round_id in range(1, 5):
            order = stable_order(question.question_id, repeat_id, round_id)
            self._event( {"run_id": run_id, "type": "order",
                                         "round_id": round_id, "roles": [roles[i] for i in order]})
            for position, idx in enumerate(order):
                visible_turns = [t for t in turns if t.round_id == round_id - 1 or
                                 (t.round_id == round_id and t.order < position)]
                raw_messages: list[AgentTurn | dict[str, Any]] = list(visible_turns)
                if message_injection and round_id == message_injection.get("round_id", 1) and idx == message_injection["receiver"]:
                    synthetic_id = "peer-message:" + hashlib.sha256(f"{run_id}:{round_id}:{idx}".encode()).hexdigest()[:12]
                    injected = {"id": synthetic_id, "role": roles[message_injection["speaker"]],
                                "answer": message_injection["wrong_option"],
                                "claims": [{"text": message_injection["text"],
                                            "answer": message_injection["wrong_option"],
                                            "citation_ids": [], "polarity": "support", "conditions": ""}]}
                    original = self.message_view(run_id, raw_messages, semantic)
                    raw_messages.append(injected)
                    if message_injection.get("repeat_message"):
                        repeated = {**injected, "id": injected["id"] + ":repeated"}
                        raw_messages.append(repeated)
                    for synthetic in [x for x in raw_messages if isinstance(x, dict)]:
                        self._event( {"run_id": run_id, "type": "injected_message",
                                                     **synthetic})
                        for i, c in enumerate(synthetic["claims"]):
                            self._event( {"run_id": run_id, "type": "claim",
                                                         "claim_id": f"{synthetic['id']}:claim-{i}", **c})
                    messages = self.message_view(run_id, raw_messages, semantic)
                    self._event( {"run_id": run_id, "type": "injection",
                                                 "round_id": round_id, "receiver": roles[idx],
                                                 "original_messages": original,
                                                 "altered_raw_messages": [x for x in raw_messages if isinstance(x, dict)],
                                                 "displayed_messages": messages})
                else:
                    messages = self.message_view(run_id, raw_messages, semantic)
                turns.append(self._turn(run_id, question, evidence, messages, roles[idx],
                                        round_id, position, f"{run_id}:{roles[idx]}:R{round_id}"))
        latest = [next(t for t in reversed(turns) if t.role == role) for role in roles]
        host_messages = self.message_view(run_id, latest, semantic)
        host = self._turn(run_id, question, evidence, host_messages, "host", 5, 0,
                          f"{run_id}:host:first", task="host")
        trigger = self._trigger(run_id, latest, host, semantic)
        extra: list[Evidence] = []
        reviews: list[AgentTurn] = []
        final = host
        if group == "semantic_retrieval" and trigger["triggered"]:
            query_data = self.invoke(run_id, {"task": "queries", "question": question.public(),
                                              "latest_view": public_view(host_messages),
                                              "host": host.raw_response})
            queries = query_data.get("queries", [])
            if not isinstance(queries, list) or len(queries) > 2 or not all(isinstance(q, str) for q in queries):
                raise ValueError("invalid supplemental queries")
            seen = {e.chunk_id for e in evidence}
            candidates: list[dict[str, Any]] = []
            search_traces: list[dict[str, Any]] = []
            known_claims = [c.text for t in latest for c in t.claims]
            for query in queries:
                forbidden = target_pubmed_ids(question, self.retriever.evidence)
                search_results = self.retriever.search(query, limit=20, exclude=seen | forbidden)
                search_traces.append(dict(self.retriever.last_trace))
                for item in search_results:
                    if len(extra) == 4:
                        break
                    # 未知关系本身不算新信息：还须独立判定相关且产生新命题/新条件。
                    existing = known_claims + [e.text for e in evidence + extra]
                    normalized = " ".join(item.text.lower().split())
                    if normalized in {" ".join(text.lower().split()) for text in existing}:
                        decision = {"relevant": None, "novel": False,
                                    "new_claim_or_condition": False, "reason": "exact_duplicate"}
                    else:
                        decision = self.invoke(run_id, {"task": "novelty", "question": question.public(),
                                                        "candidate": item.text, "existing": existing})
                    accept = (decision.get("relevant") is True and decision.get("novel") is True
                              and decision.get("new_claim_or_condition") is True)
                    candidates.append({"query": query, "chunk_id": item.chunk_id,
                                       "decision": decision, "accepted": accept})
                    if accept:
                        extra.append(item)
                        seen.add(item.chunk_id)
            self._event( {"run_id": run_id, "type": "supplement",
                                         "queries": queries, "chunk_ids": [e.chunk_id for e in extra],
                                         "search_traces": search_traces,
                                         "candidates": candidates, "trigger": trigger})
            if extra:
                self._register_evidence(run_id, extra)
                reviews = [self._turn(run_id, question, evidence + extra, host_messages,
                                      role, 6, i, f"{run_id}:{role}:review", task="review")
                           for i, role in enumerate(roles)]
                final = self._turn(run_id, question, evidence + extra,
                                   self.message_view(run_id, reviews, True), "host", 7, 0,
                                   f"{run_id}:host:final", task="host")
        final_experts = reviews or latest
        consensus = max([sum(t.answer == label for t in final_experts) for label in question.options])
        record = {"run_id": run_id, "question_id": question.question_id, "dataset": question.dataset,
                  "repeat_id": repeat_id, "variant": variant, "group": group,
                  "gold": question.gold, "answer": final.answer,
                  "probabilities": final.probabilities, "r0_answers": [t.answer for t in r0],
                  "discussion_expert_answers": [t.answer for t in latest],
                  "expert_answers": [t.answer for t in final_experts], "roles": list(roles),
                  "options": question.options, "status": "ok", "model_revision": self.last_model_revision,
                  "consensus": "4/4" if consensus == 4 else "3/4" if consensus == 3 else "none",
                  "trigger": trigger, "supplemental_chunk_ids": [e.chunk_id for e in extra],
                  "calls_total_at_completion": self.budget.calls,
                  "tokens_total_at_completion": self.budget.tokens,
                  "r0_reused": r0_source is not None,
                  "attempt_id": self.current_attempt_id,
                  "fingerprint": self.fingerprint,
                  "b_target_answer": self.b_target_answer if variant.startswith("B:") else None}
        append_jsonl(self.result_log, record)
        return record, r0
