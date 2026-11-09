"""MedXpertQA 全输入准备：仅产生可审计材料，不启动答题实验。"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path

from evidence_mdt.data import read_json_records, normalize_question
from evidence_mdt.query_manifest import public_question_sha256
from evidence_mdt.retrieval import INITIAL_EVIDENCE_LIMIT, Retriever, MockDense
from evidence_mdt.schema import Evidence, Question
from evidence_mdt.interventions import (B_CONDITIONS, B_PARAPHRASE_COUNT, D_EVIDENCE_COUNT, D_EVIDENCE_ANGLES,
                                        validate_experiment_questions)
from . import prompts
from .corpus import chunk_rows, check_lengths, fetch_pubmed
from .materials import (B_TARGET_SELECTION_POLICY, prepare_material,
                        validate_b_annotations, validate_b_paraphrases)
from .state import Cache, ConfigurationError, canonical, digest, fatal, jsonl, save


def load_questions(path):
    questions = []
    for index, item in enumerate(read_json_records(path)):
        dataset = item.get("dataset", "MedXpertQA")
        q = Question(**item) if "stem" in item and "question_id" in item else normalize_question(item, dataset, index)
        if not q.stem.strip() or len(q.options) < 2 or (q.gold is not None and q.gold not in q.options):
            raise ConfigurationError("Invalid question: " + q.question_id)
        questions.append(q)
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ConfigurationError("Input must contain nonempty unique questions")
    try:
        validate_experiment_questions(questions)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    return questions


def prepare(questions_path: Path, output: Path, model, dense, *, interventions="none",
            seed=20261008, retries=2, timeout=60, max_records=8, fetcher=fetch_pubmed,
            tokenizer=None, source_preparation: Path | None = None, fail_fast=False, recover_minor=False):
    if interventions not in {"none", "b", "c", "d"}:
        raise ConfigurationError("interventions must be none, b, c or d")
    if retries < 0 or timeout <= 0 or max_records < 1:
        raise ConfigurationError("Invalid retry, timeout or retrieval limits")
    key_env = getattr(model, "api_key_env", None)
    if key_env and not os.environ.get(key_env):
        raise ConfigurationError("Missing environment variable " + key_env)
    questions = load_questions(questions_path)
    if interventions in {"c", "d"} and any(q.gold not in q.options for q in questions):
        raise ConfigurationError("C/D preparation requires gold for every input question")
    # 代码和数据内容参与审计，文件修改时间与运行时间不参与。
    code = {p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}
    code.update({"models.py": digest(Path(__import__("evidence_mdt.models", fromlist=["x"]).__file__).read_bytes())})
    for module in ("retrieval", "schema", "data", "query_manifest", "interventions"):
        source = Path(__import__("evidence_mdt." + module, fromlist=["x"]).__file__)
        code["evidence_mdt/" + source.name] = digest(source.read_bytes())
    config = {"questions": digest([asdict(q) for q in questions]), "code": code,
              "model": {k: getattr(model, k, None) for k in ("name", "base_url", "max_tokens", "disable_thinking", "answer_temperature")},
              "dense": type(dense).__name__, "revisions": getattr(dense, "revisions", None),
              "interventions": interventions, "seed": seed,
              "max_records": max_records, "chunk_limit": 400,
              "initial_evidence_limit": INITIAL_EVIDENCE_LIMIT}
    if interventions == "b":
        config["b_target_selection_policy"] = B_TARGET_SELECTION_POLICY
        config["b_conditions"] = list(B_CONDITIONS)
        if recover_minor:
            config["b_recovery"] = "ordered_unique_topup_v1"
    if interventions == "d":
        config.update(d_evidence_count=D_EVIDENCE_COUNT, d_evidence_angles=list(D_EVIDENCE_ANGLES))
    if source_preparation:
        source_preparation = source_preparation.resolve()
        if source_preparation == output.resolve():
            raise ConfigurationError("Source and output must differ")
        parent_path = source_preparation / "preparation_manifest.json"
        config["source_preparation"] = {"path": str(source_preparation),
                                         "manifest_sha256": digest(parent_path.read_bytes())}
    output = output.resolve()
    path = output / "preparation_manifest.json"
    if output.exists() and any(output.iterdir()) and not path.exists():
        raise ConfigurationError("Output is nonempty without a preparation manifest")
    if path.exists():
        state = json.loads(path.read_bytes())
        if state.get("fingerprint") != digest(config):
            raise ConfigurationError("Preparation configuration changed; use a new output directory")
        for name, expected in (state.get("artifacts", {}) if state["status"] == "complete" else {}).items():
            if not (output / name).exists() or digest((output / name).read_bytes()) != expected:
                raise ConfigurationError("Artifact hash mismatch: " + name)
        if state["status"] == "complete":
            return state
    else:
        state = {"fingerprint": digest(config), "config": config, "status": "running",
                 "records": [{"question_id": q.question_id, "status": "pending"} for q in questions]}
    if fail_fast:
        retries = 0
    state["execution"] = {"retries": retries, "timeout": timeout, "fail_fast": fail_fast}
    def checkpoint():
        save(path, state)
    checkpoint()
    jsonl(output / "questions.jsonl", [asdict(q) for q in questions])
    cache = Cache(output / "cache", retries)
    def call_for(q):
        def call(stage, instruction, data):
            payload = {"task": "preparation", "instruction": instruction, **data}
            def request(request_payload):
                result = model.complete(request_payload)
                response = {"content": result.content, "raw": result.raw, "usage": result.usage,
                            "wire_request": result.wire_request}
                # 原响应先落盘，再执行适用的校验并提交缓存。
                audit = output / "cache" / "model_calls.jsonl"
                audit.parent.mkdir(parents=True, exist_ok=True)
                with audit.open("ab") as stream:
                    stream.write(canonical({"question_id": q.question_id, "stage": stage, **response}) + b"\n")
                return response

            def operation():
                if stage == "b" and recover_minor:
                    return recover_b_paraphrases(cache.root, q.question_id, payload, request)
                response = request(payload)
                value = response["content"]
                if stage in {"short", "broad"}:
                    parts = validate_queries(value.get("queries"))
                    if stage == "short":
                        check_lengths(dense, " ; ".join(parts), [])
                elif stage == "b_annotation":
                    validate_b_annotations(value, [e["chunk_id"] for e in data["evidence"]], q.options)
                elif stage == "b":
                    validate_b_paraphrases(value, data["original"]["text"])
                elif stage == "c":
                    message = value.get("c_message_text")
                    if not isinstance(message, str) or not message.strip():
                        raise ValueError("Invalid intervention material structure")
                return response
            return cache.get(f"model:{q.question_id}:{stage}", {"payload": payload, "model": config["model"]}, operation)["content"]
        return call
    if source_preparation:
        from .reuse import reuse_frozen
        return reuse_frozen(source_preparation, output, questions, state, interventions, seed, call_for, checkpoint,
                            fail_fast=fail_fast)
    for q, row in zip(questions, state["records"]):
        try:
            call = call_for(q)
            stem = {"question_id": q.question_id, "stem": q.stem.split("\nAnswer Choices:", 1)[0]}
            short = call("short", prompts.SHORT, {"question": stem})
            queries = validate_queries(short.get("queries"))
            query = " ; ".join(queries)
            check_lengths(dense, query, [])
            row.update(queries=queries, retrieval_query=query, status="fetching")
            checkpoint()
            documents = {}
            snapshots = []
            def fetch(terms):
                for term in terms:
                    result = Cache(cache.root, 0).get(f"search:{q.question_id}:{term}", {"query": term, "max_records": max_records},
                        lambda: fetcher(term, cache, max_records, timeout))
                    snapshots.append({"query": term, "status": result["status"], "ids": result["ids"],
                                      "record_types": result.get("record_types", {}),
                                      "without_abstract": result.get("without_abstract", [])})
                    for evidence in result["evidence"]:
                        old = documents.get(evidence["document_id"])
                        if old and old["content_hash"] != evidence["content_hash"]:
                            raise ValueError("Conflicting abstract content")
                        documents[evidence["document_id"]] = evidence
                    row.update(snapshots=list(snapshots), abstracts=list(documents.values()))
                    checkpoint()
            fetch(queries)
            row["broad_triggered"] = len(documents) < 3
            if row["broad_triggered"]:
                broad = validate_queries(call("broad", prompts.BROAD,
                    {"question": stem, "initial_queries": queries}).get("queries"))
                row["broad_queries"] = broad
                fetch(broad)
            row.update(status="retrieved", abstract_count=len(documents))
            row.pop("error", None)
        except Exception as exc:
            row.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            if fail_fast or fatal(exc):
                state.update(status="stopped", error=str(exc), failed_question_id=q.question_id)
                checkpoint()
                raise
        checkpoint()
    # 先固定完整文献集合，再索引、冻结每题初始证据。失败题不提供实验可执行清单。
    if any(row["status"] != "retrieved" for row in state["records"]):
        state["status"] = "incomplete"
        checkpoint()
        return state
    parents = {}
    for row in state["records"]:
        for evidence in row.get("abstracts", []):
            previous = parents.get(evidence["document_id"])
            if previous and previous["content_hash"] != evidence["content_hash"]:
                raise ConfigurationError("Conflicting shared PubMed abstract")
            parents[evidence["document_id"]] = evidence
    if tokenizer is None:
        tokenizer = getattr(dense, "atok", None)
    if tokenizer is None:
        raise ConfigurationError("A tokenizer is required for physical chunking")
    children, spans, mapping = chunk_rows(list(parents.values()), tokenizer)
    evidence = [Evidence(**item) for item in children]
    if not evidence and not isinstance(dense, MockDense):
        state.update(status="incomplete", error="Empty live corpus; no executable index can be built")
        if fail_fast:
            state["status"] = "stopped"
            checkpoint()
            raise ConfigurationError(state["error"])
        checkpoint()
        return state
    jsonl(output / "abstracts.jsonl", list(parents.values()))
    jsonl(output / "corpus.jsonl", children)
    save(output / "chunk_spans.json", {"spans": spans, "parent_to_children": mapping})
    # 每次实际索引前都检查文章长度，查询阶段检查真实cross pair。
    if hasattr(dense, "vectors"):
        dense.vectors = None
    for q, row in zip(questions, state["records"]):
        if row["status"] == "retrieved":
            try:
                check_lengths(dense, row["retrieval_query"], evidence)
            except ValueError as exc:
                row.update(status="failed", error=str(exc), error_type=type(exc).__name__)
                if fail_fast:
                    state.update(status="stopped", error=str(exc), failed_question_id=q.question_id)
                    checkpoint()
                    raise
    retriever = Retriever(evidence, dense)
    index = {"corpus": str(output / "corpus.jsonl"), "sha256": digest((output / "corpus.jsonl").read_bytes()),
             "chunks": len(evidence), "indexed_chunks": len(retriever.evidence),
             "backend": "mock" if isinstance(dense, MockDense) else "medcpt",
             "model_revisions": getattr(dense, "revisions", None), "medcpt_index_built": not isinstance(dense, MockDense)}
    if not isinstance(dense, MockDense):
        vector_path = output / "index_manifest.vectors.pt"
        dense.torch.save(dense.vectors, vector_path)
        index.update(vectors=str(vector_path), vectors_sha256=digest(vector_path.read_bytes()))
    save(output / "index_manifest.json", index)
    for q, row in zip(questions, state["records"]):
        if row["status"] != "retrieved":
            continue
        try:
            base = retriever.search(row["retrieval_query"], limit=INITIAL_EVIDENCE_LIMIT)
            row.update(top10=[asdict(e) for e in base], retrieval_trace=dict(retriever.last_trace))
            checkpoint()
            def save_spec(spec):
                row["spec"] = spec
                checkpoint()
            row["spec"] = prepare_material(q, base, interventions, seed, call_for(q), row.get("spec"), save_spec)
            row["status"] = "ready"
        except Exception as exc:
            row.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            if fail_fast or fatal(exc):
                state.update(status="stopped", error=str(exc), failed_question_id=q.question_id)
                checkpoint()
                raise
        checkpoint()
    jsonl(output / "initial_top10.jsonl", [{"question_id": r["question_id"], "evidence": r.get("top10", []),
         "trace": r.get("retrieval_trace"), "status": r["status"]} for r in state["records"]])
    state["status"] = "complete" if all(r["status"] == "ready" for r in state["records"]) else "incomplete"
    if state["status"] == "complete":
        jsonl(output / "initial_queries.jsonl", [{"question_id": q.question_id, "dataset": q.dataset,
              "public_question_sha256": public_question_sha256(q), "retrieval_query": row["retrieval_query"]}
              for q, row in zip(questions, state["records"])])
        if interventions != "none":
            jsonl(output / "intervention_spec.jsonl", [r["spec"] for r in state["records"]])
    else:
        for name in ("initial_queries.jsonl", "intervention_spec.jsonl"):
            (output / name).unlink(missing_ok=True)
    state["artifacts"] = {p.name: digest(p.read_bytes()) for p in output.iterdir()
                          if p.is_file() and p.name != path.name}
    checkpoint()
    return state


def recover_b_paraphrases(cache_dir, question_id, payload, request):
    original = payload["original"]
    def normalized(text):
        return " ".join(text.casefold().split())

    seen = {normalized(original["text"])}
    selected, sources = [], []
    def collect(response):
        parts = response["content"].get("paraphrases")
        if not isinstance(parts, list):
            return
        for position, part in enumerate(parts, 1):
            if not isinstance(part, str):
                continue
            key = normalized(part)
            if key and key not in seen:
                seen.add(key)
                selected.append(part)
                sources.append({"response_sha256": digest(response["raw"]), "position": position})
                if len(selected) == B_PARAPHRASE_COUNT:
                    break

    response = None
    for name in ("prior_model_calls.jsonl", "model_calls.jsonl"):
        path = cache_dir / name
        if not path.exists():
            continue
        for row in map(json.loads, path.read_bytes().splitlines()):
            if row["question_id"] != question_id or row["stage"] != "b":
                continue
            previous = json.loads(row["wire_request"]["messages"][1]["content"])
            if previous.get("original") != original:
                continue
            response = {k: row[k] for k in ("content", "raw", "usage", "wire_request")}
            collect(response)
            if len(selected) == B_PARAPHRASE_COUNT:
                break
        if len(selected) == B_PARAPHRASE_COUNT:
            break
    if response is None:
        response = request(payload)
        collect(response)
    for _ in range(6):
        needed = B_PARAPHRASE_COUNT - len(selected)
        if needed == 0:
            break
        count = min(needed, 2)
        instruction = (
            f'Return JSON with "paraphrases" containing exactly {count} new English rewrites of original.text. '
            'Each rewrite must preserve the entire passage, including all findings, qualifications and numbers. '
            'Use different sentence structures and wording. Do not repeat the original or any existing rewrite. '
            'Do not summarize or add facts.'
        )
        response = request({**payload, "instruction": instruction, "existing_paraphrases": selected[:]})
        collect(response)
    content = {"paraphrases": selected}
    validate_b_paraphrases(content, original["text"])
    return {**response, "content": content, "paraphrase_sources": sources}


def validate_queries(value):
    if not isinstance(value, list) or len(value) not in (2, 3) or any(
            not isinstance(x, str) or not x.strip() or len(x) > 500 for x in value):
        raise ValueError("Expected two or three nonempty queries, each at most 500 characters")
    return [x.strip() for x in value]
