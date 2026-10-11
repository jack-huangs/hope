"""为固定题目准备可续跑的 PubMed 摘要快照，不做医学答案评价。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_mdt.data import load_questions, pubmed_snapshot, write_jsonl
from evidence_mdt.env import load_env
from evidence_mdt.models import create_model


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def save_json(path: Path, obj: object) -> None:
    # replace 是原子提交点；崩溃时不会留下半份 JSON 清单。
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def code_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "src/evidence_mdt/data.py",
             ROOT / "src/evidence_mdt/models.py", ROOT / "src/evidence_mdt/schema.py"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p.read_bytes()) for p in paths}


def public_model_config(model: Any, backend: str) -> dict[str, Any]:
    # 配置指纹不包含环境变量中的密钥。
    return {"backend": backend, "name": model.name,
            "base_url": getattr(model, "base_url", None),
            "max_output_tokens": getattr(model, "max_tokens", None),
            "disable_thinking": getattr(model, "disable_thinking", None)}


def query_payload(question: Any) -> dict[str, Any]:
    return {"task": "queries", "question": question.public(),
            "purpose": "Generate faithful PubMed search queries from the question stem before any agent answers.",
            "constraints": ("Return at most two concise English search queries within 64 tokens. "
                            "Use only the stem; ignore options and context. "
                            "Preserve the population, key findings, clinically important negative findings, "
                            "and the exact information being asked for. "
                            "Do not add symptoms, diagnoses, drugs, treatments, or mechanisms "
                            "that are not explicitly stated in the stem. "
                            "Prefer removing incidental details over removing essential conditions.")}


def valid_queries(content: dict[str, Any]) -> list[str]:
    queries = content.get("queries")
    if (not isinstance(queries, list) or len(queries) > 2 or
            any(not isinstance(q, str) or not q.strip() or len(q) > 500 for q in queries)):
        raise ValueError("Model must return at most two nonempty query strings (max 500 characters)")
    return [q.strip() for q in queries]


def usage_tokens(usage: Any) -> int:
    if not isinstance(usage, dict):
        return 0
    return int(usage.get("total_tokens") or 0)


def totals(manifest: dict[str, Any]) -> tuple[int, int]:
    calls = [call for row in manifest["records"] for call in row["model_calls"]]
    return len(calls), sum(usage_tokens(call.get("usage")) for call in calls)


def safe_error(exc: Exception) -> str:
    # 异常字符串有机会包含请求 URL；只保存类型与简短描述，不打印密钥。
    return f"{type(exc).__name__}: {str(exc)[:500]}"


def prepare(questions_path: Path, output: Path, max_records: int, max_calls: int,
            max_tokens: int, backend: str, *, pause_seconds: float = 1.0) -> dict[str, Any]:
    if max_records < 1 or max_calls < 0 or max_tokens < 0:
        raise ValueError("max-records must be positive; budgets must be nonnegative")
    questions_path = questions_path.resolve()
    output = output.resolve()
    questions = load_questions(questions_path)
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("Questions must be nonempty with unique question_id values")
    if backend == "deepseek":
        load_env(ROOT / ".env")
    model = create_model(backend)
    config = {"questions_sha256": digest(questions_path.read_bytes()), "code_sha256": code_hashes(),
              "model": public_model_config(model, backend), "max_records": max_records,
              "query_protocol": "public-question-only-v1"}
    fingerprint = digest(canonical(config))
    manifest_path = output / "preparation_manifest.json"
    if output.exists() and not manifest_path.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory already contains data without a manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Existing snapshot has a different input/code/model configuration; choose a new output")
        if manifest.get("status") == "complete":
            # 完成快照不可覆盖；相同配置可直接复用。
            return manifest
    else:
        records = [{"question_id": q.question_id, "dataset": q.dataset,
                    "public_question_sha256": digest(canonical(q.public())),
                    "status": "pending", "queries": [], "model_calls": [], "snapshots": []}
                   for q in questions]
        manifest = {"schema_version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                    "fingerprint": fingerprint, "config": config, "question_ids": [q.question_id for q in questions],
                    "forbidden_pubmedqa_pmids": sorted({q.question_id for q in questions
                                                         if q.dataset.lower().startswith("pubmedqa")}),
                    "records": records}
        # 仅公开题面写到快照；原输入金标文件保持在原目录。
        write_jsonl(output / "questions.jsonl", [q.public() for q in questions])
        save_json(manifest_path, manifest)

    forbidden = set(manifest["forbidden_pubmedqa_pmids"])
    stopped_for_budget = False
    for index, question in enumerate(questions):
        row = manifest["records"][index]
        if row["status"] in {"pending", "query_failed"}:
            calls, tokens = totals(manifest)
            # token 数仅在服务端回报后才已知，故是调用前软阈值。
            if calls >= max_calls or tokens >= max_tokens:
                stopped_for_budget = True
                break
            payload = query_payload(question)
            previous_attempts = len(row["model_calls"])
            try:
                result = model.complete(payload)
                call = {"request": payload, "wire_request": result.wire_request,
                        "raw_response": result.raw, "usage": result.usage,
                        "model_revision": result.raw.get("model", model.name)}
                row["model_calls"].append(call)
                # 即使响应结构不合格，其调用和已知费用也须留存。
                row["queries"] = valid_queries(result.content)
                row["snapshots"] = [{"query": q, "status": "fetch_pending"} for q in row["queries"]]
                row["status"] = "queried" if row["queries"] else "empty_queries"
                row.pop("error", None)
            except Exception as exc:
                # 无法确认是否到达服务端的异常也占一次请求预算，用量未知时记零。
                if len(row["model_calls"]) == previous_attempts:
                    row["model_calls"].append({"request": payload,
                        "wire_request": getattr(exc, "wire_request", None),
                        "raw_response": getattr(exc, "raw", None),
                        "usage": getattr(exc, "usage", {}) or {}, "error": safe_error(exc)})
                row["status"] = "query_failed"
                row["error"] = safe_error(exc)
                save_json(manifest_path, manifest)
                continue
            save_json(manifest_path, manifest)
        if row["status"] == "empty_queries":
            continue
        for query_index, entry in enumerate(row["snapshots"]):
            if entry["status"] in {"ok", "empty"}:
                continue
            query = row["queries"][query_index]
            prefix = f"q{index + 1:04d}.query{query_index + 1}"
            try:
                evidence, snapshot = pubmed_snapshot(query, max_records=max_records)
                raw_xml = snapshot.pop("_raw_xml", None)
                if raw_xml is not None:
                    raw_bytes = raw_xml.encode("utf-8")
                    if snapshot.get("raw_xml_sha256") != digest(raw_bytes):
                        raise ValueError("Downloaded XML bytes do not match snapshot hash")
                    (output / f"{prefix}.raw.xml").write_bytes(raw_bytes)
                save_json(output / f"{prefix}.snapshot.json", snapshot)
                kept = [e for e in evidence if e.document_id.removeprefix("PMID:") not in forbidden]
                rejected = [e.chunk_id for e in evidence if e.document_id.removeprefix("PMID:") in forbidden]
                write_jsonl(output / f"{prefix}.evidence.jsonl", kept)
                entry.update({"status": "ok" if evidence else "empty", "snapshot_file": f"{prefix}.snapshot.json",
                              "raw_xml_file": f"{prefix}.raw.xml" if raw_xml is not None else None,
                              "raw_xml_sha256": snapshot.get("raw_xml_sha256"),
                              "evidence_file": f"{prefix}.evidence.jsonl",
                              "kept_chunk_ids": [e.chunk_id for e in kept],
                              "excluded_target_pmids": rejected,
                              "source_queries": [query]})
            except Exception as exc:
                entry.update({"status": "fetch_failed", "error": safe_error(exc)})
            save_json(manifest_path, manifest)
            if pause_seconds:
                time.sleep(pause_seconds)
        row["status"] = "ready" if all(e["status"] in {"ok", "empty"} for e in row["snapshots"]) else "fetch_failed"
        save_json(manifest_path, manifest)

    # union 按 chunk_id；来源查询逐条保留在清单，不塞进被测模型的 Evidence。
    by_chunk: dict[str, dict[str, Any]] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for row in manifest["records"]:
        for query_index, entry in enumerate(row["snapshots"]):
            if entry["status"] not in {"ok", "empty"}:
                continue
            for line in (output / entry["evidence_file"]).read_text(encoding="utf-8").splitlines():
                evidence = json.loads(line)
                chunk_id = evidence["chunk_id"]
                if chunk_id in by_chunk and by_chunk[chunk_id]["content_hash"] != evidence["content_hash"]:
                    raise ValueError(f"Same chunk_id has conflicting content: {chunk_id}")
                by_chunk[chunk_id] = evidence
                origin = {"question_id": row["question_id"], "query_index": query_index,
                          "query": row["queries"][query_index]}
                if origin not in sources.setdefault(chunk_id, []):
                    sources[chunk_id].append(origin)
    write_jsonl(output / "corpus.jsonl", list(by_chunk.values()))
    calls, tokens = totals(manifest)
    manifest.update({"model_calls": calls, "known_tokens": tokens,
                     "corpus_chunks": len(by_chunk), "corpus_sha256": digest((output / "corpus.jsonl").read_bytes()),
                     "chunk_sources": sources,
                     "status": "budget_stopped" if stopped_for_budget else
                               "complete" if all(r["status"] in {"ready", "empty_queries"} for r in manifest["records"])
                               else "incomplete"})
    if manifest["status"] == "complete":
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    summary = {"status": manifest["status"], "questions": len(questions),
               "ready_questions": sum(r["status"] in {"ready", "empty_queries"} for r in manifest["records"]),
               "query_failed": sum(r["status"] == "query_failed" for r in manifest["records"]),
               "fetch_failed": sum(r["status"] == "fetch_failed" for r in manifest["records"]),
               "model_calls": calls, "known_tokens": tokens, "corpus_chunks": len(by_chunk),
               "corpus_sha256": manifest["corpus_sha256"]}
    save_json(output / "summary.json", summary)
    save_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--model-backend", choices=["deepseek", "mock"], default="deepseek")
    args = parser.parse_args(argv)
    result = prepare(args.questions, args.output, args.max_records, args.max_calls,
                     args.max_tokens, args.model_backend)
    print(json.dumps({"status": result["status"], "questions": len(result["question_ids"]),
                      "ready": sum(r["status"] in {"ready", "empty_queries"} for r in result["records"]),
                      "model_calls": result.get("model_calls", 0), "known_tokens": result.get("known_tokens", 0),
                      "corpus_chunks": result.get("corpus_chunks", 0)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
