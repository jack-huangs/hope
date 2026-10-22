"""对原查询不足三条独立摘要的题，准备统一规则的宽查询补充快照。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_mdt.env import load_env
from evidence_mdt.models import create_model
import prepare_relevance_corpus as base


def public_model_config(model, backend: str) -> dict:
    return {"backend": backend, "name": model.name, "base_url": getattr(model, "base_url", None),
            "max_output_tokens": getattr(model, "max_tokens", None),
            "answer_temperature": getattr(model, "answer_temperature", None),
            "timeout": getattr(model, "timeout", None),
            "disable_thinking": getattr(model, "disable_thinking", None)}


def model_payload(question: dict) -> dict:
    # 只传题干；PubMedQA CONTEXTS 可能包含研究结果，绝不能进入宽查询生成。
    return {"task": "queries", "question": {"question_id": question["question_id"],
              "dataset": question["dataset"], "stem": question["stem"].split("\nAnswer Choices:", 1)[0]},
            "purpose": "Broad PubMed abstract recall when two frozen narrow queries yielded fewer than 3 unique abstracts",
            "constraints": ("Return exactly two short PubMed Boolean queries. Each query should use only 2-3 core medical "
                            "concepts, combining synonyms with OR and concepts with AND. Do not require all case "
                            "details. Do not answer the question, mention an option, or infer the correct choice. "
                            "Use only the stem, without background, options, answer labels, or prior retrieval results.")}


def validate_queries(queries) -> list[str]:
    if (not isinstance(queries, list) or len(queries) != 2 or
            any(not isinstance(q, str) or not q.strip() or len(q) > 500 or
                not re.search(r"\bAND\b", q, flags=re.I) for q in queries)):
        raise ValueError("Model must return exactly two short PubMed Boolean queries with AND")
    return [q.strip() for q in queries]


def source_inputs(source: Path) -> tuple[dict, list[dict], list[dict], bytes]:
    raw = (source / "preparation_manifest.json").read_bytes()
    parent = json.loads(raw)
    if parent.get("status") != "complete" or parent.get("config", {}).get("sort") != "relevance":
        raise ValueError("Source must be a complete relevance snapshot")
    public = base.read_jsonl(source / "questions.jsonl")
    corpus_bytes = (source / "corpus.jsonl").read_bytes()
    if base.sha(corpus_bytes) != parent["corpus_sha256"]:
        raise ValueError("Source corpus hash mismatch")
    corpus = base.read_jsonl(source / "corpus.jsonl")
    chunk_ids = [e["chunk_id"] for e in corpus]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Source corpus has duplicate chunk IDs")
    forbidden = set(parent["forbidden_pubmedqa_pmids"])
    if len(forbidden) != sum(q["dataset"].lower().startswith("pubmedqa") for q in public):
        raise ValueError("Source target PMID set differs from PubMedQA questions")
    if any(e["document_id"].removeprefix("PMID:") in forbidden for e in corpus):
        raise ValueError("Source corpus contains a target PubMedQA PMID")
    if len(public) != len(parent["records"]):
        raise ValueError("Source public questions differ")
    selected = []
    for question, row in zip(public, parent["records"], strict=True):
        if question["question_id"] != row["question_id"] or base.sha(base.canonical(question)) != row["public_question_sha256"]:
            raise ValueError("Source public question hash mismatch")
        ids = {chunk_id for entry in row["snapshots"] if entry["status"] in {"ok", "empty"}
               for chunk_id in entry["kept_chunk_ids"]}
        if len(ids) < 3:
            selected.append({"question_id": row["question_id"], "dataset": row["dataset"],
                             "source_unique_abstracts": len(ids),
                             "public_question_sha256": row["public_question_sha256"]})
    return parent, public, selected, raw


def totals(manifest: dict) -> tuple[int, int]:
    calls = [c for r in manifest["records"] for c in r["model_calls"]]
    return len(calls), sum(int((c.get("usage") or {}).get("total_tokens") or 0) for c in calls)


def prepare(source: Path, output: Path, max_records: int = 8, max_model_calls: int = 20,
            max_tokens: int = 30000, backend: str = "deepseek", retries: int = 2,
            pause: float = 1.0, backoff: float = 1.0, reuse_source: Path | None = None,
            query_protocol: str = "stem-only-2-to-3-core-concepts-v1",
            format_repair_sha256: str | None = None) -> dict:
    if max_records < 1 or max_model_calls < 0 or max_tokens < 0 or retries < 0 or pause < 0:
        raise ValueError("Invalid limits")
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Output must differ from source")
    parent, public, selected, parent_raw = source_inputs(source)
    if backend == "deepseek":
        load_env(ROOT / ".env")
    model = create_model(backend)
    config = {"source_manifest_sha256": base.sha(parent_raw),
              "selected_sha256": base.sha(base.canonical(selected)),
              "script_sha256": base.sha(Path(__file__).read_bytes()),
              "retrieval_script_sha256": base.sha(Path(base.__file__).read_bytes()),
              "models_code_sha256": base.sha((ROOT / "src/evidence_mdt/models.py").read_bytes()),
              "model": public_model_config(model, backend), "max_records": max_records,
              "sort": "relevance", "trigger": "per-question unique source abstracts < 3",
              "query_protocol": query_protocol, "retries": retries}
    if format_repair_sha256:
        config["format_repair_sha256"] = format_repair_sha256
    fingerprint = base.sha(base.canonical(config))
    if reuse_source:
        config["reuse_manifest_sha256"] = base.sha((reuse_source / "preparation_manifest.json").read_bytes())
        fingerprint = base.sha(base.canonical(config))
    manifest_path = output / "preparation_manifest.json"
    if output.exists() and not manifest_path.exists() and any(output.iterdir()):
        raise FileExistsError("Output has files but no broad-query manifest")
    output.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Existing broad snapshot has different configuration")
        if manifest.get("status") == "complete":
            return manifest
    else:
        shutil.copyfile(source / "questions.jsonl", output / "questions.jsonl")
        selected_ids = {x["question_id"] for x in selected}
        records = [{"question_id": q["question_id"], "dataset": q["dataset"],
                    "public_question_sha256": base.sha(base.canonical(q)),
                    "status": "query_pending" if q["question_id"] in selected_ids else "not_triggered",
                    "queries": [], "model_calls": [], "snapshots": []}
                   for q in public]
        manifest = {"schema_version": 1, "status": "running", "created_at": base.now(),
                    "fingerprint": fingerprint, "config": config, "source_snapshot": str(source),
                    "source_corpus_sha256": parent["corpus_sha256"],
                    "source_model_calls": parent["source_model_calls"],
                    "source_known_tokens": parent["source_known_tokens"],
                    "new_model_calls": 0, "known_tokens": 0, "question_ids": parent["question_ids"],
                    "selected_questions": selected, "forbidden_pubmedqa_pmids": parent["forbidden_pubmedqa_pmids"],
                    "records": records}
        if reuse_source:
            from reuse_query_snapshot import reuse_rows
            reuse_rows(output, manifest, reuse_source, broad=True)
        base.save_json(manifest_path, manifest)
    forbidden = set(manifest["forbidden_pubmedqa_pmids"])
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    budget_stopped = False
    for i, (question, row) in enumerate(zip(public, manifest["records"], strict=True)):
        if row["status"] == "not_triggered":
            continue
        if row["status"] in {"query_pending", "query_failed"}:
            calls, tokens = totals(manifest)
            if calls >= max_model_calls or tokens >= max_tokens:
                budget_stopped = True
                break
            payload = model_payload(question)
            previous = len(row["model_calls"])
            started_at = base.now()
            started = time.monotonic()
            try:
                response = model.complete(payload)
                row["model_calls"].append({"request": payload, "wire_request": response.wire_request,
                                            "raw_response": response.raw, "usage": response.usage,
                                            "started_at": started_at, "elapsed_seconds": time.monotonic() - started,
                                            "usage_known": bool(response.usage),
                                            "model_revision": response.raw.get("model", model.name)})
                row["queries"] = validate_queries(response.content.get("queries"))
                row["snapshots"] = [{"query": q, "status": "fetch_pending", "attempts": []}
                                    for q in row["queries"]]
                row["status"] = "queried"
                row.pop("error", None)
            except Exception as exc:
                if len(row["model_calls"]) == previous:
                    row["model_calls"].append({"request": payload,
                        "wire_request": getattr(exc, "wire_request", None),
                        "raw_response": getattr(exc, "raw", None),
                        "usage": getattr(exc, "usage", {}) or {},
                        "started_at": started_at, "elapsed_seconds": time.monotonic() - started,
                        "usage_known": bool(getattr(exc, "usage", None)),
                        "error_type": type(exc).__name__, "error": str(exc)[:300]})
                row["status"] = "query_failed"
                row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                base.save_json(manifest_path, manifest)
                continue
            base.save_json(manifest_path, manifest)
        for j, entry in enumerate(row["snapshots"]):
            if entry["status"] in {"ok", "empty"}:
                continue
            query = row["queries"][j]
            prefix = output / f"q{i + 1:04d}.broad{j + 1}"
            search_url = base_url + "esearch.fcgi?" + urllib.parse.urlencode(
                {"db": "pubmed", "term": query, "retmax": max_records, "retmode": "json", "sort": "relevance"})
            def checkpoint() -> None:
                base.save_json(manifest_path, manifest)
            try:
                search_raw = base.request_with_retries(search_url, prefix, "esearch", 30, retries,
                                                       backoff, entry, checkpoint)
                ids = json.loads(search_raw)["esearchresult"]["idlist"]
                fetched_at = base.now()
                snapshot = {"query": query, "sort": "relevance", "ids": ids,
                            "fetched_at": fetched_at, "search_url": search_url,
                            "raw_esearch_sha256": base.sha(search_raw),
                            "raw_esearch_file": entry["attempts"][-1]["raw_file"]}
                evidence = []
                if ids:
                    fetch_url = base_url + "efetch.fcgi?" + urllib.parse.urlencode(
                        {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
                    raw_xml = base.request_with_retries(fetch_url, prefix, "efetch", 60, retries,
                                                        backoff, entry, checkpoint)
                    evidence, metadata = base.parse_xml(raw_xml, fetched_at)
                    snapshot.update({"fetch_url": fetch_url, "raw_xml_sha256": base.sha(raw_xml),
                                     "article_metadata": metadata})
                    entry["raw_xml_file"] = entry["attempts"][-1]["raw_file"]
                kept = [e for e in evidence if e["document_id"].removeprefix("PMID:") not in forbidden]
                rejected = [e["chunk_id"] for e in evidence if e["document_id"].removeprefix("PMID:") in forbidden]
                snapshot_name, evidence_name = prefix.name + ".snapshot.json", prefix.name + ".evidence.jsonl"
                base.save_json(output / snapshot_name, snapshot)
                base.write_jsonl(output / evidence_name, kept)
                entry.update({"status": "ok" if ids else "empty", "snapshot_file": snapshot_name,
                              "evidence_file": evidence_name, "raw_xml_sha256": snapshot.get("raw_xml_sha256"),
                              "kept_chunk_ids": [e["chunk_id"] for e in kept],
                              "excluded_target_pmids": rejected})
                entry.pop("error", None)
            except Exception as exc:
                entry.update({"status": "fetch_failed", "error_type": type(exc).__name__,
                              "error": str(exc)[:300]})
            checkpoint()
            if pause:
                time.sleep(pause)
        row["status"] = "ready" if all(e["status"] in {"ok", "empty"} for e in row["snapshots"]) else "fetch_failed"
        base.save_json(manifest_path, manifest)
    by_chunk = {e["chunk_id"]: e for e in base.read_jsonl(source / "corpus.jsonl")}
    sources = {cid: list(origins) for cid, origins in parent["chunk_sources"].items()}
    for row in manifest["records"]:
        for j, entry in enumerate(row["snapshots"]):
            if entry["status"] not in {"ok", "empty"}:
                continue
            for evidence in base.read_jsonl(output / entry["evidence_file"]):
                cid = evidence["chunk_id"]
                if cid in by_chunk and by_chunk[cid]["content_hash"] != evidence["content_hash"]:
                    raise ValueError(f"Conflicting source abstract for {cid}")
                by_chunk[cid] = evidence
                origin = {"question_id": row["question_id"], "query_index": j,
                          "query": row["queries"][j], "stage": "broad"}
                if origin not in sources.setdefault(cid, []):
                    sources[cid].append(origin)
    base.write_jsonl(output / "corpus.jsonl", list(by_chunk.values()))
    calls, tokens = totals(manifest)
    reused = [c for r in manifest["records"] if r.get("reused_from") for c in r["model_calls"]]
    manifest.update({"new_model_calls": calls, "known_tokens": tokens,
                     "actual_new_model_calls": calls - len(reused),
                     "reused_model_calls": len(reused),
                     "reused_known_tokens": sum(int((c.get("usage") or {}).get("total_tokens") or 0) for c in reused),
                     "corpus_chunks": len(by_chunk), "corpus_sha256": base.sha((output / "corpus.jsonl").read_bytes()),
                     "chunk_sources": sources,
                     "status": "budget_stopped" if budget_stopped else
                               "complete" if all(r["status"] in {"ready", "not_triggered"} for r in manifest["records"])
                               else "incomplete"})
    if manifest["status"] == "complete":
        manifest["completed_at"] = base.now()
    base.save_json(output / "summary.json", {"status": manifest["status"], "questions": len(public),
                   "selected": len(selected), "ready_selected": sum(r["status"] == "ready" for r in manifest["records"]),
                   "new_model_calls": calls, "known_tokens": tokens, "corpus_chunks": len(by_chunk),
                   "source_corpus_chunks": parent["corpus_chunks"], "corpus_sha256": manifest["corpus_sha256"]})
    base.save_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--max-model-calls", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=30000)
    parser.add_argument("--model-backend", choices=["deepseek", "mock"], default="deepseek")
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.max_records, args.max_model_calls,
                     args.max_tokens, args.model_backend, args.retries)
    print(json.dumps({"status": result["status"], "new_model_calls": result["new_model_calls"],
                      "corpus_chunks": result["corpus_chunks"]}, ensure_ascii=False))
