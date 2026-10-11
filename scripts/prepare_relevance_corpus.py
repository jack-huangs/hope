"""复用冻结的查询，以 PubMed Best Match 排序准备独立摘要快照。无模型调用。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, obj: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_parent(source: Path) -> tuple[dict, list[dict], list[dict], str, str]:
    raw = (source / "preparation_manifest.json").read_bytes()
    parent = json.loads(raw)
    if parent.get("status") != "complete":
        raise ValueError("Source snapshot must be complete")
    public = read_jsonl(source / "questions.jsonl")
    records = parent["records"]
    if len(public) != len(records) or not records:
        raise ValueError("Source questions and records differ")
    frozen_queries = []
    for question, row in zip(public, records, strict=True):
        if question.get("question_id") != row["question_id"] or sha(canonical(question)) != row["public_question_sha256"]:
            raise ValueError("Source public question differs from manifest")
        if row["status"] not in {"ready", "empty_queries"} or len(row["queries"]) != len(row["snapshots"]):
            raise ValueError("Source query set is incomplete")
        if not all(isinstance(q, str) and q.strip() for q in row["queries"]):
            raise ValueError("Source contains an invalid query")
        if any(s["query"] != q for q, s in zip(row["queries"], row["snapshots"], strict=True)):
            raise ValueError("Source query/snapshot mismatch")
        calls = row.get("model_calls") or []
        if not calls:
            raise ValueError("Source query has no recorded model response")
        try:
            generated = json.loads(calls[-1]["raw_response"]["choices"][0]["message"]["content"])["queries"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Source query response cannot be decoded") from exc
        if generated != row["queries"]:
            raise ValueError("Source query differs from model response")
        frozen_queries.append({"question_id": row["question_id"], "queries": row["queries"]})
    if [q["question_id"] for q in public] != parent["question_ids"]:
        raise ValueError("Source question order differs")
    return parent, public, frozen_queries, sha(raw), sha(canonical(frozen_queries))


def request_once(url: str, filename: Path, timeout: int) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        raw = response.read()
    filename.write_bytes(raw)
    return raw


def request_with_retries(url: str, prefix: Path, suffix: str, timeout: int, retries: int,
                         backoff: float, entry: dict, checkpoint: Any) -> bytes:
    # 每次请求保存原始体及结果；最后一次失败仍有清单可查。
    previous = max((int(item["attempt"]) for item in entry["attempts"]
                    if item.get("phase") == suffix), default=0)
    for attempt in range(retries + 1):
        number = previous + attempt + 1
        filename = prefix.with_name(f"{prefix.name}.{suffix}.attempt{number}.raw")
        log = {"phase": suffix, "attempt": number, "url": url, "at": now(),
               "raw_file": filename.name}
        try:
            raw = request_once(url, filename, timeout)
            log["raw_sha256"] = sha(raw)
            if suffix == "esearch":
                result = json.loads(raw)
                ids = result.get("esearchresult", {}).get("idlist")
                if not isinstance(ids, list) or any(not isinstance(x, str) for x in ids):
                    raise ValueError("ESearch response missing valid esearchresult.idlist")
            else:
                ET.fromstring(raw)
            log["status"] = "ok"
            entry["attempts"].append(log)
            checkpoint()
            return raw
        except Exception as exc:
            # HTTP 错误正文也保留，便于区分服务器临时故障与查询零命中。
            if hasattr(exc, "read"):
                try:
                    error_body = exc.read()
                    filename.write_bytes(error_body)
                    log["raw_sha256"] = sha(error_body)
                except Exception:
                    pass
            log.update({"status": "failed", "error_type": type(exc).__name__,
                        "error": str(exc)[:300]})
            if not filename.exists():
                log["raw_file"] = None
            entry["attempts"].append(log)
            checkpoint()
            if attempt == retries:
                raise
            if backoff:
                time.sleep(backoff * (2 ** attempt))
    raise RuntimeError("Unreachable retry state")


def parse_xml(raw: bytes, fetched_at: str) -> tuple[list[dict], list[dict]]:
    root = ET.fromstring(raw)
    evidence, metadata = [], []
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//MedlineCitation/PMID")
        title_node = article.find(".//ArticleTitle")
        title = "".join(title_node.itertext()) if title_node is not None else ""
        doi = next((x.text for x in article.findall(".//ArticleId") if x.attrib.get("IdType") == "doi"), None)
        abstract = " ".join("".join(x.itertext()) for x in article.findall(".//Abstract/AbstractText"))
        if pmid and abstract:
            evidence.append({"chunk_id": f"PMID:{pmid}:abstract", "document_id": f"PMID:{pmid}",
                             "text": abstract, "source": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                             "position": "abstract", "retrieved_at": fetched_at,
                             "content_hash": sha(abstract.encode()), "derived_from": None, "title": title})
            metadata.append({"pmid": pmid, "doi": doi, "title": title})
    return evidence, metadata


def prepare(source: Path, output: Path, max_records: int = 8, retries: int = 2,
            pause: float = 1.0, backoff: float = 1.0) -> dict:
    if max_records < 1 or retries < 0 or pause < 0 or backoff < 0:
        raise ValueError("Invalid retrieval limits")
    source, output = source.resolve(), output.resolve()
    if source == output:
        raise ValueError("Output must be a separate directory")
    parent, public, frozen_queries, parent_hash, query_hash = validate_parent(source)
    config = {"source_manifest_sha256": parent_hash, "frozen_queries_sha256": query_hash,
              "script_sha256": sha(Path(__file__).read_bytes()), "sort": "relevance",
              "max_records": max_records, "retries": retries}
    fingerprint = sha(canonical(config))
    path = output / "preparation_manifest.json"
    if output.exists() and not path.exists() and any(output.iterdir()):
        raise FileExistsError("Output has files but no relevance manifest")
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("Existing relevance snapshot has different configuration; use a new output")
        if manifest.get("status") == "complete":
            return manifest
    else:
        shutil.copyfile(source / "questions.jsonl", output / "questions.jsonl")
        records = [{"question_id": r["question_id"], "dataset": r["dataset"],
                    "public_question_sha256": r["public_question_sha256"], "queries": list(r["queries"]),
                    "status": "pending", "snapshots": [{"query": q, "status": "fetch_pending", "attempts": []}
                                                      for q in r["queries"]]}
                   for r in parent["records"]]
        manifest = {"schema_version": 1, "status": "running", "created_at": now(),
                    "fingerprint": fingerprint, "config": config,
                    "source_snapshot": str(source), "source_model_calls": parent["model_calls"],
                    "source_known_tokens": parent["known_tokens"], "new_model_calls": 0,
                    "question_ids": parent["question_ids"],
                    "forbidden_pubmedqa_pmids": parent["forbidden_pubmedqa_pmids"], "records": records}
        save_json(path, manifest)
    forbidden = set(manifest["forbidden_pubmedqa_pmids"])
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    for i, row in enumerate(manifest["records"]):
        for j, entry in enumerate(row["snapshots"]):
            if entry["status"] in {"ok", "empty"}:
                continue
            query = row["queries"][j]
            prefix = output / f"q{i + 1:04d}.query{j + 1}"
            search_url = base + "esearch.fcgi?" + urllib.parse.urlencode(
                {"db": "pubmed", "term": query, "retmax": max_records, "retmode": "json", "sort": "relevance"})
            def checkpoint() -> None:
                save_json(path, manifest)
            try:
                search_raw = request_with_retries(search_url, prefix, "esearch", 30, retries,
                                                  backoff, entry, checkpoint)
                ids = json.loads(search_raw)["esearchresult"]["idlist"]
                fetched_at = now()
                snapshot = {"query": query, "sort": "relevance", "ids": ids,
                            "fetched_at": fetched_at, "search_url": search_url,
                            "raw_esearch_sha256": sha(search_raw),
                            "raw_esearch_file": entry["attempts"][-1]["raw_file"]}
                evidence = []
                if ids:
                    fetch_url = base + "efetch.fcgi?" + urllib.parse.urlencode(
                        {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"})
                    raw_xml = request_with_retries(fetch_url, prefix, "efetch", 60, retries,
                                                   backoff, entry, checkpoint)
                    evidence, metadata = parse_xml(raw_xml, fetched_at)
                    snapshot.update({"fetch_url": fetch_url, "raw_xml_sha256": sha(raw_xml),
                                     "article_metadata": metadata})
                    entry["raw_xml_file"] = entry["attempts"][-1]["raw_file"]
                kept = [e for e in evidence if e["document_id"].removeprefix("PMID:") not in forbidden]
                rejected = [e["chunk_id"] for e in evidence if e["document_id"].removeprefix("PMID:") in forbidden]
                snapshot_name = prefix.name + ".snapshot.json"
                evidence_name = prefix.name + ".evidence.jsonl"
                save_json(output / snapshot_name, snapshot)
                write_jsonl(output / evidence_name, kept)
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
        save_json(path, manifest)
    by_chunk: dict[str, dict] = {}
    sources: dict[str, list[dict]] = {}
    for row in manifest["records"]:
        for j, entry in enumerate(row["snapshots"]):
            if entry["status"] not in {"ok", "empty"}:
                continue
            for evidence in read_jsonl(output / entry["evidence_file"]):
                chunk_id = evidence["chunk_id"]
                if chunk_id in by_chunk and by_chunk[chunk_id]["content_hash"] != evidence["content_hash"]:
                    raise ValueError(f"Conflicting content for {chunk_id}")
                by_chunk[chunk_id] = evidence
                origin = {"question_id": row["question_id"], "query_index": j, "query": row["queries"][j]}
                if origin not in sources.setdefault(chunk_id, []):
                    sources[chunk_id].append(origin)
    write_jsonl(output / "corpus.jsonl", list(by_chunk.values()))
    manifest.update({"status": "complete" if all(r["status"] == "ready" for r in manifest["records"])
                     else "incomplete", "corpus_chunks": len(by_chunk),
                     "corpus_sha256": sha((output / "corpus.jsonl").read_bytes()), "chunk_sources": sources})
    if manifest["status"] == "complete":
        manifest["completed_at"] = now()
    save_json(output / "summary.json", {"status": manifest["status"], "questions": len(public),
              "queries": sum(len(r["queries"]) for r in manifest["records"]),
              "ready_questions": sum(r["status"] == "ready" for r in manifest["records"]),
              "empty_queries": sum(e["status"] == "empty" for r in manifest["records"] for e in r["snapshots"]),
              "fetch_failed": sum(e["status"] == "fetch_failed" for r in manifest["records"] for e in r["snapshots"]),
              "corpus_chunks": len(by_chunk), "corpus_sha256": manifest["corpus_sha256"],
              "new_model_calls": 0, "source_model_calls": parent["model_calls"]})
    save_json(path, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original complete snapshot directory")
    parser.add_argument("--output", type=Path, required=True, help="New relevance snapshot directory")
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--retries", type=int, default=2, help="Extra attempts after initial HTTP request")
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.max_records, args.retries)
    print(json.dumps({"status": result["status"], "corpus_chunks": result["corpus_chunks"],
                      "new_model_calls": 0}, ensure_ascii=False))
