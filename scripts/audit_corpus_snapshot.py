"""只读审计已保存的 PubMed 摘要快照；不发起网络请求。"""
from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(directory: Path) -> dict:
    manifest = json.loads((directory / "preparation_manifest.json").read_text(encoding="utf-8"))
    public = lines(directory / "questions.jsonl")
    corpus = lines(directory / "corpus.jsonl")
    forbidden = set(manifest["forbidden_pubmedqa_pmids"])
    issues: list[str] = []
    details = []
    expected_union: dict[str, dict] = {}
    expected_sources: dict[str, list[dict]] = {}
    query_count = empty_queries = missing_xml = total_kept = total_excluded = 0
    if manifest.get("config", {}).get("sort") == "relevance":
        if manifest.get("new_model_calls") != 0:
            issues.append("Relevance snapshot records new model calls")
        source = Path(manifest["source_snapshot"])
        if sha((source / "preparation_manifest.json").read_bytes()) != manifest["config"]["source_manifest_sha256"]:
            issues.append("Source manifest lineage hash mismatch")
    for i, (row, question) in enumerate(zip(manifest["records"], public, strict=True)):
        label = f"q{i + 1:04d} {row['question_id']}"
        if question.get("gold") is not None or any(k.lower() in {"gold", "long_answer", "final_decision", "answer_idx"} for k in question):
            issues.append(f"{label}: public question contains answer field")
        if sha(canonical(question)) != row["public_question_sha256"]:
            issues.append(f"{label}: public question hash mismatch")
        for call in row.get("model_calls", []):
            if call["request"].get("question") != question:
                issues.append(f"{label}: model request differs from public question")
            wire = call.get("wire_request") or {}
            if wire:
                user_messages = [m["content"] for m in wire.get("messages", []) if m.get("role") == "user"]
                if len(user_messages) != 1 or json.loads(user_messages[0]).get("question") != question:
                    issues.append(f"{label}: wire request differs from public question")
        if row.get("model_calls"):
            raw = row["model_calls"][-1].get("raw_response") or {}
            try:
                generated = json.loads(raw["choices"][0]["message"]["content"])["queries"]
                if generated != row["queries"]:
                    issues.append(f"{label}: saved queries differ from model response")
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                issues.append(f"{label}: query response cannot be independently decoded")
        if len(row["queries"]) != len(row["snapshots"]):
            issues.append(f"{label}: query/snapshot count mismatch")
        unique = set()
        counts = []
        for j, entry in enumerate(row["snapshots"]):
            query_count += 1
            if entry.get("query") != row["queries"][j]:
                issues.append(f"{label} query{j + 1}: query changed")
            snapshot = json.loads((directory / entry["snapshot_file"]).read_text(encoding="utf-8"))
            for attempt in entry.get("attempts", []):
                if attempt.get("raw_file") and sha((directory / attempt["raw_file"]).read_bytes()) != attempt.get("raw_sha256"):
                    issues.append(f"{label} query{j + 1}: HTTP attempt raw hash mismatch")
            if snapshot.get("raw_esearch_file"):
                if sha((directory / snapshot["raw_esearch_file"]).read_bytes()) != snapshot.get("raw_esearch_sha256"):
                    issues.append(f"{label} query{j + 1}: ESearch raw hash mismatch")
            if snapshot.get("query") != entry["query"]:
                issues.append(f"{label} query{j + 1}: metadata query changed")
            if not snapshot.get("ids"):
                empty_queries += 1
            expected_sort = manifest.get("config", {}).get("sort")
            actual_sort = parse_qs(urlparse(snapshot["search_url"]).query).get("sort")
            if actual_sort != ([expected_sort] if expected_sort else None):
                issues.append(f"{label} query{j + 1}: unexpected search sort")
            evidence = lines(directory / entry["evidence_file"])
            if [e["chunk_id"] for e in evidence] != entry["kept_chunk_ids"]:
                issues.append(f"{label} query{j + 1}: evidence ID list mismatch")
            counts.append({"returned_pmids": len(snapshot["ids"]), "kept_abstracts": len(evidence),
                           "excluded_target_pmids": len(entry["excluded_target_pmids"])})
            total_kept += len(evidence)
            total_excluded += len(entry["excluded_target_pmids"])
            for e in evidence:
                if sha(e["text"].encode()) != e["content_hash"]:
                    issues.append(f"{label} query{j + 1}: abstract content hash mismatch {e['chunk_id']}")
                if e["document_id"].removeprefix("PMID:") in forbidden:
                    issues.append(f"{label} query{j + 1}: target PMID leaked {e['chunk_id']}")
                unique.add(e["chunk_id"])
                old = expected_union.setdefault(e["chunk_id"], e)
                if old["content_hash"] != e["content_hash"]:
                    issues.append(f"{label} query{j + 1}: conflicting abstract {e['chunk_id']}")
                origin = {"question_id": row["question_id"], "query_index": j, "query": row["queries"][j]}
                if origin not in expected_sources.setdefault(e["chunk_id"], []):
                    expected_sources[e["chunk_id"]].append(origin)
            xml_name = entry.get("raw_xml_file")
            if xml_name:
                raw_bytes = (directory / xml_name).read_bytes()
                if sha(raw_bytes) != snapshot.get("raw_xml_sha256") or sha(raw_bytes) != entry.get("raw_xml_sha256"):
                    issues.append(f"{label} query{j + 1}: XML hash mismatch")
                root = ET.fromstring(raw_bytes)
                xml_abstracts = {}
                for article in root.findall(".//PubmedArticle"):
                    pmid = article.findtext(".//MedlineCitation/PMID")
                    abstract = " ".join("".join(x.itertext()) for x in article.findall(".//Abstract/AbstractText"))
                    if pmid and abstract:
                        xml_abstracts[f"PMID:{pmid}:abstract"] = abstract
                for e in evidence:
                    if xml_abstracts.get(e["chunk_id"]) != e["text"]:
                        issues.append(f"{label} query{j + 1}: evidence differs from XML {e['chunk_id']}")
                for item in snapshot.get("article_metadata", []):
                    if f"PMID:{item['pmid']}:abstract" not in xml_abstracts:
                        issues.append(f"{label} query{j + 1}: metadata PMID absent from XML")
            elif snapshot.get("ids"):
                missing_xml += 1
                issues.append(f"{label} query{j + 1}: returned IDs but no XML")
        details.append({"question_id": row["question_id"], "dataset": row["dataset"],
                        "queries": len(row["queries"]), "per_query": counts,
                        "unique_kept_abstracts": len(unique)})
    corpus_ids = {e["chunk_id"] for e in corpus}
    if len(corpus_ids) != len(corpus) or corpus_ids != set(expected_union):
        issues.append("corpus union IDs differ from query evidence")
    if sha((directory / "corpus.jsonl").read_bytes()) != manifest["corpus_sha256"]:
        issues.append("corpus file hash mismatch")
    if manifest["chunk_sources"] != expected_sources:
        issues.append("chunk_sources provenance differs from query evidence")
    if any(e["document_id"].removeprefix("PMID:") in forbidden for e in corpus):
        issues.append("target PMID in final corpus")
    return {"status": manifest["status"], "questions": len(details), "pubmedqa_target_pmids": len(forbidden),
            "queries": query_count, "empty_queries": empty_queries, "missing_xml": missing_xml,
            "query_kept_abstracts_with_duplicates": total_kept, "excluded_target_abstracts": total_excluded,
            "union_chunks": len(corpus), "model_calls": manifest.get("model_calls", manifest.get("new_model_calls", 0)),
            "known_tokens": manifest.get("known_tokens", 0), "issues": issues, "per_question": details}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), ensure_ascii=False, indent=2))
