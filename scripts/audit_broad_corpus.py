"""只读核对宽查询补充快照、父快照和 union 的来源与原文。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import audit_corpus_snapshot as parent_audit
import prepare_relevance_corpus as base


def audit(folder: Path) -> dict:
    folder = folder.resolve()
    manifest_raw = (folder / "preparation_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    failure_hash_path = folder / "failed_response_hashes.json"
    failure_hashes = json.loads(failure_hash_path.read_bytes()) if failure_hash_path.exists() else {}
    source = Path(manifest["source_snapshot"])
    source_raw = (source / "preparation_manifest.json").read_bytes()
    parent = json.loads(source_raw)
    issues = []
    parent_result = parent_audit.audit(source)
    if parent_result["issues"]:
        issues.extend("parent: " + x for x in parent_result["issues"])
    if base.sha(source_raw) != manifest["config"]["source_manifest_sha256"]:
        issues.append("Parent manifest hash mismatch")
    source_bytes = (source / "corpus.jsonl").read_bytes()
    if base.sha(source_bytes) != manifest["source_corpus_sha256"]:
        issues.append("Parent corpus hash mismatch")
    public = base.read_jsonl(folder / "questions.jsonl")
    if (folder / "questions.jsonl").read_bytes() != (source / "questions.jsonl").read_bytes():
        issues.append("Public questions differ from parent")
    if len(public) != len(manifest["records"]):
        issues.append("Public question count mismatch")
    forbidden = set(manifest["forbidden_pubmedqa_pmids"])
    if len(forbidden) != 20 or forbidden != set(parent["forbidden_pubmedqa_pmids"]):
        issues.append("PubMedQA target PMID set differs")
    parent_corpus = base.read_jsonl(source / "corpus.jsonl")
    by_chunk = {e["chunk_id"]: e for e in parent_corpus}
    provenance = {cid: list(origins) for cid, origins in parent["chunk_sources"].items()}
    new_ids = set()
    query_count = empty = total_kept = total_excluded = 0
    expected_selected = []
    for q, source_row in zip(public, parent["records"], strict=True):
        local = {cid for entry in source_row["snapshots"] for cid in entry["kept_chunk_ids"]}
        if len(local) < 3:
            expected_selected.append((q["question_id"], len(local)))
    if expected_selected != [(x["question_id"], x["source_unique_abstracts"]) for x in manifest["selected_questions"]]:
        issues.append("Trigger set differs from frozen <3 rule")
    for q, row in zip(public, manifest["records"], strict=True):
        label = q["question_id"]
        if row["question_id"] != label or row["public_question_sha256"] != base.sha(base.canonical(q)):
            issues.append(f"{label}: public question hash mismatch")
        selected = label in {x[0] for x in expected_selected}
        if not selected:
            if row["status"] != "not_triggered" or row["model_calls"] or row["snapshots"]:
                issues.append(f"{label}: unselected question was processed")
            continue
        if row["status"] != "ready" or not 1 <= len(row["model_calls"]) <= 3 or len(row["queries"]) != 2:
            issues.append(f"{label}: selected question not ready with bounded calls and two queries")
            continue
        if len(row["model_calls"]) == 3 and row["model_calls"][-1].get("call_kind") != "query_format_repair":
            issues.append(f"{label}: third call is not an explicit format repair")
        expected_visible = {"question_id": q["question_id"], "dataset": q["dataset"],
                            "stem": q["stem"].split("\nAnswer Choices:", 1)[0]}
        for attempt_index, attempt in enumerate(row["model_calls"]):
            payload = attempt["request"]
            if attempt.get("call_kind") == "query_format_repair":
                expected_keys = {"task", "protocol", "question_id", "input_queries", "constraints"}
                if set(payload) != expected_keys or payload.get("question_id") != label or attempt_index != 2:
                    issues.append(f"{label}/{attempt_index}: invalid format-only repair payload")
                previous = json.loads(row["model_calls"][attempt_index - 1]["raw_response"]["choices"][0]["message"]["content"])["queries"]
                if payload.get("input_queries") != previous:
                    issues.append(f"{label}: repair input differs from last invalid response")
                from repair_broad_query_format_oct12 import validate_repair
                try:
                    validate_repair(previous, row["queries"])
                except ValueError:
                    issues.append(f"{label}: repair changed original query words")
            else:
                visible = payload["question"]
                if set(visible) != {"question_id", "dataset", "stem"} or visible != expected_visible:
                    issues.append(f"{label}/{attempt_index}: model request includes non-stem question data")
            wire = attempt.get("wire_request") or {}
            users = [x["content"] for x in wire.get("messages", []) if x.get("role") == "user"]
            if users and (len(users) != 1 or json.loads(users[0]) != payload):
                issues.append(f"{label}/{attempt_index}: wire request differs from saved payload")
            if attempt_index < len(row["model_calls"]) - 1:
                try:
                    prior = json.loads(attempt["raw_response"]["choices"][0]["message"]["content"])["queries"]
                    prior_valid = (isinstance(prior, list) and len(prior) == 2 and all(
                        isinstance(term, str) and term.strip() and len(term) <= 500 and re.search(r"\bAND\b", term, re.I)
                        for term in prior))
                    if prior_valid:
                        issues.append(f"{label}/{attempt_index}: prior valid query call was unnecessarily replaced")
                except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                    pass  # 结构失败也允许一次有界重试；原调用和用量仍保留。
        call = row["model_calls"][-1]
        try:
            generated = json.loads(call["raw_response"]["choices"][0]["message"]["content"])["queries"]
            if generated != row["queries"]:
                issues.append(f"{label}: queries differ from model response")
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            issues.append(f"{label}: raw model response cannot be decoded")
        if not call.get("started_at") or not isinstance(call.get("elapsed_seconds"), (int, float)):
            issues.append(f"{label}: model timing absent")
        for j, entry in enumerate(row["snapshots"]):
            query_count += 1
            if entry["query"] != row["queries"][j]:
                issues.append(f"{label}/{j}: query changed")
            snapshot = json.loads((folder / entry["snapshot_file"]).read_text(encoding="utf-8"))
            if snapshot["query"] != entry["query"] or parse_qs(urlparse(snapshot["search_url"]).query).get("sort") != ["relevance"]:
                issues.append(f"{label}/{j}: search query or sort differs")
            for attempt in entry["attempts"]:
                if attempt.get("raw_file"):
                    expected_hash = attempt.get("raw_sha256")
                    if not expected_hash and attempt.get("status") == "failed":
                        if failure_hashes.get("manifest_sha256") != base.sha(manifest_raw):
                            issues.append(f"{label}/{j}: missing failure-body hash provenance")
                        expected_hash = failure_hashes.get("files", {}).get(attempt["raw_file"])
                    if base.sha((folder / attempt["raw_file"]).read_bytes()) != expected_hash:
                        issues.append(f"{label}/{j}: raw attempt hash mismatch")
            if base.sha((folder / snapshot["raw_esearch_file"]).read_bytes()) != snapshot["raw_esearch_sha256"]:
                issues.append(f"{label}/{j}: raw ESearch hash mismatch")
            ids = json.loads((folder / snapshot["raw_esearch_file"]).read_bytes())["esearchresult"]["idlist"]
            if ids != snapshot["ids"]:
                issues.append(f"{label}/{j}: ESearch IDs differ from raw")
            if not ids:
                empty += 1
            kept = base.read_jsonl(folder / entry["evidence_file"])
            if [e["chunk_id"] for e in kept] != entry["kept_chunk_ids"]:
                issues.append(f"{label}/{j}: evidence IDs differ from entry")
            total_kept += len(kept)
            total_excluded += len(entry["excluded_target_pmids"])
            if ids:
                xml = (folder / entry["raw_xml_file"]).read_bytes()
                if base.sha(xml) != snapshot["raw_xml_sha256"] or base.sha(xml) != entry["raw_xml_sha256"]:
                    issues.append(f"{label}/{j}: XML hash mismatch")
                all_evidence, metadata = base.parse_xml(xml, snapshot["fetched_at"])
                expected_kept = [e for e in all_evidence if e["document_id"].removeprefix("PMID:") not in forbidden]
                expected_rejected = [e["chunk_id"] for e in all_evidence if e["document_id"].removeprefix("PMID:") in forbidden]
                if metadata != snapshot["article_metadata"] or kept != expected_kept or expected_rejected != entry["excluded_target_pmids"]:
                    issues.append(f"{label}/{j}: XML content/metadata/exclusion differs")
            for e in kept:
                cid = e["chunk_id"]
                if base.sha(e["text"].encode()) != e["content_hash"] or e["document_id"].removeprefix("PMID:") in forbidden:
                    issues.append(f"{label}/{j}: content hash or target exclusion invalid {cid}")
                if cid in by_chunk and by_chunk[cid]["content_hash"] != e["content_hash"]:
                    issues.append(f"{label}/{j}: conflicting abstract {cid}")
                if cid not in by_chunk:
                    new_ids.add(cid)
                by_chunk[cid] = e
                origin = {"question_id": label, "query_index": j, "query": row["queries"][j], "stage": "broad"}
                if origin not in provenance.setdefault(cid, []):
                    provenance[cid].append(origin)
    union_bytes = (folder / "corpus.jsonl").read_bytes()
    union = base.read_jsonl(folder / "corpus.jsonl")
    if base.sha(union_bytes) != manifest["corpus_sha256"] or len(union) != len(by_chunk) or set(e["chunk_id"] for e in union) != set(by_chunk):
        issues.append("Union corpus hash or IDs differ")
    if any(e != by_chunk[e["chunk_id"]] for e in union):
        issues.append("Union abstract differs from source evidence")
    if manifest["chunk_sources"] != provenance:
        issues.append("Union provenance differs")
    if any(e["document_id"].removeprefix("PMID:") in forbidden for e in union):
        issues.append("Target PMID leaked into union")
    reported_calls = sum(len(r["model_calls"]) for r in manifest["records"])
    reported_tokens = sum(int((c.get("usage") or {}).get("total_tokens") or 0)
                          for r in manifest["records"] for c in r["model_calls"])
    if reported_calls != manifest["new_model_calls"] or reported_tokens != manifest["known_tokens"]:
        issues.append("Model call or usage totals differ")
    return {"status": manifest["status"], "parent_status": parent_result["status"],
            "questions": len(public), "selected": len(expected_selected), "queries": query_count,
            "empty_queries": empty, "new_model_calls": reported_calls, "known_tokens": reported_tokens,
            "source_chunks": len(parent_corpus), "new_chunks": len(new_ids), "union_chunks": len(union),
            "query_kept_with_duplicates": total_kept, "excluded_target_abstracts": total_excluded,
            "target_pmids": len(forbidden), "issues": issues}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder), ensure_ascii=False, indent=2))
