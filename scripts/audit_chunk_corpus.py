"""只读审计分片语料对父摘要的逐字映射、来源和 MedCPT 长度。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def audit(source: Path, target: Path, model_manifest: Path) -> dict:
    parent_raw = (source / "preparation_manifest.json").read_bytes()
    parent = json.loads(parent_raw)
    child = json.loads((target / "preparation_manifest.json").read_bytes())
    parent_bytes = (source / "corpus.jsonl").read_bytes()
    child_bytes = (target / "corpus.jsonl").read_bytes()
    issues = []
    if child["config"]["parent_manifest_sha256"] != sha(parent_raw):
        issues.append("parent manifest hash mismatch")
    if child["config"]["parent_corpus_sha256"] != sha(parent_bytes):
        issues.append("parent corpus hash mismatch")
    if child["config"]["questions_sha256"] != sha((source / "questions.jsonl").read_bytes()):
        issues.append("parent questions hash mismatch")
    if (target / "questions.jsonl").read_bytes() != (source / "questions.jsonl").read_bytes():
        issues.append("public questions changed")
    if child["corpus_sha256"] != sha(child_bytes):
        issues.append("child corpus hash mismatch")
    parents = {x["chunk_id"]: x for x in read_jsonl(source / "corpus.jsonl")}
    children = {x["chunk_id"]: x for x in read_jsonl(target / "corpus.jsonl")}
    if len(parents) != len(read_jsonl(source / "corpus.jsonl")) or len(children) != len(read_jsonl(target / "corpus.jsonl")):
        issues.append("duplicate chunk ID")
    if set(child["parent_to_children"]) != set(parents) or set(child["chunk_sources"]) != set(children):
        issues.append("parent or child mapping IDs differ")
    spans: dict[str, list[dict]] = {}
    for item in child["chunk_spans"]:
        spans.setdefault(item["parent_chunk_id"], []).append(item)
    if set(spans) != set(parents):
        issues.append("span parent IDs differ")
    split_count = 0
    for pid, original in parents.items():
        parts = spans.get(pid, [])
        if not parts:
            continue
        if len(parts) > 1:
            split_count += 1
        ids = [x["child_chunk_id"] for x in parts]
        if ids != child["parent_to_children"][pid]:
            issues.append(f"{pid}: child order mismatch")
        if parts[0]["char_start"] != 0 or parts[-1]["char_end"] != len(original["text"]):
            issues.append(f"{pid}: boundary mismatch")
        if any(a["char_end"] != b["char_start"] for a, b in zip(parts, parts[1:])):
            issues.append(f"{pid}: gaps or overlaps")
        if "".join(children[x]["text"] for x in ids) != original["text"]:
            issues.append(f"{pid}: parent text cannot be reconstructed")
        for part in parts:
            item = children[part["child_chunk_id"]]
            if (item["text"] != original["text"][part["char_start"]:part["char_end"]] or
                    sha(item["text"].encode()) != item["content_hash"] or
                    sha(item["text"].encode()) != part["child_content_sha256"] or
                    item["document_id"] != original["document_id"] or
                    child["chunk_sources"][item["chunk_id"]] != parent["chunk_sources"][pid]):
                issues.append(f"{pid}: child text/hash/document/provenance mismatch")
    forbidden = set(child["forbidden_pubmedqa_pmids"])
    if forbidden != set(parent["forbidden_pubmedqa_pmids"]) or len(forbidden) != 20:
        issues.append("target PMID set mismatch")
    if any(x["document_id"].removeprefix("PMID:") in forbidden for x in children.values()):
        issues.append("target PMID leaked")
    models = json.loads(model_manifest.read_text(encoding="utf-8"))["models"]
    article = models["ncbi/MedCPT-Article-Encoder"]
    if article["revision"] != child["tokenizer_revision"]:
        issues.append("tokenizer revision mismatch")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(article["snapshot_path"], use_fast=True, local_files_only=True)
    lengths = {cid: len(tokenizer(x["title"] + " [SEP] " + x["text"],
                                  add_special_tokens=False)["input_ids"]) for cid, x in children.items()}
    if max(lengths.values()) > 400 or any(lengths[x["child_chunk_id"]] != x["title_sep_text_tokens"]
                                          for x in child["chunk_spans"]):
        issues.append("token count mismatch or over 400")
    return {"parent_chunks": len(parents), "child_chunks": len(children), "split_parents": split_count,
            "max_tokens": max(lengths.values()), "target_pmids": len(forbidden),
            "parent_manifest_sha256": sha(parent_raw), "child_corpus_sha256": sha(child_bytes),
            "issues": issues}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.target, args.model_manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
