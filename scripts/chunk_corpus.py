"""将长 PubMed 摘要离线切片，保留逐字区间与来源映射。"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def token_count(tokenizer, title: str, text: str) -> int:
    return len(tokenizer(title + " [SEP] " + text, add_special_tokens=False)["input_ids"])


def spans(title: str, text: str, tokenizer, limit: int = 400) -> list[tuple[int, int]]:
    """按句末空白优先切分；词边界和 tokenizer offset 作超长单句后备。"""
    if token_count(tokenizer, title, text) <= limit:
        return [(0, len(text))]
    prefix_tokens = len(tokenizer(title + " [SEP] ", add_special_tokens=False)["input_ids"])
    budget = limit - prefix_tokens - 4  # 给连接处的分词变化留余量，最终仍逐片实测。
    if budget < 1:
        raise ValueError("Title leaves no room for any abstract token")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if not offsets:
        raise ValueError("Long abstract has no token offsets")
    token_ends = [end for _start, end in offsets]
    sentence_ends = [m.end() for m in re.finditer(r"(?<=[.!?。！？])\s+|\n+", text)]
    word_ends = [m.end() for m in re.finditer(r"\s+", text)]
    out = []
    start = 0
    while start < len(text):
        if token_count(tokenizer, title, text[start:]) <= limit:
            out.append((start, len(text)))
            break
        token_index = bisect.bisect_right(token_ends, start)
        candidate_index = min(len(token_ends) - 1, token_index + budget - 1)
        max_end = token_ends[candidate_index]
        if max_end <= start:
            raise ValueError("Tokenizer offsets did not advance")
        # 有足够近的句末就用句末；否则选择最近词边界，最后才在字符边界切。
        floor = start + (max_end - start) // 2
        sentence = [end for end in sentence_ends if floor <= end <= max_end]
        words = [end for end in word_ends if start < end <= max_end]
        end = sentence[-1] if sentence else words[-1] if words else max_end
        while token_count(tokenizer, title, text[start:end]) > limit:
            prior = bisect.bisect_left(token_ends, end) - 1
            if prior < 0 or token_ends[prior] <= start:
                raise ValueError("A single token exceeds the chunk limit")
            end = token_ends[prior]
        out.append((start, end))
        start = end
    if out[0][0] != 0 or out[-1][1] != len(text) or any(a[1] != b[0] for a, b in zip(out, out[1:])):
        raise AssertionError("Chunk spans do not cover original text exactly")
    if "".join(text[a:b] for a, b in out) != text:
        raise AssertionError("Chunk text cannot reconstruct parent")
    return out


def chunk_rows(rows: list[dict[str, Any]], tokenizer, limit: int = 400
               ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    children = []
    span_records = []
    parent_to_children = {}
    used_ids: set[str] = set()
    for parent in rows:
        parent_id, original = parent["chunk_id"], parent["text"]
        parent_hash = sha(original.encode("utf-8"))
        if parent.get("content_hash") and parent["content_hash"] != parent_hash:
            raise ValueError(f"Parent content hash mismatch: {parent_id}")
        segments = spans(parent.get("title", ""), original, tokenizer, limit)
        child_ids = []
        for start, end in segments:
            piece = original[start:end]
            if len(segments) == 1:
                child = dict(parent)  # 短摘要完全保留原 ID、原元数据和内容。
            else:
                child = dict(parent)
                child_id = "chunk:" + sha(f"{parent_id}:{start}:{end}".encode("utf-8"))[:20]
                child.update(chunk_id=child_id, text=piece,
                             position=f"{parent.get('position', 'abstract')} chars[{start}:{end}]",
                             content_hash=sha(piece.encode("utf-8")), derived_from=None)
            if child["chunk_id"] in used_ids:
                raise ValueError(f"Duplicate child chunk ID: {child['chunk_id']}")
            used_ids.add(child["chunk_id"])
            count = token_count(tokenizer, child.get("title", ""), child["text"])
            if count > limit:
                raise AssertionError(f"Chunk exceeds {limit} tokens")
            children.append(child)
            child_ids.append(child["chunk_id"])
            span_records.append({"parent_chunk_id": parent_id, "child_chunk_id": child["chunk_id"],
                                 "parent_content_sha256": parent_hash, "char_start": start,
                                 "char_end": end, "child_content_sha256": sha(piece.encode("utf-8")),
                                 "title_sep_text_tokens": count})
        if "".join(c["text"] for c in children[-len(segments):]) != original:
            raise AssertionError(f"Parent cannot be reconstructed: {parent_id}")
        parent_to_children[parent_id] = child_ids
    return children, span_records, parent_to_children


def prepare(source: Path, output: Path, tokenizer, revision: str, limit: int = 400) -> dict[str, Any]:
    source, output = source.resolve(), output.resolve()
    source_manifest_path = source / "preparation_manifest.json"
    source_corpus_path = source / "corpus.jsonl"
    source_questions_path = source / "questions.jsonl"
    parent_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("status") != "complete":
        raise ValueError("Parent preparation is incomplete")
    if sha(source_corpus_path.read_bytes()) != parent_manifest["corpus_sha256"]:
        raise ValueError("Parent corpus hash mismatch")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Fast tokenizer with offset mapping is required")
    script_hash = sha(Path(__file__).read_bytes())
    config = {"parent_manifest_sha256": sha(source_manifest_path.read_bytes()),
              "parent_corpus_sha256": sha(source_corpus_path.read_bytes()),
              "questions_sha256": sha(source_questions_path.read_bytes()),
              "chunker_sha256": script_hash, "tokenizer_revision": revision,
              "max_title_sep_text_tokens": limit}
    fingerprint = sha(canonical(config))
    config_path = output / "chunk_config.json"
    if output.exists() and any(output.iterdir()):
        if not config_path.exists():
            raise FileExistsError("Output has files without chunk configuration")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != {"fingerprint": fingerprint, **config}:
            raise ValueError("Existing output uses different corpus, tokenizer, or chunker")
        manifest_path = output / "preparation_manifest.json"
        if manifest_path.exists():
            saved = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (saved.get("status") == "complete" and
                    sha((output / "corpus.jsonl").read_bytes()) == saved.get("corpus_sha256") and
                    sha((output / "questions.jsonl").read_bytes()) == config["questions_sha256"]):
                return saved
        raise ValueError("Existing chunk output is incomplete or changed")
    parents = read_jsonl(source_corpus_path)
    children, spans_manifest, mapping = chunk_rows(parents, tokenizer, limit)
    if set(parent_manifest["chunk_sources"]) != set(mapping):
        raise ValueError("Parent source mapping does not match corpus IDs")
    child_sources = {child_id: parent_manifest["chunk_sources"][parent_id]
                     for parent_id, child_ids in mapping.items() for child_id in child_ids}
    forbidden = set(parent_manifest["forbidden_pubmedqa_pmids"])
    if any(row["document_id"].removeprefix("PMID:") in forbidden for row in children):
        raise ValueError("Forbidden PubMedQA target PMID in output")
    corpus_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in children).encode("utf-8")
    manifest = {"schema_version": 1, "status": "complete", "fingerprint": fingerprint,
                "question_ids": parent_manifest["question_ids"],
                "forbidden_pubmedqa_pmids": parent_manifest["forbidden_pubmedqa_pmids"],
                "corpus_chunks": len(children), "corpus_sha256": sha(corpus_bytes),
                "chunk_sources": child_sources, "parent_to_children": mapping,
                "chunk_spans": spans_manifest, "config": config,
                "tokenizer_revision": revision,
                "note": "Physical slices of one source document; derived_from is not synthetic provenance."}
    output.mkdir(parents=True, exist_ok=True)
    def write_atomic(name: str, data: bytes) -> None:
        temp = output / (name + ".tmp")
        temp.write_bytes(data)
        os.replace(temp, output / name)
    write_atomic("chunk_config.json", json.dumps({"fingerprint": fingerprint, **config},
                                                 ensure_ascii=False, indent=2).encode("utf-8"))
    write_atomic("questions.jsonl", source_questions_path.read_bytes())
    write_atomic("corpus.jsonl", corpus_bytes)
    write_atomic("preparation_manifest.json", json.dumps(manifest, ensure_ascii=False,
                                                         indent=2).encode("utf-8"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/pilot_broad_oct09")
    parser.add_argument("--output", type=Path, default=ROOT / "data/pilot_chunks_oct09")
    parser.add_argument("--model-manifest", type=Path, default=ROOT / "indexes/model_download_manifest.json")
    args = parser.parse_args()
    model = json.loads(args.model_manifest.read_text(encoding="utf-8"))["models"]["ncbi/MedCPT-Article-Encoder"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model["snapshot_path"], use_fast=True, local_files_only=True)
    manifest = prepare(args.source, args.output, tokenizer, model["revision"])
    print(json.dumps({"status": manifest["status"], "chunks": manifest["corpus_chunks"],
                      "fingerprint": manifest["fingerprint"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
