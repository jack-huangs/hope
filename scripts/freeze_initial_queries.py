"""把已冻结的每题两条 PubMed 查询合为短初始检索输入；不读取答案。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"Frozen query artifact differs: {path}")
        return
    temp = path.with_name(path.name + ".tmp")
    temp.write_bytes(content)
    os.replace(temp, path)


def freeze(snapshot: Path, output: Path, tokenizer_revision: str) -> tuple[Path, Path]:
    snapshot = snapshot.resolve()
    parent_path = snapshot / "preparation_manifest.json"
    parent_raw = parent_path.read_bytes()
    parent = json.loads(parent_raw)
    if parent.get("status") != "complete":
        raise ValueError("Source snapshot is not complete")
    public = [json.loads(x) for x in (snapshot / "questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(public) != len(parent["records"]):
        raise ValueError("Question count differs from source records")
    rows, original = [], []
    for question, record in zip(public, parent["records"], strict=True):
        digest = sha(canonical(question))
        queries = record["queries"]
        if (question["question_id"] != record["question_id"] or digest != record["public_question_sha256"] or
                len(queries) != 2 or any(not isinstance(q, str) or not q.strip() for q in queries)):
            raise ValueError("Source queries or public question changed")
        retrieval_query = " ; ".join(queries)
        rows.append({"question_id": record["question_id"], "dataset": record["dataset"],
                     "public_question_sha256": digest, "retrieval_query": retrieval_query})
        original.append({"question_id": record["question_id"], "queries": queries})
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
    sidecar = {"source_manifest_sha256": sha(parent_raw), "original_queries": original,
               "original_queries_sha256": sha(canonical(original)), "query_manifest_sha256": sha(data),
               "separator": " ; ", "script_sha256": sha(Path(__file__).read_bytes()),
               "tokenizer_revision": tokenizer_revision, "question_count": len(rows)}
    sidecar_path = output.with_suffix(".provenance.json")
    write_once(output, data)
    write_once(sidecar_path, json.dumps(sidecar, ensure_ascii=False, indent=2).encode("utf-8"))
    return output, sidecar_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    args = parser.parse_args()
    for path in freeze(args.snapshot, args.output, args.tokenizer_revision):
        print(path)
