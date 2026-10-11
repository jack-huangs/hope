"""以公开题面检查固定语料的初始 BM25/MedCPT 检索；不读取答案、不调用 LLM。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_mdt.retrieval import MedCPTDense, MockDense, Retriever, load_corpus, tokens


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: Path, obj: object) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    os.replace(temp, path)


def validate_inputs(snapshot: Path, backend: str, index_manifest: Path | None) -> tuple[list[dict], list, dict, object, dict]:
    manifest = json.loads((snapshot / "preparation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Corpus preparation must be complete")
    questions_path, corpus_path = snapshot / "questions.jsonl", snapshot / "corpus.jsonl"
    public = read_jsonl(questions_path)
    if len(public) != len(manifest["question_ids"]) or [q["question_id"] for q in public] != manifest["question_ids"]:
        raise ValueError("Public question order differs from preparation manifest")
    forbidden_keys = {"gold", "long_answer", "final_decision", "answer_idx", "label"}
    if any(forbidden_keys & {k.lower() for k in q} for q in public):
        raise ValueError("Question file contains answer fields")
    corpus = load_corpus(corpus_path)
    corpus_hash = sha(corpus_path.read_bytes())
    if corpus_hash != manifest["corpus_sha256"]:
        raise ValueError("Corpus hash differs from preparation manifest")
    if any(e.document_id.removeprefix("PMID:") in set(manifest["forbidden_pubmedqa_pmids"]) for e in corpus):
        raise ValueError("PubMedQA target PMID leaked into corpus")
    index_data = {}
    if backend == "medcpt":
        if index_manifest is None:
            raise ValueError("MedCPT audit requires --index-manifest")
        index_data = json.loads(index_manifest.read_text(encoding="utf-8"))
        if index_data.get("sha256") != corpus_hash or index_data.get("backend") != "medcpt":
            raise ValueError("MedCPT index does not match corpus")
        vector = Path(index_data["vectors"])
        if sha(vector.read_bytes()) != index_data["vectors_sha256"]:
            raise ValueError("MedCPT vector hash mismatch")
        model_manifest = ROOT / "indexes/model_download_manifest.json"
        dense = MedCPTDense(model_manifest if model_manifest.exists() else None)
        dense.load_vectors(vector)
    elif backend == "mock":
        dense = MockDense()
    else:
        raise ValueError("backend must be medcpt or mock")
    return public, corpus, manifest, dense, index_data


def audit(snapshot: Path, output: Path, backend: str = "medcpt",
          index_manifest: Path | None = None, query_manifest: Path | None = None) -> dict:
    snapshot, output = snapshot.resolve(), output.resolve()
    public, corpus, manifest, dense, index_data = validate_inputs(snapshot, backend, index_manifest)
    frozen_query_rows = read_jsonl(query_manifest) if query_manifest else []
    if query_manifest:
        if len(frozen_query_rows) != len(public):
            raise ValueError("Query manifest count differs from public questions")
        for question, frozen in zip(public, frozen_query_rows, strict=True):
            digest = sha(json.dumps(question, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode())
            if (frozen.get("question_id") != question["question_id"] or
                    frozen.get("dataset") != question["dataset"] or
                    frozen.get("public_question_sha256") != digest or
                    not isinstance(frozen.get("retrieval_query"), str) or
                    not frozen["retrieval_query"].strip()):
                raise ValueError("Query manifest differs from public question")
    config = {"questions_sha256": sha((snapshot / "questions.jsonl").read_bytes()),
              "corpus_sha256": manifest["corpus_sha256"],
              "preparation_manifest_sha256": sha((snapshot / "preparation_manifest.json").read_bytes()),
              "retrieval_code_sha256": sha((ROOT / "src/evidence_mdt/retrieval.py").read_bytes()),
              "audit_code_sha256": sha(Path(__file__).read_bytes()), "backend": backend,
              "index_manifest_sha256": sha(index_manifest.read_bytes()) if index_manifest else None,
              "query_manifest_sha256": sha(query_manifest.read_bytes()) if query_manifest else None,
              "model_revisions": index_data.get("model_revisions"),
              "settings": {"bm25_top": 50, "dense_top": 50, "rrf_k": 60, "rerank_top": 30, "return_top": 6}}
    fingerprint = sha(json.dumps(config, sort_keys=True, ensure_ascii=False).encode())
    config_path = output / "config.json"
    if output.exists() and not config_path.exists() and any(output.iterdir()):
        raise FileExistsError("Output contains files without audit configuration")
    output.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("Existing retrieval audit uses different configuration")
        if (output / "summary.json").exists():
            return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    else:
        save_json(config_path, {"fingerprint": fingerprint, **config})
    retriever = Retriever(corpus, dense)
    by_id = {e.chunk_id: e for e in corpus}
    provenance = manifest["chunk_sources"]
    records = []
    for position, question in enumerate(public):
        qid = question["question_id"]
        query = (frozen_query_rows[position]["retrieval_query"] if query_manifest else
                 question["stem"] + " " + question.get("context", ""))
        query_encoder_tokens = (len(dense.qtok(query, add_special_tokens=True)["input_ids"])
                                if backend == "medcpt" else len(tokens(query)))
        reranker_query_tokens = (len(dense.rtok(query, add_special_tokens=False)["input_ids"])
                                 if backend == "medcpt" else len(tokens(query)))
        # 与运行器一致：额外防御性排除目标文献和逐题上下文原文。
        exclude = set()
        if question["dataset"].lower().startswith("pubmedqa"):
            exclude = {e.chunk_id for e in corpus if e.document_id in {qid, f"PMID:{qid}"} or
                       (question.get("context") and e.text.strip() in question["context"])}
        selected = retriever.search(query, limit=6, exclude=exclude)
        trace = retriever.last_trace
        score_maps = {key: {x["chunk_id"]: x["score"] for x in trace.get(key, [])}
                      for key in ("bm25_top50", "dense_top50", "rrf_top30", "reranked")}
        stem_words = set(tokens(question["stem"]))
        top6 = []
        for rank, item in enumerate(selected, 1):
            source_queries = provenance.get(item.chunk_id, [])
            top6.append({"rank": rank, "evidence": asdict(item),
                         "scores": {key: values.get(item.chunk_id) for key, values in score_maps.items()},
                         "source_queries": source_queries,
                         "from_this_question": any(x["question_id"] == qid for x in source_queries),
                         "stem_token_overlap": len(stem_words & set(tokens(item.title + " " + item.text)))})
        records.append({"question_id": qid, "dataset": question["dataset"],
                        "public_question_sha256": sha(json.dumps(question, sort_keys=True, ensure_ascii=False,
                                                                  separators=(",", ":")).encode()),
                        "retrieval_query": query, "excluded_chunk_ids": sorted(exclude),
                        "query_lengths": {"characters": len(query),
                                          "query_encoder_untruncated_tokens": query_encoder_tokens,
                                          "query_encoder_limit": 64,
                                          "reranker_query_untruncated_tokens": reranker_query_tokens},
                        "trace": trace, "top6": top6,
                        "citation_coverage": {"returned": len(top6),
                            "with_url": sum(bool(x["evidence"]["source"]) for x in top6),
                            "with_content_hash": sum(bool(x["evidence"]["content_hash"]) for x in top6),
                            "from_this_question": sum(x["from_this_question"] for x in top6)},
                        "review_flags": {"empty_top6": not top6,
                            "no_same_question_source": not any(x["from_this_question"] for x in top6),
                            "zero_stem_token_overlap": bool(top6) and all(x["stem_token_overlap"] == 0 for x in top6)}})
    write_jsonl(output / "retrieval_trace.jsonl", records)
    summary = {"backend": backend, "fingerprint": fingerprint, "questions": len(records),
               "corpus_chunks": len(by_id),
               "top6_count_distribution": {str(i): sum(len(r["top6"]) == i for r in records) for i in range(7)},
               "empty_top6_ids": [r["question_id"] for r in records if r["review_flags"]["empty_top6"]],
               "no_same_question_source_ids": [r["question_id"] for r in records
                                                if r["review_flags"]["no_same_question_source"]],
               "zero_stem_token_overlap_ids": [r["question_id"] for r in records
                                               if r["review_flags"]["zero_stem_token_overlap"]],
               "query_encoder_over_64_ids": [r["question_id"] for r in records
                                             if r["query_lengths"]["query_encoder_untruncated_tokens"] > 64],
               "max_query_encoder_tokens": max(r["query_lengths"]["query_encoder_untruncated_tokens"]
                                               for r in records),
               "max_reranker_query_tokens": max(r["query_lengths"]["reranker_query_untruncated_tokens"]
                                                for r in records),
               "citations_with_url": sum(r["citation_coverage"]["with_url"] for r in records),
               "citations_with_hash": sum(r["citation_coverage"]["with_content_hash"] for r in records),
               "total_citations": sum(len(r["top6"]) for r in records),
               "note": "Flags are gold-blind lexical/provenance checks, not medical relevance judgments."}
    save_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=["medcpt", "mock"], default="medcpt")
    parser.add_argument("--index-manifest", type=Path)
    parser.add_argument("--query-manifest", type=Path, help="Frozen public-question short retrieval queries")
    args = parser.parse_args()
    print(json.dumps(audit(args.snapshot, args.output, args.backend, args.index_manifest,
                           args.query_manifest), ensure_ascii=False, indent=2))
