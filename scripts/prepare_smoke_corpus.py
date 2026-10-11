"""为五题真实烟测准备可回放的 PubMed 小语料。运行会调用 DeepSeek 与 PubMed。"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_mdt.data import load_questions, pubmed_snapshot, write_jsonl
from evidence_mdt.env import load_env
from evidence_mdt.models import create_model
from evidence_mdt.schema import Budget


INPUT = ROOT / "data" / "pilot"
OUTPUT = ROOT / "data" / "smoke"


def save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    # 固定题号与顺序；任何模型结果都不参与选题。
    questions = load_questions(INPUT / "MedQA.jsonl")[:3] + load_questions(INPUT / "PubMedQA.jsonl")[:2]
    if len(questions) != 5:
        raise ValueError("Pilot selection must contain 3 MedQA and 2 PubMedQA questions")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "questions.jsonl", questions)
    load_env(ROOT / ".env")
    model = create_model("deepseek")
    budget = Budget(max_calls=10, max_tokens=20000)
    forbidden_pmids = {q.question_id for q in questions if q.dataset.lower().startswith("pubmedqa")}
    all_evidence = {}
    records = []
    for q in questions:
        # 只给模型公开题面；Question.public() 不含 gold、LONG_ANSWER 或隐藏标签。
        payload = {"task": "queries", "question": q.public(),
                   "purpose": "Initial neutral PubMed abstract search before any MDT answer",
                   "constraints": ("Return at most two short medical entity queries for PubMed. "
                                   "Do not answer the question, select an option, or include a conclusion. "
                                   "Use only the visible question and background; no gold label is available.")}
        row = {"question_id": q.question_id, "dataset": q.dataset,
               "public_question_sha256": hashlib.sha256(
                   json.dumps(q.public(), sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
               "status": "pending", "queries": [], "snapshots": []}
        records.append(row)
        attempted = False
        charged = False
        try:
            budget.before_call()
            attempted = True
            response = model.complete(payload)
            budget.after_call(response.usage)
            charged = True
            row["model_call"] = {"request": payload, "wire_request": response.wire_request,
                                 "raw_response": response.raw, "usage": response.usage,
                                 "model_revision": response.raw.get("model", model.name)}
            if budget.tokens > budget.max_tokens:
                raise RuntimeError("Known token budget exceeded")
            queries = response.content.get("queries")
            if (not isinstance(queries, list) or len(queries) > 2 or
                    any(not isinstance(query, str) or not query.strip() or len(query) > 120
                        for query in queries)):
                raise ValueError("Model must return at most two short nonempty queries")
            row["queries"] = queries
            row["status"] = "queried" if queries else "empty_queries"
        except Exception as exc:
            if attempted and not charged:
                usage = getattr(exc, "usage", {})
                budget.after_call(usage)
                row["model_call"] = {"request": payload, "wire_request": getattr(exc, "wire_request", None),
                                     "raw_response": getattr(exc, "raw", None), "usage": usage,
                                     "error": repr(exc)}
            row["status"] = "query_failed"
            row["error"] = repr(exc)
            # 模型失败也记录，后续题仍按预定题号尝试；不改选题。
            save_json(OUTPUT / "preparation_manifest.json", {"records": records,
                      "model_calls": budget.calls, "known_tokens": budget.tokens})
            continue
        for query_index, query in enumerate(row["queries"]):
            prefix = f"{q.question_id}.query{query_index + 1}"
            try:
                evidence, snapshot = pubmed_snapshot(query, max_records=8)
                raw_xml = snapshot.pop("_raw_xml", None)
                if raw_xml is not None:
                    (OUTPUT / f"{prefix}.raw.xml").write_text(raw_xml, encoding="utf-8")
                save_json(OUTPUT / f"{prefix}.snapshot.json", snapshot)
                kept = []
                rejected = []
                for item in evidence:
                    # PubMedQA 的目标研究 PMID 在所有五题的语料中一律排除。
                    if item.document_id.removeprefix("PMID:") in forbidden_pmids:
                        rejected.append(item.chunk_id)
                    else:
                        all_evidence[item.chunk_id] = item
                        kept.append(item.chunk_id)
                row["snapshots"].append({"query": query, "status": "ok" if evidence else "empty",
                                         "snapshot_file": f"{prefix}.snapshot.json",
                                         "raw_xml_file": f"{prefix}.raw.xml" if raw_xml else None,
                                         "raw_xml_sha256": snapshot.get("raw_xml_sha256"),
                                         "kept_chunk_ids": kept, "excluded_target_pmids": rejected})
            except Exception as exc:
                row["snapshots"].append({"query": query, "status": "fetch_failed", "error": repr(exc)})
            # NCBI 公共接口需要节制请求速度。
            time.sleep(1)
        save_json(OUTPUT / "preparation_manifest.json", {"records": records,
                  "model_calls": budget.calls, "known_tokens": budget.tokens})
    corpus = list(all_evidence.values())
    write_jsonl(OUTPUT / "corpus.jsonl", corpus)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(),
                "purpose": "Five-question software/cost smoke only; not a 50-question comparison",
                "question_ids": [q.question_id for q in questions],
                "question_selection": "first 3 fixed MedQA pilot + first 2 fixed PubMedQA pilot",
                "forbidden_pubmedqa_pmids": sorted(forbidden_pmids),
                "model_calls": budget.calls, "known_tokens": budget.tokens,
                "corpus_chunks": len(corpus),
                "corpus_sha256": hashlib.sha256((OUTPUT / "corpus.jsonl").read_bytes()).hexdigest(),
                "records": records}
    save_json(OUTPUT / "preparation_manifest.json", manifest)
    print(OUTPUT / "preparation_manifest.json")


if __name__ == "__main__":
    main()
