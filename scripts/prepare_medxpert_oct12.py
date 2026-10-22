"""准备30道MedXpertQA与原20道PubMedQA及检索快照；从不运行答题。"""
from __future__ import annotations
import copy
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evidence_mdt.data import load_questions, normalize_question
from evidence_mdt.env import load_env
from evidence_mdt.models import create_model
import prepare_question_corpus as query
import prepare_relevance_corpus as base
import prepare_broad_corpus as broad
from prepare_intervention_selection import branch_record
from freeze_initial_queries import freeze

SEED = 20261012


def once(path: Path, data: bytes):
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f"Frozen artifact differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(data)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def jsonl_bytes(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def select(output: Path):
    pilot = ROOT / "data/pilot"
    raw = (pilot / "MedXpertQA.raw").read_bytes()
    original = (pilot / "MedXpertQA.jsonl").read_bytes()
    old = load_questions(pilot / "MedXpertQA.jsonl")
    # 抽样步骤只读ID，不检查选项、金标或模型表现。
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    expected = json.loads((pilot / "selection_manifest.json").read_bytes())["sources"]["MedXpertQA"]
    if base.sha(raw) != expected["raw_sha256"]:
        raise ValueError("Official raw data differs from original selection manifest")
    ids = [str(r["id"]) for r in records]
    if len(ids) != len(set(ids)) or len(old) != 20:
        raise ValueError("Invalid source ID pool or old selection")
    old_ids = [q.question_id for q in old]
    if not set(old_ids) <= set(ids):
        raise ValueError("Original selected IDs are absent from the official source")
    pool = [qid for qid in ids if qid not in set(old_ids)]
    extra = random.Random(SEED).sample(pool, 10)
    lookup = {str(r["id"]): r for r in records}
    normalized = [asdict(normalize_question(lookup[qid], "MedXpertQA", 0)) for qid in extra]
    mx = original + (b"" if original.endswith(b"\n") else b"\n") + jsonl_bytes(normalized)
    pqa = (pilot / "PubMedQA.jsonl").read_bytes()
    once(output / "MedXpertQA.jsonl", mx)
    once(output / "PubMedQA.jsonl", pqa)
    once(output / "pilot50.jsonl", mx + pqa)
    manifest = {"seed": SEED, "algorithm": "retain old20; random.Random(seed).sample(source-order remaining IDs,10)",
                "source_url": "https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA/resolve/main/Text/test.jsonl",
                "source_sha256": base.sha(raw), "source_n": len(ids), "remaining_pool_n": len(pool),
                "old_selection_sha256": base.sha(original), "retained_ids": old_ids, "added_ids": extra,
                "pubmedqa_sha256": base.sha(pqa), "pubmedqa_preserved_byte_for_byte": True,
                "clinical_n": 30, "pubmedqa_n": 20, "pilot50_sha256": base.sha(mx + pqa)}
    once(output / "selection_manifest.json", json_bytes(manifest))
    intervention = branch_record(output, "medxpertqa_oct12", "MedXpertQA.jsonl")
    intervention["branch_selected"] = True
    intervention["note"] = "仅冻结15MX+10PQA题号；未制作干预材料或运行B/C。PQA独立固定流沿用原seed。"
    once(output / "intervention_selection.json", json_bytes(intervention))
    return load_questions(output / "pilot50.jsonl")


def prepare_queries(questions, output):
    load_env(ROOT / ".env")
    model = create_model("deepseek")
    old_source = ROOT / "data/pilot_corpus_oct09"
    old_raw = (old_source / "preparation_manifest.json").read_bytes()
    old = json.loads(old_raw)
    old_rows = {r["question_id"]: r for r in old["records"]}
    config = {"script_sha256": base.sha(Path(__file__).read_bytes()), "query_script_sha256": base.sha(Path(query.__file__).read_bytes()),
              "questions_sha256": base.sha(base.canonical([q.public() for q in questions])),
              "model": query.public_model_config(model, "deepseek"), "old_manifest_sha256": base.sha(old_raw),
              "max_calls": 100, "max_tokens": 200000, "query_protocol": "stem-only-strip-terminal-answer-choices-v1"}
    path = output / "preparation_manifest.json"
    fingerprint = base.sha(base.canonical(config))
    output.mkdir(parents=True, exist_ok=True)
    if path.exists():
        manifest = json.loads(path.read_bytes())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Query preparation configuration differs")
        if manifest["status"] == "complete":
            # 已完成清单不可再次写completed_at，否则下游父哈希失效。
            return manifest
    else:
        rows = []
        for q in questions:
            row = {"question_id": q.question_id, "dataset": q.dataset,
                   "public_question_sha256": base.sha(base.canonical(q.public())),
                   "status": "pending", "queries": [], "model_calls": [], "snapshots": []}
            if q.dataset.lower().startswith("pubmedqa"):
                prior = old_rows[q.question_id]
                if row["public_question_sha256"] != prior["public_question_sha256"]:
                    raise ValueError("PQA public input differs")
                row.update(copy.deepcopy(prior))
                row["query_origin"] = {"source": str(old_source), "manifest_sha256": base.sha(old_raw),
                                       "model_config": old["config"]["model"], "historical": True}
            rows.append(row)
        manifest = {"status": "running", "fingerprint": fingerprint, "config": config,
                    "created_at": base.now(), "question_ids": [q.question_id for q in questions],
                    "forbidden_pubmedqa_pmids": [q.question_id for q in questions if q.dataset.lower().startswith("pubmedqa")],
                    "records": rows}
        once(output / "questions.jsonl", jsonl_bytes([q.public() for q in questions]))
        base.save_json(path, manifest)
    for q, row in zip(questions, manifest["records"], strict=True):
        if row["status"] == "ready":
            continue
        # 本作业不无限重试查询失败；重新启动前须核实记录。
        if row["model_calls"]:
            raise RuntimeError(f"Query failed and requires review: {q.question_id}")
        calls = [c for r in manifest["records"] if not r.get("query_origin") for c in r["model_calls"]]
        if len(calls) >= 100 or sum(query.usage_tokens(c.get("usage")) for c in calls) >= 200000:
            raise RuntimeError("Query budget reached")
        payload = query.query_payload(q)
        started = time.monotonic()
        stamp = base.now()
        try:
            response = model.complete(payload)
            row["model_calls"].append({"request": payload, "wire_request": response.wire_request,
                "raw_response": response.raw, "usage": response.usage, "started_at": stamp,
                "elapsed_seconds": time.monotonic() - started,
                "model_revision": response.raw.get("model", model.name)})
            row["queries"] = query.valid_queries(response.content)
            if len(row["queries"]) != 2:
                raise ValueError("Exactly two initial queries required")
            # 此目录只冻结查询，实际网络下载全部直接使用relevance。
            row["snapshots"] = [{"query": term, "status": "query_only"} for term in row["queries"]]
            row["status"] = "ready"
        except Exception as exc:
            if not row["model_calls"]:
                row["model_calls"].append({"request": payload, "usage": getattr(exc, "usage", {}) or {},
                    "raw_response": getattr(exc, "raw", None), "wire_request": getattr(exc, "wire_request", None),
                    "error_type": type(exc).__name__})
            row["status"] = "query_failed"
            base.save_json(path, manifest)
            raise
        base.save_json(path, manifest)
        print(f"query ready: {q.question_id}", flush=True)
    calls, tokens = query.totals(manifest)
    historical = [c for r in manifest["records"] if r.get("query_origin") for c in r["model_calls"]]
    manifest.update({"status": "complete", "model_calls": calls, "known_tokens": tokens,
                    "actual_new_model_calls": calls - len(historical), "reused_model_calls": len(historical),
                    "reused_known_tokens": sum(query.usage_tokens(c.get("usage")) for c in historical),
                    "completed_at": base.now()})
    base.save_json(path, manifest)
    return manifest


def main():
    dataset = ROOT / "data/medxpert_pilot_oct12"
    questions = select(dataset)
    source = ROOT / "data/medxpert_queries_oct12"
    prepare_queries(questions, source)
    relevance = ROOT / "data/medxpert_relevance_oct12"
    result = base.prepare(source, relevance, reuse_source=ROOT / "data/pilot_relevance_oct09")
    if result["status"] != "complete":
        raise RuntimeError("Relevance snapshot incomplete")
    revision = json.loads((ROOT / "indexes/model_download_manifest.json").read_bytes())["models"]["ncbi/MedCPT-Query-Encoder"]["revision"]
    freeze(relevance, relevance / "initial_queries.jsonl", revision)
    final = broad.prepare(relevance, ROOT / "data/medxpert_broad_oct12", max_model_calls=70,
                          max_tokens=180000, reuse_source=ROOT / "data/pilot_broad_oct09")
    print(json.dumps({"status": final["status"], "abstracts": final["corpus_chunks"],
                      "actual_new_broad_calls": final["actual_new_model_calls"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
