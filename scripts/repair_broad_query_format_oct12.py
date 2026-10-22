"""仅对四题已有宽查询做一次语法修复，词数完全保留后下载；不作答。"""
import copy
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path
import prepare_broad_corpus as broad
import prepare_relevance_corpus as base
import prepare_medxpert_oct12 as preparation
import resume_medxpert_transport_oct12 as transport
import finish_medxpert_oct12 as finish
from evidence_mdt.env import load_env
from evidence_mdt.models import create_model

ROOT = preparation.ROOT
PROTOCOL = "stem-only-broad-plus-word-preserving-format-repair-v1"
EXPECTED = {"Text-592", "Text-1740", "Text-1039", "Text-5"}


def words(text):
    return Counter(token.lower() for token in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", text)
                   if token.upper() != "AND")


def validate_repair(original, repaired):
    repaired = broad.validate_queries(repaired)
    if len(original) != 2 or any(words(before) != words(after) for before, after in zip(original, repaired, strict=True)):
        raise ValueError("Repair added, removed, or replaced original query words")
    return repaired


def main():
    load_env(ROOT / ".env")
    model = create_model("deepseek")
    old = ROOT / "data/medxpert_broad_oct12_v2"
    target = ROOT / "data/medxpert_broad_oct12_v3"
    old_raw = (old / "preparation_manifest.json").read_bytes()
    parent = json.loads(old_raw)
    script_hash = base.sha(Path(__file__).read_bytes())
    path = target / "preparation_manifest.json"
    if not target.exists():
        if {r["question_id"] for r in parent["records"] if r["status"] == "query_failed"} != EXPECTED:
            raise ValueError("Unexpected repair set")
        shutil.copytree(old, target)
        manifest = copy.deepcopy(parent)
        manifest["config"]["script_sha256"] = base.sha(Path(broad.__file__).read_bytes())
        manifest["config"]["query_protocol"] = PROTOCOL
        manifest["config"]["format_repair_sha256"] = script_hash
        manifest["fingerprint"] = base.sha(base.canonical(manifest["config"]))
        manifest["repair_migration"] = {"source": str(old), "source_manifest_sha256": base.sha(old_raw),
                                        "protocol": PROTOCOL, "repair_ids": sorted(EXPECTED), "at": base.now()}
        base.save_json(path, manifest)
    else:
        manifest = json.loads(path.read_bytes())
        if manifest["repair_migration"]["source_manifest_sha256"] != base.sha(old_raw):
            raise ValueError("Repair source changed")
    for row in manifest["records"]:
        if row["question_id"] not in EXPECTED or row["status"] in {"ready", "queried"}:
            continue
        if len(row["model_calls"]) != 2:
            raise RuntimeError("One format repair already attempted; no repeated repair")
        previous = row["model_calls"][-1]["raw_response"]
        original = json.loads(previous["choices"][0]["message"]["content"])["queries"]
        payload = {"task": "queries", "protocol": "syntax-only-format-repair-v1", "question_id": row["question_id"],
                   "input_queries": original,
                   "constraints": "Format each of the two supplied queries into 2-3 phrases joined with uppercase AND. "
                       "Return exactly two strings in the same order. Keep EVERY original word exactly once for each occurrence; "
                       "do not add, remove, replace, or reorder words. You may only add AND, quotes and parentheses. "
                       "Do not add synonyms, diagnoses, treatments, answers, OR, NOT, or explanations."}
        started, stamp = time.monotonic(), base.now()
        try:
            response = model.complete(payload)
            row["model_calls"].append({"call_kind": "query_format_repair", "request": payload,
                "wire_request": response.wire_request, "raw_response": response.raw, "usage": response.usage,
                "started_at": stamp, "elapsed_seconds": time.monotonic() - started,
                "model_revision": response.raw.get("model", model.name)})
            row["queries"] = validate_repair(original, response.content.get("queries"))
            row["snapshots"] = [{"query": term, "status": "fetch_pending", "attempts": []} for term in row["queries"]]
            row["status"] = "queried"
            row.pop("error", None)
        except Exception as exc:
            if len(row["model_calls"]) == 2:
                row["model_calls"].append({"call_kind": "query_format_repair", "request": payload,
                    "usage": getattr(exc, "usage", {}) or {}, "raw_response": getattr(exc, "raw", None),
                    "wire_request": getattr(exc, "wire_request", None), "error_type": type(exc).__name__})
            row["status"] = "format_repair_failed"
            row["error"] = type(exc).__name__ + ": " + str(exc)[:200]
            base.save_json(path, manifest)
            raise
        base.save_json(path, manifest)
        print("word-preserving repair passed: " + row["question_id"], flush=True)
    original_request = base.request_with_retries
    base.request_once = transport.curl_request
    def annotated(url, prefix, suffix, timeout, retries, backoff, entry, checkpoint):
        before = len(entry["attempts"])
        def save():
            for attempt in entry["attempts"][before:]:
                attempt["transport"] = "curl.exe"
            checkpoint()
        return original_request(url, prefix, suffix, timeout, retries, backoff, entry, save)
    base.request_with_retries = annotated
    result = broad.prepare(ROOT / "data/medxpert_relevance_oct12_v2", target, max_model_calls=70,
        max_tokens=180000, reuse_source=ROOT / "data/pilot_broad_oct09", query_protocol=PROTOCOL,
        format_repair_sha256=script_hash)
    if result["status"] != "complete":
        raise RuntimeError("Repaired query snapshot retrieval is incomplete")
    print("broad v3 complete", flush=True)
    finish.main()


if __name__ == "__main__":
    main()
