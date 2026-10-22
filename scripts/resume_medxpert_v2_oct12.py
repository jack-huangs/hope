"""保留首遍失败历史，迁移父清单元数据后进行有限curl恢复。"""
import copy
import json
import shutil
import sys
from pathlib import Path
import prepare_relevance_corpus as base
import prepare_broad_corpus as broad
import prepare_medxpert_oct12 as preparation
import resume_medxpert_transport_oct12 as transport
import finish_medxpert_oct12 as finish
from freeze_initial_queries import freeze

ROOT = preparation.ROOT


def migrate():
    old = ROOT / "data/medxpert_relevance_oct12"
    target = ROOT / "data/medxpert_relevance_oct12_v2"
    source = ROOT / "data/medxpert_queries_oct12"
    parent, public, frozen, parent_hash, frozen_hash = base.validate_parent(source)
    raw = (old / "preparation_manifest.json").read_bytes()
    previous = json.loads(raw)
    if previous["config"]["frozen_queries_sha256"] != frozen_hash:
        raise ValueError("Frozen queries changed")
    if [(q["question_id"], base.sha(base.canonical(q))) for q in public] != [
            (r["question_id"], r["public_question_sha256"]) for r in previous["records"]]:
        raise ValueError("Public question hashes changed")
    if previous["config"]["sort"] != "relevance" or previous["config"]["max_records"] != 8 or previous["config"]["retries"] != 2:
        raise ValueError("Retrieval parameters differ")
    if target.exists():
        saved = json.loads((target / "migration.json").read_bytes())
        if saved["old_manifest_sha256"] != base.sha(raw) or saved["new_parent_manifest_sha256"] != parent_hash:
            raise ValueError("Migration inputs differ")
        return target
    shutil.copytree(old, target)
    current = copy.deepcopy(previous)
    current["config"]["source_manifest_sha256"] = parent_hash
    current["fingerprint"] = base.sha(base.canonical(current["config"]))
    migration = {"reason": "query-only complete timestamp rewritten during two configuration-only recovery attempts",
                 "old_manifest_sha256": base.sha(raw), "old_parent_manifest_sha256": previous["config"]["source_manifest_sha256"],
                 "new_parent_manifest_sha256": parent_hash, "old_fingerprint": previous["fingerprint"],
                 "new_fingerprint": current["fingerprint"], "frozen_queries_sha256": frozen_hash,
                 "query_public_and_retrieval_parameters_unchanged": True, "at": base.now(),
                 "changed_config_fields": ["source_manifest_sha256"],
                 "preserved_files_sha256": {p.name: base.sha(p.read_bytes()) for p in old.iterdir() if p.is_file()}}
    current["migration_provenance"] = migration
    base.save_json(target / "preparation_manifest.json", current)
    base.save_json(target / "migration.json", migration)
    return target


def freeze_short(relevance):
    model = json.loads((ROOT / "indexes/model_download_manifest.json").read_bytes())["models"]["ncbi/MedCPT-Query-Encoder"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model["snapshot_path"], local_files_only=True)
    original_path = relevance / "original_initial_queries.jsonl"
    freeze(relevance, original_path, model["revision"])
    rows = base.read_jsonl(original_path)
    sources = json.loads((relevance / "preparation_manifest.json").read_bytes())["records"]
    revisions = []
    for row, source in zip(rows, sources, strict=True):
        old = row["retrieval_query"]
        old_tokens = len(tokenizer(old)["input_ids"])
        if old_tokens > 64:
            if row["question_id"] != "Text-281" or not set(source["queries"][1].lower().split()) <= set(source["queries"][0].lower().split()):
                raise ValueError("Unexpected overlong query")
            row["retrieval_query"] = source["queries"][0]
            new_tokens = len(tokenizer(row["retrieval_query"])["input_ids"])
            if new_tokens > 64:
                raise ValueError("Revised query still overlong")
            revisions.append({"question_id": row["question_id"], "original_query": old, "revised_query": row["retrieval_query"],
                              "original_tokens": old_tokens, "revised_tokens": new_tokens,
                              "rule": "second query whitespace word set subset of first; retain full first query",
                              "scope": "local initial retrieval only; original PubMed searches unchanged",
                              "not_a_semantic_equivalence_claim": True})
    data = preparation.jsonl_bytes(rows)
    preparation.once(relevance / "initial_queries.jsonl", data)
    preparation.once(relevance / "initial_queries.revision.json", preparation.json_bytes({
        "source_manifest_sha256": base.sha((relevance / "preparation_manifest.json").read_bytes()),
        "original_query_file_sha256": base.sha(original_path.read_bytes()),
        "revised_query_file_sha256": base.sha(data), "tokenizer_revision": model["revision"], "revisions": revisions}))


def main():
    relevance = migrate()
    base.request_once = transport.curl_request
    original = base.request_with_retries
    def annotated(url, prefix, suffix, timeout, retries, backoff, entry, checkpoint):
        before = len(entry["attempts"])
        def save():
            for attempt in entry["attempts"][before:]:
                attempt["transport"] = "curl.exe"
            checkpoint()
        return original(url, prefix, suffix, timeout, retries, backoff, entry, save)
    base.request_with_retries = annotated
    for number in range(1, 3):
        result = base.prepare(ROOT / "data/medxpert_queries_oct12", relevance,
                              reuse_source=ROOT / "data/pilot_relevance_oct09")
        print(f"relevance curl pass{number}: {result['status']}", flush=True)
        if result["status"] == "complete":
            break
    if result["status"] != "complete":
        raise RuntimeError("Two relevance recovery passes exhausted")
    freeze_short(relevance)
    for number in range(1, 3):
        result = broad.prepare(relevance, ROOT / "data/medxpert_broad_oct12_v2", max_model_calls=70,
                    max_tokens=180000, reuse_source=ROOT / "data/pilot_broad_oct09")
        print(f"broad curl pass{number}: {result['status']}", flush=True)
        if result["status"] == "complete":
            break
    if result["status"] != "complete":
        raise RuntimeError("Two broad passes exhausted")
    finish.main()


if __name__ == "__main__":
    main()
