"""配对模式只复制已验证的冻结产物，不下载、分词或重排。"""
from dataclasses import asdict
import json
from pathlib import Path
import shutil

from evidence_mdt.data import load_questions
from evidence_mdt.schema import Evidence
from evidence_mdt.retrieval import INITIAL_EVIDENCE_LIMIT
from .materials import prepare_material
from .state import ConfigurationError, digest, fatal, jsonl, save


def reuse_frozen(source, output, questions, state, mode, seed, call_for, checkpoint, *, fail_fast=False):
    parent = json.loads((source / "preparation_manifest.json").read_bytes())
    if parent.get("status") != "complete":
        raise ConfigurationError("Source preparation must be complete")
    if parent.get("config", {}).get("initial_evidence_limit") != INITIAL_EVIDENCE_LIMIT:
        raise ConfigurationError("Source initial evidence limit differs; prepare a new frozen evidence snapshot")
    artifacts = parent.get("artifacts", {})
    required = {"questions.jsonl", "initial_queries.jsonl", "corpus.jsonl", "initial_top10.jsonl", "index_manifest.json"}
    if not required <= artifacts.keys():
        raise ConfigurationError("Source preparation lacks frozen artifacts")
    for name, expected in artifacts.items():
        if Path(name).name != name or not (source / name).is_file() or digest((source / name).read_bytes()) != expected:
            raise ConfigurationError("Source artifact hash mismatch: " + name)
    original = load_questions(source / "questions.jsonl")
    if [asdict(q) for q in original] != [asdict(q) for q in questions]:
        raise ConfigurationError("Source question set, order or content differs")
    if [r["question_id"] for r in parent["records"]] != [q.question_id for q in questions]:
        raise ConfigurationError("Source record order differs")
    names = required | {name for name in ("abstracts.jsonl", "chunk_spans.json", "index_manifest.vectors.pt") if name in artifacts}
    # 可执行查询只在所有材料完成后导出，认证失败或中断也不能留下半成品入口。
    (output / "initial_queries.jsonl").unlink(missing_ok=True)
    for name in names - {"initial_queries.jsonl"}:
        shutil.copyfile(source / name, output / name)
    index = json.loads((output / "index_manifest.json").read_bytes())
    index["corpus"] = str(output / "corpus.jsonl")
    if index.get("backend") == "medcpt":
        if "index_manifest.vectors.pt" not in names:
            raise ConfigurationError("Source lacks dense vectors")
        index["vectors"] = str(output / "index_manifest.vectors.pt")
    save(output / "index_manifest.json", index)
    state["source_preparation"] = {"path": str(source), "fingerprint": parent["fingerprint"],
                                    "frozen_artifacts_sha256": {name: artifacts[name] for name in names}}
    top10 = [json.loads(line) for line in (source / "initial_top10.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if [r["question_id"] for r in top10] != [q.question_id for q in questions]:
        raise ConfigurationError("Source frozen top10 question order differs")
    for question, row, frozen in zip(questions, state["records"], top10):
        try:
            base = [Evidence(**item) for item in frozen["evidence"]]
            row.update(top10=frozen["evidence"], retrieval_trace=frozen.get("trace"))
            def save_spec(spec):
                row["spec"] = spec
                checkpoint()
            row["spec"] = prepare_material(question, base, mode, seed, call_for(question), row.get("spec"), save_spec)
            row["status"] = "ready"
            row.pop("error", None)
        except Exception as exc:
            row.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            if fail_fast or fatal(exc):
                state.update(status="stopped", error=str(exc), failed_question_id=question.question_id)
                checkpoint()
                raise
        checkpoint()
    state["status"] = "complete" if all(r["status"] == "ready" for r in state["records"]) else "incomplete"
    if state["status"] == "complete":
        if mode != "none":
            jsonl(output / "intervention_spec.jsonl", [r["spec"] for r in state["records"]])
        shutil.copyfile(source / "initial_queries.jsonl", output / "initial_queries.jsonl")
    elif state["status"] != "complete":
        # 保留源冻结材料供审计，失败批次不提供可执行query/spec。
        (output / "initial_queries.jsonl").unlink(missing_ok=True)
        (output / "intervention_spec.jsonl").unlink(missing_ok=True)
    state["artifacts"] = {p.name: digest(p.read_bytes()) for p in output.iterdir()
                          if p.is_file() and p.name != "preparation_manifest.json"}
    checkpoint()
    return state
