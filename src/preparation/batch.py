"""任意题量的共用检索 → B/C/D → 离线验收，不生成批次专用代码。"""
from dataclasses import asdict
import json
from pathlib import Path

from .audit import verify_preparation
from .state import ConfigurationError, digest, jsonl, save


def prepare_batch(questions_path: Path, output: Path, model, dense=None, *,
                  modes=("b", "c", "d"), source_preparation=None, **kwargs):
    from . import load_questions, prepare
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or any(m not in {"b", "c", "d"} for m in modes):
        raise ConfigurationError("Batch modes must be unique choices from b, c, d")
    questions = load_questions(questions_path)
    if any(m in modes for m in ("c", "d")) and any(q.gold not in q.options for q in questions):
        raise ConfigurationError("C/D preparation requires gold for every input question")
    output = output.resolve()
    source = Path(source_preparation).resolve() if source_preparation else output / "shared"
    if source_preparation and (source == output or output in source.parents):
        raise ConfigurationError("External source must be outside the batch output")
    code = {p.name: digest(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}
    config = {"questions": digest([asdict(q) for q in questions]), "modes": list(modes),
              "source": {"path": str(source),
                         "manifest_sha256": digest((source / "preparation_manifest.json").read_bytes())}
                        if source_preparation else None,
              "code": code, "seed": kwargs.get("seed", 20261008),
              "max_records": kwargs.get("max_records", 8),
              "model": {k: getattr(model, k, None) for k in
                        ("name", "base_url", "max_tokens", "disable_thinking", "answer_temperature")}}
    if kwargs.get("recover_minor"):
        config["recover_minor"] = True
    path = output / "batch_manifest.json"
    if path.exists():
        if json.loads(path.read_bytes())["fingerprint"] != digest(config):
            raise ConfigurationError("Batch configuration changed; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        # sample 命令可先在同一根目录写入题目和抽样清单。
        if set(p.name for p in output.iterdir()) - {"questions.jsonl", "selection_manifest.json"}:
            raise ConfigurationError("Batch output is nonempty without a batch manifest")
        if (output / "questions.jsonl").exists() and load_questions(output / "questions.jsonl") != questions:
            raise ConfigurationError("Batch output contains different questions")
    summary = {"fingerprint": digest(config), "config": config, "status": "running",
               "questions": len(questions), "stages": {}, "answer_teams_started": False}
    def checkpoint():
        save(path, summary)
    def record(mode, state):
        summary["stages"][mode] = {"status": state["status"],
            "ready": sum(r["status"] == "ready" for r in state["records"]),
            "failures": [{"question_id": r["question_id"], "error": r.get("error")}
                         for r in state["records"] if r["status"] != "ready"]}
        checkpoint()
    checkpoint()
    jsonl(output / "questions.jsonl", [asdict(q) for q in questions])
    try:
        if not source_preparation:
            summary["active_stage"] = "shared"
            checkpoint()
            shared = prepare(questions_path, source, model, dense, interventions="none", **kwargs)
            record("shared", shared)
            if shared["status"] != "complete":
                summary["status"] = "incomplete"
                return summary
        summary["shared_verification"] = verify_preparation(source)
        for mode in modes:
            summary["active_stage"] = mode
            checkpoint()
            state = prepare(questions_path, output / mode, model, None, interventions=mode,
                            source_preparation=source, **kwargs)
            record(mode, state)
            if state["status"] == "complete":
                summary["stages"][mode]["verification"] = verify_preparation(output / mode)
                for name in ("corpus.jsonl", "initial_queries.jsonl", "initial_top10.jsonl"):
                    if (source / name).read_bytes() != (output / mode / name).read_bytes():
                        raise ConfigurationError("Shared artifacts differ: " + mode + "/" + name)
        summary["status"] = "complete" if all(
            s["status"] == "complete" for s in summary["stages"].values()) else "incomplete"
    except Exception as exc:
        summary.update(status="stopped", error_type=type(exc).__name__, error=str(exc),
                       failed_stage=summary.get("active_stage"))
        raise
    finally:
        summary.pop("active_stage", None)
        # 统计本目录实际保存的响应；复用源的历史调用不重复计费。
        summary["saved_model_responses"] = {}
        for stage in ("shared", *modes):
            calls = output / stage / "cache" / "model_calls.jsonl"
            if calls.exists():
                rows = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines() if line.strip()]
                summary["saved_model_responses"][stage] = {
                    "responses": len(rows),
                    "known_tokens": sum(int(r.get("usage", {}).get("total_tokens") or 0) for r in rows)}
        checkpoint()
        lines = ["# 材料准备报告", "", f"状态：{summary['status']}；题数：{len(questions)}。", "",
                 "| 阶段 | 状态 | 完成题数 |", "|---|---|---|"]
        lines += [f"| {mode} | {s['status']} | {s['ready']} |" for mode, s in summary["stages"].items()]
        lines += ["", "本流程只准备材料，未启动专家答题。验收检查文件完整性和输入结构，不核实医学真假。"]
        (output / "preparation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
