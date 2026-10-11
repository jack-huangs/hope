"""只读汇总烟测日志；输出不包含请求、原始响应或密钥。"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from evidence_mdt.metrics import summarize


GROUP_NAMES = {
    "ordinary": "普通 MDT",
    "semantic": "语义合并",
    "semantic_retrieval": "语义合并＋补检索",
    "shared_recruitment": "共享角色招募",
}


def read_jsonl(path: Path, warnings: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # 运行中的进程可能尚未写完最后一行；绝不在报告中回显原文。
            warnings.append(f"{path.name} 第 {index} 行尚不能解析，已略过")
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def numeric(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def call_group(call: dict[str, Any], assignments: dict[str, dict[str, Any]]) -> str:
    run_id = call.get("run_id", "")
    if run_id in assignments:
        return assignments[run_id]["group"]
    if isinstance(run_id, str) and run_id.endswith(":recruit"):
        return "shared_recruitment"
    return "unassigned"


def call_stats(calls: list[dict[str, Any]], assignments: dict[str, dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(lambda: {
        "calls": 0, "failed_calls": 0, "known_total_tokens": 0,
        "known_cache_hit_input_tokens": 0, "known_cache_miss_input_tokens": 0,
        "known_output_tokens": 0, "calls_without_total_usage": 0,
        "calls_without_cache_breakdown": 0, "calls_without_output_usage": 0,
        "latencies_seconds": [], "calls_without_latency": 0,
    })
    for call in calls:
        request = call.get("request") or {}
        task = request.get("task", "unknown") if isinstance(request, dict) else "unknown"
        bucket = buckets[(call_group(call, assignments), str(task))]
        bucket["calls"] += 1
        bucket["failed_calls"] += int(call.get("status") == "failed" or "error" in call)
        usage = call.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        total = numeric(usage.get("total_tokens"))
        if total is None:
            bucket["calls_without_total_usage"] += 1
        else:
            bucket["known_total_tokens"] += int(total)
        hit = numeric(usage.get("prompt_cache_hit_tokens"))
        miss = numeric(usage.get("prompt_cache_miss_tokens"))
        if hit is None or miss is None:
            bucket["calls_without_cache_breakdown"] += 1
        else:
            bucket["known_cache_hit_input_tokens"] += int(hit)
            bucket["known_cache_miss_input_tokens"] += int(miss)
        output = numeric(usage.get("completion_tokens"))
        if output is None:
            bucket["calls_without_output_usage"] += 1
        else:
            bucket["known_output_tokens"] += int(output)
        elapsed = numeric(call.get("elapsed_seconds"))
        if elapsed is None:
            bucket["calls_without_latency"] += 1
        else:
            bucket["latencies_seconds"].append(elapsed)
    out = {}
    for (group, task), item in sorted(buckets.items()):
        latencies = item.pop("latencies_seconds")
        item["known_latency_seconds_sum"] = round(sum(latencies), 3)
        item["known_latency_seconds_mean"] = round(mean(latencies), 3) if latencies else None
        item["latency_n"] = len(latencies)
        out[f"{group}/{task}"] = item
    return out


def make_report(run_dir: Path, preparation_manifest: Path | None = None) -> dict[str, Any]:
    warnings: list[str] = []
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    assignments = read_jsonl(run_dir / "assignments.jsonl", warnings)
    results = read_jsonl(run_dir / "results.jsonl", warnings)
    failures = read_jsonl(run_dir / "failures.jsonl", warnings)
    events = read_jsonl(run_dir / "events.jsonl", warnings)
    calls = read_jsonl(run_dir / "calls.jsonl", warnings)
    assigned_by_id = {a["run_id"]: a for a in assignments if isinstance(a.get("run_id"), str)}
    completed_ids = {r["run_id"] for r in results if r.get("status", "ok") == "ok" and r.get("run_id") in assigned_by_id}
    failed_ids = {f.get("run_id") for f in failures if f.get("run_id") in assigned_by_id}
    status = {
        "assigned": len(assigned_by_id), "completed": len(completed_ids),
        "currently_failed": len(failed_ids - completed_ids),
        "pending": len(set(assigned_by_id) - completed_ids - failed_ids),
        "failure_attempts_recorded": len(failures),
        "completed_after_prior_failure": len(completed_ids & failed_ids),
    }
    validations = Counter(e.get("run_id") for e in events if e.get("type") == "response_validation_failed")
    schema_failures = Counter(f.get("run_id") for f in failures if any(
        word in str(f.get("error", "")).lower() for word in
        ("answer label", "probabilit", "citation", "claim", "conditions", "invalid json", "host must")))
    format_issues = sum(max(validations[run_id], schema_failures[run_id])
                        for run_id in set(validations) | set(schema_failures))
    repair_calls = sum((c.get("request") or {}).get("task") == "repair" for c in calls)
    repair_checked = sum(e.get("type") == "response_repair_checked" for e in events)
    supplements = [e for e in events if e.get("type") == "supplement"]
    natural = [r for r in results if r.get("variant", "natural") == "natural"]
    natural_assignments = [a for a in assignments if a.get("variant", "natural") == "natural"]
    by_dataset_group = summarize(natural, natural_assignments, failures)
    calls_by_group_task = call_stats(calls, assigned_by_id)
    all_calls = list(calls_by_group_task.values())
    usage = {field: sum(row[field] for row in all_calls) for field in (
        "calls", "failed_calls", "known_total_tokens", "known_cache_hit_input_tokens",
        "known_cache_miss_input_tokens", "known_output_tokens", "calls_without_total_usage",
        "calls_without_cache_breakdown", "calls_without_output_usage")}
    usage["known_latency_seconds_sum"] = round(sum(row["known_latency_seconds_sum"] for row in all_calls), 3)
    prep = None
    if preparation_manifest and preparation_manifest.exists():
        source = json.loads(preparation_manifest.read_text(encoding="utf-8"))
        prep = {"question_count": len(source.get("question_ids", [])),
                "corpus_chunks": source.get("corpus_chunks"),
                "preparation_model_calls": source.get("model_calls"),
                "preparation_known_tokens": source.get("known_tokens")}
    if status["currently_failed"]:
        next_step = ("先排查最新失败调用与校验事件；若更改代码、模型或数据，用新指纹目录重跑；"
                     "未更改科学配置时保留旧日志续跑。")
    elif status["pending"]:
        next_step = "继续同指纹运行至分配完成，再检查每组完整性、补检索和调用成本。"
    else:
        next_step = "检查可见证据与日志、人工复核失败及补检索；再决定是否进入正式先导。"
    return {
        "note": "技术烟测；样本很小，不能证明医学效果或机制有效。只统计日志中已返回的用量。当前烟测同题共享 R0 及语义判定 cache；表中是执行增量成本，不是每组单独部署成本。",
        "model_backend": config.get("model_backend", "unknown"),
        "model_name": config.get("model_name"),
        "fingerprint": config.get("fingerprint"),
        "status": status, "usage": usage, "calls_by_group_task": calls_by_group_task,
        "format": {"issues_observed": format_issues,
                   "validation_events": sum(validations.values()),
                   "schema_failure_attempts": sum(schema_failures.values()),
                   "repair_calls": repair_calls, "repair_checked_events": repair_checked,
                   "note": "失败尝试累计保留；旧版失败若无校验事件，按失败错误计入问题次数。"},
        "supplement": {"criteria_met_completed_runs": sum(bool(r.get("trigger", {}).get("triggered")) for r in natural),
                       "eligible_triggered_completed_runs": sum(r.get("group") == "semantic_retrieval"
                           and bool(r.get("trigger", {}).get("triggered")) for r in natural),
                       "supplement_attempt_events": len(supplements),
                       "new_chunks_in_attempt_events": sum(len(e.get("chunk_ids") or []) for e in supplements),
                       "completed_runs_with_new_chunks": sum(bool(r.get("supplemental_chunk_ids")) for r in natural)},
        "natural_by_dataset_group": by_dataset_group,
        "preparation": prep, "warnings": warnings, "next_step": next_step,
    }


def render_markdown(report: dict[str, Any]) -> str:
    state, usage, fmt, supplement = (report[key] for key in ("status", "usage", "format", "supplement"))
    lines = ["# 五题烟测技术验收", "", report["note"], "",
             f"模型：{report['model_backend']} / {report.get('model_name') or '未知'}。",
             (f"分配 {state['assigned']}，已完成 {state['completed']}，当前失败 {state['currently_failed']}，"
              f"待跑 {state['pending']}；累计失败尝试 {state['failure_attempts_recorded']}，"
              f"失败后完成 {state['completed_after_prior_failure']}。"), "",
             "## 调用与格式", "",
             (f"调用 {usage['calls']} 次（失败调用 {usage['failed_calls']}）；已知总 token "
              f"{usage['known_total_tokens']}，其中缓存命中输入 {usage['known_cache_hit_input_tokens']}、"
              f"缓存未命中输入 {usage['known_cache_miss_input_tokens']}、输出 {usage['known_output_tokens']}。"
              f"缺总用量 {usage['calls_without_total_usage']} 次，缺缓存拆分 "
              f"{usage['calls_without_cache_breakdown']} 次；已知调用耗时合计 "
              f"{usage['known_latency_seconds_sum']:.1f} 秒。"),
             (f"格式／校验问题 {fmt['issues_observed']} 次；修复调用 {fmt['repair_calls']} 次，"
              f"修复后校验事件 {fmt['repair_checked_events']} 次。失败尝试保留在累计数中。"), "",
             "| 组别 / 任务 | 调用 | 失败 | 已知 token | 命中输入 | 未命中输入 | 输出 | 平均耗时秒 |", 
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, row in report["calls_by_group_task"].items():
        group, task = key.split("/", 1)
        lines.append(f"| {GROUP_NAMES.get(group, group)} / {task} | {row['calls']} | {row['failed_calls']} | "
                     f"{row['known_total_tokens']} | {row['known_cache_hit_input_tokens']} | "
                     f"{row['known_cache_miss_input_tokens']} | {row['known_output_tokens']} | "
                     f"{row['known_latency_seconds_mean'] if row['known_latency_seconds_mean'] is not None else '未知'} |")
    lines += ["", "## 结果与补检索", "",
              (f"已完成且满足补检索条件 {supplement['criteria_met_completed_runs']} 组次（所有组）；"
               f"其中补检索组满足条件 {supplement['eligible_triggered_completed_runs']} 组次；实际补检索事件 "
               f"{supplement['supplement_attempt_events']} 次，事件中新增片段 "
               f"{supplement['new_chunks_in_attempt_events']} 个；已完成且取得新片段 "
               f"{supplement['completed_runs_with_new_chunks']} 组次。事件包含失败尝试。"), "",
              "| 题集 / 组别 | 分配 | 完成 | 有效概率 n | 准确率（完成） | Brier |", 
              "|---|---:|---:|---:|---:|---:|"]
    for key, row in report["natural_by_dataset_group"].items():
        dataset, group, _variant = key.split("/", 2)
        accuracy = row["accuracy_completed_with_gold"]
        brier = row["brier"]
        lines.append(f"| {dataset} / {GROUP_NAMES.get(group, group)} | {row['n_assigned']} | "
                     f"{row['n_completed']} | {row['brier_denominator']} | "
                     f"{f'{accuracy:.3f}' if accuracy is not None else '—'} | "
                     f"{f'{brier:.3f}' if brier is not None else '—'} |")
    lines += ["", "每个格子的 n 很小；准确率和 Brier 只作流程与成本烟测参考，不能判断机制有效。", ""]
    prep = report.get("preparation")
    if prep:
        lines += [(f"语料准备：{prep['question_count']} 题、{prep['corpus_chunks']} 片段；"
                   f"准备阶段模型调用 {prep['preparation_model_calls']} 次，"
                   f"已知 token {prep['preparation_known_tokens']}（未计入上表运行调用）。"), ""]
    lines += [f"下一步：{report['next_step']}", ""]
    if report["warnings"]:
        lines += ["读取提醒：" + "；".join(report["warnings"]), ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="从本地日志生成不含原始请求的烟测验收报告")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--preparation-manifest", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    report = make_report(args.run_dir, args.preparation_manifest)
    prefix = args.output_prefix or args.run_dir / "smoke_acceptance"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    prefix.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
