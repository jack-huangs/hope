"""只读检查预设的 MedQA 普通组 30 题×2 次难度门槛。"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from analyze_common_errors import read_jsonl


def decide(run_dir: Path, questions_path: Path | None = None) -> dict:
    warnings: list[str] = []
    problems: list[str] = []
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config = {}
        problems.append("缺少 config.json")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            config = {}
            problems.append("config.json 无法解析")
    source_path = questions_path or (Path(config["questions"]) if config.get("questions") else None)
    if source_path is None or not source_path.exists():
        problems.append("题目文件不存在")
        source = []
    else:
        expected_hash = config.get("questions_sha256")
        observed_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if not expected_hash or expected_hash != observed_hash:
            problems.append("题目文件 SHA-256 与 config 不一致")
        source = read_jsonl(source_path, warnings)
    source_medqa = [q for q in source if q.get("dataset") == "MedQA"]
    source_ids = [str(q.get("question_id")) for q in source_medqa]
    source_by_id = {str(q.get("question_id")): q for q in source_medqa}
    if len(source_ids) != len(source_by_id):
        problems.append("题目文件含重复 MedQA ID")
    if len(source_by_id) > 30:
        problems.append("题目文件超过原定 30 道 MedQA")
    fingerprint = config.get("fingerprint")
    if not fingerprint:
        problems.append("config 缺少 fingerprint")
    assignments = read_jsonl(run_dir / "assignments.jsonl", warnings)
    results = read_jsonl(run_dir / "results.jsonl", warnings)
    # 同一次配置的全部分配与结果都应带同一指纹，不能让旁支记录混入。
    for row in assignments + results:
        if row.get("fingerprint") != fingerprint:
            problems.append(f"fingerprint 不一致：{row.get('run_id')}")
    selected = lambda x: (x.get("dataset") == "MedQA" and x.get("group") == "ordinary"
                          and x.get("variant", "natural") == "natural"
                          and x.get("repeat_id") in (1, 2))
    assigned = [x for x in assignments if selected(x)]
    done = [x for x in results if selected(x) and x.get("status", "ok") == "ok"]
    questions = sorted(source_by_id)
    assigned_keys = Counter((str(x.get("question_id")), x.get("repeat_id")) for x in assigned)
    result_keys = Counter((str(x.get("question_id")), x.get("repeat_id")) for x in done)
    expected_keys = {(q, repeat) for q in questions for repeat in (1, 2)}
    for row in assigned + done:
        if str(row.get("question_id")) not in source_by_id:
            problems.append(f"题目 ID 未见于原始题目文件：{row.get('run_id')}")
    if any(x.get("dataset") == "MedQA" and x.get("group") == "ordinary"
           and x.get("variant", "natural") == "natural" and x.get("repeat_id") not in (1, 2)
           for x in assignments + results):
        problems.append("MedQA 普通组含预设范围外 repeat")
    if any(count > 1 for count in assigned_keys.values()):
        problems.append("存在重复分配键")
    if any(count > 1 for count in result_keys.values()):
        problems.append("存在重复成功结果键")
    if set(result_keys) - expected_keys:
        problems.append("成功结果包含未分配的题目或 repeat")
    for row in done:
        opts = row.get("options")
        gold, answer = row.get("gold"), row.get("answer")
        if (not isinstance(opts, dict) or len(opts) != 4 or
                any(not isinstance(k, str) or not isinstance(v, str) or not v.strip()
                    for k, v in opts.items()) or gold not in opts or answer not in opts):
            problems.append(f"选项或 gold/answer 无效：{row.get('run_id')}")
        original = source_by_id.get(str(row.get("question_id")))
        if original and (opts != original.get("options") or gold != original.get("gold")):
            problems.append(f"选项或 gold 与原始题不一致：{row.get('run_id')}")
    if warnings:
        problems.append("输入日志缺失或无法完整解析")
    complete = (len(questions) == 30 and len(assigned_keys) == 60
                and set(assigned_keys) == expected_keys and len(result_keys) == 60
                and set(result_keys) == expected_keys and not problems)
    correct = sum(row["answer"] == row["gold"] for row in done) if complete else None
    status = "invalid" if problems else ("ready" if complete else "pending")
    return {"status": status, "dataset": "MedQA", "group": "ordinary", "variant": "natural",
            "required_questions": 30, "required_repeats": [1, 2], "required_successes": 60,
            "assigned_questions_seen": len(questions), "assigned_keys_seen": len(assigned_keys),
            "successful_keys_seen": len(result_keys), "problems": problems, "warnings": warnings,
            "correct_of_60": correct, "switch_to_medxpertqa_text": correct >= 54 if complete else None,
            "clinical_dataset_for_next_stage": ("MedXpertQA Text" if correct >= 54 else "MedQA") if complete else None,
            "pubmedqa_handling": "独立统计，不参与临床题集切换门槛"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--questions", type=Path, help="显式提供题目文件，仍须符合 config 的 SHA-256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = decide(args.run_dir, args.questions)
    output = args.output or args.run_dir / "analysis" / "difficulty_decision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
