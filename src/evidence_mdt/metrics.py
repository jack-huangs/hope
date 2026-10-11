"""探索性评价；缺失的运行和无效概率会单独计数。"""
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _valid_probabilities(row: dict[str, Any], require_options: bool = True) -> bool:
    probs = row.get("probabilities")
    gold = row.get("gold")
    if not isinstance(probs, dict) or not probs or gold not in probs:
        return False
    expected = row.get("options")
    if require_options and not isinstance(expected, (dict, list, tuple)):
        return False
    if isinstance(expected, (dict, list, tuple)) and set(probs) != set(expected):
        return False
    if any(not isinstance(p, (float, int)) or isinstance(p, bool)
           or not math.isfinite(p) or not 0 <= p <= 1 for p in probs.values()):
        return False
    return abs(sum(probs.values()) - 1) <= 0.01


def brier(probabilities: dict[str, float], gold: str) -> float:
    """多类 Brier：对所有类别求和，不按类别数再除。"""
    if not _valid_probabilities({"probabilities": probabilities, "gold": gold}, require_options=False):
        raise ValueError("Complete valid probability vector required")
    return sum((p - (1.0 if label == gold else 0.0)) ** 2
               for label, p in probabilities.items())


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _run_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("dataset"), row.get("question_id"), row.get("repeat_id"),
            row.get("variant", "natural"), row.get("group"))


def _assigned_count(rows: list[dict[str, Any]], assignments: list[dict[str, Any]] | None,
                    bucket: tuple[str, str, str]) -> int | None:
    if assignments is None:
        return None
    return len({_run_key(a) for a in assignments
                if (a.get("dataset"), a.get("group"), a.get("variant", "natural")) == bucket})


def _joint_error(row: dict[str, Any]) -> bool:
    experts = row.get("expert_answers")
    probs = row["probabilities"]
    answer = row.get("answer")
    return (isinstance(experts, list) and len(experts) == 4
            and len(set(experts)) == 1 and experts[0] != row["gold"]
            and answer == experts[0] and probs.get(answer, 0) >= 0.8)


def _flip_counts(row: dict[str, Any]) -> tuple[int, int]:
    initial, final = row.get("r0_answers"), row.get("expert_answers")
    if not isinstance(initial, list) or not isinstance(final, list) or len(initial) != 4 or len(final) != 4:
        return 0, 0
    eligible = sum(a == row["gold"] for a in initial)
    flips = sum(a == row["gold"] and b != row["gold"] for a, b in zip(initial, final))
    return flips, eligible


def _diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """每个格子附样本量；小样本只给描述值，不作校准保证。"""
    confident = sorted(rows, key=lambda r: r["probabilities"][r["answer"]], reverse=True)
    coverage = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        n = max(1, math.ceil(len(confident) * fraction))
        selected = confident[:n]
        coverage.append({"requested_fraction": fraction, "n": n,
                         "actual_coverage": n / len(confident),
                         "error_rate": mean(r["answer"] != r["gold"] for r in selected)})
    bins = []
    for lo, hi in ((0.0, 0.5), (0.5, 0.8), (0.8, 1.0)):
        selected = [r for r in rows if lo <= r["probabilities"][r["answer"]]
                    and (r["probabilities"][r["answer"]] < hi or hi == 1.0)]
        bins.append({"confidence_range": [lo, hi], "n": len(selected),
                     "mean_reported_confidence": _mean_or_none([r["probabilities"][r["answer"]] for r in selected]),
                     "observed_accuracy": _mean_or_none([float(r["answer"] == r["gold"]) for r in selected])})
    return {"risk_coverage": coverage, "reliability": bins,
            "sharpness_mean_max_probability": mean(max(r["probabilities"].values()) for r in rows)}


def summarize(records: list[dict[str, Any]],
              assignments: list[dict[str, Any]] | None = None,
              failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """按题集、方法和干预分别报告，所有分母都显式给出。"""
    keys = [_run_key(row) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate completed trial key in results")
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("dataset") and row.get("group"):
            buckets[(row["dataset"], row["group"], row.get("variant", "natural"))].append(row)
    if assignments is not None:
        for row in assignments:
            if row.get("dataset") and row.get("group"):
                buckets.setdefault((row["dataset"], row["group"], row.get("variant", "natural")), [])
    out = {}
    for bucket, rows in sorted(buckets.items()):
        dataset, group, variant = bucket
        assigned = _assigned_count(rows, assignments, bucket)
        assigned_rows = [a for a in assignments or [] if
                         (a.get("dataset"), a.get("group"), a.get("variant", "natural")) == bucket]
        completed_keys = {_run_key(r) for r in rows}
        failed_ids = {f.get("run_id") for f in failures or [] if f.get("run_id")}
        failed_keys = {_run_key(f) for f in failures or [] if all(f.get(field) is not None
                       for field in ("dataset", "question_id", "repeat_id", "group"))}
        n_failed = len({_run_key(a) for a in assigned_rows if _run_key(a) not in completed_keys
                        and (a.get("run_id") in failed_ids or _run_key(a) in failed_keys)})
        labelled = [r for r in rows if r.get("gold") is not None]
        valid_answer = [r for r in labelled if r.get("answer") is not None and
                        r.get("answer") in (r.get("options") or r.get("probabilities") or {})]
        valid = [r for r in labelled if _valid_probabilities(r)]
        valid_for_confidence = [r for r in valid if r.get("answer") in r["probabilities"]]
        flips, eligible = zip(*(_flip_counts(r) for r in labelled)) if labelled else ([], [])
        common_errors = sum(_joint_error(r) for r in valid_for_confidence)
        error_questions = {r.get("question_id") for r in valid_for_confidence if _joint_error(r)}
        assigned_questions = ({a.get("question_id") for a in assignments
                              if (a.get("dataset"), a.get("group"), a.get("variant", "natural")) == bucket}
                             if assignments is not None else None)
        n_completed = len(rows)
        result = {
            "n_assigned": assigned, "n_completed": n_completed,
            "n_failed_or_pending": (max(0, assigned - n_completed) if assigned is not None else None),
            "n_failed": n_failed if assigned is not None and failures is not None else None,
            "n_pending": (max(0, assigned - n_completed - n_failed)
                          if assigned is not None and failures is not None else None),
            "failure_state_note": "Failed means an uncompleted assigned trial with a recorded failure; it may be retried.",
            "n_questions_completed": len({r.get("question_id") for r in rows}),
            "n_with_gold": len(labelled), "n_valid_answer": len(valid_answer),
            "n_valid_probability": len(valid),
            "n_valid_answer_and_probability": len(valid_for_confidence),
            "brier": _mean_or_none([brier(r["probabilities"], r["gold"]) for r in valid]),
            "brier_denominator": len(valid),
            "accuracy_completed_with_gold": _mean_or_none([float(r.get("answer") == r["gold"]) for r in labelled]),
            "accuracy_assigned": (sum(r.get("answer") == r["gold"] for r in labelled) / assigned
                                  if assigned else None),
            "accuracy_assigned_note": "Unfinished/failed trials count as not correct; null when assignment manifest is absent.",
            "true_answer_probability": _mean_or_none([r["probabilities"][r["gold"]] for r in valid]),
            "high_confidence_joint_error_count": common_errors,
            "high_confidence_joint_error_question_count": len(error_questions),
            "high_confidence_joint_error_question_rate_assigned":
                (len(error_questions) / len(assigned_questions) if assigned_questions else None),
            "high_confidence_joint_error_question_note":
                "A question counts once if any completed repeat meets the criterion; trial count is reported separately.",
            "high_confidence_joint_error_rate_assigned": (common_errors / assigned if assigned else None),
            "high_confidence_joint_error_rate_completed": (common_errors / len(labelled) if labelled else None),
            "r0_correct_expert_count": sum(eligible),
            "r0_correct_to_final_wrong_count": sum(flips),
            "r0_correct_to_final_wrong_rate": (sum(flips) / sum(eligible) if sum(eligible) else None),
            "probability_diagnostics": _diagnostics(valid_for_confidence) if valid_for_confidence else None,
        }
        out[f"{dataset}/{group}/{variant}"] = result
    return out


def paired_cluster_bootstrap(records: list[dict[str, Any]], group_a: str, group_b: str,
                             iterations: int = 1000, seed: int = 20261008,
                             dataset: str | None = None, variant: str = "natural") -> dict[str, Any]:
    """同题、同 repeat、同 variant 配对；以题目为簇重采样。"""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    datasets = sorted({r["dataset"] for r in records if r.get("dataset") and
                       r.get("variant", "natural") == variant})
    if dataset is not None:
        datasets = [dataset]
    result: dict[str, Any] = {}
    for name in datasets:
        indexed: dict[tuple[str, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in records:
            if (row.get("dataset") != name or row.get("variant", "natural") != variant
                    or row.get("group") not in {group_a, group_b}):
                continue
            key = (row.get("question_id"), row.get("repeat_id"))
            if row["group"] in indexed[key]:
                raise ValueError(f"Duplicate result for {name}/{key}/{row['group']}")
            indexed[key][row["group"]] = row
        by_question: dict[str, list[float]] = defaultdict(list)
        for (question_id, _repeat), pair in indexed.items():
            if all(g in pair and _valid_probabilities(pair[g]) for g in (group_a, group_b)):
                if pair[group_a]["gold"] != pair[group_b]["gold"]:
                    raise ValueError("Gold labels disagree within pair")
                by_question[question_id].append(
                    brier(pair[group_b]["probabilities"], pair[group_b]["gold"])
                    - brier(pair[group_a]["probabilities"], pair[group_a]["gold"]))
        if not by_question:
            continue
        clusters = list(by_question.values())
        # 一道题的所有 repeat 同时入样；题内 repeat 数不同也按原始记录权重。
        observed = mean(delta for cluster in clusters for delta in cluster)
        rng = random.Random(seed)
        samples = sorted(mean(delta for cluster in rng.choices(clusters, k=len(clusters))
                              for delta in cluster) for _ in range(iterations))
        result[name] = {"paired_questions": len(clusters),
                        "paired_trials": sum(map(len, clusters)),
                        "brier_difference_b_minus_a": observed,
                        "bootstrap_95_interval": [samples[int(0.025 * (iterations - 1))],
                                                  samples[int(0.975 * (iterations - 1))]],
                        "note": ("Fewer than five paired questions: the interval is descriptive and has little inferential value. "
                                 if len(clusters) < 5 else "Exploratory interval. ")
                                + "Resampling unit is question, with paired repeats kept together."}
    if not result:
        raise ValueError("No valid paired question/repeat trials")
    return result


def b_target_probability_change(records: list[dict[str, Any]]) -> dict[str, Any]:
    """B 干预相对同题、同 repeat、同方法的 B:base 条件。"""
    baseline = {(r.get("dataset"), r.get("question_id"), r.get("repeat_id"), r.get("group")): r
                for r in records if r.get("variant") == "B:base"}
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    missing: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in records:
        variant = row.get("variant", "")
        if not variant.startswith("B:") or variant == "B:base":
            continue
        key = (row.get("dataset"), row.get("group"), variant)
        base = baseline.get((row.get("dataset"), row.get("question_id"), row.get("repeat_id"), row.get("group")))
        target = row.get("b_target_answer") or row.get("target_answer") or row.get("target_option")
        if (not base or not target or not _valid_probabilities(base)
                or not _valid_probabilities(row) or target not in base["probabilities"]
                or target not in row["probabilities"]):
            missing[key] += 1
            continue
        grouped[key].append(row["probabilities"][target] - base["probabilities"][target])
    keys = sorted(set(grouped) | set(missing))
    return {"/".join(key): {"n_valid_pairs": len(grouped[key]), "n_missing_or_invalid": missing[key],
                            "mean_delta_target_probability": _mean_or_none(grouped[key]),
                            "note": "Descriptive B stress test; position and length were not controlled."}
            for key in keys}


def cost_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """仅汇总日志真实记载的用量；未知价格不换算货币。"""
    by_run: dict[str, dict[str, Any]] = defaultdict(lambda: {"calls": 0, "failed_calls": 0,
                                                               "tokens_recorded": 0, "calls_without_usage": 0,
                                                               "elapsed_seconds_recorded": 0.0})
    for call in calls:
        run_id = call.get("run_id")
        if not run_id:
            continue
        item = by_run[run_id]
        item["calls"] += 1
        item["failed_calls"] += int("error" in call)
        usage = call.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), (int, float)):
            item["tokens_recorded"] += usage["total_tokens"]
        else:
            item["calls_without_usage"] += 1
        if isinstance(call.get("elapsed_seconds"), (int, float)):
            item["elapsed_seconds_recorded"] += call["elapsed_seconds"]
    return dict(by_run)


def load_results(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
