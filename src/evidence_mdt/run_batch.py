"""使用同一批冻结材料运行普通 MDT-RAG 的 B/C/D 条件，共用一次基线。"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.error

from preparation.audit import verify_batch
from preparation.state import digest, jsonl, save
from .data import load_questions, read_json_records
from .env import load_env
from .intervention_preflight import prepare_interventions
from .interventions import B_CONDITIONS, C_CONDITIONS, D_CONDITIONS, b_evidence, d_evidence
from .metrics import summarize, b_target_probability_change, d_target_probability_change
from .models import create_model
from .retrieval import INITIAL_EVIDENCE_LIMIT
from .runner import Runner, append_jsonl
from .schema import Budget, Evidence

CONDITIONS_BY_MODE = {"b": B_CONDITIONS[1:], "c": C_CONDITIONS, "d": D_CONDITIONS}


def read_rows(path):
    return read_json_records(path) if path.exists() else []


class FrozenRetrieval:
    def __init__(self, rows, queries):
        self.lookup = {queries[r["question_id"]]: r for r in rows}

    def search(self, query, **kwargs):
        row = self.lookup[query]
        self.last_trace = row.get("trace") or {}
        return [Evidence(**e) for e in row["evidence"]]


class NoRetrieval:
    def __init__(self, evidence):
        self.evidence = evidence

    def search(self, *args, **kwargs):
        raise RuntimeError("Runtime retrieval is disabled for ordinary batch experiments")


class CallLimit:
    def __init__(self, calls, tokens, max_calls, max_tokens):
        self.calls, self.tokens = calls, tokens
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def reserve(self):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("Execution stopped after a fatal error")
            if self.calls >= self.max_calls or self.tokens >= self.max_tokens:
                self.stop.set()
                raise RuntimeError("API budget exhausted")
            self.calls += 1

    def charge(self, usage):
        with self.lock:
            self.tokens += int((usage or {}).get("total_tokens") or 0)


class QuestionBudget(Budget):
    def __init__(self, limit, calls, tokens):
        super().__init__(limit.max_calls, limit.max_tokens, calls, tokens)
        self.limit = limit

    def before_call(self):
        self.limit.reserve()

    def after_call(self, usage):
        super().after_call(usage)
        self.limit.charge(usage)


def run_batch(preparation, output, model, *, modes=("b", "c", "d"), workers=4, attempts=2,
              max_calls=30000, max_tokens=300000000, fail_fast=False, recover_minor=False):
    preparation, output = Path(preparation).resolve(), Path(output).resolve()
    if workers < 1 or attempts < 1 or max_calls < 1 or max_tokens < 1:
        raise ValueError("Workers, attempts and execution budgets must be positive")
    if fail_fast:
        workers = attempts = 1
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes) or set(modes) - CONDITIONS_BY_MODE.keys():
        raise ValueError("Run modes must be unique choices from b, c, d")
    modes = tuple(mode for mode in CONDITIONS_BY_MODE if mode in modes)
    verified = verify_batch(preparation)
    missing = set(modes) - verified["modes"].keys()
    if missing:
        raise ValueError("Complete preparations are required for: " + ", ".join(sorted(missing)))
    questions = load_questions(preparation / "questions.jsonl")
    # 每个模式都有经 verify_batch 核对的同一份快照，也适用于复用外部准备目录的批次。
    common = preparation / modes[0]
    queries = {r["question_id"]: r["retrieval_query"] for r in read_rows(common / "initial_queries.jsonl")}
    frozen = read_rows(common / "initial_top10.jsonl")
    retriever = FrozenRetrieval(frozen, queries)
    specs, preflight, injections = {}, {}, {}
    for mode in modes:
        _, specs[mode], bases, _, mode_injections, preflight[mode] = prepare_interventions(
            questions, retriever, mode, preparation / mode / "intervention_spec.jsonl",
            1, B_CONDITIONS, queries)
        injections.update(mode_injections)
    d_audits = {row["question_id"]: row["passages"] for row in preflight.get("d", {}).get("d_assignments", [])}
    variants = ("B:base",) + tuple(f"{mode.upper()}:{condition}" for mode in modes
                                   for condition in CONDITIONS_BY_MODE[mode])
    source = Path(__file__).resolve().parent
    config = {"protocol": "ordinary_batch_shared_base_v1", "preparation": str(preparation),
              "preparation_manifest_sha256": digest((preparation / "batch_manifest.json").read_bytes()),
              "questions_sha256": digest((preparation / "questions.jsonl").read_bytes()),
              "questions": [q.question_id for q in questions], "variants": list(variants), "modes": list(modes),
              "groups": ["ordinary"], "repeats": 1, "initial_evidence_limit": INITIAL_EVIDENCE_LIMIT,
              "runtime_retrieval": False, "semantic_merging": False,
              "fail_fast": fail_fast,
              "recover_minor": recover_minor,
              "model": {k: getattr(model, k, None) for k in
                        ("name", "base_url", "max_tokens", "answer_temperature", "disable_thinking")},
              "code_sha256": {p.name: digest(p.read_bytes()) for p in source.glob("*.py")},
              "experiment_plan_sha256": digest((source.parents[1] / "EXPERIMENT_PLAN.md").read_bytes()),
              "preflight": preflight}
    config["fingerprint"] = digest(config)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_bytes()) != config:
        raise ValueError("Scientific configuration changed; use a new output directory")
    save(config_path, config)
    fingerprint = config["fingerprint"]
    folders = {q.question_id: output / "questions" / f"q{i:04d}" for i, q in enumerate(questions, 1)}
    assignments = [{"run_id": f"{q.question_id}:repeat-1:{v}:ordinary", "question_id": q.question_id,
                    "dataset": q.dataset, "repeat_id": 1, "variant": v, "group": "ordinary",
                    "fingerprint": fingerprint, "log_directory": str(folders[q.question_id])}
                   for q in questions for v in variants]
    jsonl(output / "assignments.jsonl", assignments)
    jsonl(output / "questions.jsonl", [asdict(q) for q in questions])
    completed = {}
    call_count = token_count = 0
    for folder in folders.values():
        for r in read_rows(folder / "results.jsonl"):
            if r.get("fingerprint") != fingerprint or r.get("group") != "ordinary":
                raise ValueError("Existing result fingerprint/group differs")
            if r["run_id"] in completed:
                raise ValueError("Duplicate successful result")
            completed[r["run_id"]] = r
        for r in read_rows(folder / "calls.jsonl"):
            call_count += 1
            token_count += int((r.get("usage") or {}).get("total_tokens") or 0)
    limit = CallLimit(call_count, token_count, max_calls, max_tokens)
    append_jsonl(output / "budget_limits.jsonl", {"at": time.time(), "max_calls": max_calls,
                 "max_tokens": max_tokens, "calls_at_start": call_count, "tokens_at_start": token_count,
                 "workers": workers, "attempts_per_pending_variant": attempts})
    lock = threading.Lock()
    terminal = []
    def checkpoint(status="running"):
        records = [completed[a["run_id"]] for a in assignments if a["run_id"] in completed]
        jsonl(output / "results.jsonl", records)
        summary = {"status": status, "planned": len(assignments), "completed": len(records),
                   "question_count": len(questions), "by_condition": dict(Counter(r["variant"] for r in records)),
                   "calls": limit.calls, "known_tokens": limit.tokens, "workers": workers,
                   "terminal_failures": list(terminal), "updated_at": time.time(),
                   "group": "ordinary", "semantic_merging": False, "runtime_retrieval": False}
        save(output / "summary.json", summary)
        return summary
    checkpoint()
    def work(q):
        folder = folders[q.question_id]
        previous = read_rows(folder / "calls.jsonl")
        budget = QuestionBudget(limit, len(previous), sum(int((r.get("usage") or {}).get("total_tokens") or 0) for r in previous))
        runner = Runner(copy.copy(model), NoRetrieval(bases[q.question_id]), folder, budget)
        runner.fingerprint = fingerprint
        runner.fail_fast = fail_fast and not recover_minor
        r0 = None
        bspec = specs.get("b", {}).get(q.question_id, {})
        dspec = specs.get("d", {}).get(q.question_id, {})
        for variant in variants:
            rid = f"{q.question_id}:repeat-1:{variant}:ordinary"
            if rid in completed:
                if variant == "B:base":
                    r0 = runner.load_r0(rid)
                    if len(r0) != 4:
                        raise ValueError("Completed baseline lacks replayable R0")
                continue
            if limit.stop.is_set():
                return
            evidence = bases[q.question_id]
            if variant.startswith("B:") and variant != "B:base":
                evidence = b_evidence(evidence, variant.split(":")[1], bspec["b_target_id"], bspec["b_paraphrases"])
            if variant == "D:evidence_error":
                evidence = d_evidence(evidence, [row["text"] for row in dspec["d_evidence_passages"]], q.question_id)
            injection = injections[q.question_id, 1, variant.split(":")[1]] if variant.startswith("C:") else None
            runner.b_target_answer = bspec.get("target_answer") if variant.startswith("B:") else None
            runner.d_target_answer = (dspec.get("d_wrong_option")
                                      if variant in {"B:base", "D:evidence_error"} else None)
            runner.d_evidence_audit = d_audits[q.question_id] if variant == "D:evidence_error" else []
            runner.d_error_explanation = "\n".join(f"{row['angle']}: {row['error_explanation']}"
                                                   for row in runner.d_evidence_audit if row["error_explanation"])
            for attempt in range(1, attempts + 1):
                succeeded = False
                try:
                    result, current_r0 = runner.run_group(q, 1, "ordinary", evidence, variant,
                                                        r0 if variant.startswith("C:") else None, injection)
                    succeeded = True
                    if variant == "B:base":
                        r0 = current_r0
                    with lock:
                        completed[rid] = result
                        summary = checkpoint()
                        print(json.dumps({"completed": summary["completed"], "planned": len(assignments),
                                          "run_id": rid, "calls": limit.calls}), flush=True)
                    break
                except Exception as exc:
                    if succeeded:
                        # 结果已落盘后只允许恢复汇总，不能因报告写入失败重新调用团队。
                        raise
                    failure = {"run_id": rid, "question_id": q.question_id, "variant": variant,
                               "group": "ordinary", "dataset": q.dataset, "repeat_id": 1,
                               "fingerprint": fingerprint, "attempt_id": runner.current_attempt_id,
                               "attempt_this_invocation": attempt, "error": repr(exc), "at": time.time()}
                    append_jsonl(folder / "failures.jsonl", failure)
                    fatal = fail_fast or (
                        isinstance(exc, urllib.error.HTTPError) and exc.code in {400, 401, 403, 404, 422})
                    if fatal:
                        limit.stop.set()
                    if fatal or limit.stop.is_set() or attempt == attempts:
                        with lock:
                            terminal.append(failure)
                            checkpoint()
                        # C requires the successful baseline R0; other failures do not erase successes.
                        if variant == "B:base" or limit.stop.is_set():
                            return
                        break
                    time.sleep(min(2 ** attempt, 4))
    def run_question(q):
        try:
            work(q)
        except Exception as exc:
            limit.stop.set()
            with lock:
                terminal.append({"question_id": q.question_id, "error": repr(exc)})
                checkpoint()

    if fail_fast:
        for q in questions:
            run_question(q)
            if limit.stop.is_set():
                break
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed(pool.submit(run_question, q) for q in questions):
                future.result()
    with lock:
        status = "stopped" if fail_fast and limit.stop.is_set() else "incomplete"
        summary = checkpoint("complete" if len(completed) == len(assignments) else status)
    records = list(completed.values())
    save(output / "report.json", {"outcomes": summarize(records, assignments),
                                  "b_target_probability_change": b_target_probability_change(records),
                                  "d_target_probability_change": d_target_probability_change(records),
                                  "execution": summary})
    return summary


def main(argv=None, *, default_modes=("b", "c", "d"), module_name="evidence_mdt.run_batch"):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preparation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", choices=tuple(CONDITIONS_BY_MODE), default=default_modes)
    p.add_argument("--model", choices=("deepseek", "qwen"), default="deepseek")
    p.add_argument("--model-name")
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--fail-fast", action="store_true", help="Run sequentially and stop at the first error")
    p.add_argument("--recover-minor", action="store_true", help="Allow the usual single response-format repair")
    p.add_argument("--max-calls", type=int, default=30000)
    p.add_argument("--max-tokens", type=int, default=300000000)
    p.add_argument("--max-output-tokens", type=int, default=6000)
    p.add_argument("--wait-preparation", action="store_true")
    p.add_argument("--frozen-worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if not a.frozen_worker:
        snapshot = a.output.resolve() / "software_snapshot"
        if not snapshot.exists():
            project = Path(__file__).resolve().parents[2]
            for package in ("evidence_mdt", "preparation"):
                target = snapshot / "src" / package
                target.mkdir(parents=True, exist_ok=True)
                for path in (project / "src" / package).glob("*.py"):
                    shutil.copyfile(path, target / path.name)
            shutil.copyfile(project / "EXPERIMENT_PLAN.md", snapshot / "EXPERIMENT_PLAN.md")
            save(snapshot / "source_hashes.json", {str(f.relative_to(snapshot)): digest(f.read_bytes())
                 for f in snapshot.rglob("*") if f.is_file()})
        expected = json.loads((snapshot / "source_hashes.json").read_bytes())
        if any(digest((snapshot / name).read_bytes()) != value for name, value in expected.items()):
            raise ValueError("Frozen software snapshot hash mismatch")
        if not (snapshot / "src" / Path(*module_name.split(".")).with_suffix(".py")).is_file():
            raise ValueError("This entry point is absent from the frozen software; use a new output directory")
        env = dict(os.environ, PYTHONPATH=str(snapshot / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
        return subprocess.call([sys.executable, "-u", "-m", module_name, *argv,
                                "--frozen-worker"], env=env)
    load_env(a.env_file)
    if a.wait_preparation:
        path = a.preparation / "batch_manifest.json"
        while True:
            state = json.loads(path.read_bytes()) if path.exists() else {}
            if state.get("status") == "complete":
                break
            if state.get("status") in {"incomplete", "stopped"}:
                raise ValueError("Preparation stopped or incomplete; repair/resume preparation first")
            print(json.dumps({"waiting_for_preparation": str(path), "status": state.get("status")}), flush=True)
            time.sleep(20)
    model = create_model(a.model, a.model_name)
    model.max_tokens = a.max_output_tokens
    model.timeout = 180
    if not os.environ.get(model.api_key_env):
        p.error("Missing environment variable " + model.api_key_env)
    state = run_batch(a.preparation, a.output, model, modes=a.modes, workers=a.workers, attempts=a.attempts,
                      max_calls=a.max_calls, max_tokens=a.max_tokens, fail_fast=a.fail_fast,
                      recover_minor=a.recover_minor)
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
