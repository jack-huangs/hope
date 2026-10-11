from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .budget_resume import budget_event, reconcile_config
from .data import (load_questions, local_text_corpus, normalize_question, prepare_pilot,
                   pubmed_snapshot, read_json_records, write_jsonl)
from .env import load_env
from .interventions import B_CONDITIONS, B_EXTENDED_CONDITIONS, b_evidence, c_evidence
from .intervention_preflight import prepare_interventions
from .metrics import (b_target_probability_change, cost_summary, load_results,
                      paired_cluster_bootstrap, summarize)
from .models import create_model
from .query_manifest import initial_queries, validate_medcpt_query_lengths
from .retrieval import MedCPTDense, MockDense, Retriever, load_corpus
from .runner import GROUPS, Runner, append_jsonl
from .schema import Budget


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evidence-mdt")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Normalize local questions/text or snapshot PubMed abstracts")
    prepare.add_argument("--source", choices=["questions", "local-text", "pubmed", "pilot"], required=True)
    prepare.add_argument("--input", type=Path)
    prepare.add_argument("--dataset")
    prepare.add_argument("--query")
    prepare.add_argument("--max-records", type=int, default=20)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=20261008)

    index = sub.add_parser("index", help="Validate corpus and build the selected dense index")
    index.add_argument("--corpus", type=Path, required=True)
    index.add_argument("--backend", choices=["mock", "medcpt"], required=True)
    index.add_argument("--output", type=Path, required=True)

    run = sub.add_parser("run", help="Run three paired MDT methods")
    run.add_argument("--questions", type=Path, required=True)
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model-backend", choices=["mock", "deepseek", "qwen"], required=True)
    run.add_argument("--model-name")
    run.add_argument("--env-file", type=Path, default=Path(".env"))
    run.add_argument("--retrieval-backend", choices=["mock", "medcpt"], required=True)
    run.add_argument("--index-manifest", type=Path)
    run.add_argument("--query-manifest", type=Path,
                     help="每题固定的短检索查询；按公开题面哈希校验")
    run.add_argument("--repeats", type=int, default=2)
    run.add_argument("--max-calls", type=int, default=50000)
    run.add_argument("--max-tokens", type=int, default=10000000)
    run.add_argument("--interventions", choices=["none", "b", "c", "both"], default="none")
    run.add_argument("--intervention-spec", type=Path)
    run.add_argument("--b-extended", action="store_true",
                     help="显式增加 literal5/paraphrase5；首轮默认仅三个 B 条件")
    run.add_argument("--max-questions", type=int)
    run.add_argument("--groups", choices=["all", "ordinary"], default="all",
                     help="ordinary 仅用于 MedQA 难度预筛；all 用于三组主比较")

    evaluate = sub.add_parser("evaluate", help="Summarize outcomes and paired Brier differences")
    evaluate.add_argument("--results", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    return parser


def prepare_command(args) -> None:
    if args.source == "pilot":
        prepare_pilot(args.output, args.seed)
    elif args.source == "questions":
        if not args.input or not args.dataset:
            raise ValueError("questions requires --input and --dataset")
        records = read_json_records(args.input)
        questions = [normalize_question(item, args.dataset, i) for i, item in enumerate(records)]
        write_jsonl(args.output, questions)
    elif args.source == "local-text":
        if not args.input:
            raise ValueError("local-text requires --input directory")
        write_jsonl(args.output, local_text_corpus(args.input))
    else:
        if not args.query:
            raise ValueError("pubmed requires --query")
        evidence, snapshot = pubmed_snapshot(args.query, args.max_records)
        raw_xml = snapshot.pop("_raw_xml", None)
        write_jsonl(args.output, evidence)
        if raw_xml is not None:
            args.output.with_suffix(".raw.xml").write_text(raw_xml, encoding="utf-8")
        args.output.with_suffix(".snapshot.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


def dense_backend(name: str):
    if name == "mock":
        return MockDense()
    manifest = Path("indexes/model_download_manifest.json")
    return MedCPTDense(manifest if manifest.exists() else None)


def index_command(args) -> None:
    corpus = load_corpus(args.corpus)
    retriever = Retriever(corpus, dense_backend(args.backend))
    digest = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"corpus": str(args.corpus.resolve()), "sha256": digest,
                "chunks": len(corpus), "indexed_chunks": len(retriever.evidence),
                "backend": args.backend, "model_revisions": getattr(retriever.dense, "revisions", None),
                "medcpt_index_built": args.backend == "medcpt"}
    if args.backend == "medcpt":
        vector_path = args.output.with_suffix(".vectors.pt")
        retriever.dense.torch.save(retriever.dense.vectors, vector_path)
        manifest["vectors"] = str(vector_path.resolve())
        manifest["vectors_sha256"] = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(args.output)


def run_command(args) -> None:
    load_env(args.env_file)
    questions = load_questions(args.questions)
    if args.max_questions is not None:
        questions = questions[:args.max_questions]
    query_lookup = initial_queries(questions, args.query_manifest)
    corpus = load_corpus(args.corpus)
    model = create_model(args.model_backend, args.model_name)
    dense = dense_backend(args.retrieval_backend)
    if args.query_manifest and args.retrieval_backend == "medcpt":
        validate_medcpt_query_lengths(query_lookup, dense.qtok)
    if args.index_manifest:
        manifest = json.loads(args.index_manifest.read_text(encoding="utf-8"))
        if manifest["sha256"] != hashlib.sha256(args.corpus.read_bytes()).hexdigest() or manifest["backend"] != args.retrieval_backend:
            raise ValueError("Index manifest does not match corpus/backend")
        if args.retrieval_backend == "medcpt":
            if manifest.get("model_revisions") != dense.revisions:
                raise ValueError("Dense index model revisions differ from loaded MedCPT models")
            vector_path = Path(manifest["vectors"])
            if (manifest.get("vectors_sha256") and
                    hashlib.sha256(vector_path.read_bytes()).hexdigest() != manifest["vectors_sha256"]):
                raise ValueError("Dense index vector hash mismatch")
            dense.load_vectors(vector_path)
            if len(dense.vectors) != manifest.get("indexed_chunks", len(corpus)):
                raise ValueError("Dense index vector count mismatch")
    retriever = Retriever(corpus, dense)
    runner = Runner(model, retriever, args.output, Budget(args.max_calls, args.max_tokens))
    args.output.mkdir(parents=True, exist_ok=True)
    if args.b_extended and args.interventions not in {"b", "both"}:
        raise ValueError("--b-extended requires B interventions")
    b_conditions = B_CONDITIONS + (B_EXTENDED_CONDITIONS if args.b_extended else ())
    chosen: set[str] = set()
    specs: dict[str, dict] = {}
    initial_bases = {}
    initial_traces = {}
    c_injections = {}
    intervention_manifest = None
    if args.interventions != "none":
        if not args.intervention_spec:
            raise ValueError("B/C needs --intervention-spec with investigator-supplied text")
        (chosen, specs, initial_bases, initial_traces, c_injections,
         intervention_manifest) = prepare_interventions(
             questions, retriever, args.interventions, args.intervention_spec,
             args.repeats, b_conditions, query_lookup)
        intervention_manifest.update({
            "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
            "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
            "index_manifest_sha256": (hashlib.sha256(args.index_manifest.read_bytes()).hexdigest()
                                      if args.index_manifest else None),
            "retrieval_revisions": getattr(dense, "revisions", None)})
    intervention_manifest_bytes = (json.dumps(intervention_manifest, sort_keys=True,
                                              ensure_ascii=False).encode("utf-8")
                                   if intervention_manifest else None)
    # 指纹覆盖数据、代码、提示词、实验规格与设置。变化后必须使用新目录。
    source_dir = Path(__file__).parent
    code_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob("*.py"))}
    spec_file = Path(__file__).resolve().parents[2] / "EXPERIMENT_PLAN.md"
    config = {"questions": str(args.questions.resolve()), "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
              "corpus": str(args.corpus.resolve()), "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
              "model_backend": args.model_backend, "model_name": model.name,
              "model_base_url": getattr(model, "base_url", None),
              "retrieval_backend": args.retrieval_backend, "index_manifest": str(args.index_manifest.resolve()) if args.index_manifest else None,
              "initial_query_strategy": "explicit_manifest" if args.query_manifest else "stem_plus_context",
              "query_manifest_sha256": (hashlib.sha256(args.query_manifest.read_bytes()).hexdigest()
                                        if args.query_manifest else None),
              "index_manifest_sha256": (hashlib.sha256(args.index_manifest.read_bytes()).hexdigest()
                                        if args.index_manifest else None),
              "repeats": args.repeats, "groups": args.groups,
              "interventions": args.interventions, "max_questions": args.max_questions,
              "b_conditions": list(b_conditions) if args.interventions in {"b", "both"} else [],
              "intervention_selection_seed": 20261008 if args.interventions != "none" else None,
              "intervention_manifest_sha256": (hashlib.sha256(intervention_manifest_bytes).hexdigest()
                                                if intervention_manifest_bytes else None),
              "intervention_spec_sha256": (hashlib.sha256(args.intervention_spec.read_bytes()).hexdigest()
                                            if args.intervention_spec else None),
              "code_sha256": code_hashes,
              "experiment_plan_sha256": hashlib.sha256(spec_file.read_bytes()).hexdigest(),
              "retrieval_revisions": getattr(dense, "revisions", None),
              "model_temperature": getattr(model, "answer_temperature", None),
              "model_max_output_tokens": getattr(model, "max_tokens", None)}
    config_path = args.output / "config.json"
    existing = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else None
    policy_path = Path(__file__).resolve().parents[2] / "budget_resume_policy.json"
    policy_bytes = policy_path.read_bytes() if existing and "max_calls" in existing else b"{}"
    policy = json.loads(policy_bytes)
    config, migration = reconcile_config(existing, config, policy,
                                          hashlib.sha256(policy_bytes).hexdigest())
    fingerprint = config["fingerprint"]
    runner.fingerprint = fingerprint
    if existing is None:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    if intervention_manifest is not None:
        manifest_path = args.output / "intervention_manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != intervention_manifest:
            raise ValueError("Frozen intervention manifest differs from prior run")
        manifest_path.write_text(json.dumps(intervention_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if migration:
        migration_path = args.output / "execution_migrations.jsonl"
        previous_migrations = read_json_records(migration_path) if migration_path.exists() else []
        if not any(item.get("policy_sha256") == migration["policy_sha256"] and
                   item.get("legacy_fingerprint_retained") == fingerprint
                   for item in previous_migrations):
            append_jsonl(migration_path, migration)
    if runner.call_log.exists():
        previous = [json.loads(line) for line in runner.call_log.read_text(encoding="utf-8").splitlines() if line.strip()]
        # 网络失败也消耗了调用尝试，续跑不能把它从预算中抹掉。
        runner.budget.calls = len(previous)
        runner.budget.tokens = sum(int((x.get("usage") or {}).get("total_tokens") or 0) for x in previous)
    limit_path = args.output / "budget_limits.jsonl"
    prior_limits = read_json_records(limit_path) if limit_path.exists() else []
    old_calls = (prior_limits[-1]["new_max_calls"] if prior_limits else config.get("max_calls"))
    old_tokens = (prior_limits[-1]["new_max_tokens"] if prior_limits else config.get("max_tokens"))
    append_jsonl(limit_path, budget_event(old_calls, old_tokens, args.max_calls, args.max_tokens,
                                          runner.budget.calls, runner.budget.tokens,
                                          fingerprint, code_hashes["cli.py"]))
    completed = {r["run_id"] for r in load_results(runner.result_log)
                 if r.get("status") == "ok" and r.get("fingerprint") == fingerprint} if runner.result_log.exists() else set()
    groups = ("ordinary",) if args.groups == "ordinary" else GROUPS
    # 运行前一次性登记全部分配，预算耗尽或中断也不会丢失分母。
    assignments = []
    for q in questions:
        for repeat_id in range(1, args.repeats + 1):
            variants = ["natural"]
            if q.question_id in chosen:
                if args.interventions in {"b", "both"}:
                    variants += ["B:" + condition for condition in b_conditions]
                if args.interventions in {"c", "both"}:
                    variants += ["C:evidence_error", "C:message_error", "C:repeated_message"]
            for variant in variants:
                for group in groups:
                    assignments.append({"run_id": f"{q.question_id}:repeat-{repeat_id}:{variant}:{group}",
                                        "dataset": q.dataset, "question_id": q.question_id,
                                        "repeat_id": repeat_id, "variant": variant, "group": group,
                                        "fingerprint": fingerprint})
    assignment_path = args.output / "assignments.jsonl"
    if assignment_path.exists() and read_json_records(assignment_path) != assignments:
        raise ValueError("Assignment manifest differs from prior run")
    write_jsonl(assignment_path, assignments)

    def do_variant(q, repeat_id, variant, evidence, injection=None, reuse_r0=None):
        first_r0 = reuse_r0
        runner.b_target_answer = (specs.get(q.question_id, {}).get("target_answer") or
                                  specs.get(q.question_id, {}).get("target_option")) if variant.startswith("B:") else None
        for group in groups:
            run_id = f"{q.question_id}:repeat-{repeat_id}:{variant}:{group}"
            if run_id in completed:
                if first_r0 is None:
                    first_r0 = runner.load_r0(run_id)
                    if len(first_r0) != 4:
                        raise RuntimeError("Completed baseline lacks R0 turns")
                continue
            try:
                result, r0 = runner.run_group(q, repeat_id, group, evidence, variant,
                                              first_r0, injection)
            except BaseException as exc:
                append_jsonl(args.output / "failures.jsonl", {"run_id": run_id,
                             "dataset": q.dataset, "question_id": q.question_id,
                             "repeat_id": repeat_id, "variant": variant, "group": group,
                             "fingerprint": fingerprint, "error": repr(exc),
                             "calls": runner.budget.calls, "tokens": runner.budget.tokens})
                raise
            result["fingerprint"] = fingerprint
            completed.add(result["run_id"])
            if first_r0 is None:
                first_r0 = r0
        return first_r0

    try:
        for q in questions:
            excluded = set()
            if q.dataset.lower().startswith("pubmedqa"):
                excluded = {e.chunk_id for e in corpus if e.document_id in {q.question_id, f"PMID:{q.question_id}"} or
                            (q.context and e.text.strip() in q.context)}
            if q.question_id in initial_bases:
                base = initial_bases[q.question_id]
                trace = initial_traces[q.question_id]
            else:
                base = retriever.search(query_lookup[q.question_id], limit=6, exclude=excluded)
                trace = retriever.last_trace
            append_jsonl(args.output / "retrieval_snapshots.jsonl",
                         {"question_id": q.question_id, "dataset": q.dataset,
                          "fingerprint": fingerprint, "trace": trace})
            # 空快照仍可运行；专家会看到无证据的事实并自行给出不确定概率。
            for repeat_id in range(1, args.repeats + 1):
                natural_r0 = do_variant(q, repeat_id, "natural", base)
                if q.question_id not in chosen:
                    continue
                spec = specs.get(q.question_id)
                if not spec:
                    raise ValueError(f"Missing intervention specification for {q.question_id}")
                if args.interventions in {"b", "both"}:
                    for condition in b_conditions:
                        exposed = b_evidence(base, condition, spec["b_target_id"],
                                             spec.get("b_paraphrases"))
                        do_variant(q, repeat_id, "B:" + condition, exposed)
                if args.interventions in {"c", "both"}:
                    perturbed = c_evidence(base, spec["c_evidence_text"], q.question_id)
                    do_variant(q, repeat_id, "C:evidence_error", perturbed)
                    for name, repeated in (("message_error", False), ("repeated_message", True)):
                        injection = c_injections[(q.question_id, repeat_id, name)]
                        do_variant(q, repeat_id, "C:" + name, base, injection, natural_r0)
    except Exception:
        raise
    print(runner.result_log)


def evaluate_command(args) -> None:
    # 全部失败时 results.jsonl 可能尚不存在，仍需生成含完整分母的报告。
    records = load_results(args.results) if args.results.exists() else []
    assignment_path = args.results.parent / "assignments.jsonl"
    assignments = read_json_records(assignment_path) if assignment_path.exists() else None
    failure_path = args.results.parent / "failures.jsonl"
    failures = read_json_records(failure_path) if failure_path.exists() else []
    call_path = args.results.parent / "calls.jsonl"
    calls = read_json_records(call_path) if call_path.exists() else []
    natural = [r for r in records if r.get("variant") == "natural"]
    config_path = args.results.parent / "config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    mode = run_config.get("model_backend", "unknown")
    report = {"note": ("Mock records are software fixtures, not medical results." if mode == "mock" else
                       "Exploratory research results; not clinical validation or medical advice."),
              "model_backend": mode,
              "natural": summarize(natural, [a for a in assignments if a["variant"] == "natural"] if assignments else None, failures),
              "all_variants": summarize(records, assignments, failures),
              "b_target_probability_change": b_target_probability_change(records),
              "cost_by_run": cost_summary(calls), "paired": {}}
    for other in ("semantic", "semantic_retrieval"):
        try:
            report["paired"][other] = paired_cluster_bootstrap(natural, "ordinary", other)
        except ValueError as exc:
            report["paired"][other] = {"unavailable_reason": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


def main(argv: list[str] | None = None) -> None:
    args = args_parser().parse_args(argv)
    if args.command == "prepare":
        prepare_command(args)
    elif args.command == "index":
        index_command(args)
    elif args.command == "run":
        run_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)


if __name__ == "__main__":
    main()
