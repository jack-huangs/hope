"""统一准备命令；完成后不会启动任何答题实验。"""
import argparse
import json
from pathlib import Path
import sys

from evidence_mdt.env import load_env
from evidence_mdt.models import create_model
from evidence_mdt.retrieval import MedCPTDense
from . import prepare


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and argv[0] in {"sample", "batch", "verify"} else "single"
    parser = argparse.ArgumentParser(description=__doc__ +
        " Commands: sample (explicit selection), batch (shared retrieval + B/C/D), verify (offline audit).")
    if command == "sample":
        from .selection import select_questions
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--count", type=int, required=True)
        parser.add_argument("--seed", type=int, required=True)
        parser.add_argument("--exclude", type=Path, action="append", default=[],
                            help="Prior question file or directory containing questions.jsonl; repeatable")
        args = parser.parse_args(argv)
        result = select_questions(args.source, args.output, args.count, args.seed, args.exclude)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if command == "verify":
        from .audit import verify_batch, verify_preparation
        from .state import save
        parser.add_argument("--preparation", type=Path, required=True)
        parser.add_argument("--report", type=Path)
        args = parser.parse_args(argv)
        verify = verify_batch if (args.preparation / "batch_manifest.json").exists() else verify_preparation
        result = verify(args.preparation)
        if args.report:
            target = args.report.resolve()
            if target == args.preparation.resolve() or args.preparation.resolve() in target.parents:
                parser.error("Write audit reports outside the frozen preparation directory")
            save(target, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("deepseek", "qwen"), required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--source-preparation", type=Path,
                        help="Reuse a complete frozen preparation; generate only current intervention materials")
    if command == "batch":
        parser.add_argument("--modes", nargs="+", choices=("b", "c", "d"), default=["b", "c", "d"])
    else:
        parser.add_argument("--interventions", choices=("none", "b", "c", "d"), required=True)
    parser.add_argument("--model-manifest", type=Path, default=Path("indexes/model_download_manifest.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first preparation error without retrying")
    parser.add_argument("--recover-minor", action="store_true", help="Fill missing B rewrites with bounded calls")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int,
                        help="Model output limit; defaults: B 12000, D 6000, otherwise 3000")
    args = parser.parse_args(argv)
    if args.max_output_tokens is not None and args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    load_env(args.env_file)
    model = create_model(args.model, args.model_name)
    model.timeout = args.timeout
    preparing_b = "b" in args.modes if command == "batch" else args.interventions == "b"
    preparing_d = "d" in args.modes if command == "batch" else args.interventions == "d"
    model.max_tokens = args.max_output_tokens or (12000 if preparing_b else 6000 if preparing_d else 3000)
    model.answer_temperature = 0
    # 缺少认证在载入大模型前报告。
    import os
    if not os.environ.get(model.api_key_env):
        parser.error("Missing environment variable " + model.api_key_env)
    if not args.source_preparation and not args.model_manifest.is_file():
        parser.error("Local MedCPT model manifest does not exist")
    dense = None if args.source_preparation else MedCPTDense(args.model_manifest)
    if command == "batch":
        from .batch import prepare_batch
        state = prepare_batch(args.questions, args.output, model, dense, modes=args.modes,
                              seed=args.seed, retries=args.retries, timeout=args.timeout,
                              max_records=args.max_records, source_preparation=args.source_preparation,
                              fail_fast=args.fail_fast, recover_minor=args.recover_minor)
        print(json.dumps({"status": state["status"], "questions": state["questions"],
                          "stages": state["stages"]}, ensure_ascii=False))
        return 0 if state["status"] == "complete" else 1
    state = prepare(args.questions, args.output, model, dense, interventions=args.interventions,
                    seed=args.seed, retries=args.retries, timeout=args.timeout, max_records=args.max_records,
                    source_preparation=args.source_preparation, fail_fast=args.fail_fast,
                    recover_minor=args.recover_minor)
    print(json.dumps({"status": state["status"], "questions": len(state["records"]),
                      "ready": sum(r["status"] == "ready" for r in state["records"])}, ensure_ascii=False))
    return 0 if state["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
