"""直接把题干和选项发给 DeepSeek；不检索、不讨论、不做格式修复。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evidence_mdt.env import load_env


def make_request(question: dict, model: str, max_tokens: int) -> dict:
    options = "\n".join(f"{key}. {text}" for key, text in question["options"].items())
    return {
        "model": model,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": (
            f"{question['stem']}\n\n{options}\n\n"
            "Which option is correct? Return only the option letter."
        )}],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/pilot/MedQA.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--question-id", help="只运行指定题号；不传则运行全部题目")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--base-url", help="兼容 Chat Completions 的 API 地址，例如 http://127.0.0.1:18080/v1")
    parser.add_argument("--model", help="覆盖默认模型名")
    parser.add_argument("--local", action="store_true", help="本地 llama.cpp 接口，无需 DeepSeek 密钥；关闭思考")
    parser.add_argument("--include-context", action="store_true", help="同时提供题目自带背景，适用于 PubMedQA；不是 RAG")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1:
        parser.error("repeats 和 max-tokens 必须大于零")
    load_env(ROOT / ".env")
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key and not args.local:
        parser.error("请在 .env 配置 DEEPSEEK_API_KEY")
    model = args.model or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
    if args.local and (not args.base_url or not args.model):
        parser.error("--local 需要显式提供 --base-url 和 --model")
    url = (args.base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    if args.question_id is not None:
        questions = [q for q in questions if str(q["question_id"]) == args.question_id]
    if not questions:
        parser.error("未找到题目")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # 独占创建，避免覆盖已有结果；每题立即落盘。
    with args.output.open("x", encoding="utf-8") as output:
        for question in questions:
            for repeat in range(1, args.repeats + 1):
                payload = make_request(question, model, args.max_tokens)
                if args.include_context and question.get("context"):
                    payload["messages"][0]["content"] = (
                        question["context"] + "\n\n" + payload["messages"][0]["content"])
                if args.local:
                    payload.pop("thinking", None)
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                record = {"question_id": question["question_id"], "repeat_id": repeat,
                          "request": payload, "gold": question.get("gold")}
                try:
                    headers = {"Content-Type": "application/json"}
                    if not args.local:
                        headers["Authorization"] = "Bearer " + key
                    request = urllib.request.Request(
                        url + "/chat/completions",
                        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers=headers,
                    )
                    with urllib.request.urlopen(request, timeout=120) as response:
                        raw = json.load(response)
                    record["response"] = raw
                    choice = raw["choices"][0]
                    answer = (choice["message"].get("content") or "").strip()
                    record.update(answer=answer, usage=raw.get("usage", {}),
                                  finish_reason=choice.get("finish_reason"))
                    valid = answer in question["options"] and choice.get("finish_reason") == "stop"
                    record["status"] = "ok" if valid else "invalid_output"
                    record["correct"] = (answer == question["gold"] if valid and question.get("gold")
                                         else None)
                except Exception as exc:
                    record.update(status="error", error=type(exc).__name__, correct=None)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                print(f"题号 {question['question_id']} 第 {repeat} 次："
                      f"{record.get('answer', '')} [{record['status']}]", flush=True)


if __name__ == "__main__":
    main()
