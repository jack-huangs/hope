from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ModelResult:
    """一次模型响应：结构化内容供决策，原始响应和用量供审计。"""
    content: dict[str, Any]
    raw: dict[str, Any]
    usage: dict[str, Any]
    wire_request: dict[str, Any] | None = None


class ModelResponseError(ValueError):
    """服务端已响应但 JSON 不能解析；仍保留原文和 token 用量。"""
    def __init__(self, message: str, raw: dict[str, Any], wire_request: dict[str, Any]):
        super().__init__(message)
        self.raw = raw
        self.wire_request = wire_request
        self.usage = raw.get("usage") or {}


class Model:
    name: str

    def complete(self, payload: dict[str, Any]) -> ModelResult:
        raise NotImplementedError


class MockModel(Model):
    """只用于离线验证流程，不代表医疗能力或研究结果。"""
    name = "mock"

    def complete(self, payload: dict[str, Any]) -> ModelResult:
        task = payload["task"]
        if task == "semantic":
            a, b = payload["a"], payload["b"]
            same_text = " ".join(a["text"].lower().split()) == " ".join(b["text"].lower().split())
            content = {"relation": ("equivalent" if same_text and a.get("conditions") == b.get("conditions")
                                    else "condition_different" if same_text else "unknown")}
        elif task == "semantic_batch":
            content = {"relations": ["equivalent" if p["a"] == p["b"] else
                                     "condition_different" if p["a"]["text"] == p["b"]["text"] else "unknown"
                                     for p in payload["pairs"]]}
        elif task == "queries":
            content = {"queries": [payload["question"]["stem"] + " counter evidence",
                                   payload["question"]["stem"] + " distinguish alternatives"]}
        elif task == "recruit":
            content = {"roles": ["diagnostic specialist", "clinical epidemiologist",
                                 "pharmacology specialist", "general medicine specialist"]}
        elif task == "novelty":
            candidate = " ".join(payload["candidate"].lower().split())
            known = {" ".join(x.lower().split()) for x in payload["existing"]}
            content = {"relevant": True, "novel": candidate not in known,
                       "new_claim_or_condition": candidate not in known}
        else:
            labels = list(payload["question"]["options"])
            seed = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).digest()
            chosen = labels[seed[0] % len(labels)]
            rest = 0.3 / (len(labels) - 1)
            probs = {k: (0.7 if k == chosen else rest) for k in labels}
            cites = [x["chunk_id"] for x in payload.get("evidence", [])[:1]]
            content = {"answer": chosen, "probabilities": probs, "citations": cites,
                       "claims": [{"text": "Fixture claim " + chosen, "answer": chosen,
                                   "citation_ids": cites, "polarity": "support", "conditions": ""}],
                       "opinion": "Mock output for software testing only.",
                       "unexplained_contradiction": False, "missing_distinction": False}
        return ModelResult(content, {"fixture": True, "content": content}, {"total_tokens": 0}, payload)


class OpenAICompatibleModel(Model):
    """调用兼容 Chat Completions 的服务，并保存实际请求参数。"""
    def __init__(self, name: str, base_url: str, api_key_env: str, timeout: int = 120,
                 answer_temperature: float = 0.3, max_tokens: int = 800,
                 disable_thinking: bool = False):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.answer_temperature = answer_temperature
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking

    def complete(self, payload: dict[str, Any]) -> ModelResult:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"Missing environment variable {self.api_key_env}")
        task = payload["task"]
        if task in {"answer", "review", "host", "repair"}:
            original = payload.get("original_request", payload) if task == "repair" else payload
            labels = list(original["question"]["options"])
            label_text = json.dumps(labels, ensure_ascii=False)
            probability_shape = json.dumps({label: "number in [0,1]" for label in labels}, ensure_ascii=False)
            visible_ids = [item["chunk_id"] for item in original.get("evidence", [])]
            visible_ids += [item["id"] for item in original.get("messages", [])]
            system = (
                "Return exactly one JSON object. This is a research response, not clinical advice. "
                f"The only valid answer and claims[].answer values are option KEYS {label_text}; "
                "never put option text there. "
                f"probabilities must have exactly these keys and numeric values: {probability_shape}; "
                "every value must be between 0 and 1 and the values must sum to 1. "
                f"citations and claims[].citation_ids may contain only visible IDs {json.dumps(visible_ids, ensure_ascii=False)}. "
                "Return answer, probabilities, citations (array of IDs), claims (array of objects with "
                "text, answer, citation_ids, polarity, conditions), and opinion (short string). "
                "Each claim polarity must be support, contradict, limit, or uncertain; "
                "conditions must be one STRING (use an empty string if none), never an array. "
                "Put each distinct reason, limitation, and counterevidence in claims; "
                "opinion is only a short audit summary and must contain no unique reason. "
                "For host responses also return unexplained_contradiction and missing_distinction as booleans. "
                "Use only the provided question, options, visible evidence, and messages. "
                "Do not use a gold label or assume an option is correct from its key."
            )
            if task == "repair":
                system += (" Correct only the structure of original_response using validation_error. "
                           "Preserve its substantive judgment and every original numeric probability exactly. "
                           "Map option text to its unique key if needed; never change the selected option. "
                           "Do not add new medical reasoning or calculate a new probability vector. "
                           "If the response cannot be faithfully repaired, return an empty JSON object. "
                           "Return no commentary or wrapper.")
        else:
            system = ("Return exactly one JSON object. For semantic return "
                      "{\"relation\": one of equivalent, complementary, contradictory, condition_different, unknown}. "
                      "Equivalent requires "
                      "the same medical proposition, rationale, population, conditions and polarity, even "
                      "when wording or language differs. For semantic_batch return "
                      "{\"relations\":[one relation label per input pair in the same order]}. "
                      "Uncertain pairs are unknown. For recruit return "
                      "{\"roles\":[four distinct medical role strings]}. For novelty return "
                      "{\"relevant\":boolean,\"novel\":boolean,\"new_claim_or_condition\":boolean}; "
                      "the last flag is true only for a distinct relevant claim or condition. "
                      "For queries return {\"queries\":[zero to two strings]}. Use only supplied material.")
        request_body = {"model": self.name,
                        "temperature": 0 if payload["task"] in {"semantic", "semantic_batch", "novelty", "recruit", "queries"} else self.answer_temperature,
                        "max_tokens": self.max_tokens,
                        "response_format": {"type": "json_object"},
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
        if self.disable_thinking:
            request_body["thinking"] = {"type": "disabled"}
        url = self.base_url + "/chat/completions"
        request = urllib.request.Request(url, data=json.dumps(request_body).encode("utf-8"),
                                         headers={"Authorization": "Bearer " + key,
                                                  "Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
        try:
            content = json.loads(raw["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ModelResponseError("Model did not return valid JSON", raw, request_body) from exc
        if not isinstance(content, dict):
            raise ModelResponseError("Model JSON must be an object", raw, request_body)
        return ModelResult(content, raw, raw.get("usage") or {}, request_body)


def create_model(backend: str, model_name: str | None = None) -> Model:
    if backend == "mock":
        return MockModel()
    if backend == "deepseek":
        return OpenAICompatibleModel(model_name or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
                                     os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                                     "DEEPSEEK_API_KEY",
                                     answer_temperature=float(os.environ.get("MDT_ANSWER_TEMPERATURE", "0.3")),
                                     max_tokens=int(os.environ.get("MDT_MAX_OUTPUT_TOKENS", "800")),
                                     disable_thinking=True)
    if backend == "qwen":
        return OpenAICompatibleModel(model_name or os.environ.get("QWEN_MODEL", "qwen-plus"),
                                     os.environ.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                                     "QWEN_API_KEY",
                                     answer_temperature=float(os.environ.get("MDT_ANSWER_TEMPERATURE", "0.3")),
                                     max_tokens=int(os.environ.get("MDT_MAX_OUTPUT_TOKENS", "800")))
    raise ValueError(f"Unknown model backend {backend}")
