"""本地只读模型对话查看器：python scripts/log_viewer.py，浏览器访问127.0.0.1:8765。"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).with_name("log_viewer_static")
SHARED_TASKS = {"recruit"}


def redact(value: Any) -> Any:
    """请求头不参与展示；保留token统计，仅遮蔽凭据字段。"""
    if isinstance(value, dict):
        return {k: "[已隐藏]" if re.fullmatch(
            r"authorization|password|api[_-]?key|access[_-]?token|secret", k, re.I)
                else redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def request_data(row: dict) -> dict:
    payload = row.get("request") or {}
    return payload if isinstance(payload, dict) else {}


def identity(row: dict) -> tuple[str, str, int | str | None]:
    payload = request_data(row)
    original = payload.get("original_request") or payload
    question = original.get("question") or payload.get("question") or {}
    rid = str(row.get("run_id") or "")
    qid = str(row.get("question_id") or question.get("question_id") or "")
    match = re.match(r"^(.*?):repeat-([^:]+)(?::|$)", rid)
    if match:
        qid = qid or match[1]
    repeat = row.get("repeat_id") or (match[2] if match else None)
    if not qid and rid:
        qid = rid.split(":", 1)[0]
    if not rid:
        rid = f"{qid}:repeat-{repeat or 1}:direct"
    return qid, rid, repeat


def is_recruit(row: dict) -> bool:
    return request_data(row).get("task") in SHARED_TASKS or str(row.get("run_id", "")).endswith(":recruit")


def small_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except (ValueError, UnicodeError):
                continue
    return rows


@dataclass
class BatchIndex:
    signature: tuple[int, ...]
    entries: list[dict]
    cases: dict[str, dict]
    warnings: list[str]


class LogStore:
    def __init__(self, runs_root: Path):
        self.root = runs_root.resolve()
        self.cache: dict[str, BatchIndex] = {}
        self.event_cache: dict[str, tuple[tuple[int, int], dict[str, list[dict]], list[str]]] = {}
        self.lock = threading.RLock()

    def batch_path(self, name: str) -> Path:
        # 仅接受runs的直接子目录，不提供通用文件读取接口。
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("无效的实验批次")
        target = (self.root / name).resolve()
        if target.parent != self.root or not target.is_dir():
            raise ValueError("实验批次不存在")
        return target

    @staticmethod
    def log_path(folder: Path) -> Path | None:
        if (folder / "calls.jsonl").is_file():
            return folder / "calls.jsonl"
        direct = folder / "results.jsonl"
        if direct.is_file():
            with direct.open(encoding="utf-8-sig") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    return direct if isinstance(row, dict) and "request" in row else None
        return None

    def batches(self) -> list[dict]:
        found = []
        if not self.root.is_dir():
            return found
        for folder in self.root.iterdir():
            if not folder.is_dir() or folder.resolve().parent != self.root:
                continue
            path = self.log_path(folder)
            if path:
                stat = path.stat()
                found.append({"id": folder.name, "name": folder.name,
                              "modified": stat.st_mtime, "bytes": stat.st_size,
                              "log_file": path.name})
        return sorted(found, key=lambda b: b["modified"], reverse=True)

    def index(self, name: str) -> BatchIndex:
        folder = self.batch_path(name)
        path = self.log_path(folder)
        if path is None:
            raise ValueError("此批次没有模型调用日志")
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        for related in (folder / "results.jsonl", folder / "failures.jsonl"):
            info = related.stat() if related.is_file() else None
            signature += (info.st_mtime_ns, info.st_size) if info else (0, 0)
        with self.lock:
            if name in self.cache and self.cache[name].signature == signature:
                return self.cache[name]
            entries: list[dict] = []
            cases: dict[str, dict] = {}
            recruits: dict[str, dict] = {}
            warnings = []
            with path.open("rb") as f:
                line_no = 0
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    line_no += 1
                    # 正在写入的半行下一次刷新再读，绝不修改日志。
                    if not line.endswith(b"\n"):
                        try:
                            json.loads(line)
                        except (ValueError, UnicodeError):
                            warnings.append(f"第 {line_no} 行尚未写完，已暂时跳过")
                            break
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError()
                    except (ValueError, UnicodeError):
                        warnings.append(f"第 {line_no} 行无法解析，已跳过")
                        continue
                    qid, rid, repeat = identity(row)
                    payload = request_data(row)
                    original = payload.get("original_request") or payload
                    question = original.get("question") or {}
                    if not isinstance(question, dict):
                        question = {}
                    choices = (row.get("response") or {}).get("choices") or []
                    finish = choices[0].get("finish_reason") if choices else row.get("finish_reason")
                    usage = row.get("usage") or (row.get("response") or {}).get("usage") or {}
                    shared = is_recruit(row)
                    entry = {"line": line_no, "offset": offset, "length": len(line), "run_id": rid,
                             "question_id": qid, "repeat_id": repeat,
                             "task": payload.get("task", "direct"), "role": original.get("role", ""),
                             "round": original.get("round_id"), "status": row.get("status", "unknown"),
                             "finish_reason": finish, "elapsed_seconds": row.get("elapsed_seconds"),
                             "usage": usage, "started_at": row.get("started_at"),
                             "attempt_id": row.get("attempt_id"), "shared": shared}
                    entries.append(entry)
                    if shared:
                        recruits[rid] = {"batch":name, "run_id":rid, "question_id":qid,
                                         "repeat_id":repeat,"stem":question.get("stem", ""),
                                         "dataset":question.get("dataset", ""),"calls":0,
                                         "anomalies":0,"status":"logged","answer":None,
                                         "gold":None,"attempt_ids":[]}
                        continue
                    if rid not in cases:
                        cases[rid] = {"batch": name, "run_id": rid, "question_id": qid,
                                      "repeat_id": repeat, "stem": question.get("stem", ""),
                                      "dataset": question.get("dataset", ""), "calls": 0,
                                      "anomalies": 0, "status": "logged", "answer": None,
                                      "gold": None, "attempt_ids": []}
                    case = cases[rid]
                    case["calls"] += 1
                    case["anomalies"] += row.get("status") not in {"ok", None} or finish == "length"
                    if row.get("attempt_id") and row["attempt_id"] not in case["attempt_ids"]:
                        case["attempt_ids"].append(row["attempt_id"])
                    if row.get("gold") is not None:
                        case["gold"] = row["gold"]
                    if path.name == "results.jsonl":
                        case["status"] = "complete" if row.get("status") == "ok" else "failed"
                        case["answer"] = row.get("answer")
            # 仅招募后就中断的题，也应能查到已有调用。
            for rid, recruit in recruits.items():
                if not any(c["question_id"] == recruit["question_id"] and
                           str(c["repeat_id"]) == str(recruit["repeat_id"]) for c in cases.values()):
                    cases[rid] = recruit
            # 结果与失败按run_id对齐；保留所有attempt，成功优先表示该run最终已有结果。
            for row in small_jsonl(folder / "failures.jsonl"):
                qid, rid, _ = identity(row)
                if rid in cases:
                    cases[rid]["status"] = "failed"
            if path.name != "results.jsonl":
                for row in small_jsonl(folder / "results.jsonl"):
                    _, rid, _ = identity(row)
                    if rid in cases:
                        cases[rid].update(status="complete", answer=row.get("answer"),
                                          gold=row.get("gold"), dataset=row.get("dataset", ""))
            for case in cases.values():
                case["shared_calls"] = sum(e["shared"] and e["question_id"] == case["question_id"]
                                           and str(e["repeat_id"]) == str(case["repeat_id"]) for e in entries)
            index = BatchIndex(signature, entries, cases, warnings)
            self.cache[name] = index
            return index

    def search(self, query: str = "", batch: str = "all") -> dict:
        names = [b["id"] for b in self.batches()] if batch == "all" else [batch]
        result = []
        warnings = []
        query = query.strip().lower()
        for name in names:
            idx = self.index(name)
            warnings.extend(f"{name}：{w}" for w in idx.warnings)
            for case in idx.cases.values():
                if (not query or query == case["question_id"].lower()
                        or (not query.isdecimal() and (query in case["run_id"].lower()
                            or query in case["stem"].lower() or query in case["dataset"].lower()))):
                    result.append(case)
        return {"cases": result[:400], "total": len(result), "warnings": warnings[:20],
                "limited": len(result) > 400}

    def call(self, name: str, line: int) -> dict:
        idx = self.index(name)
        entry = next((e for e in idx.entries if e["line"] == line), None)
        if entry is None:
            raise ValueError("找不到该次调用")
        path = self.log_path(self.batch_path(name))
        with path.open("rb") as f:
            f.seek(entry["offset"])
            return redact(json.loads(f.read(entry["length"])))

    def case(self, name: str, run_id: str) -> dict:
        idx = self.index(name)
        meta = idx.cases.get(run_id)
        if meta is None:
            raise ValueError("找不到该run_id；可先按题号搜索")
        selected = [e for e in idx.entries if e["run_id"] == run_id or
                    (e["shared"] and e["question_id"] == meta["question_id"] and
                     str(e["repeat_id"]) == str(meta["repeat_id"]))]
        first = next((e for e in selected if not e["shared"]), selected[0])
        row = self.call(name, first["line"])
        payload = request_data(row)
        original = payload.get("original_request") or payload
        question = original.get("question") or {}
        # 直接问答脚本未单独存题目；实际消息中仍完整保存了题干与选项。
        folder = self.batch_path(name)
        results = [r for r in small_jsonl(folder / "results.jsonl") if identity(r)[1] == run_id]
        failures = [r for r in small_jsonl(folder / "failures.jsonl") if identity(r)[1] == run_id]
        has_unknown_attempt = any(not e["attempt_id"] for e in selected if not e["shared"])
        return {"meta": meta, "question": question, "calls": selected,
                "results": redact(results), "failures": redact(failures), "warnings": idx.warnings,
                "attempt_note": ("该批calls未记录attempt_id；以下保留同一run_id的全部调用顺序，不强行归入某次重试。"
                                 if has_unknown_attempt else "调用按原始日志顺序展示，并保留各自的attempt_id。"),
                "totals": {"calls": len(selected), "shared_calls": sum(e["shared"] for e in selected),
                           "tokens": sum((e["usage"].get("total_tokens") or 0) for e in selected)},
                "source": f"runs/{name}/{self.log_path(folder).name}"}

    def graph(self, name: str, run_id: str, attempt: str = "") -> dict:
        # Graph building is deliberately separate from experiment code and never invokes a model.
        try:
            from log_viewer_graph import build_graph
        except ModuleNotFoundError:
            from scripts.log_viewer_graph import build_graph
        case = self.case(name, run_id)
        path = self.batch_path(name) / "events.jsonl"
        rows, warnings = [], []
        if path.is_file():
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            with self.lock:
                cached = self.event_cache.get(name)
                if cached is None or cached[0] != signature:
                    entries: dict[str, list[dict]] = {}
                    notes = []
                    with path.open("rb") as f:
                        line = 0
                        while True:
                            offset = f.tell()
                            raw = f.readline()
                            if not raw:
                                break
                            line += 1
                            try:
                                row = json.loads(raw)
                                if not isinstance(row, dict):
                                    raise ValueError()
                            except (ValueError, UnicodeError):
                                notes.append(f"events 第 {line} 行未完成或无法解析，已跳过")
                                continue
                            rid = str(row.get("run_id", ""))
                            entries.setdefault(rid, []).append({"offset": offset, "length": len(raw), "line": line})
                    cached = (signature, entries, notes)
                    self.event_cache[name] = cached
                warnings = cached[2][:20]
                with path.open("rb") as f:
                    for entry in cached[1].get(run_id, []):
                        f.seek(entry["offset"])
                        rows.append({**json.loads(f.read(entry["length"])), "_line": entry["line"]})
        calls = [(entry, self.call(name, entry["line"])) for entry in case["calls"] if not entry["shared"]]
        return redact(build_graph(case, rows, calls, attempt, warnings))


def make_handler(store: LogStore):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, body: bytes, kind: str, status=200):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            def arg(key, default=""):
                return query.get(key, [default])[0]
            try:
                if parsed.path == "/api/batches":
                    data = {"batches": store.batches()}
                elif parsed.path == "/api/search":
                    data = store.search(arg("q"), arg("batch", "all"))
                elif parsed.path == "/api/case":
                    data = store.case(arg("batch"), arg("run_id"))
                elif parsed.path == "/api/call":
                    data = store.call(arg("batch"), int(arg("line", "0")))
                elif parsed.path == "/api/graph":
                    data = store.graph(arg("batch"), arg("run_id"), arg("attempt"))
                elif parsed.path in {"/", "/index.html", "/app.js", "/graph.js", "/style.css", "/graph.css"}:
                    filename = "index.html" if parsed.path == "/" else parsed.path[1:]
                    kind = {"html":"text/html; charset=utf-8", "js":"text/javascript; charset=utf-8", "css":"text/css; charset=utf-8"}[filename.rsplit(".", 1)[1]]
                    self.respond((ASSETS / filename).read_bytes(), kind)
                    return
                else:
                    self.respond(b"Not found", "text/plain", 404)
                    return
                self.respond(json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self.respond(json.dumps({"error":str(exc)},ensure_ascii=False).encode(), "application/json; charset=utf-8",400)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(LogStore(args.runs)))
    print(f"模型对话查看器：http://127.0.0.1:{args.port} （只读本地日志）", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
