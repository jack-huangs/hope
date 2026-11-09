"""原子检查点与内容寻址缓存；时间仅作审计，不参与依赖指纹。"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


def jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    temporary.replace(path)


class ConfigurationError(RuntimeError):
    """认证和配置错误必须停止整个准备任务。"""


def fatal(exc):
    return isinstance(exc, ConfigurationError) or (
        isinstance(exc, urllib.error.HTTPError) and exc.code in {400, 401, 403, 404, 422})


class Cache:
    def __init__(self, root: Path, retries=2):
        self.root, self.retries = root, retries

    def get(self, key, inputs, operation):
        fingerprint = digest(inputs)
        path = self.root / (digest(key) + ".json")
        if path.exists():
            row = json.loads(path.read_bytes())
            if row["input_sha256"] == fingerprint and row.get("status") == "ok":
                if digest(row["value"]) != row["value_sha256"]:
                    raise ConfigurationError(f"Cache content hash mismatch: {key}")
                return row["value"]
        # 失败不伪装成零命中；每次尝试即时写盘，续跑只重试未完成步骤。
        for attempt in range(self.retries + 1):
            started = time.time()
            try:
                value = operation()
                save(path, {"input_sha256": fingerprint, "status": "ok", "value": value,
                            "value_sha256": digest(value)})
                return value
            except Exception as exc:
                log = {"key": key, "attempt": attempt + 1, "at": started,
                       "error_type": type(exc).__name__, "error": str(exc)[:500]}
                if hasattr(exc, "raw"):
                    log.update(raw=exc.raw, wire_request=exc.wire_request)
                self.root.mkdir(parents=True, exist_ok=True)
                with (self.root / "failures.jsonl").open("ab") as stream:
                    stream.write(canonical(log) + b"\n")
                if fatal(exc):
                    raise ConfigurationError(str(exc)) from exc
                if attempt == self.retries:
                    raise
                time.sleep(min(2 ** attempt, 4))
