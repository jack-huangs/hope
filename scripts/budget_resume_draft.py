"""预算续跑兼容逻辑草稿；当前 CLI 尚未导入此文件。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


LEGACY_BUDGET_KEYS = {"max_calls", "max_tokens"}


def digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def scientific_part(config: dict[str, Any], *, ignore_cli_hash: bool = False) -> dict[str, Any]:
    """执行限额不属于实验条件；迁移审查另行核对 CLI 哈希。"""
    filtered = {key: value for key, value in config.items()
                if key not in LEGACY_BUDGET_KEYS | {"fingerprint"}}
    if ignore_cli_hash:
        filtered["code_sha256"] = {key: value for key, value in filtered["code_sha256"].items()
                                    if key != "cli.py"}
    return filtered


def new_scientific_config(proposed: dict[str, Any]) -> dict[str, Any]:
    config = scientific_part(proposed)
    config["fingerprint"] = digest(config)
    return config


def reconcile_config(existing: dict[str, Any] | None, proposed: dict[str, Any],
                     approved_legacy_cli_hash: str, approved_new_cli_hash: str
                     ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """返回实际沿用的 config 与可审计迁移记录；不改旧文件。"""
    wanted = new_scientific_config(proposed)
    if existing is None:
        return wanted, None
    if not LEGACY_BUDGET_KEYS.intersection(existing):
        if existing != wanted:
            raise ValueError("Existing scientific run configuration differs")
        return existing, None
    old_hash = existing.get("code_sha256", {}).get("cli.py")
    new_hash = proposed.get("code_sha256", {}).get("cli.py")
    if (old_hash, new_hash) != (approved_legacy_cli_hash, approved_new_cli_hash):
        raise ValueError("CLI migration hash pair is not approved")
    old_unsigned = {key: value for key, value in existing.items() if key != "fingerprint"}
    if existing.get("fingerprint") != digest(old_unsigned):
        raise ValueError("Legacy configuration fingerprint is invalid")
    if scientific_part(existing, ignore_cli_hash=True) != scientific_part(proposed, ignore_cli_hash=True):
        raise ValueError("Scientific parameter, data, model, prompt, or core code changed")
    migration = {"old_cli_sha256": old_hash, "new_cli_sha256": new_hash,
                 "legacy_fingerprint_retained": existing["fingerprint"],
                 "reason": "Only CLI budget accounting changed; all scientific fields matched exactly.",
                 "recorded_at_utc": datetime.now(timezone.utc).isoformat()}
    return existing, migration


def budget_limit_event(old_max_calls: int | None, old_max_tokens: int | None,
                       new_max_calls: int, new_max_tokens: int,
                       consumed_calls: int, consumed_tokens: int,
                       fingerprint: str, cli_sha256: str) -> dict[str, Any]:
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint, "cli_sha256": cli_sha256,
            "old_max_calls": old_max_calls, "old_max_tokens": old_max_tokens,
            "new_max_calls": new_max_calls, "new_max_tokens": new_max_tokens,
            "consumed_calls_at_start": consumed_calls,
            "consumed_known_tokens_at_start": consumed_tokens}


def completed_ids(results: list[dict[str, Any]], fingerprint: str) -> set[str]:
    """迁移后仍使用旧 fingerprint 跳过成功组。"""
    return {row["run_id"] for row in results if row.get("status") == "ok"
            and row.get("fingerprint") == fingerprint}
