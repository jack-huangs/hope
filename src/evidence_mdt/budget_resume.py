"""执行预算与科学配置分离；仅允许审定的旧 CLI 迁移。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


EXECUTION_FIELDS = {"max_calls", "max_tokens"}
EXECUTION_CODE = {"cli.py", "budget_resume.py"}


def digest(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def science(config: dict[str, Any], *, legacy_compare: bool = False) -> dict[str, Any]:
    result = {key: value for key, value in config.items()
              if key not in EXECUTION_FIELDS | {"fingerprint"}}
    if legacy_compare:
        result["code_sha256"] = {name: value for name, value in result["code_sha256"].items()
                                 if name not in EXECUTION_CODE}
    return result


def reconcile_config(existing: dict[str, Any] | None, proposed: dict[str, Any],
                     policy: dict[str, str], policy_sha256: str
                     ) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """新运行用无预算指纹；旧运行保留原文件与原指纹。"""
    proposed_science = science(proposed)
    wanted = {**proposed_science, "fingerprint": digest(proposed_science)}
    if existing is None:
        return wanted, None
    if not EXECUTION_FIELDS.intersection(existing):
        if existing != wanted:
            raise ValueError("Existing scientific run configuration differs")
        return existing, None
    old_code = existing.get("code_sha256", {})
    new_code = proposed.get("code_sha256", {})
    if "budget_resume.py" in old_code or set(new_code) != set(old_code) | {"budget_resume.py"}:
        raise ValueError("Legacy migration permits only the new budget helper")
    if (old_code.get("cli.py") != policy.get("old_cli_sha256")
            or new_code.get("cli.py") != policy.get("new_cli_sha256")
            or new_code.get("budget_resume.py") != policy.get("new_budget_helper_sha256")):
        raise ValueError("CLI budget migration hash pair is not approved")
    unsigned = {key: value for key, value in existing.items() if key != "fingerprint"}
    if existing.get("fingerprint") != digest(unsigned):
        raise ValueError("Legacy configuration fingerprint is invalid")
    if science(existing, legacy_compare=True) != science(proposed, legacy_compare=True):
        raise ValueError("Scientific parameters, data, model, prompt, or core code changed")
    migration = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                 "old_cli_sha256": old_code["cli.py"], "new_cli_sha256": new_code["cli.py"],
                 "new_budget_helper_sha256": new_code["budget_resume.py"],
                 "policy_sha256": policy_sha256,
                 "legacy_fingerprint_retained": existing["fingerprint"],
                 "reason": "Only reviewed CLI budget accounting changed; all scientific fields matched."}
    return existing, migration


def budget_event(old_calls: int | None, old_tokens: int | None,
                 new_calls: int, new_tokens: int, consumed_calls: int, consumed_tokens: int,
                 fingerprint: str, cli_sha256: str) -> dict[str, Any]:
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint, "cli_sha256": cli_sha256,
            "old_max_calls": old_calls, "old_max_tokens": old_tokens,
            "new_max_calls": new_calls, "new_max_tokens": new_tokens,
            "consumed_calls_at_start": consumed_calls,
            "consumed_known_tokens_at_start": consumed_tokens}
