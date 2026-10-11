import copy

import pytest

from scripts.budget_resume_draft import (budget_limit_event, completed_ids, digest,
                                         new_scientific_config, reconcile_config)


OLD_CLI = "a60baa8e1c46eaa998280e141504c3bf9d416e42e1b80437585016bb9b9a9433"
APPROVED_NEW_CLI = "approved-new-cli-hash-fixture"


def configs():
    old = {"questions_sha256": "question-hash", "corpus_sha256": "corpus-hash",
           "model_name": "fixture-model", "model_temperature": 0.3,
           "experiment_plan_sha256": "plan-hash", "groups": "all", "repeats": 1,
           "code_sha256": {"cli.py": OLD_CLI, "runner.py": "same-core-hash",
                           "models.py": "same-prompt-hash"},
           "max_calls": 100, "max_tokens": 1000}
    old["fingerprint"] = digest(old)
    proposed = copy.deepcopy(old)
    proposed.pop("fingerprint")
    proposed["code_sha256"]["cli.py"] = APPROVED_NEW_CLI
    proposed["max_calls"] = 200
    proposed["max_tokens"] = 2000
    return old, proposed


def test_approved_budget_increase_retains_legacy_fingerprint_and_completed_rows():
    old, proposed = configs()
    retained, migration = reconcile_config(old, proposed, OLD_CLI, APPROVED_NEW_CLI)
    assert retained == old
    assert migration["old_cli_sha256"] == OLD_CLI
    assert migration["new_cli_sha256"] == APPROVED_NEW_CLI
    assert migration["legacy_fingerprint_retained"] == old["fingerprint"]
    rows = [{"run_id": "q:repeat-1:natural:ordinary", "status": "ok",
             "fingerprint": old["fingerprint"]}]
    assert completed_ids(rows, retained["fingerprint"]) == {rows[0]["run_id"]}
    assert rows[0]["fingerprint"] == old["fingerprint"]
    event = budget_limit_event(100, 1000, 200, 2000, 70, 800,
                               retained["fingerprint"], APPROVED_NEW_CLI)
    assert (event["consumed_calls_at_start"], event["consumed_known_tokens_at_start"]) == (70, 800)
    assert (event["old_max_calls"], event["new_max_calls"]) == (100, 200)


@pytest.mark.parametrize("path,value", [
    (("questions_sha256",), "other-questions"),
    (("model_temperature",), 0.7),
    (("experiment_plan_sha256",), "other-plan"),
    (("code_sha256", "runner.py"), "other-core"),
    (("code_sha256", "models.py"), "other-prompt"),
])
def test_scientific_change_is_rejected(path, value):
    old, proposed = configs()
    if len(path) == 1:
        proposed[path[0]] = value
    else:
        proposed[path[0]][path[1]] = value
    with pytest.raises(ValueError, match="Scientific"):
        reconcile_config(old, proposed, OLD_CLI, APPROVED_NEW_CLI)


def test_unknown_cli_change_and_tampered_legacy_fingerprint_rejected():
    old, proposed = configs()
    with pytest.raises(ValueError, match="not approved"):
        reconcile_config(old, proposed, OLD_CLI, "other-cli")
    old["fingerprint"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint"):
        reconcile_config(old, proposed, OLD_CLI, APPROVED_NEW_CLI)


def test_new_config_fingerprint_ignores_budget_limits():
    _, proposed = configs()
    first = new_scientific_config(proposed)
    proposed["max_calls"] = 9999
    proposed["max_tokens"] = 999999
    assert new_scientific_config(proposed) == first
    assert "max_calls" not in first and "max_tokens" not in first
