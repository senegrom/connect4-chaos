#!/usr/bin/env python3
"""Focused fail-closed tests for perfect-chaos-main-recovery.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("perfect-chaos-main-recovery.py")
SPEC = importlib.util.spec_from_file_location("perfect_chaos_main_recovery", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def state(role: str, shards: int = 64, workers: int = 4) -> dict[str, object]:
    return {
        "role": role,
        "sourceRun": 12345,
        "sourceSha": "a" * 40,
        "sourceArtifact": f"perfect-chaos-{role}-18-source-round",
        "existingRejections": 100,
        "cumulativeRejections": 200,
        "prepareShards": shards,
        "prepareWorkers": workers,
        "shardCount": 256,
    }


def encoded(value: dict[str, object]) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def run(
    role: str,
    *,
    event: str = "workflow_dispatch",
    title: str | None = None,
) -> dict[str, object]:
    state_path = f".campaign/perfect-chaos-main-18/{role}.json"
    return {
        "id": 98765,
        "run_attempt": 2,
        "path": MODULE.CONTINUE_PATH,
        "head_branch": "main",
        "head_sha": "b" * 40,
        "status": "completed",
        "conclusion": "failure",
        "event": event,
        "display_title": title or f"Continue Perfect Chaos 18-piece — {state_path}",
    }


def jobs(*, prepare_conclusion: str = "failure") -> dict[str, object]:
    return {
        "jobs": [
            {
                "id": 111,
                "name": MODULE.PREPARE_JOB,
                "status": "completed",
                "conclusion": prepare_conclusion,
            },
            {
                "id": 112,
                "name": "load",
                "status": "completed",
                "conclusion": "success",
            },
        ]
    }


def jobs_without_prepare() -> dict[str, object]:
    return {
        "jobs": [
            {
                "id": 112,
                "name": "load",
                "status": "completed",
                "conclusion": "failure",
            }
        ]
    }


def expect_failure(function, pattern: str) -> None:
    try:
        function()
    except RuntimeError as error:
        if pattern not in str(error):
            raise AssertionError(f"Expected {pattern!r} in {error!r}") from error
    else:
        raise AssertionError(f"Expected RuntimeError containing {pattern!r}")


def decide(
    role: str,
    *,
    shards: int = 64,
    workers: int = 4,
    current_bytes: bytes | None = None,
    run_bytes: bytes | None = None,
    jobs_payload: dict[str, object] | None = None,
    run_payload: dict[str, object] | None = None,
    changed_paths: list[str] | None = None,
):
    value = state(role, shards, workers)
    exact = encoded(value)
    return MODULE.decide_recovery(
        run_payload or run(role),
        jobs_payload or jobs(),
        value,
        exact if current_bytes is None else current_bytes,
        exact if run_bytes is None else run_bytes,
        changed_paths=changed_paths,
    )


def test_failed_preparation_doubles_sharding() -> None:
    decision, updated = decide("yellow")
    assert decision["action"] == "adapted"
    assert decision["handled"] is True
    assert decision["oldProfile"] == {"prepareShards": 64, "prepareWorkers": 4}
    assert decision["newProfile"] == {"prepareShards": 128, "prepareWorkers": 4}
    assert updated is not None
    assert updated["prepareShards"] == 128
    assert updated["prepareWorkers"] == 4
    assert updated["sourceRun"] == 12345
    assert updated["cumulativeRejections"] == 200


def test_shards_are_capped_before_workers_drop() -> None:
    decision, updated = decide("red", shards=200, workers=8)
    assert decision["newProfile"] == {"prepareShards": 256, "prepareWorkers": 8}
    assert updated is not None and updated["prepareShards"] == 256


def test_workers_drop_only_after_maximum_sharding() -> None:
    decision, updated = decide("red", shards=256, workers=8)
    assert decision["newProfile"] == {"prepareShards": 256, "prepareWorkers": 4}
    assert updated is not None and updated["prepareWorkers"] == 4


def test_exhausted_profile_blocks_identical_retry() -> None:
    decision, updated = decide("yellow", shards=256, workers=1)
    assert decision["action"] == "exhausted"
    assert decision["handled"] is True
    assert updated is None


def test_stale_state_is_not_adapted() -> None:
    value = state("red")
    current = encoded(value)
    stale = encoded({**value, "cumulativeRejections": 201})
    decision, updated = MODULE.decide_recovery(
        run("red"), jobs(), value, current, stale
    )
    assert decision["action"] == "stale-state"
    assert decision["handled"] is False
    assert updated is None


def test_non_prepare_failure_is_not_adapted() -> None:
    decision, updated = decide("red", jobs_payload=jobs(prepare_conclusion="success"))
    assert decision["action"] == "not-prepare-failure"
    assert decision["handled"] is False
    assert updated is None


def test_failure_before_prepare_exists_is_retryable() -> None:
    decision, updated = decide("yellow", jobs_payload=jobs_without_prepare())
    assert decision["action"] == "not-prepare-failure"
    assert decision["handled"] is False
    assert updated is None


def test_push_run_binds_to_exact_changed_role_state() -> None:
    payload = run("yellow", event="push", title="Advance Yellow proof state")
    decision, updated = decide(
        "yellow",
        run_payload=payload,
        changed_paths=[".campaign/perfect-chaos-main-18/yellow.json"],
    )
    assert decision["role"] == "yellow"
    assert decision["action"] == "adapted"
    assert updated is not None


def test_ambiguous_dispatched_title_fails_closed() -> None:
    title = (
        ".campaign/perfect-chaos-main-18/red.json and "
        ".campaign/perfect-chaos-main-18/yellow.json"
    )
    expect_failure(
        lambda: decide("red", run_payload=run("red", title=title)),
        "exactly one role-state path",
    )


def test_ambiguous_push_change_fails_closed() -> None:
    payload = run("red", event="push", title="Advance campaign")
    expect_failure(
        lambda: decide(
            "red",
            run_payload=payload,
            changed_paths=[
                ".campaign/perfect-chaos-main-18/red.json",
                ".campaign/perfect-chaos-main-18/yellow.json",
            ],
        ),
        "exactly one role-state file",
    )


def test_state_schema_cannot_smuggle_proof_changes() -> None:
    value = state("red")
    value["unexpected"] = True
    exact = encoded(value)
    expect_failure(
        lambda: MODULE.decide_recovery(run("red"), jobs(), value, exact, exact),
        "State schema mismatch",
    )


def main() -> None:
    tests = [
        test_failed_preparation_doubles_sharding,
        test_shards_are_capped_before_workers_drop,
        test_workers_drop_only_after_maximum_sharding,
        test_exhausted_profile_blocks_identical_retry,
        test_stale_state_is_not_adapted,
        test_non_prepare_failure_is_not_adapted,
        test_failure_before_prepare_exists_is_retryable,
        test_push_run_binds_to_exact_changed_role_state,
        test_ambiguous_dispatched_title_fails_closed,
        test_ambiguous_push_change_fails_closed,
        test_state_schema_cannot_smuggle_proof_changes,
    ]
    for test in tests:
        test()
    print({"tests": len(tests), "status": "pass"})


if __name__ == "__main__":
    main()
