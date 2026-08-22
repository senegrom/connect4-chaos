#!/usr/bin/env python3
"""Focused fail-closed tests for cancelled exact preparation recovery."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("perfect-chaos-main-cancelled-recovery.py")
SPEC = importlib.util.spec_from_file_location(
    "perfect_chaos_main_cancelled_recovery",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RECOVERY = MODULE.RECOVERY


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


def run(role: str, conclusion: str = "cancelled") -> dict[str, object]:
    state_path = f".campaign/perfect-chaos-main-18/{role}.json"
    return {
        "id": 98765,
        "run_attempt": 1,
        "path": RECOVERY.CONTINUE_PATH,
        "head_branch": "main",
        "head_sha": "b" * 40,
        "status": "completed",
        "conclusion": conclusion,
        "event": "workflow_dispatch",
        "display_title": f"Continue Perfect Chaos 18-piece — {state_path}",
    }


def jobs(prepare_conclusion: str = "cancelled") -> dict[str, object]:
    return {
        "jobs": [
            {
                "id": 111,
                "name": RECOVERY.PREPARE_JOB,
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
    run_payload: dict[str, object] | None = None,
    jobs_payload: dict[str, object] | None = None,
    run_bytes: bytes | None = None,
):
    value = state(role, shards, workers)
    current = encoded(value)
    return MODULE.decide_cancelled_recovery(
        run_payload or run(role),
        jobs_payload or jobs(),
        value,
        current,
        current if run_bytes is None else run_bytes,
    )


def test_cancelled_preparation_doubles_sharding() -> None:
    decision, updated = decide("yellow")
    assert decision["format"] == MODULE.FORMAT
    assert decision["action"] == "adapted"
    assert decision["runConclusion"] == "cancelled"
    assert decision["prepareConclusion"] == "cancelled"
    assert decision["normalisedConclusion"] == "failure"
    assert decision["oldProfile"] == {"prepareShards": 64, "prepareWorkers": 4}
    assert decision["newProfile"] == {"prepareShards": 128, "prepareWorkers": 4}
    assert updated is not None
    assert updated["prepareShards"] == 128
    assert updated["prepareWorkers"] == 4
    assert updated["sourceRun"] == 12345
    assert updated["cumulativeRejections"] == 200


def test_stale_cancelled_run_cannot_mutate_current_state() -> None:
    value = state("red")
    stale = encoded({**value, "cumulativeRejections": 201})
    decision, updated = decide("red", run_bytes=stale)
    assert decision["action"] == "stale-state"
    assert decision["handled"] is False
    assert updated is None


def test_exhausted_cancelled_profile_remains_terminal() -> None:
    decision, updated = decide("yellow", shards=256, workers=1)
    assert decision["action"] == "exhausted"
    assert decision["handled"] is True
    assert updated is None


def test_failed_workflow_is_not_mislabelled_as_cancelled() -> None:
    expect_failure(
        lambda: decide("red", run_payload=run("red", conclusion="failure")),
        "completed cancelled workflow run",
    )


def test_unrelated_cancellation_is_not_adapted() -> None:
    expect_failure(
        lambda: decide("red", jobs_payload=jobs(prepare_conclusion="success")),
        "preparation job itself",
    )


def test_missing_prepare_job_is_rejected() -> None:
    expect_failure(
        lambda: decide(
            "yellow",
            jobs_payload={
                "jobs": [
                    {
                        "id": 112,
                        "name": "load",
                        "status": "completed",
                        "conclusion": "cancelled",
                    }
                ]
            },
        ),
        "exactly one",
    )


def test_duplicate_prepare_identity_is_rejected() -> None:
    duplicate = jobs()["jobs"] * 2
    expect_failure(
        lambda: decide("yellow", jobs_payload={"jobs": duplicate}),
        "exactly one",
    )


def main() -> None:
    tests = [
        test_cancelled_preparation_doubles_sharding,
        test_stale_cancelled_run_cannot_mutate_current_state,
        test_exhausted_cancelled_profile_remains_terminal,
        test_failed_workflow_is_not_mislabelled_as_cancelled,
        test_unrelated_cancellation_is_not_adapted,
        test_missing_prepare_job_is_rejected,
        test_duplicate_prepare_identity_is_rejected,
    ]
    for case in tests:
        case()
    print({"tests": len(tests), "status": "pass"})


if __name__ == "__main__":
    main()
