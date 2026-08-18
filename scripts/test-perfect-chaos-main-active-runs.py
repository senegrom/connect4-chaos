#!/usr/bin/env python3
"""Focused tests for perfect-chaos-main-active-runs.py."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("perfect-chaos-main-active-runs.py")
SPEC = importlib.util.spec_from_file_location("perfect_chaos_main_active_runs", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

RED_SHA = "a" * 40
YELLOW_SHA = "b" * 40
OTHER_SHA = "c" * 40


def run(
    run_id: int,
    workflow: str,
    *,
    event: str,
    head_sha: str,
    display_title: str,
    status: str = "in_progress",
) -> dict[str, object]:
    return {
        "id": run_id,
        "path": f".github/workflows/{workflow}",
        "event": event,
        "status": status,
        "head_branch": "main",
        "head_sha": head_sha,
        "display_title": display_title,
        "created_at": "2026-08-18T00:00:00Z",
        "updated_at": "2026-08-18T00:01:00Z",
    }


def payload(*records: tuple[str, list[dict[str, object]]]) -> dict[str, object]:
    values = {workflow: runs for workflow, runs in records}
    return {
        "workflows": [
            {"workflow": workflow, "runs": values.get(workflow, [])}
            for workflow in MODULE.WORKFLOWS
        ]
    }


def classify(value: dict[str, object]) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as directory:
        return MODULE.classify_active_runs(
            Path(directory),
            value,
            state_commits={"red": RED_SHA, "yellow": YELLOW_SHA},
        )


def expect_failure(function, pattern: str) -> None:
    try:
        function()
    except RuntimeError as error:
        if pattern not in str(error):
            raise AssertionError(f"Expected {pattern!r} in {error!r}") from error
    else:
        raise AssertionError(f"Expected RuntimeError containing {pattern!r}")


def test_push_runs_bind_to_exact_state_commits() -> None:
    result = classify(
        payload(
            (
                MODULE.CONTINUE_WORKFLOW,
                [
                    run(
                        10,
                        MODULE.CONTINUE_WORKFLOW,
                        event="push",
                        head_sha=RED_SHA,
                        display_title="Start Red 18-piece refinement",
                    ),
                    run(
                        11,
                        MODULE.CONTINUE_WORKFLOW,
                        event="push",
                        head_sha=YELLOW_SHA,
                        display_title="Start Yellow 18-piece refinement",
                    ),
                ],
            )
        )
    )
    assert result["activeRoles"] == ["red", "yellow"]
    continuations = next(
        item for item in result["workflows"] if item["workflow"] == MODULE.CONTINUE_WORKFLOW
    )
    assert continuations["active"][0]["roles"] == ["red"]
    assert continuations["active"][1]["roles"] == ["yellow"]


def test_dispatched_run_requires_exact_state_path_in_title() -> None:
    title = (
        "Continue Perfect Chaos 18-piece — "
        ".campaign/perfect-chaos-main-18/yellow.json"
    )
    result = classify(
        payload(
            (
                MODULE.CONTINUE_WORKFLOW,
                [
                    run(
                        20,
                        MODULE.CONTINUE_WORKFLOW,
                        event="workflow_dispatch",
                        head_sha=OTHER_SHA,
                        display_title=title,
                    )
                ],
            )
        )
    )
    assert result["activeRoles"] == ["yellow"]


def test_bootstrap_run_reserves_both_roles() -> None:
    workflow = sorted(MODULE.BOOTSTRAP_WORKFLOWS)[0]
    result = classify(
        payload(
            (
                workflow,
                [
                    run(
                        30,
                        workflow,
                        event="workflow_run",
                        head_sha=OTHER_SHA,
                        display_title="Bootstrap exact 18-piece refinement",
                    )
                ],
            )
        )
    )
    assert result["activeRoles"] == ["red", "yellow"]


def test_completed_runs_are_ignored() -> None:
    result = classify(
        payload(
            (
                MODULE.CONTINUE_WORKFLOW,
                [
                    run(
                        40,
                        MODULE.CONTINUE_WORKFLOW,
                        event="push",
                        head_sha=RED_SHA,
                        display_title="Completed Red run",
                        status="completed",
                    )
                ],
            )
        )
    )
    assert result["activeRoles"] == []


def test_unknown_push_sha_fails_closed() -> None:
    expect_failure(
        lambda: classify(
            payload(
                (
                    MODULE.CONTINUE_WORKFLOW,
                    [
                        run(
                            50,
                            MODULE.CONTINUE_WORKFLOW,
                            event="push",
                            head_sha=OTHER_SHA,
                            display_title="Unknown continuation",
                        )
                    ],
                )
            )
        ),
        "match exactly one current role-state commit",
    )


def test_ambiguous_dispatch_title_fails_closed() -> None:
    title = (
        ".campaign/perfect-chaos-main-18/red.json and "
        ".campaign/perfect-chaos-main-18/yellow.json"
    )
    expect_failure(
        lambda: classify(
            payload(
                (
                    MODULE.CONTINUE_WORKFLOW,
                    [
                        run(
                            60,
                            MODULE.CONTINUE_WORKFLOW,
                            event="workflow_dispatch",
                            head_sha=OTHER_SHA,
                            display_title=title,
                        )
                    ],
                )
            )
        ),
        "exactly one exact role-state path",
    )


def test_missing_workflow_record_fails_closed() -> None:
    value = payload()
    value["workflows"] = value["workflows"][:-1]
    expect_failure(lambda: classify(value), "missing supervised workflows")


def main() -> None:
    tests = [
        test_push_runs_bind_to_exact_state_commits,
        test_dispatched_run_requires_exact_state_path_in_title,
        test_bootstrap_run_reserves_both_roles,
        test_completed_runs_are_ignored,
        test_unknown_push_sha_fails_closed,
        test_ambiguous_dispatch_title_fails_closed,
        test_missing_workflow_record_fails_closed,
    ]
    for test in tests:
        test()
    print({"tests": len(tests), "status": "pass"})


if __name__ == "__main__":
    main()
