#!/usr/bin/env python3
"""Validate an in-place retry of one pinned 7×7 Classic certificate run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, NoReturn

FORMAT = "connect4-perfect-classic-7x7-failed-run-recovery-v1"
LAUNCH_FORMAT = "connect4-perfect-classic-7x7-launch-v1"
SHA_RE = re.compile(r"[0-9a-f]{40}")
ROLE_WORKFLOWS = {
    "role1": {
        "name": "Compute perfect classic 7x7 role 1 certificate",
        "path": ".github/workflows/compute-perfect-classic-7x7-role1.yml",
    },
    "role2": {
        "name": "Compute perfect classic 7x7 role 2 certificate",
        "path": ".github/workflows/compute-perfect-classic-7x7-role2.yml",
    },
}


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def require_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        fail(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def load_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"Could not parse {label} {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{label} must contain a JSON object: {path}")
    return value


def extract_run(event: dict[str, Any]) -> dict[str, Any]:
    run = event.get("workflow_run", event)
    if not isinstance(run, dict):
        fail("Event must contain a workflow_run object")
    return run


def validate_launch(launch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if launch.get("format") != LAUNCH_FORMAT:
        fail(f"launch.format must be {LAUNCH_FORMAT!r}")
    source_sha = launch.get("sourceSha")
    if not isinstance(source_sha, str) or SHA_RE.fullmatch(source_sha) is None:
        fail("launch.sourceSha must be a lowercase 40-character commit SHA")
    roles = launch.get("roles")
    if not isinstance(roles, dict) or set(roles) != set(ROLE_WORKFLOWS):
        fail("launch.roles must contain exactly role1 and role2")

    validated: dict[str, dict[str, Any]] = {}
    run_ids: set[int] = set()
    for role in ROLE_WORKFLOWS:
        record = roles[role]
        if not isinstance(record, dict):
            fail(f"launch.roles.{role} must be an object")
        run_id = require_integer(record.get("id"), f"launch.roles.{role}.id", 1, 10**15)
        if run_id in run_ids:
            fail("The two launch roles cannot share one workflow run id")
        run_ids.add(run_id)
        head_sha = record.get("head_sha")
        if head_sha != source_sha:
            fail(f"launch.roles.{role}.head_sha must equal launch.sourceSha")
        validated[role] = {
            "runId": run_id,
            "headSha": head_sha,
        }
    return validated


def decide_recovery(
    launch: dict[str, Any],
    event: dict[str, Any],
    *,
    maximum_attempts: int,
) -> dict[str, Any]:
    require_integer(maximum_attempts, "maximum_attempts", 1, 20)
    pinned = validate_launch(launch)
    run = extract_run(event)
    run_id = require_integer(run.get("id"), "workflow_run.id", 1, 10**15)

    role_matches = [
        role
        for role, workflow in ROLE_WORKFLOWS.items()
        if run.get("name") == workflow["name"]
        or run.get("path") == workflow["path"]
    ]
    if not role_matches:
        return {
            "format": FORMAT,
            "action": "ignore",
            "reason": "unsupported-workflow",
            "runId": run_id,
        }
    if len(role_matches) != 1:
        fail(f"Workflow identity is ambiguous: matches={role_matches}")
    role = role_matches[0]
    workflow = ROLE_WORKFLOWS[role]
    if run.get("name") != workflow["name"] or run.get("path") != workflow["path"]:
        fail(f"{role}: workflow name and path do not identify the same pinned calculation")

    base = {
        "format": FORMAT,
        "role": role,
        "runId": run_id,
        "headSha": pinned[role]["headSha"],
        "workflowName": workflow["name"],
        "workflowPath": workflow["path"],
    }
    if run_id != pinned[role]["runId"]:
        fail(
            f"{role}: event run {run_id} does not match pinned run "
            f"{pinned[role]['runId']}"
        )
    if run.get("head_branch") != "main":
        return {**base, "action": "ignore", "reason": "not-main"}
    if run.get("head_sha") != pinned[role]["headSha"]:
        fail(f"{role}: event head SHA does not match the pinned launch SHA")
    if run.get("status") != "completed":
        return {**base, "action": "ignore", "reason": "run-not-completed"}
    conclusion = run.get("conclusion")
    if conclusion != "failure":
        return {**base, "action": "ignore", "reason": f"conclusion-{conclusion}"}

    attempt = require_integer(run.get("run_attempt"), "workflow_run.run_attempt", 1, 100)
    if attempt >= maximum_attempts:
        return {
            **base,
            "action": "ignore",
            "reason": "retry-budget-exhausted",
            "runAttempt": attempt,
            "maximumAttempts": maximum_attempts,
        }
    return {
        **base,
        "action": "rerun-failed-jobs",
        "runAttempt": attempt,
        "expectedNextAttempt": attempt + 1,
        "maximumAttempts": maximum_attempts,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    launch = load_object(arguments.launch, "launch manifest")
    event = load_object(arguments.event, "workflow-run event")
    result = decide_recovery(
        launch,
        event,
        maximum_attempts=arguments.maximum_attempts,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, newline="\n")
    print(encoded, end="")


if __name__ == "__main__":
    main()
