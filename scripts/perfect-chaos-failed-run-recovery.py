#!/usr/bin/env python3
"""Decide whether a failed exact Perfect Chaos run may be retried in place.

A retry is sound only when the failed run can be tied to exactly one role and
the role-state bytes at the run's head commit are identical to the currently
committed state.  The caller remains responsible for checking that no newer
run for that role is active immediately before invoking GitHub's failed-job
rerun API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, NoReturn

FORMAT = "connect4-chaos-failed-run-recovery-v1"
CAMPAIGN_ROOT = Path(".campaign/perfect-chaos-main-18")
CONTINUE_WORKFLOW_PATH = ".github/workflows/continue-perfect-chaos-18-main.yml"
ROLES = ("red", "yellow")
STATE_KEYS = {
    "role",
    "sourceRun",
    "sourceSha",
    "sourceArtifact",
    "existingRejections",
    "cumulativeRejections",
    "prepareShards",
    "prepareWorkers",
    "shardCount",
}
SHA_RE = re.compile(r"[0-9a-f]{40}")
ARTIFACT_RE = re.compile(r"[A-Za-z0-9._-]+")


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


def git_bytes(root: Path, commit: str, relative: Path) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative.as_posix()}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        fail(
            f"Could not read {relative} at {commit}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def changed_role_for_push(root: Path, head_sha: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            head_sha,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        fail(f"Could not inspect push commit {head_sha}: {result.stderr.strip()}")
    changed = {Path(line) for line in result.stdout.splitlines() if line}
    matches = [
        role
        for role in ROLES
        if CAMPAIGN_ROOT / f"{role}.json" in changed
    ]
    if len(matches) != 1:
        fail(
            "A failed push continuation must change exactly one role state; "
            f"head={head_sha}, matches={matches}"
        )
    return matches[0]


def role_from_dispatch_title(title: str) -> str:
    matches = [
        role
        for role in ROLES
        if (CAMPAIGN_ROOT / f"{role}.json").as_posix() in title
    ]
    if len(matches) != 1:
        fail(
            "A failed workflow dispatch must name exactly one role-state path; "
            f"title={title!r}, matches={matches}"
        )
    return matches[0]


def validate_state_bytes(encoded: bytes, role: str, label: str) -> dict[str, Any]:
    try:
        state = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"Could not parse {label}: {error}")
    if not isinstance(state, dict):
        fail(f"{label} must contain a JSON object")
    missing = sorted(STATE_KEYS.difference(state))
    unknown = sorted(set(state).difference(STATE_KEYS))
    if missing or unknown:
        fail(f"{label}: missing={missing}, unknown={unknown}")
    if state["role"] != role:
        fail(f"{label}: expected role {role!r}, found {state.get('role')!r}")
    require_integer(state["sourceRun"], f"{label}.sourceRun", 1, 10**15)
    if not isinstance(state["sourceSha"], str) or SHA_RE.fullmatch(state["sourceSha"]) is None:
        fail(f"{label}.sourceSha must be a lowercase 40-character commit SHA")
    if (
        not isinstance(state["sourceArtifact"], str)
        or ARTIFACT_RE.fullmatch(state["sourceArtifact"]) is None
    ):
        fail(f"{label}.sourceArtifact must be a safe artifact name")
    existing = require_integer(
        state["existingRejections"], f"{label}.existingRejections", 0, 10_000_000
    )
    cumulative = require_integer(
        state["cumulativeRejections"], f"{label}.cumulativeRejections", 1, 10_000_000
    )
    if cumulative <= existing:
        fail(f"{label}.cumulativeRejections must exceed existingRejections")
    require_integer(state["prepareShards"], f"{label}.prepareShards", 1, 512)
    require_integer(state["prepareWorkers"], f"{label}.prepareWorkers", 1, 16)
    require_integer(state["shardCount"], f"{label}.shardCount", 1, 512)
    return state


def closure_candidate_present(root: Path, role: str) -> bool:
    directory = root / CAMPAIGN_ROOT / "closure-candidates"
    if not directory.exists():
        return False
    if directory.is_symlink() or not directory.is_dir():
        fail(f"Closure candidate root must be a regular directory: {directory}")
    matches = sorted(directory.glob(f"{role}-*.json"))
    for path in matches:
        if path.is_symlink() or not path.is_file():
            fail(f"Closure candidate must be a regular file: {path}")
    if len(matches) > 1:
        fail(f"Expected at most one {role} closure candidate; found {matches}")
    return len(matches) == 1


def extract_run(event: dict[str, Any]) -> dict[str, Any]:
    run = event.get("workflow_run", event)
    if not isinstance(run, dict):
        fail("Event must contain a workflow_run object")
    return run


def decide_recovery(
    root: Path,
    event: dict[str, Any],
    *,
    maximum_attempts: int,
) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        fail(f"Repository root must be a regular directory: {root}")
    require_integer(maximum_attempts, "maximum_attempts", 1, 20)
    run = extract_run(event)

    run_id = require_integer(run.get("id"), "workflow_run.id", 1, 10**15)
    path = run.get("path")
    status = run.get("status")
    conclusion = run.get("conclusion")
    if path != CONTINUE_WORKFLOW_PATH:
        return {
            "format": FORMAT,
            "action": "ignore",
            "reason": "unsupported-workflow",
            "runId": run_id,
        }
    if status != "completed":
        return {
            "format": FORMAT,
            "action": "ignore",
            "reason": "run-not-completed",
            "runId": run_id,
        }
    if conclusion != "failure":
        return {
            "format": FORMAT,
            "action": "ignore",
            "reason": f"conclusion-{conclusion}",
            "runId": run_id,
        }
    if run.get("head_branch") != "main":
        return {
            "format": FORMAT,
            "action": "ignore",
            "reason": "not-main",
            "runId": run_id,
        }

    head_sha = run.get("head_sha")
    if not isinstance(head_sha, str) or SHA_RE.fullmatch(head_sha) is None:
        fail("workflow_run.head_sha must be a lowercase 40-character commit SHA")
    event_name = run.get("event")
    if event_name == "push":
        role = changed_role_for_push(root, head_sha)
    elif event_name == "workflow_dispatch":
        title = run.get("display_title")
        if not isinstance(title, str) or not title:
            fail("A failed workflow dispatch has no display_title")
        role = role_from_dispatch_title(title)
    else:
        return {
            "format": FORMAT,
            "action": "ignore",
            "reason": f"unsupported-event-{event_name}",
            "runId": run_id,
        }

    state_relative = CAMPAIGN_ROOT / f"{role}.json"
    state_path = root / state_relative
    if state_path.is_symlink() or not state_path.is_file():
        fail(f"Current role state must be a regular file: {state_path}")
    current_bytes = state_path.read_bytes()
    historical_bytes = git_bytes(root, head_sha, state_relative)
    current_state = validate_state_bytes(current_bytes, role, f"current {state_relative}")
    validate_state_bytes(historical_bytes, role, f"{state_relative} at {head_sha}")

    base = {
        "format": FORMAT,
        "role": role,
        "runId": run_id,
        "headSha": head_sha,
        "statePath": state_relative.as_posix(),
        "stateSha256": hashlib.sha256(current_bytes).hexdigest(),
        "cumulativeRejectedRoots": current_state["cumulativeRejections"],
    }
    if current_bytes != historical_bytes:
        return {
            **base,
            "action": "ignore",
            "reason": "state-advanced",
        }
    if closure_candidate_present(root, role):
        return {
            **base,
            "action": "ignore",
            "reason": "closure-candidate-present",
        }

    run_attempt = require_integer(run.get("run_attempt"), "workflow_run.run_attempt", 1, 100)
    if run_attempt >= maximum_attempts:
        return {
            **base,
            "action": "ignore",
            "reason": "retry-budget-exhausted",
            "runAttempt": run_attempt,
            "maximumAttempts": maximum_attempts,
        }
    return {
        **base,
        "action": "rerun-failed-jobs",
        "runAttempt": run_attempt,
        "expectedNextAttempt": run_attempt + 1,
        "maximumAttempts": maximum_attempts,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--maximum-attempts", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    event = load_object(arguments.event, "workflow-run event")
    result = decide_recovery(
        arguments.root.resolve(),
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
