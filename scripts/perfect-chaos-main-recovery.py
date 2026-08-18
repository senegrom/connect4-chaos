#!/usr/bin/env python3
"""Adapt an exact Perfect Chaos preparation after a bound, verified failure.

The script is deliberately pure at its decision boundary. It accepts one
completed workflow run and its latest-attempt jobs, binds the run to exactly one
committed role-state file, and changes only preparation performance parameters.
Proof identity, rejection counts, source artifacts, and solver semantics remain
unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, NoReturn

FORMAT = "connect4-chaos-main-preparation-recovery-v1"
STATE_ROOT = Path(".campaign/perfect-chaos-main-18")
ROLES = ("red", "yellow")
CONTINUE_PATH = ".github/workflows/continue-perfect-chaos-18-main.yml"
PREPARE_JOB = "round / round / prepare"
SHA_RE = re.compile(r"[0-9a-f]{40}")
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


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


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


def require_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        fail(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def validate_state(state: dict[str, Any], role: str) -> dict[str, Any]:
    missing = sorted(STATE_KEYS.difference(state))
    unknown = sorted(set(state).difference(STATE_KEYS))
    if missing or unknown:
        fail(f"State schema mismatch: missing={missing}, unknown={unknown}")
    if state["role"] != role:
        fail(f"State role mismatch: expected {role!r}, found {state.get('role')!r}")
    require_int(state["sourceRun"], "sourceRun", 1, 10**15)
    if not isinstance(state["sourceSha"], str) or SHA_RE.fullmatch(state["sourceSha"]) is None:
        fail("sourceSha must be a lowercase 40-character commit SHA")
    if not isinstance(state["sourceArtifact"], str) or not re.fullmatch(
        r"[A-Za-z0-9._-]+", state["sourceArtifact"]
    ):
        fail("sourceArtifact must be a safe artifact name")
    existing = require_int(state["existingRejections"], "existingRejections", 0, 10_000_000)
    cumulative = require_int(
        state["cumulativeRejections"], "cumulativeRejections", 1, 10_000_000
    )
    if cumulative <= existing:
        fail("cumulativeRejections must exceed existingRejections")
    require_int(state["prepareShards"], "prepareShards", 1, 256)
    require_int(state["prepareWorkers"], "prepareWorkers", 1, 16)
    require_int(state["shardCount"], "shardCount", 1, 512)
    return dict(state)


def exact_state_paths_in_title(title: str) -> list[str]:
    return [
        role
        for role in ROLES
        if (STATE_ROOT / f"{role}.json").as_posix() in title
    ]


def role_from_run(run: dict[str, Any], changed_paths: list[str] | None = None) -> str:
    event = run.get("event")
    title = run.get("display_title")
    if not isinstance(title, str) or not title:
        fail("Workflow run has no display title")

    title_roles = exact_state_paths_in_title(title)
    if event == "workflow_dispatch":
        if len(title_roles) != 1:
            fail(
                "A failed dispatched continuation must name exactly one role-state path; "
                f"title={title!r}, matches={title_roles}"
            )
        return title_roles[0]

    if event == "push":
        if changed_paths is None:
            fail("Push-triggered recovery requires exact changed paths")
        allowed = {
            (STATE_ROOT / f"{role}.json").as_posix(): role
            for role in ROLES
        }
        matches = [allowed[path] for path in changed_paths if path in allowed]
        if len(matches) != 1:
            fail(
                "A failed push continuation must change exactly one role-state file; "
                f"matches={matches}"
            )
        return matches[0]

    fail(f"Unsupported continuation event for preparation recovery: {event!r}")


def latest_prepare_failed(jobs_payload: dict[str, Any]) -> bool:
    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, list):
        fail("Jobs payload must contain a jobs array")
    prepare_jobs = [job for job in jobs if isinstance(job, dict) and job.get("name") == PREPARE_JOB]
    if len(prepare_jobs) > 1:
        fail(f"Expected at most one {PREPARE_JOB!r} job, found {len(prepare_jobs)}")
    if not prepare_jobs:
        return False
    prepare = prepare_jobs[0]
    return prepare.get("status") == "completed" and prepare.get("conclusion") == "failure"


def next_profile(shards: int, workers: int) -> tuple[int, int] | None:
    if shards < 256:
        return min(256, max(shards + 1, shards * 2)), workers
    if workers > 1:
        return shards, max(1, workers // 2)
    return None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decide_recovery(
    run: dict[str, Any],
    jobs_payload: dict[str, Any],
    current_state: dict[str, Any],
    current_bytes: bytes,
    run_state_bytes: bytes,
    *,
    changed_paths: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    run_id = require_int(run.get("id"), "run.id", 1, 10**15)
    run_attempt = require_int(run.get("run_attempt"), "run.run_attempt", 1, 10_000)
    if run.get("path") != CONTINUE_PATH:
        fail(f"Unsupported failed workflow path: {run.get('path')!r}")
    if run.get("head_branch") != "main":
        fail("Failed preparation is not bound to main")
    head_sha = run.get("head_sha")
    if not isinstance(head_sha, str) or SHA_RE.fullmatch(head_sha) is None:
        fail("Failed run has an invalid head SHA")
    if run.get("status") != "completed" or run.get("conclusion") != "failure":
        fail("Preparation recovery requires a completed failed workflow run")

    role = role_from_run(run, changed_paths)
    state = validate_state(current_state, role)
    state_path = (STATE_ROOT / f"{role}.json").as_posix()
    base = {
        "format": FORMAT,
        "runId": run_id,
        "runAttempt": run_attempt,
        "runSha": head_sha,
        "role": role,
        "statePath": state_path,
        "expectedStateSha256": sha256(current_bytes),
        "oldProfile": {
            "prepareShards": state["prepareShards"],
            "prepareWorkers": state["prepareWorkers"],
        },
    }

    if current_bytes != run_state_bytes:
        return ({**base, "action": "stale-state", "handled": False}, None)
    if not latest_prepare_failed(jobs_payload):
        return ({**base, "action": "not-prepare-failure", "handled": False}, None)

    profile = next_profile(state["prepareShards"], state["prepareWorkers"])
    if profile is None:
        return (
            {
                **base,
                "action": "exhausted",
                "handled": True,
                "reason": "Preparation already uses 256 shards and one worker.",
            },
            None,
        )

    shards, workers = profile
    updated = dict(state)
    updated["prepareShards"] = shards
    updated["prepareWorkers"] = workers
    encoded = (json.dumps(updated, indent=2) + "\n").encode()
    decision = {
        **base,
        "action": "adapted",
        "handled": True,
        "newProfile": {
            "prepareShards": shards,
            "prepareWorkers": workers,
        },
        "updatedStateSha256": sha256(encoded),
    }
    return decision, updated


def changed_paths_for_push(root: Path, head_sha: str) -> list[str]:
    parents = subprocess.check_output(
        ["git", "-C", str(root), "rev-list", "--parents", "-n", "1", head_sha],
        text=True,
    ).split()
    if len(parents) != 2:
        fail(f"Push-triggered recovery requires one parent; commit={head_sha}")
    return subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", parents[1], head_sha],
        text=True,
    ).splitlines()


def state_at_run(root: Path, head_sha: str, path: Path) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "show", f"{head_sha}:{path.as_posix()}"],
        )
    except subprocess.CalledProcessError as error:
        fail(f"Could not read {path} at failed run {head_sha}: {error}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--state-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    root = arguments.root.resolve()
    if not root.is_dir():
        fail(f"Repository root does not exist: {root}")
    run = load_object(arguments.run, "workflow run")
    jobs = load_object(arguments.jobs, "workflow jobs")
    head_sha = run.get("head_sha")
    if not isinstance(head_sha, str) or SHA_RE.fullmatch(head_sha) is None:
        fail("Workflow run has an invalid head SHA")
    changed_paths = changed_paths_for_push(root, head_sha) if run.get("event") == "push" else None
    role = role_from_run(run, changed_paths)
    relative = STATE_ROOT / f"{role}.json"
    current_path = root / relative
    current_bytes = current_path.read_bytes()
    current_state = load_object(current_path, "current role state")
    run_bytes = state_at_run(root, head_sha, relative)
    decision, updated = decide_recovery(
        run,
        jobs,
        current_state,
        current_bytes,
        run_bytes,
        changed_paths=changed_paths,
    )

    arguments.decision.parent.mkdir(parents=True, exist_ok=True)
    arguments.decision.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", newline="\n")
    arguments.state_output.parent.mkdir(parents=True, exist_ok=True)
    if updated is None:
        arguments.state_output.unlink(missing_ok=True)
    else:
        arguments.state_output.write_text(json.dumps(updated, indent=2) + "\n", newline="\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
