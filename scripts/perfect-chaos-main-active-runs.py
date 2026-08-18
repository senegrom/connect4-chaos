#!/usr/bin/env python3
"""Classify active main-branch Perfect Chaos workflows by starting role.

The GitHub API inspection remains in the calling workflow.  This module is a
pure, fail-closed validator for the returned run metadata.  It maps active
continuations to Red or Yellow without guessing from free-form titles:

* push-triggered continuations must be bound to the commit that most recently
  changed exactly one role-state file;
* workflow-dispatched continuations must either be bound to exactly one current
  role-state commit or carry exactly one exact role-state path in their run name;
* if the SHA and title identify different roles, classification fails closed;
* bootstrap workflows cover both roles.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, NoReturn

FORMAT = "connect4-chaos-main-active-runs-v1"
STATE_ROOT = Path(".campaign/perfect-chaos-main-18")
ROLES = ("red", "yellow")
BOOTSTRAP_WORKFLOWS = {
    "start-perfect-chaos-18-from-main.yml",
    "advance-perfect-chaos-18-main-bootstrap.yml",
}
CONTINUE_WORKFLOW = "continue-perfect-chaos-18-main.yml"
WORKFLOWS = (*sorted(BOOTSTRAP_WORKFLOWS), CONTINUE_WORKFLOW)
SHA_RE = re.compile(r"[0-9a-f]{40}")


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def require_int(value: Any, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"{label} must be an integer of at least {minimum}")
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


def git_state_commits(root: Path) -> dict[str, str]:
    commits: dict[str, str] = {}
    for role in ROLES:
        relative = STATE_ROOT / f"{role}.json"
        path = root / relative
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            fail(f"Role state must be a regular file: {relative}")
        result = subprocess.run(
            ["git", "-C", str(root), "log", "-1", "--format=%H", "--", str(relative)],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
        if SHA_RE.fullmatch(commit) is None:
            fail(f"Could not determine the committed SHA for {relative}")
        commits[role] = commit
    return commits


def validate_state_commits(value: dict[str, str]) -> dict[str, str]:
    unknown = sorted(set(value).difference(ROLES))
    if unknown:
        fail(f"Unknown role commit bindings: {unknown}")
    for role, commit in value.items():
        if SHA_RE.fullmatch(commit) is None:
            fail(f"State commit for {role} is not a lowercase 40-character SHA")
    return dict(value)


def roles_from_dispatch_title(title: str) -> list[str]:
    return [
        role
        for role in ROLES
        if (STATE_ROOT / f"{role}.json").as_posix() in title
    ]


def roles_from_state_sha(head_sha: str, state_commits: dict[str, str]) -> list[str]:
    return [role for role, commit in state_commits.items() if commit == head_sha]


def role_from_dispatch(
    head_sha: str,
    title: str,
    state_commits: dict[str, str],
) -> str:
    sha_matches = roles_from_state_sha(head_sha, state_commits)
    title_matches = roles_from_dispatch_title(title)
    if len(sha_matches) > 1:
        fail(
            "A workflow-dispatched continuation matches more than one current "
            f"role-state commit; head_sha={head_sha}, matches={sha_matches}"
        )
    if len(title_matches) > 1:
        fail(
            "A workflow-dispatched continuation names more than one exact role-state "
            f"path; title={title!r}, matches={title_matches}"
        )
    combined = sorted(set(sha_matches).union(title_matches))
    if len(combined) != 1:
        fail(
            "A workflow-dispatched continuation must identify exactly one role by its "
            "current state commit or exact role-state path; "
            f"head_sha={head_sha}, title={title!r}, "
            f"sha_matches={sha_matches}, title_matches={title_matches}"
        )
    return combined[0]


def role_from_push_sha(head_sha: str, state_commits: dict[str, str]) -> str:
    matches = roles_from_state_sha(head_sha, state_commits)
    if len(matches) != 1:
        fail(
            "A push-triggered continuation must match exactly one current role-state "
            f"commit; head_sha={head_sha}, matches={matches}"
        )
    return matches[0]


def classify_active_runs(
    root: Path,
    payload: dict[str, Any],
    *,
    state_commits: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw_workflows = payload.get("workflows")
    if not isinstance(raw_workflows, list):
        fail("Active-run payload must contain a workflows array")

    bindings = (
        validate_state_commits(state_commits)
        if state_commits is not None
        else git_state_commits(root)
    )
    expected = set(WORKFLOWS)
    seen_workflows: set[str] = set()
    seen_runs: set[int] = set()
    active_roles: set[str] = set()
    classified_workflows: list[dict[str, Any]] = []

    for workflow_record in raw_workflows:
        if not isinstance(workflow_record, dict):
            fail("Each workflow record must be an object")
        workflow = workflow_record.get("workflow")
        runs = workflow_record.get("runs")
        if workflow not in expected:
            fail(f"Unsupported supervised workflow: {workflow!r}")
        if workflow in seen_workflows:
            fail(f"Duplicate supervised workflow record: {workflow}")
        seen_workflows.add(workflow)
        if not isinstance(runs, list):
            fail(f"{workflow}: runs must be an array")

        active: list[dict[str, Any]] = []
        for run in runs:
            if not isinstance(run, dict):
                fail(f"{workflow}: every run must be an object")
            status = run.get("status")
            if not isinstance(status, str) or not status:
                fail(f"{workflow}: run status is missing")
            if status == "completed":
                continue

            run_id = require_int(run.get("id"), f"{workflow}.run.id")
            if run_id in seen_runs:
                fail(f"Run {run_id} appears in more than one workflow record")
            seen_runs.add(run_id)
            path = run.get("path")
            expected_path = f".github/workflows/{workflow}"
            if path != expected_path:
                fail(f"Run {run_id} path mismatch: expected {expected_path}, found {path!r}")
            if run.get("head_branch") != "main":
                fail(f"Run {run_id} is not bound to main")
            head_sha = run.get("head_sha")
            if not isinstance(head_sha, str) or SHA_RE.fullmatch(head_sha) is None:
                fail(f"Run {run_id} has an invalid head SHA")
            event = run.get("event")
            title = run.get("display_title")
            if not isinstance(title, str) or not title:
                fail(f"Run {run_id} has no display title")

            if workflow in BOOTSTRAP_WORKFLOWS:
                roles = list(ROLES)
            elif event == "push":
                roles = [role_from_push_sha(head_sha, bindings)]
            elif event == "workflow_dispatch":
                roles = [role_from_dispatch(head_sha, title, bindings)]
            else:
                fail(f"Run {run_id} uses unsupported event {event!r}")

            active_roles.update(roles)
            active.append(
                {
                    "id": run_id,
                    "workflow": workflow,
                    "event": event,
                    "status": status,
                    "headSha": head_sha,
                    "displayTitle": title,
                    "roles": roles,
                    "createdAt": run.get("created_at"),
                    "updatedAt": run.get("updated_at"),
                }
            )

        active.sort(key=lambda item: item["id"])
        classified_workflows.append({"workflow": workflow, "active": active})

    missing = sorted(expected.difference(seen_workflows))
    if missing:
        fail(f"Active-run payload is missing supervised workflows: {missing}")
    classified_workflows.sort(key=lambda item: item["workflow"])
    return {
        "format": FORMAT,
        "stateCommits": bindings,
        "activeRoles": sorted(active_roles),
        "workflows": classified_workflows,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    root = arguments.root.resolve()
    if not root.is_dir():
        fail(f"Repository root does not exist: {root}")
    payload = load_object(arguments.input, "active-run payload")
    result = classify_active_runs(root, payload)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, newline="\n")
    print(encoded, end="")


if __name__ == "__main__":
    main()
