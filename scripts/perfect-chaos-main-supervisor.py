#!/usr/bin/env python3
"""Validate and select one stalled exact Perfect Chaos campaign state on main.

The selector is deliberately pure and fail-closed. GitHub API inspection and
workflow dispatch remain in the calling workflow.  This module validates only
committed repository state and chooses the oldest unresolved role that is not
already covered by an active exact run, so Red and Yellow can progress fairly
and independently.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable, NoReturn

FORMAT = "connect4-chaos-main-supervisor-selection-v1"
STATE_ROOT = Path(".campaign/perfect-chaos-main-18")
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


def require_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        fail(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def normalize_active_roles(active_roles: Iterable[str] | None) -> set[str]:
    selected = set(active_roles or ())
    unknown = sorted(selected.difference(ROLES))
    if unknown:
        fail(f"Unknown active Perfect Chaos roles: {unknown}")
    return selected


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


def validate_state(path: Path, role: str) -> dict[str, Any]:
    state = load_object(path, "campaign state")
    missing = sorted(STATE_KEYS.difference(state))
    unknown = sorted(set(state).difference(STATE_KEYS))
    if missing or unknown:
        fail(f"{path}: missing={missing}, unknown={unknown}")
    if state["role"] != role or path.name != f"{role}.json":
        fail(f"{path}: expected role {role!r}, found {state.get('role')!r}")

    require_int(state["sourceRun"], f"{path}.sourceRun", 1, 10**15)
    if not isinstance(state["sourceSha"], str) or SHA_RE.fullmatch(state["sourceSha"]) is None:
        fail(f"{path}.sourceSha must be a lowercase 40-character commit SHA")
    if (
        not isinstance(state["sourceArtifact"], str)
        or ARTIFACT_RE.fullmatch(state["sourceArtifact"]) is None
    ):
        fail(f"{path}.sourceArtifact must be a safe artifact name")

    existing = require_int(
        state["existingRejections"], f"{path}.existingRejections", 0, 10_000_000
    )
    cumulative = require_int(
        state["cumulativeRejections"], f"{path}.cumulativeRejections", 1, 10_000_000
    )
    if cumulative <= existing:
        fail(f"{path}.cumulativeRejections must exceed existingRejections")
    require_int(state["prepareShards"], f"{path}.prepareShards", 1, 512)
    require_int(state["prepareWorkers"], f"{path}.prepareWorkers", 1, 16)
    require_int(state["shardCount"], f"{path}.shardCount", 1, 512)
    return state


def validate_closure_candidate(path: Path, role: str) -> dict[str, Any]:
    candidate = load_object(path, "closure candidate")
    required = {
        "format": "connect4-chaos-auto-advance-decision-v1",
        "role": role,
        "fromPieces": 16,
        "targetPieces": 18,
        "closedCandidate": True,
        "newRejectedRoots": 0,
    }
    for field, expected in required.items():
        if candidate.get(field) != expected:
            fail(f"{path}: {field} must be {expected!r}")
    require_int(
        candidate.get("cumulativeRejectedRoots"),
        f"{path}.cumulativeRejectedRoots",
        1,
        10_000_000,
    )
    return candidate


def git_commit_time(root: Path, relative: Path) -> int:
    result = subprocess.run(
        ["git", "-C", str(root), "log", "-1", "--format=%ct", "--", str(relative)],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value.isdigit():
        fail(f"Could not determine the committed timestamp for {relative}")
    return int(value)


def inspect_states(
    root: Path,
    *,
    commit_times: dict[str, int] | None = None,
    active_roles: Iterable[str] | None = None,
) -> dict[str, Any]:
    campaign_root = root / STATE_ROOT
    closure_root = campaign_root / "closure-candidates"
    active = normalize_active_roles(active_roles)
    records: dict[str, dict[str, Any]] = {}
    selectable: list[tuple[int, int, str]] = []

    for order, role in enumerate(ROLES):
        relative = STATE_ROOT / f"{role}.json"
        path = root / relative
        candidates = sorted(closure_root.glob(f"{role}-*.json")) if closure_root.exists() else []
        for candidate_path in candidates:
            validate_closure_candidate(candidate_path, role)

        if not path.exists():
            records[role] = {
                "role": role,
                "statePath": relative.as_posix(),
                "present": False,
                "active": role in active,
                "closedCandidate": bool(candidates),
            }
            continue

        state = validate_state(path, role)
        if commit_times is None:
            committed_at = git_commit_time(root, relative)
        else:
            if role not in commit_times:
                fail(f"Missing injected commit time for {role}")
            committed_at = require_int(
                commit_times[role], f"commit_times[{role!r}]", 0, 10**12
            )

        record = {
            "role": role,
            "statePath": relative.as_posix(),
            "present": True,
            "active": role in active,
            "closedCandidate": bool(candidates),
            "committedAt": committed_at,
            "sourceRun": state["sourceRun"],
            "sourceSha": state["sourceSha"],
            "sourceArtifact": state["sourceArtifact"],
            "existingRejections": state["existingRejections"],
            "cumulativeRejections": state["cumulativeRejections"],
            "prepareShards": state["prepareShards"],
            "prepareWorkers": state["prepareWorkers"],
            "shardCount": state["shardCount"],
        }
        records[role] = record
        if not candidates and role not in active:
            selectable.append((committed_at, order, role))

    selected = None
    if selectable:
        _, _, role = min(selectable)
        selected = records[role]

    return {
        "format": FORMAT,
        "activeRoles": sorted(active),
        "states": records,
        "selected": selected,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--active-role", action="append", default=[], choices=ROLES)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    root = arguments.root.resolve()
    if not root.is_dir():
        fail(f"Repository root does not exist: {root}")
    result = inspect_states(root, active_roles=arguments.active_role)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, newline="\n")
    print(encoded, end="")


if __name__ == "__main__":
    main()
