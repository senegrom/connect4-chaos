#!/usr/bin/env python3
"""Validate exact Perfect Chaos closure candidates before release promotion.

The proof-producing workflows commit one role state plus, after a zero-
counterexample round, one closure-candidate decision.  This module performs a
pure repository-side validation of that handoff.  It does not download proof
artifacts and therefore cannot certify a release by itself; it establishes
that the committed identities are unambiguous, internally consistent, and
ready for the artifact/replay gate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, NoReturn

FORMAT = "connect4-chaos-promotion-readiness-v1"
DECISION_FORMAT = "connect4-chaos-auto-advance-decision-v1"
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
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
ARTIFACT_RE = re.compile(r"[A-Za-z0-9._-]+")
CANDIDATE_RE = re.compile(r"(red|yellow)-([1-9][0-9]*)\.json")


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def require_integer(
    value: Any,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
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


def validate_state(path: Path, role: str) -> dict[str, Any]:
    state = load_object(path, "campaign state")
    missing = sorted(STATE_KEYS.difference(state))
    unknown = sorted(set(state).difference(STATE_KEYS))
    if missing or unknown:
        fail(f"{path}: missing={missing}, unknown={unknown}")
    if state["role"] != role or path.name != f"{role}.json":
        fail(f"{path}: expected role {role!r}, found {state.get('role')!r}")

    require_integer(state["sourceRun"], f"{path}.sourceRun", 1, 10**15)
    if not isinstance(state["sourceSha"], str) or SHA_RE.fullmatch(state["sourceSha"]) is None:
        fail(f"{path}.sourceSha must be a lowercase 40-character commit SHA")
    if (
        not isinstance(state["sourceArtifact"], str)
        or ARTIFACT_RE.fullmatch(state["sourceArtifact"]) is None
    ):
        fail(f"{path}.sourceArtifact must be a safe artifact name")

    existing = require_integer(
        state["existingRejections"], f"{path}.existingRejections", 0, 10_000_000
    )
    cumulative = require_integer(
        state["cumulativeRejections"], f"{path}.cumulativeRejections", 1, 10_000_000
    )
    if cumulative <= existing:
        fail(f"{path}.cumulativeRejections must exceed existingRejections")
    require_integer(state["prepareShards"], f"{path}.prepareShards", 1, 512)
    require_integer(state["prepareWorkers"], f"{path}.prepareWorkers", 1, 16)
    require_integer(state["shardCount"], f"{path}.shardCount", 1, 512)
    return state


def validate_candidate(
    path: Path,
    *,
    role: str,
    filename_cumulative: int,
    state: dict[str, Any],
    from_pieces: int,
    target_pieces: int,
) -> dict[str, Any]:
    candidate = load_object(path, "closure candidate")
    expected = {
        "format": DECISION_FORMAT,
        "role": role,
        "fromPieces": from_pieces,
        "targetPieces": target_pieces,
        "newRejectedRoots": 0,
        "closedCandidate": True,
        "nextState": None,
    }
    for field, target in expected.items():
        if candidate.get(field) != target:
            fail(f"{path}: {field} must be {target!r}")

    run = require_integer(candidate.get("run"), f"{path}.run", 1, 10**15)
    run_sha = candidate.get("runSha")
    if not isinstance(run_sha, str) or SHA_RE.fullmatch(run_sha) is None:
        fail(f"{path}.runSha must be a lowercase 40-character commit SHA")

    cumulative = require_integer(
        candidate.get("cumulativeRejectedRoots"),
        f"{path}.cumulativeRejectedRoots",
        1,
        10_000_000,
    )
    existing = require_integer(
        candidate.get("existingRejectedRoots"),
        f"{path}.existingRejectedRoots",
        0,
        10_000_000,
    )
    if existing != cumulative:
        fail(f"{path}: a zero-counterexample closure must preserve its cumulative count")
    if filename_cumulative != cumulative:
        fail(
            f"{path}: filename count {filename_cumulative} does not match "
            f"candidate count {cumulative}"
        )
    if state["cumulativeRejections"] != cumulative:
        fail(
            f"{path}: candidate count {cumulative} does not match the committed "
            f"{role} state count {state['cumulativeRejections']}"
        )

    result_artifact = f"perfect-chaos-{role}-{target_pieces}-{cumulative}-round"
    if candidate.get("resultArtifact") != result_artifact:
        fail(f"{path}.resultArtifact must be {result_artifact!r}")

    checksums = candidate.get("checksums")
    if not isinstance(checksums, dict):
        fail(f"{path}.checksums must be an object")
    checksum_fields = ("roundManifest", "evidenceManifest", "closureReplay")
    unknown_checksums = sorted(set(checksums).difference(checksum_fields))
    missing_checksums = sorted(set(checksum_fields).difference(checksums))
    if missing_checksums or unknown_checksums:
        fail(
            f"{path}.checksums: missing={missing_checksums}, "
            f"unknown={unknown_checksums}"
        )
    for field in checksum_fields:
        digest = checksums[field]
        if not isinstance(digest, str) or DIGEST_RE.fullmatch(digest) is None:
            fail(f"{path}.checksums.{field} must be a lowercase SHA-256 digest")

    return {
        "present": True,
        "candidatePath": path.as_posix(),
        "statePath": (path.parent.parent / f"{role}.json").as_posix(),
        "run": run,
        "runSha": run_sha,
        "cumulativeRejectedRoots": cumulative,
        "resultArtifact": result_artifact,
        "evidenceArtifact": f"perfect-chaos-{role}-{target_pieces}-{cumulative}-evidence",
        "checksums": {field: checksums[field] for field in checksum_fields},
    }


def inspect_readiness(
    campaign_root: Path,
    *,
    from_pieces: int,
    target_pieces: int,
) -> dict[str, Any]:
    if campaign_root.is_symlink() or not campaign_root.is_dir():
        fail(f"Campaign root must be a regular directory: {campaign_root}")
    require_integer(from_pieces, "from_pieces", 0, 64)
    require_integer(target_pieces, "target_pieces", 1, 64)
    if target_pieces != from_pieces + 2:
        fail("target_pieces must be exactly two greater than from_pieces")

    states = {
        role: validate_state(campaign_root / f"{role}.json", role)
        for role in ROLES
    }
    candidate_root = campaign_root / "closure-candidates"
    if candidate_root.exists() and (candidate_root.is_symlink() or not candidate_root.is_dir()):
        fail(f"Closure candidate root must be a regular directory: {candidate_root}")

    discovered: dict[str, tuple[Path, int]] = {}
    if candidate_root.is_dir():
        for path in sorted(candidate_root.iterdir()):
            if path.name.startswith("."):
                continue
            match = CANDIDATE_RE.fullmatch(path.name)
            if match is None:
                fail(f"Unexpected closure candidate entry: {path}")
            role = match.group(1)
            cumulative = int(match.group(2))
            if role in discovered:
                fail(
                    f"Expected at most one {role} closure candidate; found "
                    f"{discovered[role][0]} and {path}"
                )
            discovered[role] = (path, cumulative)

    roles: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        if role not in discovered:
            roles[role] = {
                "present": False,
                "statePath": (campaign_root / f"{role}.json").as_posix(),
                "cumulativeRejectedRoots": states[role]["cumulativeRejections"],
            }
            continue
        path, filename_cumulative = discovered[role]
        roles[role] = validate_candidate(
            path,
            role=role,
            filename_cumulative=filename_cumulative,
            state=states[role],
            from_pieces=from_pieces,
            target_pieces=target_pieces,
        )

    return {
        "format": FORMAT,
        "campaignRoot": campaign_root.as_posix(),
        "fromPieces": from_pieces,
        "targetPieces": target_pieces,
        "ready": all(roles[role]["present"] for role in ROLES),
        "roles": roles,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--from-pieces", type=int, required=True)
    parser.add_argument("--target-pieces", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    result = inspect_readiness(
        arguments.campaign_root,
        from_pieces=arguments.from_pieces,
        target_pieces=arguments.target_pieces,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, newline="\n")
    print(encoded, end="")
    if arguments.require_ready and not result["ready"]:
        raise SystemExit("Both role closure candidates are not present yet.")


if __name__ == "__main__":
    main()
