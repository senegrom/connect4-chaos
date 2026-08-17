#!/usr/bin/env python3
"""Validate an independently audited Perfect Chaos round and derive its next state.

This utility is deliberately conservative.  It accepts only byte-identical
producer/auditor accounting, validates all published digests, and emits a next
campaign state only when the round found at least one new losing root.  A
zero-counterexample round is reported as a closure *candidate* and never
silently promoted to a full Perfect claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

FORMAT = "connect4-chaos-frontier-classification-merged-v1"
DECISION_FORMAT = "connect4-chaos-auto-advance-decision-v1"
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
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
ARTIFACT_RE = re.compile(r"[A-Za-z0-9._-]+")


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"Could not read JSON from {path}: {error}")
    if not isinstance(value, dict):
        fail(f"{path} must contain a JSON object")
    return value


def require_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"{label} must be an integer >= {minimum}")
    return value


def require_text(value: Any, label: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        fail(f"{label} has an invalid value: {value!r}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sums(directory: Path) -> dict[str, str]:
    sums_path = directory / "SHA256SUMS"
    if not sums_path.is_file():
        fail(f"Missing {sums_path}")
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(sums_path.read_text().splitlines(), 1):
        if not raw_line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw_line)
        if match is None:
            fail(f"{sums_path}:{line_number}: malformed checksum line")
        expected, relative_text = match.groups()
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative_text in entries:
            fail(f"{sums_path}:{line_number}: unsafe or duplicate path {relative_text!r}")
        target = directory.joinpath(*relative.parts)
        if not target.is_file() or target.is_symlink():
            fail(f"Checksum target is missing or unsafe: {target}")
        actual = sha256(target)
        if actual != expected:
            fail(f"Checksum mismatch for {target}: expected {expected}, found {actual}")
        entries[relative_text] = expected
    if not entries:
        fail(f"{sums_path} contains no checksums")
    return entries


def same_bytes(first: Path, second: Path, label: str) -> None:
    if not first.is_file() or not second.is_file():
        fail(f"Missing {label} file in producer or auditor artifact")
    if first.read_bytes() != second.read_bytes():
        fail(f"Producer and independent auditor disagree on {label}")


def validate_state(path: Path, role: str) -> dict[str, Any]:
    state = load_json(path)
    missing = sorted(STATE_KEYS.difference(state))
    unknown = sorted(set(state).difference(STATE_KEYS))
    if missing or unknown:
        fail(f"{path}: missing={missing}, unknown={unknown}")
    if state["role"] != role:
        fail(f"{path}: expected role {role!r}, found {state['role']!r}")
    require_int(state["sourceRun"], "state.sourceRun", 1)
    require_text(state["sourceSha"], "state.sourceSha", GIT_SHA_RE)
    require_text(state["sourceArtifact"], "state.sourceArtifact", ARTIFACT_RE)
    existing = require_int(state["existingRejections"], "state.existingRejections")
    cumulative = require_int(state["cumulativeRejections"], "state.cumulativeRejections", 1)
    if cumulative <= existing:
        fail("state.cumulativeRejections must be greater than state.existingRejections")
    require_int(state["prepareShards"], "state.prepareShards", 1)
    workers = require_int(state["prepareWorkers"], "state.prepareWorkers", 1)
    if workers > 16:
        fail("state.prepareWorkers must be <= 16")
    shards = require_int(state["shardCount"], "state.shardCount", 1)
    if shards > 512:
        fail("state.shardCount must be <= 512")
    return state


def validate_summary(
    summary: dict[str, Any],
    classification: dict[str, Any],
    state: dict[str, Any],
    role: str,
    from_pieces: int,
    target_pieces: int,
    round_directory: Path,
) -> tuple[int, int, int]:
    for label, value in (("summary", summary), ("classification", classification)):
        if value.get("format") != FORMAT:
            fail(f"{label} has an unsupported format")
        if value.get("role") != role:
            fail(f"{label} has the wrong role")
        if value.get("fromPieces") != from_pieces or value.get("targetPieces") != target_pieces:
            fail(f"{label} has incompatible piece boundaries")
        if value.get("classificationComplete") is not True or value.get("policyConflicts") != 0:
            fail(f"{label} is incomplete or conflicted")

    # classification.json is the pre-accounting record.  Every shared field
    # must agree exactly with campaign-summary.json.
    for key, value in classification.items():
        if key not in summary or summary[key] != value:
            fail(f"campaign-summary.json disagrees with classification.json at {key}")

    existing = require_int(summary.get("existingRejectedRoots"), "summary.existingRejectedRoots")
    new = require_int(summary.get("newRejectedRoots"), "summary.newRejectedRoots")
    cumulative = require_int(summary.get("cumulativeRejectedRoots"), "summary.cumulativeRejectedRoots")
    if existing != state["cumulativeRejections"]:
        fail(
            "The round did not start from the state file's certified cumulative rejection count: "
            f"expected {state['cumulativeRejections']}, found {existing}"
        )
    if cumulative != existing + new:
        fail("The round has inconsistent cumulative rejection accounting")
    if summary.get("rejectionProgress") != new or summary.get("rejectedRoots") != new:
        fail("The round has inconsistent new-rejection accounting")

    input_roots = require_int(summary.get("inputRoots"), "summary.inputRoots", 1)
    safe_roots = require_int(summary.get("safeInputRoots"), "summary.safeInputRoots")
    if safe_roots + new != input_roots:
        fail("safe and rejected roots do not partition the input frontier")
    require_int(summary.get("safePolicyEntries"), "summary.safePolicyEntries")
    require_int(summary.get("safeFrontierStates"), "summary.safeFrontierStates")

    boundary = from_pieces
    expected_paths = {
        "rejected": f"new-reject-{boundary}.bin",
        "policy": f"{from_pieces}-{target_pieces}.policy.bin",
        "frontier": f"{from_pieces}-{target_pieces}.frontier.bin",
    }
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("summary.artifacts must be an object")
    for kind, expected_path in expected_paths.items():
        record = artifacts.get(kind)
        if not isinstance(record, dict) or record.get("path") != expected_path:
            fail(f"summary.artifacts.{kind} has the wrong path")
        expected_bytes = require_int(record.get("bytes"), f"summary.artifacts.{kind}.bytes")
        expected_digest = require_text(
            record.get("sha256"), f"summary.artifacts.{kind}.sha256", SHA256_RE
        )
        target = round_directory / expected_path
        if not target.is_file() or target.stat().st_size != expected_bytes or sha256(target) != expected_digest:
            fail(f"Published {kind} artifact does not match its summary")

    cumulative_file = round_directory / f"reject-{boundary}.bin"
    if not cumulative_file.is_file():
        fail(f"Missing cumulative rejection table {cumulative_file}")
    return existing, new, cumulative


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--round-directory", type=Path, required=True)
    parser.add_argument("--evidence-directory", type=Path, required=True)
    parser.add_argument("--role", choices=("red", "yellow"), required=True)
    parser.add_argument("--from-pieces", type=int, required=True)
    parser.add_argument("--target-pieces", type=int, required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-sha", required=True)
    parser.add_argument("--result-artifact", required=True)
    parser.add_argument("--next-state", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.from_pieces < 0 or arguments.target_pieces <= arguments.from_pieces:
        fail("Piece boundaries are invalid")
    run_id = require_int(arguments.run_id, "run-id", 1)
    run_sha = require_text(arguments.run_sha, "run-sha", GIT_SHA_RE)
    result_artifact = require_text(arguments.result_artifact, "result-artifact", ARTIFACT_RE)

    state = validate_state(arguments.state, arguments.role)
    round_sums = validate_sums(arguments.round_directory)
    evidence_sums = validate_sums(arguments.evidence_directory)

    boundary = arguments.from_pieces
    shared_files = (
        "campaign-summary.json",
        "classification.json",
        f"new-reject-{boundary}.bin",
        f"reject-{boundary}.bin",
    )
    for name in shared_files:
        same_bytes(
            arguments.round_directory / name,
            arguments.evidence_directory / name,
            name,
        )
        if name not in round_sums or name not in evidence_sums:
            fail(f"{name} is not covered by both checksum manifests")
    if not (arguments.evidence_directory / "raw-shard-audit.json").is_file():
        fail("The independent evidence artifact has no raw-shard audit")

    summary = load_json(arguments.round_directory / "campaign-summary.json")
    classification = load_json(arguments.round_directory / "classification.json")
    existing, new, cumulative = validate_summary(
        summary,
        classification,
        state,
        arguments.role,
        arguments.from_pieces,
        arguments.target_pieces,
        arguments.round_directory,
    )

    expected_artifact = (
        f"perfect-chaos-{arguments.role}-{arguments.target_pieces}-"
        f"{state['cumulativeRejections']}-round"
    )
    if result_artifact != expected_artifact:
        fail(f"Expected result artifact {expected_artifact!r}, found {result_artifact!r}")

    next_state: dict[str, Any] | None = None
    if new > 0:
        next_state = {
            "role": arguments.role,
            "sourceRun": run_id,
            "sourceSha": run_sha,
            "sourceArtifact": result_artifact,
            "existingRejections": existing,
            "cumulativeRejections": cumulative,
            "prepareShards": state["prepareShards"],
            "prepareWorkers": state["prepareWorkers"],
            "shardCount": state["shardCount"],
        }
        arguments.next_state.parent.mkdir(parents=True, exist_ok=True)
        arguments.next_state.write_text(json.dumps(next_state, indent=2) + "\n")
    elif arguments.next_state.exists():
        arguments.next_state.unlink()

    decision = {
        "format": DECISION_FORMAT,
        "role": arguments.role,
        "fromPieces": arguments.from_pieces,
        "targetPieces": arguments.target_pieces,
        "run": run_id,
        "runSha": run_sha,
        "resultArtifact": result_artifact,
        "existingRejectedRoots": existing,
        "newRejectedRoots": new,
        "cumulativeRejectedRoots": cumulative,
        "closedCandidate": new == 0,
        "nextState": next_state,
        "checksums": {
            "roundManifest": sha256(arguments.round_directory / "SHA256SUMS"),
            "evidenceManifest": sha256(arguments.evidence_directory / "SHA256SUMS"),
        },
    }
    arguments.decision.parent.mkdir(parents=True, exist_ok=True)
    arguments.decision.write_text(json.dumps(decision, indent=2) + "\n")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
