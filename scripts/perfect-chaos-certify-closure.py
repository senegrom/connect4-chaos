#!/usr/bin/env python3
"""Construct a durable certificate from an independently audited Chaos closure.

The script accepts no game-theoretic shortcut. It verifies the transition
candidate, producer and independent evidence bytes, binary table metadata,
raw-shard accounting, and the complete adversarial replay before emitting a
compact certificate JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

from perfect_chaos_tables import (
    FRONTIER_MAGIC,
    FRONTIER_RECORD_SIZE,
    POLICY_MAGIC,
    POLICY_RECORD_SIZE,
    ROLE_CODES,
    read_table,
)

DECISION_FORMAT = "connect4-chaos-auto-advance-decision-v1"
SUMMARY_FORMAT = "connect4-chaos-frontier-classification-merged-v1"
AUDIT_FORMAT = "connect4-chaos-independent-sharded-round-audit-v1"
CERTIFICATE_FORMAT = "connect4-chaos-certified-prefix-closure-v2"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
ARTIFACT_RE = re.compile(r"[A-Za-z0-9._-]+")


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular, non-symlink file")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"Could not parse {label}: {error}")
    if not isinstance(value, dict):
        fail(f"{label} must contain a JSON object")
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
    if not sums_path.is_file() or sums_path.is_symlink():
        fail(f"Missing safe checksum manifest {sums_path}")
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
            fail(f"{sums_path}:{line_number}: unsafe or duplicate path")
        target = directory.joinpath(*relative.parts)
        if not target.is_file() or target.is_symlink():
            fail(f"Checksum target is missing or unsafe: {target}")
        actual = sha256(target)
        if actual != expected:
            fail(f"Checksum mismatch for {target}")
        entries[relative_text] = expected
    if not entries:
        fail(f"{sums_path} contains no checksums")
    return entries


def same_bytes(first: Path, second: Path, label: str) -> None:
    if not first.is_file() or not second.is_file():
        fail(f"Missing producer or evidence copy of {label}")
    if first.read_bytes() != second.read_bytes():
        fail(f"Producer and independent evidence disagree on {label}")


def artifact_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    artifact_id = require_int(value.get("id"), f"{label}.id", 1)
    name = require_text(value.get("name"), f"{label}.name", ARTIFACT_RE)
    size = require_int(value.get("size_in_bytes"), f"{label}.size_in_bytes")
    digest = require_text(value.get("digest"), f"{label}.digest")
    if not digest.startswith("sha256:") or SHA256_RE.fullmatch(digest[7:]) is None:
        fail(f"{label}.digest must be a SHA-256 artifact digest")
    workflow_run = value.get("workflow_run")
    if not isinstance(workflow_run, dict):
        fail(f"{label}.workflow_run must be an object")
    run_id = require_int(workflow_run.get("id"), f"{label}.workflow_run.id", 1)
    head_sha = require_text(workflow_run.get("head_sha"), f"{label}.workflow_run.head_sha", GIT_SHA_RE)
    return {
        "id": artifact_id,
        "name": name,
        "bytes": size,
        "digest": digest,
        "run": run_id,
        "sha": head_sha,
    }


def regular_file_names(directory: Path) -> list[str]:
    if directory.is_symlink() or not directory.is_dir():
        fail(f"Missing safe directory {directory}")
    names: list[str] = []
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            fail(f"Unexpected non-file entry in {directory}: {path.name}")
        names.append(path.name)
    return sorted(names)


def validate_replay(
    producer: Path,
    evidence: Path,
    producer_sums: dict[str, str],
    evidence_sums: dict[str, str],
    role: str,
    target_pieces: int,
) -> tuple[dict[str, Any], str]:
    producer_name = f"{role}-{target_pieces}-replay.json"
    evidence_name = "closure-replay.json"
    if producer_name not in producer_sums or evidence_name not in evidence_sums:
        fail("Closure replay is not covered by both checksum manifests")
    published = load_json(producer / producer_name, "published closure replay")
    audited = load_json(evidence / evidence_name, "independent closure replay")
    if published.get("replay") != audited.get("replay"):
        fail("Published and independently reproduced closure replays differ")
    replay = audited.get("replay")
    if not isinstance(replay, dict) or replay.get("role") != role:
        fail("Closure replay has the wrong role")
    segments = replay.get("segments")
    if not isinstance(segments, list) or not segments:
        fail("Closure replay contains no segments")
    expected_from = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            fail(f"Closure replay segment {index} is invalid")
        if segment.get("fromPieces") != expected_from:
            fail(f"Closure replay segment {index} is not contiguous")
        frontier = require_int(
            segment.get("frontierPieces"),
            f"replay.segments[{index}].frontierPieces",
            1,
        )
        if frontier <= expected_from:
            fail(f"Closure replay segment {index} does not advance")
        expected_from = frontier
    if expected_from != target_pieces:
        fail("Closure replay does not reach the certified frontier")
    return replay, sha256(evidence / evidence_name)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    candidate = load_json(arguments.candidate, "closure candidate")
    if candidate.get("format") != DECISION_FORMAT:
        fail("Closure candidate has an unsupported format")
    role = require_text(candidate.get("role"), "candidate.role")
    if role not in ROLE_CODES:
        fail("Closure candidate has an unsupported role")
    from_pieces = require_int(candidate.get("fromPieces"), "candidate.fromPieces")
    target_pieces = require_int(candidate.get("targetPieces"), "candidate.targetPieces", 1)
    if target_pieces <= from_pieces:
        fail("Closure candidate has invalid piece boundaries")
    run_id = require_int(candidate.get("run"), "candidate.run", 1)
    run_sha = require_text(candidate.get("runSha"), "candidate.runSha", GIT_SHA_RE)
    result_artifact = require_text(
        candidate.get("resultArtifact"), "candidate.resultArtifact", ARTIFACT_RE
    )
    existing = require_int(
        candidate.get("existingRejectedRoots"), "candidate.existingRejectedRoots"
    )
    new = require_int(candidate.get("newRejectedRoots"), "candidate.newRejectedRoots")
    cumulative = require_int(
        candidate.get("cumulativeRejectedRoots"), "candidate.cumulativeRejectedRoots"
    )
    if new != 0 or cumulative != existing or candidate.get("closedCandidate") is not True:
        fail("Candidate is not a zero-counterexample closure")
    if candidate.get("nextState") is not None:
        fail("Closure candidate may not contain a next campaign state")
    expected_result = f"perfect-chaos-{role}-{target_pieces}-{existing}-round"
    if result_artifact != expected_result:
        fail(f"Closure candidate names the wrong result artifact: {result_artifact}")

    checksums = candidate.get("checksums")
    if not isinstance(checksums, dict):
        fail("candidate.checksums must be an object")
    producer_sums = validate_sums(arguments.producer)
    evidence_sums = validate_sums(arguments.evidence)
    round_manifest = require_text(
        checksums.get("roundManifest"), "checksums.roundManifest", SHA256_RE
    )
    evidence_manifest = require_text(
        checksums.get("evidenceManifest"), "checksums.evidenceManifest", SHA256_RE
    )
    if sha256(arguments.producer / "SHA256SUMS") != round_manifest:
        fail("Producer checksum manifest does not match the closure candidate")
    if sha256(arguments.evidence / "SHA256SUMS") != evidence_manifest:
        fail("Evidence checksum manifest does not match the closure candidate")

    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, dict):
        fail("candidate.artifacts must be an object")
    producer_identity = artifact_identity(artifacts.get("producer"), "artifacts.producer")
    evidence_identity = artifact_identity(
        artifacts.get("independentEvidence"), "artifacts.independentEvidence"
    )
    if producer_identity["run"] != run_id or evidence_identity["run"] != run_id:
        fail("Candidate artifacts do not belong to the certified workflow run")
    if producer_identity["sha"] != run_sha or evidence_identity["sha"] != run_sha:
        fail("Candidate artifacts do not belong to the certified commit")
    if producer_identity["name"] != result_artifact:
        fail("Producer artifact name differs from the closure candidate")
    expected_evidence = f"perfect-chaos-{role}-{target_pieces}-{existing}-evidence"
    if evidence_identity["name"] != expected_evidence:
        fail("Independent evidence artifact has the wrong name")

    boundary = from_pieces
    shared = (
        "campaign-summary.json",
        "classification.json",
        f"new-reject-{boundary}.bin",
        f"reject-{boundary}.bin",
    )
    for name in shared:
        if name not in producer_sums or name not in evidence_sums:
            fail(f"{name} is not covered by both checksum manifests")
        same_bytes(arguments.producer / name, arguments.evidence / name, name)

    summary = load_json(arguments.producer / "campaign-summary.json", "campaign summary")
    classification = load_json(arguments.producer / "classification.json", "classification")
    for label, value in (("summary", summary), ("classification", classification)):
        if value.get("format") != SUMMARY_FORMAT:
            fail(f"{label} has an unsupported format")
        if (
            value.get("role") != role
            or value.get("fromPieces") != from_pieces
            or value.get("targetPieces") != target_pieces
        ):
            fail(f"{label} has incompatible identity")
        if value.get("classificationComplete") is not True or value.get("policyConflicts") != 0:
            fail(f"{label} is incomplete or conflicted")
    for key, value in classification.items():
        if summary.get(key) != value:
            fail(f"Campaign summary differs from classification at {key}")
    expected_accounting = {
        "existingRejectedRoots": existing,
        "newRejectedRoots": 0,
        "cumulativeRejectedRoots": cumulative,
        "rejectionProgress": 0,
        "rejectedRoots": 0,
    }
    for key, value in expected_accounting.items():
        if summary.get(key) != value:
            fail(f"Campaign summary has invalid {key}")
    input_roots = require_int(summary.get("inputRoots"), "summary.inputRoots", 1)
    safe_roots = require_int(summary.get("safeInputRoots"), "summary.safeInputRoots")
    safe_policy = require_int(summary.get("safePolicyEntries"), "summary.safePolicyEntries")
    safe_frontier = require_int(summary.get("safeFrontierStates"), "summary.safeFrontierStates")
    if safe_roots != input_roots:
        fail("A closure candidate must classify every input root as safe")

    audit_name = (
        "raw-shard-audit.json"
        if "raw-shard-audit.json" in evidence_sums
        else "audit-report.json"
    )
    if audit_name not in evidence_sums:
        fail("Independent evidence contains no raw-shard audit report")
    audit = load_json(arguments.evidence / audit_name, "raw-shard audit")
    expected_audit = {
        "format": AUDIT_FORMAT,
        "status": "pass",
        "role": role,
        "fromPieces": from_pieces,
        "targetPieces": target_pieces,
        "inputRoots": input_roots,
        "existingRejectedRoots": existing,
        "newRejectedRoots": 0,
        "cumulativeRejectedRoots": cumulative,
        "safeInputRoots": safe_roots,
        "safePolicyEntries": safe_policy,
        "safeFrontierStates": safe_frontier,
        "policyConflicts": 0,
    }
    for key, value in expected_audit.items():
        if audit.get(key) != value:
            fail(f"Independent audit has invalid {key}")
    shards = require_int(audit.get("shards"), "audit.shards", 1)

    expected_files = {
        "newRejectSha256": f"new-reject-{boundary}.bin",
        "cumulativeRejectSha256": f"reject-{boundary}.bin",
        "policySha256": f"{from_pieces}-{target_pieces}.policy.bin",
        "frontierSha256": f"{from_pieces}-{target_pieces}.frontier.bin",
    }
    proof_tables = audit.get("proofTables")
    if not isinstance(proof_tables, dict):
        fail("Independent audit has no proof-table identities")
    for key, relative in expected_files.items():
        expected = require_text(
            proof_tables.get(key), f"audit.proofTables.{key}", SHA256_RE
        )
        target = arguments.producer / relative
        if (
            relative not in producer_sums
            or producer_sums[relative] != expected
            or sha256(target) != expected
        ):
            fail(f"Proof-table identity mismatch for {relative}")

    rejected = read_table(
        arguments.producer / f"new-reject-{boundary}.bin",
        FRONTIER_MAGIC,
        FRONTIER_RECORD_SIZE,
    )
    cumulative_table = read_table(
        arguments.producer / f"reject-{boundary}.bin",
        FRONTIER_MAGIC,
        FRONTIER_RECORD_SIZE,
    )
    policy = read_table(
        arguments.producer / f"{from_pieces}-{target_pieces}.policy.bin",
        POLICY_MAGIC,
        POLICY_RECORD_SIZE,
    )
    frontier = read_table(
        arguments.producer / f"{from_pieces}-{target_pieces}.frontier.bin",
        FRONTIER_MAGIC,
        FRONTIER_RECORD_SIZE,
    )
    expected_role_code = ROLE_CODES[role]
    for label, table in (
        ("new rejections", rejected),
        ("cumulative rejections", cumulative_table),
    ):
        if table.role != expected_role_code or table.boundary != boundary:
            fail(f"{label} table has incompatible metadata")
    if rejected.records or len(cumulative_table.records) != cumulative:
        fail("Rejection-table counts disagree with the closure accounting")
    for label, table in (("policy", policy), ("frontier", frontier)):
        if table.role != expected_role_code or table.boundary != target_pieces:
            fail(f"{label} table has incompatible metadata")
    if len(policy.records) != safe_policy or len(frontier.records) != safe_frontier:
        fail("Policy or frontier table count differs from the independently audited count")

    replay, replay_digest = validate_replay(
        arguments.producer,
        arguments.evidence,
        producer_sums,
        evidence_sums,
        role,
        target_pieces,
    )
    candidate_replay_digest = require_text(
        checksums.get("closureReplay"), "checksums.closureReplay", SHA256_RE
    )
    if candidate_replay_digest != replay_digest:
        fail("Closure replay digest differs from the transition candidate")

    prepared = arguments.producer / f"{role}-prepared"
    assembled = arguments.producer / "assembled" / role
    prepared_names = regular_file_names(prepared)
    assembled_names = regular_file_names(assembled)
    if prepared_names != assembled_names:
        fail("Prepared and assembled closure file sets differ")
    for name in prepared_names:
        same_bytes(prepared / name, assembled / name, f"assembled/{name}")
    for relative in (
        f"{from_pieces}-{target_pieces}.policy.bin",
        f"{from_pieces}-{target_pieces}.frontier.bin",
    ):
        same_bytes(arguments.producer / relative, prepared / relative, relative)

    segments = replay["segments"]
    certificate = {
        "format": CERTIFICATE_FORMAT,
        "role": role,
        "fromPieces": from_pieces,
        "frontierPieces": target_pieces,
        "cumulativeRejectedRoots": cumulative,
        "source": producer_identity,
        "independentEvidence": evidence_identity,
        "classification": {
            "shards": shards,
            "inputRoots": input_roots,
            "safeInputRoots": safe_roots,
            "safePolicyEntries": safe_policy,
            "safeFrontierStates": safe_frontier,
            "newRejectedRoots": 0,
            "classificationComplete": True,
            "policyConflicts": 0,
        },
        "proofTables": {key: proof_tables[key] for key in sorted(expected_files)},
        "replay": {
            "segments": len(segments),
            "finalFrontierStates": segments[-1].get("frontierStates"),
            "finalPolicyEntries": segments[-1].get("policyEntries"),
            "sha256": replay_digest,
        },
        "transitionDecisionSha256": sha256(arguments.candidate),
        "manifests": {
            "producerSha256": sha256(arguments.producer / "SHA256SUMS"),
            "evidenceSha256": sha256(arguments.evidence / "SHA256SUMS"),
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(certificate, indent=2) + "\n")
    print(json.dumps(certificate, indent=2))


if __name__ == "__main__":
    main()
