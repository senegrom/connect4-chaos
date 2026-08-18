#!/usr/bin/env python3
"""Write and verify strict, directory-relative SHA-256 artifact manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

DIGEST_LINE = re.compile(r"([0-9a-f]{64})  (.+)")
TRANSIENT_DIRECTORY = re.compile(r"\.incremental-repair-[0-9]+-[0-9]+")


def safe_relative(value: str, label: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise RuntimeError(f"{label} must be a non-empty POSIX relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Unsafe {label}: {value!r}.")
    if path.as_posix() != value:
        raise RuntimeError(f"Non-canonical {label}: {value!r}.")
    return path


def manifest_path(root: Path, value: str) -> Path:
    relative = safe_relative(value, "manifest path")
    return ensure_no_symlink(root, relative)


def transient_artifact_path(relative: PurePosixPath) -> bool:
    """Return whether a path belongs to exact-repair scratch space.

    Incremental repair work directories contain only intermediate partitions and
    regenerated fragments. Durable policy, frontier, rejection, checkpoint, and
    replay files live outside them. GitHub excludes hidden directories from
    artifact uploads by default, so these paths cannot be certificate identity.
    Only the explicit boundary-labelled scratch convention is excluded.
    """
    return any(TRANSIENT_DIRECTORY.fullmatch(part) for part in relative.parts)


def ensure_no_symlink(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RuntimeError(
                f"Artifact paths may not traverse symlinks: {relative.as_posix()!r}."
            )
    return current


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_files(root: Path, manifest: Path) -> list[tuple[str, Path]]:
    records: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        posix = relative.as_posix()
        if path.is_symlink():
            raise RuntimeError(f"Artifact trees may not contain symlinks: {posix!r}.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(f"Artifact tree contains a non-regular file: {posix!r}.")
        if path == manifest:
            continue
        if transient_artifact_path(PurePosixPath(posix)):
            continue
        records.append((posix, path))
    records.sort(key=lambda item: item[0])
    return records


def write_manifest(root: Path, manifest: Path) -> None:
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(f".{manifest.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(
            f"Refusing to overwrite a stale manifest temporary: {temporary}."
        )
    records = artifact_files(root, manifest)
    payload = "".join(f"{sha256(path)}  {relative}\n" for relative, path in records)
    temporary.write_text(payload, newline="\n")
    os.replace(temporary, manifest)
    print(f"Wrote {len(records)} checksum record(s) to {manifest}.")


def read_manifest(root: Path, manifest: Path) -> dict[str, str]:
    if manifest.is_symlink() or not manifest.is_file():
        raise RuntimeError(f"Checksum manifest is not a regular file: {manifest}.")
    selected: dict[str, str] = {}
    seen: set[str] = set()
    for line_number, raw in enumerate(manifest.read_text().splitlines(), start=1):
        match = DIGEST_LINE.fullmatch(raw)
        if not match:
            raise RuntimeError(f"Malformed checksum line {line_number} in {manifest}.")
        digest, value = match.groups()
        relative = safe_relative(value, f"manifest entry on line {line_number}")
        canonical = relative.as_posix()
        if canonical in seen:
            raise RuntimeError(f"Duplicate checksum entry: {canonical!r}.")
        seen.add(canonical)
        # Legacy staged artifacts mention hidden exact-repair scratch files that
        # GitHub correctly omitted. Ignore only this explicit transient namespace.
        if transient_artifact_path(relative):
            continue
        target = ensure_no_symlink(root, relative)
        if not target.is_file():
            raise RuntimeError(f"Checksum entry is not a regular file: {canonical!r}.")
        selected[canonical] = digest
    return selected


def verify_manifest(root: Path, manifest: Path) -> None:
    selected = read_manifest(root, manifest)
    actual = {relative: path for relative, path in artifact_files(root, manifest)}
    missing = sorted(set(selected) - set(actual))
    unlisted = sorted(set(actual) - set(selected))
    if missing:
        raise RuntimeError(f"Checksum manifest references missing file(s): {missing!r}.")
    if unlisted:
        raise RuntimeError(f"Artifact tree contains unlisted file(s): {unlisted!r}.")
    for relative in sorted(selected):
        digest = sha256(actual[relative])
        if digest != selected[relative]:
            raise RuntimeError(f"Checksum mismatch: {relative!r}.")
    print(f"Verified {len(selected)} checksum record(s) from {manifest}.")



def load_json(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular JSON file: {path}.")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object.")
    return value


def require_integer(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{label} must be an integer of at least {minimum}.")
    return value


def require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise RuntimeError(f"{label} must be a sha256-prefixed lowercase digest.")
    return value


def require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise RuntimeError(f"{label} must be a lowercase 40-character commit SHA.")
    return value


def require_artifact_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise RuntimeError(f"{label} must be a safe GitHub artifact name.")
    return value


def write_checkpoint(args: argparse.Namespace) -> None:
    result_root = args.result_directory.resolve()
    evidence_root = args.evidence_directory.resolve()
    for root, label in ((result_root, "result"), (evidence_root, "evidence")):
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"{label.capitalize()} directory is missing or unsafe: {root}.")
        verify_manifest(root, manifest_path(root, args.manifest))

    if args.role not in {"red", "yellow"}:
        raise RuntimeError("Checkpoint role must be red or yellow.")
    from_pieces = require_integer(args.from_pieces, "from-pieces")
    target_pieces = require_integer(args.target_pieces, "target-pieces", from_pieces + 1)
    if target_pieces > 42:
        raise RuntimeError("target-pieces cannot exceed 42.")

    summary = load_json(result_root / "campaign-summary.json", "campaign summary")
    classification = load_json(result_root / "classification.json", "classification")
    audit = load_json(evidence_root / "raw-shard-audit.json", "raw-shard audit")
    required_summary = {
        "format": "connect4-chaos-frontier-classification-merged-v1",
        "role": args.role,
        "fromPieces": from_pieces,
        "targetPieces": target_pieces,
        "classificationComplete": True,
        "policyConflicts": 0,
    }
    for field, expected in required_summary.items():
        if summary.get(field) != expected:
            raise RuntimeError(
                f"campaign-summary.json field {field} must be {expected!r}; "
                f"received {summary.get(field)!r}."
            )
    # The accounting summary extends the merged classification. Every
    # classification field must still agree exactly with the result summary.
    for field, value in classification.items():
        if summary.get(field) != value:
            raise RuntimeError(f"Classification and campaign summary differ at {field}.")

    required_audit = {
        "format": "connect4-chaos-independent-sharded-round-audit-v1",
        "status": "pass",
        "role": args.role,
        "fromPieces": from_pieces,
        "targetPieces": target_pieces,
        "classificationComplete": True,
        "policyConflicts": 0,
    }
    for field, expected in required_audit.items():
        if field == "classificationComplete" and field not in audit:
            continue
        if audit.get(field) != expected:
            raise RuntimeError(
                f"raw-shard-audit.json field {field} must be {expected!r}; "
                f"received {audit.get(field)!r}."
            )

    count_fields = (
        "inputRoots",
        "existingRejectedRoots",
        "newRejectedRoots",
        "cumulativeRejectedRoots",
        "safeInputRoots",
        "safePolicyEntries",
        "safeFrontierStates",
        "policyConflicts",
    )
    for field in count_fields:
        result_value = require_integer(summary.get(field), f"campaign summary {field}")
        audit_value = require_integer(audit.get(field), f"raw-shard audit {field}")
        if result_value != audit_value:
            raise RuntimeError(f"Result and independent audit differ at {field}.")
    if summary["cumulativeRejectedRoots"] != (
        summary["existingRejectedRoots"] + summary["newRejectedRoots"]
    ):
        raise RuntimeError("Campaign rejection accounting is inconsistent.")

    new_reject = f"new-reject-{from_pieces}.bin"
    cumulative_reject = f"reject-{from_pieces}.bin"
    policy = f"{from_pieces}-{target_pieces}.policy.bin"
    frontier = f"{from_pieces}-{target_pieces}.frontier.bin"
    evidence_core = (
        "campaign-summary.json",
        "classification.json",
        new_reject,
        cumulative_reject,
    )
    for relative in evidence_core:
        result_path = result_root / relative
        evidence_path = evidence_root / relative
        if (
            result_path.is_symlink()
            or evidence_path.is_symlink()
            or not result_path.is_file()
            or not evidence_path.is_file()
        ):
            raise RuntimeError(f"Result/evidence core file is missing or unsafe: {relative}.")
        if result_path.read_bytes() != evidence_path.read_bytes():
            raise RuntimeError(f"Independent evidence differs from the result: {relative}.")

    proof_paths = {
        new_reject: result_root / new_reject,
        cumulative_reject: result_root / cumulative_reject,
        policy: result_root / policy,
        frontier: result_root / frontier,
    }
    for relative, target in proof_paths.items():
        if target.is_symlink() or not target.is_file():
            raise RuntimeError(f"Proof file is missing or unsafe: {relative}.")

    checkpoint = {
        "format": "connect4-chaos-certified-refinement-checkpoint-v1",
        "role": args.role,
        "fromPieces": from_pieces,
        "targetPieces": target_pieces,
        "cumulativeRejectedRoots": summary["cumulativeRejectedRoots"],
        "result": {
            "run": require_integer(args.run, "run", 1),
            "sha": require_sha(args.source_sha, "source-sha"),
            "artifact": require_artifact_name(args.result_artifact, "result-artifact"),
            "artifactId": require_integer(args.result_artifact_id, "result-artifact-id", 1),
            "digest": require_digest(args.result_digest, "result-digest"),
        },
        "independentEvidence": {
            "artifact": require_artifact_name(args.evidence_artifact, "evidence-artifact"),
            "artifactId": require_integer(
                args.evidence_artifact_id, "evidence-artifact-id", 1
            ),
            "digest": require_digest(args.evidence_digest, "evidence-digest"),
        },
        "classification": {
            field: summary[field]
            for field in count_fields
        } | {"classificationComplete": True},
        "proofFileSha256": {
            relative: sha256(target)
            for relative, target in proof_paths.items()
        },
        "verification": {
            "roundManifest": "pass",
            "evidenceManifest": "pass",
            "evidenceCoreFilesByteIdentical": True,
            "rawShardAudit": "pass",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checkpoint, indent=2) + "\n", newline="\n")
    print(json.dumps(checkpoint), flush=True)

def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("write", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--directory", required=True, type=Path)
        subparser.add_argument("--manifest", default="SHA256SUMS")

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--result-directory", required=True, type=Path)
    checkpoint.add_argument("--evidence-directory", required=True, type=Path)
    checkpoint.add_argument("--role", required=True)
    checkpoint.add_argument("--from-pieces", required=True, type=int)
    checkpoint.add_argument("--target-pieces", required=True, type=int)
    checkpoint.add_argument("--run", required=True, type=int)
    checkpoint.add_argument("--source-sha", required=True)
    checkpoint.add_argument("--result-artifact", required=True)
    checkpoint.add_argument("--result-artifact-id", required=True, type=int)
    checkpoint.add_argument("--result-digest", required=True)
    checkpoint.add_argument("--evidence-artifact", required=True)
    checkpoint.add_argument("--evidence-artifact-id", required=True, type=int)
    checkpoint.add_argument("--evidence-digest", required=True)
    checkpoint.add_argument("--output", required=True, type=Path)
    checkpoint.add_argument("--manifest", default="SHA256SUMS")
    args = parser.parse_args()

    if args.command == "checkpoint":
        write_checkpoint(args)
        return

    root = args.directory.resolve()
    if not root.is_dir():
        raise RuntimeError(f"Artifact directory does not exist: {root}.")
    manifest = manifest_path(root, args.manifest)
    if args.command == "write":
        write_manifest(root, manifest)
    else:
        verify_manifest(root, manifest)


if __name__ == "__main__":
    main()
