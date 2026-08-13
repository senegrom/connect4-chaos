#!/usr/bin/env python3
"""Independently audit one Perfect Chaos frontier-classification round.

The producer workflow is intentionally not trusted. This verifier checks both
artifact manifests, binary table framing, predecessor/new rejection set
accounting, embedded cumulative tables, merged classification metadata, and
all hashes without importing the producer's merge code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

FRONTIER_MAGIC = b"C4CFRN1\x00"
POLICY_MAGIC = b"C4CPOL1\x00"
FRONTIER_RECORD_SIZE = 19
POLICY_RECORD_SIZE = 20
ROLE_CODES = {"red": 1, "yellow": 2}
ACCOUNTING_FIELDS = {
    "existingRejectedRoots",
    "newRejectedRoots",
    "cumulativeRejectedRoots",
    "rejectionProgress",
}


class AuditError(RuntimeError):
    """A fail-closed audit rejection."""


@dataclass(frozen=True)
class Table:
    path: Path
    magic: bytes
    version: int
    role: int
    boundary: int
    record_size: int
    count: int
    records: tuple[bytes, ...]
    size: int
    sha256: str


def fail(message: str) -> None:
    raise AuditError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_plain_tree(root: Path) -> list[Path]:
    if not root.is_dir():
        fail(f"Artifact directory does not exist: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            fail(f"Artifact contains a symlink: {path.relative_to(root)}")
        if path.is_dir():
            continue
        if not path.is_file():
            fail(f"Artifact contains a non-regular entry: {path.relative_to(root)}")
        files.append(path)
    return sorted(files)


def verify_sha256sums(root: Path) -> dict[str, str]:
    files = require_plain_tree(root)
    manifest = root / "SHA256SUMS"
    if manifest not in files:
        fail(f"Artifact is missing SHA256SUMS: {root}")

    entries: dict[str, str] = {}
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            fail(f"SHA256SUMS contains a blank line at {line_number}: {root}")
        try:
            digest, relative = raw.split("  ", 1)
        except ValueError as error:
            raise AuditError(f"Malformed SHA256SUMS line {line_number}: {raw!r}") from error
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            fail(f"Invalid SHA-256 digest on line {line_number}: {digest!r}")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative in {"", ".", "SHA256SUMS"}:
            fail(f"Unsafe SHA256SUMS path on line {line_number}: {relative!r}")
        if relative in entries:
            fail(f"Duplicate SHA256SUMS path: {relative}")
        path = root / Path(*pure.parts)
        if not path.is_file() or path.is_symlink():
            fail(f"SHA256SUMS references a missing or unsafe file: {relative}")
        actual = sha256_file(path)
        if actual != digest:
            fail(f"SHA-256 mismatch for {relative}: {actual} != {digest}")
        entries[relative] = digest

    actual_files = {
        path.relative_to(root).as_posix()
        for path in files
        if path != manifest
    }
    listed_files = set(entries)
    missing = sorted(actual_files - listed_files)
    extra = sorted(listed_files - actual_files)
    if missing or extra:
        fail(f"SHA256SUMS is not exhaustive: unlisted={missing}, missing={extra}")
    return entries


def read_table(path: Path, expected_magic: bytes, expected_record_size: int) -> Table:
    if not path.is_file() or path.is_symlink():
        fail(f"Required proof table is missing or unsafe: {path}")
    data = path.read_bytes()
    if len(data) < 16:
        fail(f"Proof table is shorter than its header: {path}")
    magic = data[:8]
    version, role, boundary, record_size = data[8:12]
    count = struct.unpack_from("<I", data, 12)[0]
    if magic != expected_magic:
        fail(f"Unexpected table magic in {path}: {magic!r}")
    if version != 1:
        fail(f"Unsupported table version in {path}: {version}")
    if record_size != expected_record_size:
        fail(f"Unexpected record size in {path}: {record_size}")
    expected_size = 16 + count * record_size
    if len(data) != expected_size:
        fail(f"Table length mismatch in {path}: {len(data)} != {expected_size}")
    records = tuple(
        data[16 + index * record_size : 16 + (index + 1) * record_size]
        for index in range(count)
    )
    if len(set(records)) != count:
        fail(f"Proof table contains duplicate records: {path}")
    return Table(
        path=path,
        magic=magic,
        version=version,
        role=role,
        boundary=boundary,
        record_size=record_size,
        count=count,
        records=records,
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        fail(f"Required JSON file is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AuditError(f"Cannot decode JSON file {path}: {error}") from error
    if not isinstance(value, dict):
        fail(f"JSON proof metadata must be an object: {path}")
    return value


def require_table_metadata(table: Table, role: int, boundary: int) -> None:
    if table.role != role or table.boundary != boundary:
        fail(
            f"Table metadata mismatch in {table.path}: "
            f"role={table.role}, boundary={table.boundary}; "
            f"expected role={role}, boundary={boundary}"
        )


def require_nonnegative_integer(mapping: dict[str, Any], field: str) -> int:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"Metadata field {field!r} is not a non-negative integer: {value!r}")
    return value


def audit(args: argparse.Namespace) -> dict[str, Any]:
    role_name = args.role
    role = ROLE_CODES[role_name]
    prior_root = args.prior_directory.resolve()
    current_root = args.current_directory.resolve()
    if prior_root == current_root:
        fail("Prior and current artifact directories must differ.")

    prior_sums = verify_sha256sums(prior_root)
    current_sums = verify_sha256sums(current_root)

    prior_summary = load_json(prior_root / "campaign-summary.json")
    summary = load_json(current_root / "campaign-summary.json")
    classification = load_json(current_root / "classification.json")

    prior_reject = read_table(
        prior_root / f"{role_name}-prepared" / f"reject-{args.from_pieces}.bin",
        FRONTIER_MAGIC,
        FRONTIER_RECORD_SIZE,
    )
    new_reject = read_table(
        current_root / f"new-reject-{args.from_pieces}.bin",
        FRONTIER_MAGIC,
        FRONTIER_RECORD_SIZE,
    )
    cumulative_reject = read_table(
        current_root / f"reject-{args.from_pieces}.bin",
        FRONTIER_MAGIC,
        FRONTIER_RECORD_SIZE,
    )
    embedded_cumulative = read_table(
        current_root / f"{role_name}-prepared" / f"reject-{args.from_pieces}.bin",
        FRONTIER_MAGIC,
        FRONTIER_RECORD_SIZE,
    )
    frontier = read_table(
        current_root / f"{args.from_pieces}-{args.target_pieces}.frontier.bin",
        FRONTIER_MAGIC,
        FRONTIER_RECORD_SIZE,
    )
    policy = read_table(
        current_root / f"{args.from_pieces}-{args.target_pieces}.policy.bin",
        POLICY_MAGIC,
        POLICY_RECORD_SIZE,
    )

    for table in (prior_reject, new_reject, cumulative_reject, embedded_cumulative):
        require_table_metadata(table, role, args.from_pieces)
    require_table_metadata(frontier, role, args.target_pieces)
    require_table_metadata(policy, role, args.target_pieces)

    prior_records = set(prior_reject.records)
    new_records = set(new_reject.records)
    cumulative_records = set(cumulative_reject.records)
    overlap = prior_records & new_records
    if overlap:
        fail(f"New rejection table overlaps the predecessor by {len(overlap)} record(s).")
    expected_cumulative = prior_records | new_records
    if cumulative_records != expected_cumulative:
        missing = len(expected_cumulative - cumulative_records)
        unexpected = len(cumulative_records - expected_cumulative)
        fail(f"Cumulative rejection set mismatch: missing={missing}, unexpected={unexpected}")
    if embedded_cumulative.sha256 != cumulative_reject.sha256:
        fail("The embedded prepared rejection table is not byte-identical to the cumulative table.")

    if prior_summary.get("role") != role_name:
        fail("Predecessor campaign-summary role does not match the audited role.")
    if prior_summary.get("fromPieces") != args.from_pieces or prior_summary.get("targetPieces") != args.target_pieces:
        fail("Predecessor campaign-summary boundaries are incompatible.")
    if prior_summary.get("classificationComplete") is not True or prior_summary.get("policyConflicts") != 0:
        fail("Predecessor campaign summary is incomplete or conflicted.")
    if prior_summary.get("cumulativeRejectedRoots") != prior_reject.count:
        fail("Predecessor cumulative rejection count does not match its proof table.")

    required_exact = {
        "role": role_name,
        "fromPieces": args.from_pieces,
        "targetPieces": args.target_pieces,
        "classificationComplete": True,
        "policyConflicts": 0,
        "duplicateRejectedRecords": 0,
    }
    for field, expected in required_exact.items():
        if summary.get(field) != expected:
            fail(f"Campaign summary field {field!r} is {summary.get(field)!r}, expected {expected!r}.")

    input_roots = require_nonnegative_integer(summary, "inputRoots")
    rejected_roots = require_nonnegative_integer(summary, "rejectedRoots")
    safe_roots = require_nonnegative_integer(summary, "safeInputRoots")
    policy_entries = require_nonnegative_integer(summary, "safePolicyEntries")
    frontier_states = require_nonnegative_integer(summary, "safeFrontierStates")
    existing = require_nonnegative_integer(summary, "existingRejectedRoots")
    new = require_nonnegative_integer(summary, "newRejectedRoots")
    cumulative = require_nonnegative_integer(summary, "cumulativeRejectedRoots")
    progress = require_nonnegative_integer(summary, "rejectionProgress")

    if input_roots != rejected_roots + safe_roots:
        fail("Input-root accounting does not partition into rejected and safe roots.")
    if rejected_roots != new_reject.count:
        fail("Merged rejectedRoots does not match the new rejection table.")
    if policy_entries != policy.count or frontier_states != frontier.count:
        fail("Merged policy/frontier counts do not match their binary tables.")
    if existing != prior_reject.count or new != new_reject.count:
        fail("Campaign rejection accounting does not match predecessor/new tables.")
    if cumulative != cumulative_reject.count or cumulative != existing + new or progress != new:
        fail("Campaign cumulative rejection accounting is inconsistent.")

    for field, value in classification.items():
        if summary.get(field) != value:
            fail(f"campaign-summary.json diverges from classification.json at {field!r}.")
    missing_accounting = sorted(ACCOUNTING_FIELDS - set(summary))
    if missing_accounting:
        fail(f"Campaign summary is missing accounting fields: {missing_accounting}")

    artifact_metadata = summary.get("artifacts")
    if not isinstance(artifact_metadata, dict):
        fail("Campaign summary has no artifact metadata object.")
    expected_artifacts = {
        "rejected": new_reject,
        "policy": policy,
        "frontier": frontier,
    }
    for name, table in expected_artifacts.items():
        value = artifact_metadata.get(name)
        if not isinstance(value, dict):
            fail(f"Campaign summary is missing {name!r} artifact metadata.")
        if value.get("bytes") != table.size or value.get("sha256") != table.sha256:
            fail(f"Campaign {name} artifact metadata does not match the binary table.")
        if value.get("path") != table.path.relative_to(current_root).as_posix():
            fail(f"Campaign {name} artifact path is not exact.")

    return {
        "format": "connect4-chaos-independent-round-audit-v1",
        "status": "pass",
        "role": role_name,
        "fromPieces": args.from_pieces,
        "targetPieces": args.target_pieces,
        "priorRejectedRoots": prior_reject.count,
        "newRejectedRoots": new_reject.count,
        "cumulativeRejectedRoots": cumulative_reject.count,
        "inputRoots": input_roots,
        "safeInputRoots": safe_roots,
        "safePolicyEntries": policy.count,
        "safeFrontierStates": frontier.count,
        "policyConflicts": summary["policyConflicts"],
        "priorManifestEntries": len(prior_sums),
        "currentManifestEntries": len(current_sums),
        "proofTables": {
            "priorRejectSha256": prior_reject.sha256,
            "newRejectSha256": new_reject.sha256,
            "cumulativeRejectSha256": cumulative_reject.sha256,
            "policySha256": policy.sha256,
            "frontierSha256": frontier.sha256,
        },
    }


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(ROLE_CODES))
    parser.add_argument("--prior-directory", required=True, type=Path)
    parser.add_argument("--current-directory", required=True, type=Path)
    parser.add_argument("--from-pieces", type=int, default=14)
    parser.add_argument("--target-pieces", type=int, default=16)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.from_pieces < 0 or args.target_pieces <= args.from_pieces:
        parser.error("target pieces must be greater than from pieces")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        report = audit(args)
    except (AuditError, OSError) as error:
        print(f"audit failed: {error}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
