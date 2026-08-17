#!/usr/bin/env python3
"""Independently reconstruct and audit a sharded Perfect Chaos round.

This verifier deliberately does not import the producer's table or merge
helpers. It validates the staged frontier, every exact shard, all binary
records, cross-shard policy consistency, the independently reconstructed
merged tables, cumulative rejection accounting, and the published result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

FRONTIER_MAGIC = b"C4CFRN1\x00"
POLICY_MAGIC = b"C4CPOL1\x00"
FRONTIER_RECORD_SIZE = 19
POLICY_RECORD_SIZE = 20
ROLE_CODES = {"red": 1, "yellow": 2}
SHARD_FORMAT = "connect4-chaos-frontier-classification-shard-v1"
MERGED_FORMAT = "connect4-chaos-frontier-classification-merged-v1"


class AuditError(RuntimeError):
    """A fail-closed audit rejection."""


@dataclass(frozen=True)
class Table:
    path: Path
    magic: bytes
    role: int
    boundary: int
    record_size: int
    records: tuple[bytes, ...]
    payload: bytes

    @property
    def count(self) -> int:
        return len(self.records)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class Shard:
    index: int
    summary: dict[str, Any]
    rejected: Table
    policy: Table
    frontier: Table


def fail(message: str) -> None:
    raise AuditError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_plain_files(root: Path, *, allow_directories: bool) -> list[Path]:
    if not root.is_dir():
        fail(f"Directory does not exist: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            fail(f"Directory contains a symlink: {path.relative_to(root)}")
        if path.is_dir():
            if not allow_directories and path != root:
                fail(f"Flat shard directory contains a subdirectory: {path.relative_to(root)}")
            continue
        if not path.is_file():
            fail(f"Directory contains a non-regular entry: {path.relative_to(root)}")
        files.append(path)
    return sorted(files)


def verify_sha256sums(root: Path) -> dict[str, str]:
    files = require_plain_files(root, allow_directories=True)
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
    actual = {
        path.relative_to(root).as_posix()
        for path in files
        if path != manifest
    }
    listed = set(entries)
    if actual != listed:
        fail(
            "SHA256SUMS is not exhaustive: "
            f"unlisted={sorted(actual - listed)}, missing={sorted(listed - actual)}"
        )
    return entries


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


def record_key(record: bytes, record_size: int) -> tuple[int, int, int, int, int]:
    mover, opponent = struct.unpack_from("<QQ", record, 0)
    rows = record[16]
    columns = record[17]
    ai_turn = record[18] if record_size == FRONTIER_RECORD_SIZE else 1
    return rows, columns, ai_turn, mover, opponent


def action_key(record: bytes) -> tuple[int, int]:
    return record[18], record[19]


def mirror_mask(mask: int, rows: int, columns: int) -> int:
    stride = rows + 1
    mirrored = 0
    for column in range(columns):
        target = columns - 1 - column
        for row in range(rows):
            source = 1 << (column * stride + row)
            if mask & source:
                mirrored |= 1 << (target * stride + row)
    return mirrored


def validate_record(record: bytes, record_size: int, boundary: int, path: Path) -> None:
    if len(record) != record_size:
        fail(f"Record has the wrong size in {path}")
    mover, opponent = struct.unpack_from("<QQ", record, 0)
    rows = record[16]
    columns = record[17]
    if (rows, columns) not in {(6, 7), (7, 6)}:
        fail(f"Record has an invalid board orientation in {path}")
    stride = rows + 1
    used_bits = columns * stride
    occupied = mover | opponent
    if used_bits > 63 or occupied >> used_bits:
        fail(f"Record exceeds the sentinel mask boundary in {path}")
    if mover & opponent:
        fail(f"Record has overlapping mover/opponent masks in {path}")
    for column in range(columns):
        sentinel = 1 << (column * stride + rows)
        if occupied & sentinel:
            fail(f"Record sets a sentinel bit in {path}")
        column_bits = (occupied >> (column * stride)) & ((1 << rows) - 1)
        if column_bits & (column_bits + 1):
            fail(f"Record violates gravity in {path}")
    reflected_mover = mirror_mask(mover, rows, columns)
    reflected_opponent = mirror_mask(opponent, rows, columns)
    if reflected_mover < mover or (
        reflected_mover == mover and reflected_opponent < opponent
    ):
        fail(f"Record is not horizontally canonical in {path}")
    pieces = occupied.bit_count()
    if record_size == FRONTIER_RECORD_SIZE:
        if record[18] not in (0, 1):
            fail(f"Frontier record has an invalid side-to-move flag in {path}")
        if pieces != boundary:
            fail(f"Frontier record has the wrong piece count in {path}")
    else:
        if pieces >= boundary:
            fail(f"Policy record lies at or beyond its frontier in {path}")
        action_type, action_column = action_key(record)
        if action_type > 3:
            fail(f"Policy record has an invalid action type in {path}")
        if action_type == 0:
            if action_column >= columns:
                fail(f"Policy drop column is out of range in {path}")
        elif action_column != 0:
            fail(f"Policy transform action uses a nonzero column in {path}")


def read_table(
    path: Path,
    expected_magic: bytes,
    expected_record_size: int,
    expected_role: int,
    expected_boundary: int,
) -> Table:
    if not path.is_file() or path.is_symlink():
        fail(f"Required proof table is missing or unsafe: {path}")
    payload = path.read_bytes()
    if len(payload) < 16:
        fail(f"Proof table is shorter than its header: {path}")
    magic = payload[:8]
    version, role, boundary, record_size = payload[8:12]
    count = struct.unpack_from("<I", payload, 12)[0]
    if magic != expected_magic:
        fail(f"Unexpected table magic in {path}: {magic!r}")
    if version != 1 or record_size != expected_record_size:
        fail(f"Unsupported table header in {path}")
    if role != expected_role or boundary != expected_boundary:
        fail(f"Table role/boundary mismatch in {path}")
    expected_length = 16 + count * record_size
    if len(payload) != expected_length:
        fail(f"Table length mismatch in {path}: {len(payload)} != {expected_length}")
    records = tuple(
        payload[16 + index * record_size : 16 + (index + 1) * record_size]
        for index in range(count)
    )
    for record in records:
        validate_record(record, record_size, boundary, path)
    keys = [record_key(record, record_size) for record in records]
    if any(first >= second for first, second in zip(keys, keys[1:])):
        fail(f"Table records are not strictly sorted: {path}")
    return Table(path, magic, role, boundary, record_size, records, payload)


def encode_table(
    magic: bytes,
    role: int,
    boundary: int,
    record_size: int,
    records: Iterable[bytes],
) -> bytes:
    ordered = tuple(records)
    header = bytearray(16)
    header[:8] = magic
    header[8] = 1
    header[9] = role
    header[10] = boundary
    header[11] = record_size
    struct.pack_into("<I", header, 12, len(ordered))
    return bytes(header) + b"".join(ordered)


def nonnegative_integer(mapping: dict[str, Any], field: str, context: str) -> int:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{context} field {field!r} is not a non-negative integer: {value!r}")
    return value


def verify_artifact_metadata(
    root: Path,
    table: Table,
    metadata: Any,
    expected_name: str,
    context: str,
) -> None:
    if not isinstance(metadata, dict):
        fail(f"{context} is missing artifact metadata for {expected_name}")
    if metadata.get("path") != expected_name:
        fail(f"{context} has an inexact artifact path for {expected_name}")
    if metadata.get("bytes") != len(table.payload) or metadata.get("sha256") != table.sha256:
        fail(f"{context} artifact metadata does not match {expected_name}")
    if table.path.resolve() != (root / expected_name).resolve():
        fail(f"{context} artifact path escaped its root: {expected_name}")


def read_shards(
    root: Path,
    source: Table,
    role_name: str,
    role: int,
    target: int,
    shard_count: int,
) -> list[Shard]:
    files = require_plain_files(root, allow_directories=False)
    expected_names = {
        f"{kind}-{index}.{extension}"
        for index in range(shard_count)
        for kind, extension in (
            ("summary", "json"),
            ("rejected", "bin"),
            ("policy", "bin"),
            ("frontier", "bin"),
        )
    }
    actual_names = {path.name for path in files}
    if actual_names != expected_names:
        fail(
            "Shard file set is not exact: "
            f"unexpected={sorted(actual_names - expected_names)}, "
            f"missing={sorted(expected_names - actual_names)}"
        )

    shards: list[Shard] = []
    target_reject_digest: Any = object()
    for index in range(shard_count):
        context = f"Shard {index}"
        summary = load_json(root / f"summary-{index}.json")
        expected_roots = len(source.records[index::shard_count])
        exact = {
            "format": SHARD_FORMAT,
            "role": role_name,
            "fromPieces": source.boundary,
            "targetPieces": target,
            "shardIndex": index,
            "shardCount": shard_count,
            "inputRoots": expected_roots,
            "classificationComplete": True,
            "policyConflicts": 0,
        }
        for field, expected in exact.items():
            if summary.get(field) != expected:
                fail(f"{context} has invalid {field}: {summary.get(field)!r} != {expected!r}")
        if expected_roots == 0:
            fail(f"{context} is empty; shard count is too large")

        digest = summary.get("targetRejectSha256")
        if index == 0:
            target_reject_digest = digest
        elif digest != target_reject_digest:
            fail("Classification shards used different target rejection frontiers")

        rejected = read_table(
            root / f"rejected-{index}.bin",
            FRONTIER_MAGIC,
            FRONTIER_RECORD_SIZE,
            role,
            source.boundary,
        )
        policy = read_table(
            root / f"policy-{index}.bin",
            POLICY_MAGIC,
            POLICY_RECORD_SIZE,
            role,
            target,
        )
        frontier = read_table(
            root / f"frontier-{index}.bin",
            FRONTIER_MAGIC,
            FRONTIER_RECORD_SIZE,
            role,
            target,
        )
        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, dict):
            fail(f"{context} has no artifact metadata object")
        verify_artifact_metadata(
            root, rejected, artifacts.get("rejected"), f"rejected-{index}.bin", context
        )
        verify_artifact_metadata(
            root, policy, artifacts.get("policy"), f"policy-{index}.bin", context
        )
        verify_artifact_metadata(
            root, frontier, artifacts.get("frontier"), f"frontier-{index}.bin", context
        )

        safe = nonnegative_integer(summary, "safeInputRoots", context)
        rejected_count = nonnegative_integer(summary, "rejectedRoots", context)
        policy_count = nonnegative_integer(summary, "safePolicyEntries", context)
        frontier_count = nonnegative_integer(summary, "safeFrontierStates", context)
        for field in (
            "attempts",
            "splitEvents",
            "maximumSplitDepth",
            "safeLeaves",
            "rejectedLeaves",
        ):
            nonnegative_integer(summary, field, context)
        if safe + rejected_count != expected_roots:
            fail(f"{context} does not account for every input root")
        if rejected.count != rejected_count:
            fail(f"{context} rejection count does not match its table")
        if policy.count != policy_count:
            fail(f"{context} policy count does not match its table")
        if frontier.count != frontier_count:
            fail(f"{context} frontier count does not match its table")
        shards.append(Shard(index, summary, rejected, policy, frontier))
    return shards


def merge_frontier_records(records: Iterable[bytes]) -> tuple[tuple[bytes, ...], int]:
    selected: dict[tuple[int, int, int, int, int], bytes] = {}
    total = 0
    for record in records:
        total += 1
        selected[record_key(record, FRONTIER_RECORD_SIZE)] = record
    merged = tuple(selected[key] for key in sorted(selected))
    return merged, total - len(merged)


def merge_policy_records(records: Iterable[bytes]) -> tuple[tuple[bytes, ...], int, int]:
    selected: dict[tuple[int, int, int, int, int], bytes] = {}
    total = 0
    duplicate = 0
    conflicts = 0
    for record in records:
        total += 1
        key = record_key(record, POLICY_RECORD_SIZE)
        previous = selected.get(key)
        if previous is None:
            selected[key] = record
            continue
        duplicate += 1
        if action_key(previous) != action_key(record):
            conflicts += 1
            if action_key(record) < action_key(previous):
                selected[key] = record
    merged = tuple(selected[key] for key in sorted(selected))
    if total - len(merged) != duplicate:
        fail("Internal policy duplicate accounting mismatch")
    return merged, duplicate, conflicts


def file_metadata(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(payload), "sha256": sha256_bytes(payload)}


def exact_bytes(path: Path, expected: bytes, context: str) -> None:
    if not path.is_file() or path.is_symlink():
        fail(f"{context} is missing or unsafe: {path}")
    actual = path.read_bytes()
    if actual != expected:
        fail(
            f"{context} is not byte-identical to the independent reconstruction: "
            f"actual={sha256_bytes(actual)}, expected={sha256_bytes(expected)}"
        )


def audit(args: argparse.Namespace) -> dict[str, Any]:
    role_name = args.role
    role = ROLE_CODES[role_name]
    stage = args.stage_directory.resolve()
    shards_root = args.shard_directory.resolve()
    result = args.result_directory.resolve()
    if len({stage, shards_root, result}) != 3:
        fail("Stage, shard, and result directories must be distinct")

    stage_manifest = verify_sha256sums(stage)
    result_manifest = verify_sha256sums(result)
    source = read_table(
        stage / role_name / f"{args.from_pieces - 2}-{args.from_pieces}.frontier.bin",
        FRONTIER_MAGIC,
        FRONTIER_RECORD_SIZE,
        role,
        args.from_pieces,
    )
    existing = read_table(
        stage / role_name / f"reject-{args.from_pieces}.bin",
        FRONTIER_MAGIC,
        FRONTIER_RECORD_SIZE,
        role,
        args.from_pieces,
    )
    if not source.records:
        fail("The staged input frontier is empty")

    shards = read_shards(
        shards_root,
        source,
        role_name,
        role,
        args.target_pieces,
        args.shard_count,
    )

    rejected_records = [record for shard in shards for record in shard.rejected.records]
    policy_records = [record for shard in shards for record in shard.policy.records]
    frontier_records = [record for shard in shards for record in shard.frontier.records]
    merged_rejected, duplicate_rejected = merge_frontier_records(rejected_records)
    merged_policy, duplicate_policy, policy_conflicts = merge_policy_records(policy_records)
    merged_frontier, duplicate_frontier = merge_frontier_records(frontier_records)
    if policy_conflicts:
        fail(f"Cross-shard policy conflicts detected: {policy_conflicts}")

    new_reject_payload = encode_table(
        FRONTIER_MAGIC, role, args.from_pieces, FRONTIER_RECORD_SIZE, merged_rejected
    )
    policy_payload = encode_table(
        POLICY_MAGIC, role, args.target_pieces, POLICY_RECORD_SIZE, merged_policy
    )
    frontier_payload = encode_table(
        FRONTIER_MAGIC, role, args.target_pieces, FRONTIER_RECORD_SIZE, merged_frontier
    )
    exact_bytes(
        result / f"new-reject-{args.from_pieces}.bin",
        new_reject_payload,
        "Published new rejection table",
    )
    exact_bytes(
        result / f"{args.from_pieces}-{args.target_pieces}.policy.bin",
        policy_payload,
        "Published policy table",
    )
    exact_bytes(
        result / f"{args.from_pieces}-{args.target_pieces}.frontier.bin",
        frontier_payload,
        "Published frontier table",
    )

    existing_by_key = {
        record_key(record, FRONTIER_RECORD_SIZE): record for record in existing.records
    }
    new_by_key = {
        record_key(record, FRONTIER_RECORD_SIZE): record for record in merged_rejected
    }
    overlap = set(existing_by_key) & set(new_by_key)
    if overlap:
        fail(f"New rejection roots overlap the staged rejection set: {len(overlap)}")
    cumulative_records = tuple(
        value
        for key, value in sorted({**existing_by_key, **new_by_key}.items())
    )
    cumulative_payload = encode_table(
        FRONTIER_MAGIC, role, args.from_pieces, FRONTIER_RECORD_SIZE, cumulative_records
    )
    exact_bytes(
        result / f"reject-{args.from_pieces}.bin",
        cumulative_payload,
        "Published cumulative rejection table",
    )
    exact_bytes(
        result / f"{role_name}-prepared" / f"reject-{args.from_pieces}.bin",
        cumulative_payload,
        "Embedded cumulative rejection table",
    )

    stage_role_files = {
        path.relative_to(stage / role_name).as_posix(): path
        for path in require_plain_files(stage / role_name, allow_directories=False)
    }
    prepared_role_files = {
        path.relative_to(result / f"{role_name}-prepared").as_posix(): path
        for path in require_plain_files(
            result / f"{role_name}-prepared", allow_directories=False
        )
    }
    if set(stage_role_files) != set(prepared_role_files):
        fail("Prepared role file set does not exactly preserve the staged role directory")
    for relative, stage_path in stage_role_files.items():
        if relative == f"reject-{args.from_pieces}.bin":
            continue
        if stage_path.read_bytes() != prepared_role_files[relative].read_bytes():
            fail(f"Prepared role changed staged proof file: {relative}")

    summaries = [shard.summary for shard in shards]

    def sum_field(field: str) -> int:
        return sum(nonnegative_integer(summary, field, "Shard summary") for summary in summaries)

    target_reject_digest = summaries[0].get("targetRejectSha256")
    expected_classification = {
        "format": MERGED_FORMAT,
        "role": role_name,
        "fromPieces": args.from_pieces,
        "targetPieces": args.target_pieces,
        "shards": args.shard_count,
        "inputRoots": source.count,
        "rejectedRoots": len(merged_rejected),
        "safeInputRoots": sum_field("safeInputRoots"),
        "classificationComplete": True,
        "safePolicyEntries": len(merged_policy),
        "safeFrontierStates": len(merged_frontier),
        "policyConflicts": 0,
        "duplicateRejectedRecords": duplicate_rejected,
        "duplicateFrontierRecords": duplicate_frontier,
        "attempts": sum_field("attempts"),
        "splitEvents": sum_field("splitEvents"),
        "maximumSplitDepth": max(
            nonnegative_integer(summary, "maximumSplitDepth", "Shard summary")
            for summary in summaries
        ),
        "safeLeaves": sum_field("safeLeaves"),
        "rejectedLeaves": sum_field("rejectedLeaves"),
        "targetRejectSha256": target_reject_digest,
        "artifacts": {
            "rejected": file_metadata(
                f"new-reject-{args.from_pieces}.bin", new_reject_payload
            ),
            "policy": file_metadata(
                f"{args.from_pieces}-{args.target_pieces}.policy.bin", policy_payload
            ),
            "frontier": file_metadata(
                f"{args.from_pieces}-{args.target_pieces}.frontier.bin", frontier_payload
            ),
        },
    }
    actual_classification = load_json(result / "classification.json")
    if actual_classification != expected_classification:
        fail("classification.json is not identical to the independent merged summary")

    expected_campaign = {
        **expected_classification,
        "existingRejectedRoots": existing.count,
        "newRejectedRoots": len(merged_rejected),
        "cumulativeRejectedRoots": len(cumulative_records),
        "rejectionProgress": len(merged_rejected),
    }
    actual_campaign = load_json(result / "campaign-summary.json")
    for field, expected in expected_campaign.items():
        if actual_campaign.get(field) != expected:
            fail(f"campaign-summary.json has invalid {field}: {actual_campaign.get(field)!r}")

    if actual_campaign.get("cumulativeRejectedRoots") != (
        actual_campaign.get("existingRejectedRoots", -1)
        + actual_campaign.get("newRejectedRoots", -1)
    ):
        fail("Campaign cumulative rejection accounting is inconsistent")

    return {
        "format": "connect4-chaos-independent-sharded-round-audit-v1",
        "status": "pass",
        "role": role_name,
        "fromPieces": args.from_pieces,
        "targetPieces": args.target_pieces,
        "shards": args.shard_count,
        "inputRoots": source.count,
        "existingRejectedRoots": existing.count,
        "newRejectedRoots": len(merged_rejected),
        "cumulativeRejectedRoots": len(cumulative_records),
        "safeInputRoots": expected_classification["safeInputRoots"],
        "safePolicyEntries": len(merged_policy),
        "safeFrontierStates": len(merged_frontier),
        "duplicateRejectedRecords": duplicate_rejected,
        "duplicatePolicyRecords": duplicate_policy,
        "duplicateFrontierRecords": duplicate_frontier,
        "policyConflicts": policy_conflicts,
        "stageManifestEntries": len(stage_manifest),
        "resultManifestEntries": len(result_manifest),
        "proofTables": {
            "newRejectSha256": sha256_bytes(new_reject_payload),
            "cumulativeRejectSha256": sha256_bytes(cumulative_payload),
            "policySha256": sha256_bytes(policy_payload),
            "frontierSha256": sha256_bytes(frontier_payload),
        },
    }


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", required=True, choices=sorted(ROLE_CODES))
    parser.add_argument("--stage-directory", required=True, type=Path)
    parser.add_argument("--shard-directory", required=True, type=Path)
    parser.add_argument("--result-directory", required=True, type=Path)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--from-pieces", type=int, default=14)
    parser.add_argument("--target-pieces", type=int, default=16)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.shard_count < 1:
        parser.error("shard count must be positive")
    if args.from_pieces < 2 or args.target_pieces <= args.from_pieces:
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
