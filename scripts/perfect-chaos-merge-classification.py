#!/usr/bin/env python3
"""Merge and verify a complete set of Perfect Chaos classification shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from perfect_chaos_tables import (
    FRONTIER_MAGIC,
    FRONTIER_RECORD_SIZE,
    POLICY_MAGIC,
    POLICY_RECORD_SIZE,
    ROLE_CODES,
    file_summary,
    merge_records,
    read_table,
    write_table,
)

FORMAT = "connect4-chaos-frontier-classification-shard-v1"
MERGED_FORMAT = "connect4-chaos-frontier-classification-merged-v1"


def verify_artifact(path: Path, metadata: dict, magic: bytes, size: int, role: int, boundary: int):
    if metadata.get("bytes") != path.stat().st_size:
        raise RuntimeError(f"Artifact byte count mismatch: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata.get("sha256") != digest:
        raise RuntimeError(f"Artifact checksum mismatch: {path}")
    table = read_table(path, magic, size)
    if table.role != role or table.boundary != boundary:
        raise RuntimeError(f"Artifact metadata mismatch: {path}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=sorted(ROLE_CODES))
    parser.add_argument("--target-pieces", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--frontier", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    if args.shard_count < 1:
        raise RuntimeError("shard-count must be positive.")
    role_code = ROLE_CODES[args.role]
    source = read_table(args.input, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
    if source.role != role_code:
        raise RuntimeError("Input frontier has the wrong role.")
    if args.target_pieces <= source.boundary or args.target_pieces > 42:
        raise RuntimeError("Target boundary must be greater than the input boundary and at most 42.")

    rejected_records: list[bytes] = []
    policy_records: list[bytes] = []
    frontier_records: list[bytes] = []
    summaries: list[dict] = []
    target_reject_digest: str | None = None

    for shard_index in range(args.shard_count):
        summary_path = args.directory / f"summary-{shard_index}.json"
        rejected_path = args.directory / f"rejected-{shard_index}.bin"
        policy_path = args.directory / f"policy-{shard_index}.bin"
        frontier_path = args.directory / f"frontier-{shard_index}.bin"
        for path in (summary_path, rejected_path, policy_path, frontier_path):
            if not path.is_file():
                raise RuntimeError(f"Missing classification shard artifact: {path}")

        summary = json.loads(summary_path.read_text())
        expected_roots = len(source.records[shard_index :: args.shard_count])
        expected = {
            "format": FORMAT,
            "role": args.role,
            "fromPieces": source.boundary,
            "targetPieces": args.target_pieces,
            "shardIndex": shard_index,
            "shardCount": args.shard_count,
            "inputRoots": expected_roots,
        }
        for field, value in expected.items():
            if summary.get(field) != value:
                raise RuntimeError(
                    f"Shard {shard_index} has invalid {field}: "
                    f"expected {value!r}, received {summary.get(field)!r}."
                )
        if expected_roots == 0:
            raise RuntimeError(f"Shard {shard_index} is empty; reduce shard-count.")

        digest = summary.get("targetRejectSha256")
        if target_reject_digest is None:
            target_reject_digest = digest
        elif digest != target_reject_digest:
            raise RuntimeError("Classification shards used different target rejection frontiers.")

        artifacts = summary.get("artifacts") or {}
        rejected = verify_artifact(
            rejected_path,
            artifacts.get("rejected") or {},
            FRONTIER_MAGIC,
            FRONTIER_RECORD_SIZE,
            role_code,
            source.boundary,
        )
        policy = verify_artifact(
            policy_path,
            artifacts.get("policy") or {},
            POLICY_MAGIC,
            POLICY_RECORD_SIZE,
            role_code,
            args.target_pieces,
        )
        frontier = verify_artifact(
            frontier_path,
            artifacts.get("frontier") or {},
            FRONTIER_MAGIC,
            FRONTIER_RECORD_SIZE,
            role_code,
            args.target_pieces,
        )
        if summary.get("classificationComplete") is not True:
            raise RuntimeError(f"Shard {shard_index} is not a complete classification.")
        if summary.get("policyConflicts") != 0:
            raise RuntimeError(
                f"Shard {shard_index} contains conflicting policy actions: "
                f"{summary.get('policyConflicts')!r}."
            )
        if summary.get("safeInputRoots", 0) + summary.get("rejectedRoots", 0) != expected_roots:
            raise RuntimeError(f"Shard {shard_index} did not account for every input root.")
        if len(rejected.records) != summary.get("rejectedRoots"):
            raise RuntimeError(f"Shard {shard_index} rejection count mismatch.")
        if len(policy.records) != summary.get("safePolicyEntries"):
            raise RuntimeError(f"Shard {shard_index} policy count mismatch.")
        if len(frontier.records) != summary.get("safeFrontierStates"):
            raise RuntimeError(f"Shard {shard_index} frontier count mismatch.")

        rejected_records.extend(rejected.records)
        policy_records.extend(policy.records)
        frontier_records.extend(frontier.records)
        summaries.append(summary)

    merged_rejected, _ = merge_records(
        rejected_records,
        FRONTIER_RECORD_SIZE,
    )
    merged_policy, policy_conflicts = merge_records(policy_records, POLICY_RECORD_SIZE)
    merged_frontier, _ = merge_records(
        frontier_records,
        FRONTIER_RECORD_SIZE,
    )
    if policy_conflicts:
        raise RuntimeError(
            f"Conflicting Perfect Chaos policy actions across classification shards: "
            f"{policy_conflicts}."
        )
    write_table(
        args.rejected,
        FRONTIER_MAGIC,
        role_code,
        source.boundary,
        FRONTIER_RECORD_SIZE,
        merged_rejected,
    )
    write_table(
        args.policy,
        POLICY_MAGIC,
        role_code,
        args.target_pieces,
        POLICY_RECORD_SIZE,
        merged_policy,
    )
    write_table(
        args.frontier,
        FRONTIER_MAGIC,
        role_code,
        args.target_pieces,
        FRONTIER_RECORD_SIZE,
        merged_frontier,
    )

    def sum_field(field: str) -> int:
        return sum(int(summary.get(field, 0)) for summary in summaries)

    summary = {
        "format": MERGED_FORMAT,
        "role": args.role,
        "fromPieces": source.boundary,
        "targetPieces": args.target_pieces,
        "shards": args.shard_count,
        "inputRoots": len(source.records),
        "rejectedRoots": len(merged_rejected),
        "safeInputRoots": sum_field("safeInputRoots"),
        "classificationComplete": True,
        "safePolicyEntries": len(merged_policy),
        "safeFrontierStates": len(merged_frontier),
        "policyConflicts": policy_conflicts,
        "duplicateRejectedRecords": len(rejected_records) - len(merged_rejected),
        "duplicateFrontierRecords": len(frontier_records) - len(merged_frontier),
        "attempts": sum_field("attempts"),
        "splitEvents": sum_field("splitEvents"),
        "maximumSplitDepth": max(int(summary.get("maximumSplitDepth", 0)) for summary in summaries),
        "safeLeaves": sum_field("safeLeaves"),
        "rejectedLeaves": sum_field("rejectedLeaves"),
        "targetRejectSha256": target_reject_digest,
        "artifacts": {
            "rejected": file_summary(args.rejected),
            "policy": file_summary(args.policy),
            "frontier": file_summary(args.frontier),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
