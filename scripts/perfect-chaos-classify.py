#!/usr/bin/env python3
"""Fail-closed, resumable classifier for one deterministic Perfect Chaos frontier shard.

The native solver classifies every selected root against an optional rejected target
frontier. Resource failures are bisected deterministically. Mathematical rejections
are emitted separately from safe policy/frontier records so a workflow can either
refine the earlier prefix or merge a completed segment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

from perfect_chaos_tables import (
    FRONTIER_MAGIC,
    FRONTIER_RECORD_SIZE,
    POLICY_MAGIC,
    POLICY_RECORD_SIZE,
    ROLE_CODES,
    action_key,
    file_summary,
    read_table,
    record_key,
    write_table,
)


def splittable(details: str, code: int) -> bool:
    return (
        "Prefix graph exceeded its state limit." in details
        or "std::bad_alloc" in details
        or code == 137
        or code == -9
    )



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=sorted(ROLE_CODES))
    parser.add_argument("--target-pieces", required=True, type=int)
    parser.add_argument("--reject-frontier", type=Path)
    parser.add_argument("--shard-index", required=True, type=int)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--maximum-states", default=4_000_000, type=int)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--frontier", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()

    source = read_table(args.input, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
    role_code = ROLE_CODES[args.role]
    if source.role != role_code:
        raise RuntimeError("Input frontier has the wrong role.")
    if args.target_pieces <= source.boundary or args.target_pieces > 42:
        raise RuntimeError("Target boundary must be greater than the input boundary and at most 42.")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise RuntimeError("Invalid shard index/count.")
    if args.maximum_states < 10_000:
        raise RuntimeError("maximum-states must be at least 10,000.")

    if args.reject_frontier:
        rejected_target = read_table(args.reject_frontier, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
        if rejected_target.role != role_code or rejected_target.boundary != args.target_pieces:
            raise RuntimeError("Target rejection frontier has incompatible metadata.")

    selected = list(source.records[args.shard_index :: args.shard_count])
    if not selected:
        raise RuntimeError("The selected deterministic shard is empty.")

    rejected_by_key: dict[tuple[int, int, int, int, int], bytes] = {}
    safe_policy_by_key: dict[tuple[int, int, int, int, int], bytes] = {}
    safe_frontier_by_key: dict[tuple[int, int, int, int, int], bytes] = {}
    safe_input_roots = 0
    attempts = 0
    split_events = 0
    maximum_depth = 0
    safe_leaves = 0
    rejected_leaves = 0
    policy_conflicts = 0

    with tempfile.TemporaryDirectory(prefix="perfect-chaos-classify-") as temporary:
        root = Path(temporary)

        def classify(records: list[bytes], label: str, depth: int) -> None:
            nonlocal attempts, split_events, maximum_depth, safe_leaves, rejected_leaves
            nonlocal policy_conflicts, safe_input_roots
            attempts += 1
            maximum_depth = max(maximum_depth, depth)
            prefix = root / label.replace(".", "-")
            input_path = prefix.with_suffix(".input.bin")
            policy_path = prefix.with_suffix(".policy.bin")
            frontier_path = prefix.with_suffix(".frontier.bin")
            rejected_path = prefix.with_suffix(".rejected.bin")
            write_table(
                input_path,
                FRONTIER_MAGIC,
                role_code,
                source.boundary,
                FRONTIER_RECORD_SIZE,
                records,
            )
            command = [
                str(args.solver),
                "extend",
                "--input-frontier",
                str(input_path),
                "--frontier-pieces",
                str(args.target_pieces),
                "--maximum-states",
                str(args.maximum_states),
                "--policy",
                str(policy_path),
                "--frontier",
                str(frontier_path),
                "--rejected",
                str(rejected_path),
            ]
            if args.reject_frontier:
                command.extend(["--reject-frontier", str(args.reject_frontier)])
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode == 0:
                policy = read_table(policy_path, POLICY_MAGIC, POLICY_RECORD_SIZE)
                frontier = read_table(frontier_path, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
                if policy.role != role_code or policy.boundary != args.target_pieces:
                    raise RuntimeError("Safe policy has incompatible metadata.")
                if frontier.role != role_code or frontier.boundary != args.target_pieces:
                    raise RuntimeError("Safe frontier has incompatible metadata.")
                for record in policy.records:
                    key = record_key(record)
                    prior = safe_policy_by_key.get(key)
                    if prior is not None and action_key(prior) != action_key(record):
                        policy_conflicts += 1
                    if prior is None or action_key(record) < action_key(prior):
                        safe_policy_by_key[key] = record
                for record in frontier.records:
                    safe_frontier_by_key[record_key(record)] = record
                safe_input_roots += len(records)
                safe_leaves += 1
                return
            if rejected_path.exists():
                rejected = read_table(rejected_path, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
                if rejected.role != role_code or rejected.boundary != source.boundary:
                    raise RuntimeError("Native rejection certificate has incompatible metadata.")
                rejected_keys = {record_key(record) for record in rejected.records}
                if not rejected_keys:
                    raise RuntimeError("Native solver returned an empty rejection certificate.")
                for record in rejected.records:
                    rejected_by_key[record_key(record)] = record
                rejected_leaves += 1
                remaining = [
                    record for record in records if record_key(record) not in rejected_keys
                ]
                if len(remaining) + len(rejected_keys) != len(records):
                    raise RuntimeError("Native rejection certificate is not a subset of the input shard.")
                if remaining:
                    classify(remaining, f"{label}.safe", depth + 1)
                return
            details = f"{completed.stderr}\n{completed.stdout}"
            if splittable(details, completed.returncode) and len(records) > 1:
                middle = (len(records) + 1) // 2
                split_events += 1
                classify(records[:middle], f"{label}.0", depth + 1)
                classify(records[middle:], f"{label}.1", depth + 1)
                return
            single = " A single root exceeds the resource boundary." if len(records) == 1 else ""
            raise RuntimeError(f"Shard failed without a rejection certificate.{single}\n{details}")

        classify(selected, f"{args.shard_index:03d}", 0)

    write_table(
        args.rejected,
        FRONTIER_MAGIC,
        role_code,
        source.boundary,
        FRONTIER_RECORD_SIZE,
        rejected_by_key.values(),
    )
    write_table(
        args.policy,
        POLICY_MAGIC,
        role_code,
        args.target_pieces,
        POLICY_RECORD_SIZE,
        safe_policy_by_key.values(),
    )
    write_table(
        args.frontier,
        FRONTIER_MAGIC,
        role_code,
        args.target_pieces,
        FRONTIER_RECORD_SIZE,
        safe_frontier_by_key.values(),
    )
    rejected_count = read_table(args.rejected, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
    policy_count = read_table(args.policy, POLICY_MAGIC, POLICY_RECORD_SIZE)
    frontier_count = read_table(args.frontier, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
    if safe_input_roots + len(rejected_count.records) != len(selected):
        raise RuntimeError("Classification did not account for every input root.")
    summary = {
        "format": "connect4-chaos-frontier-classification-shard-v1",
        "role": args.role,
        "fromPieces": source.boundary,
        "targetPieces": args.target_pieces,
        "shardIndex": args.shard_index,
        "shardCount": args.shard_count,
        "inputRoots": len(selected),
        "rejectedRoots": len(rejected_count.records),
        "safeInputRoots": safe_input_roots,
        "classificationComplete": True,
        "safePolicyEntries": len(policy_count.records),
        "safeFrontierStates": len(frontier_count.records),
        "policyConflicts": policy_conflicts,
        "attempts": attempts,
        "splitEvents": split_events,
        "maximumSplitDepth": maximum_depth,
        "safeLeaves": safe_leaves,
        "rejectedLeaves": rejected_leaves,
        "maximumStatesPerLeaf": args.maximum_states,
        "targetRejectSha256": (
            hashlib.sha256(args.reject_frontier.read_bytes()).hexdigest()
            if args.reject_frontier
            else None
        ),
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
