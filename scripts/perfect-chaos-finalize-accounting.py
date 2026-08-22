#!/usr/bin/env python3
"""Validate one Perfect Chaos refinement result and write exact cumulative accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from perfect_chaos_tables import (
    FRONTIER_MAGIC,
    FRONTIER_RECORD_SIZE,
    ROLE_CODES,
    read_table,
    write_table,
)

MERGED_FORMAT = "connect4-chaos-frontier-classification-merged-v1"


def require_integer(value: object, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"Classification field {field} must be an integer >= {minimum}.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=sorted(ROLE_CODES))
    parser.add_argument("--from-pieces", required=True, type=int)
    parser.add_argument("--target-pieces", required=True, type=int)
    parser.add_argument("--existing", required=True, type=Path)
    parser.add_argument("--discovered", required=True, type=Path)
    parser.add_argument("--classification", required=True, type=Path)
    parser.add_argument("--cumulative", required=True, type=Path)
    parser.add_argument("--campaign-summary", required=True, type=Path)
    args = parser.parse_args()

    if not 0 <= args.from_pieces < args.target_pieces <= 42:
        raise RuntimeError("Invalid Perfect Chaos accounting boundaries.")

    role_code = ROLE_CODES[args.role]
    existing = read_table(args.existing, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
    discovered = read_table(args.discovered, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
    for name, table in (("existing", existing), ("discovered", discovered)):
        if table.role != role_code or table.boundary != args.from_pieces:
            raise RuntimeError(f"The {name} rejection table has incompatible role or boundary metadata.")

    existing_records = set(existing.records)
    discovered_records = set(discovered.records)
    overlap = existing_records & discovered_records
    if overlap:
        raise RuntimeError(
            f"The newly discovered rejection table overlaps the predecessor by {len(overlap)} record(s)."
        )

    summary = json.loads(args.classification.read_text())
    expected = {
        "format": MERGED_FORMAT,
        "role": args.role,
        "fromPieces": args.from_pieces,
        "targetPieces": args.target_pieces,
        "classificationComplete": True,
        "policyConflicts": 0,
    }
    for field, value in expected.items():
        if summary.get(field) != value:
            raise RuntimeError(
                f"Classification field {field} mismatch: expected {value!r}, "
                f"received {summary.get(field)!r}."
            )

    input_roots = require_integer(summary.get("inputRoots"), "inputRoots", 1)
    safe_roots = require_integer(summary.get("safeInputRoots"), "safeInputRoots")
    rejected_roots = require_integer(summary.get("rejectedRoots"), "rejectedRoots")
    if input_roots != safe_roots + rejected_roots:
        raise RuntimeError("The classification does not account for every input root exactly once.")
    if rejected_roots != len(discovered.records):
        raise RuntimeError("The classification rejection count does not match the discovered table.")
    if require_integer(summary.get("duplicateRejectedRecords"), "duplicateRejectedRecords") != 0:
        raise RuntimeError("The classification contains duplicate rejection records across shards.")
    require_integer(summary.get("safePolicyEntries"), "safePolicyEntries")
    require_integer(summary.get("safeFrontierStates"), "safeFrontierStates")

    # Direct rejection-frontier classification records the predecessor digest.
    # Rebuilt-frontier classification does not consume that table and records
    # null. A present digest must be exact; absence is covered independently by
    # the staged checkpoint and cumulative-table comparison.
    recorded_digest = summary.get("targetRejectSha256")
    if recorded_digest is not None:
        expected_digest = hashlib.sha256(args.existing.read_bytes()).hexdigest()
        if recorded_digest != expected_digest:
            raise RuntimeError("Classification shards recorded the wrong predecessor rejection table.")

    write_table(
        args.cumulative,
        FRONTIER_MAGIC,
        role_code,
        args.from_pieces,
        FRONTIER_RECORD_SIZE,
        [*existing.records, *discovered.records],
    )
    cumulative = read_table(args.cumulative, FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
    expected_count = len(existing.records) + len(discovered.records)
    if len(cumulative.records) != expected_count:
        raise RuntimeError("Cumulative rejection accounting lost or duplicated records.")

    campaign = {
        **summary,
        "existingRejectedRoots": len(existing.records),
        "newRejectedRoots": len(discovered.records),
        "cumulativeRejectedRoots": len(cumulative.records),
        "rejectionProgress": len(discovered.records),
    }
    args.campaign_summary.parent.mkdir(parents=True, exist_ok=True)
    args.campaign_summary.write_text(json.dumps(campaign, indent=2) + "\n")
    print(json.dumps(campaign), flush=True)


if __name__ == "__main__":
    main()
