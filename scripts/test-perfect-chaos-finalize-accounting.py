#!/usr/bin/env python3
"""Regression tests for Perfect Chaos producer-side accounting validation."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from perfect_chaos_tables import (
    FRONTIER_MAGIC,
    FRONTIER_RECORD_SIZE,
    ROLE_CODES,
    read_table,
    write_table,
)

ROLE = "red"
FROM = 2
TARGET = 4


def frontier_record(mover: int, opponent: int, ai_turn: int = 1) -> bytes:
    return struct.pack("<QQBBB", mover, opponent, 6, 7, ai_turn)


def canonical_records() -> tuple[bytes, bytes, bytes]:
    centre = 1 << 21
    return (
        frontier_record(centre, 1 << 14),
        frontier_record(1 << 14, centre),
        frontier_record(centre, 1 << 7),
    )


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(root: Path) -> dict[str, Path]:
    first, second, third = canonical_records()
    paths = {
        "existing": root / "existing.bin",
        "discovered": root / "discovered.bin",
        "classification": root / "classification.json",
        "cumulative": root / "cumulative.bin",
        "campaign": root / "campaign.json",
    }
    write_table(
        paths["existing"], FRONTIER_MAGIC, ROLE_CODES[ROLE], FROM,
        FRONTIER_RECORD_SIZE, [first, second],
    )
    write_table(
        paths["discovered"], FRONTIER_MAGIC, ROLE_CODES[ROLE], FROM,
        FRONTIER_RECORD_SIZE, [third],
    )
    write_json(
        paths["classification"],
        {
            "format": "connect4-chaos-frontier-classification-merged-v1",
            "role": ROLE,
            "fromPieces": FROM,
            "targetPieces": TARGET,
            "shards": 2,
            "inputRoots": 3,
            "rejectedRoots": 1,
            "safeInputRoots": 2,
            "classificationComplete": True,
            "safePolicyEntries": 5,
            "safeFrontierStates": 8,
            "policyConflicts": 0,
            "duplicateRejectedRecords": 0,
            "duplicateFrontierRecords": 3,
            "targetRejectSha256": None,
        },
    )
    return paths


def invoke(script: Path, paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--role", ROLE,
            "--from-pieces", str(FROM),
            "--target-pieces", str(TARGET),
            "--existing", str(paths["existing"]),
            "--discovered", str(paths["discovered"]),
            "--classification", str(paths["classification"]),
            "--cumulative", str(paths["cumulative"]),
            "--campaign-summary", str(paths["campaign"]),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def require_success(result: subprocess.CompletedProcess[str]) -> None:
    if result.returncode != 0:
        raise AssertionError(f"validator rejected valid accounting:\n{result.stderr}")


def require_failure(result: subprocess.CompletedProcess[str], marker: str) -> None:
    if result.returncode == 0:
        raise AssertionError(f"validator accepted invalid accounting: {result.stdout}")
    if marker not in result.stderr:
        raise AssertionError(f"missing failure marker {marker!r}: {result.stderr}")


def main() -> int:
    script = Path(__file__).with_name("perfect-chaos-finalize-accounting.py")
    with tempfile.TemporaryDirectory(prefix="perfect-chaos-accounting-") as temporary:
        base = Path(temporary)

        paths = fixture(base / "valid-rebuilt-frontier")
        require_success(invoke(script, paths))
        campaign = json.loads(paths["campaign"].read_text())
        cumulative = read_table(paths["cumulative"], FRONTIER_MAGIC, FRONTIER_RECORD_SIZE)
        if campaign["cumulativeRejectedRoots"] != 3 or len(cumulative.records) != 3:
            raise AssertionError("valid cumulative accounting has the wrong record count")

        paths = fixture(base / "valid-direct-rejection-frontier")
        summary = json.loads(paths["classification"].read_text())
        summary["targetRejectSha256"] = hashlib.sha256(paths["existing"].read_bytes()).hexdigest()
        write_json(paths["classification"], summary)
        require_success(invoke(script, paths))

        paths = fixture(base / "overlap")
        first, _, _ = canonical_records()
        write_table(
            paths["discovered"], FRONTIER_MAGIC, ROLE_CODES[ROLE], FROM,
            FRONTIER_RECORD_SIZE, [first],
        )
        require_failure(invoke(script, paths), "overlaps the predecessor")

        paths = fixture(base / "duplicates")
        summary = json.loads(paths["classification"].read_text())
        summary["duplicateRejectedRecords"] = 1
        write_json(paths["classification"], summary)
        require_failure(invoke(script, paths), "duplicate rejection records")

        paths = fixture(base / "digest")
        summary = json.loads(paths["classification"].read_text())
        summary["targetRejectSha256"] = "0" * 64
        write_json(paths["classification"], summary)
        require_failure(invoke(script, paths), "wrong predecessor rejection table")

        paths = fixture(base / "accounting")
        summary = json.loads(paths["classification"].read_text())
        summary["inputRoots"] = 4
        write_json(paths["classification"], summary)
        require_failure(invoke(script, paths), "every input root exactly once")

    print("perfect-chaos-finalize-accounting: all regression cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
