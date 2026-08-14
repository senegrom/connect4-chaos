#!/usr/bin/env python3
"""Integration tests for the reusable Perfect Chaos root-status catalog."""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from pathlib import Path

from perfect_chaos_tables import (
    FRONTIER_MAGIC,
    FRONTIER_RECORD_SIZE,
    write_table,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "perfect-chaos-root-catalog.py"
ROLE = 1
BOUNDARY = 2
ROWS = 6
COLUMNS = 7


def bit(column: int, row: int = 0) -> int:
    return 1 << (column * (ROWS + 1) + row)


def mirror(mask: int) -> int:
    column_mask = (1 << ROWS) - 1
    output = 0
    for column in range(COLUMNS):
        output |= (
            (mask >> (column * (ROWS + 1))) & column_mask
        ) << ((COLUMNS - 1 - column) * (ROWS + 1))
    return output


def record(mover_column: int, opponent_column: int, ai_turn: bool = True) -> bytes:
    mover = bit(mover_column)
    opponent = bit(opponent_column)
    reflected = (mirror(mover), mirror(opponent))
    if reflected < (mover, opponent):
        mover, opponent = reflected
    return struct.pack("<QQBBB", mover, opponent, ROWS, COLUMNS, int(ai_turn))


def table(path: Path, records: list[bytes]) -> None:
    write_table(
        path,
        FRONTIER_MAGIC,
        ROLE,
        BOUNDARY,
        FRONTIER_RECORD_SIZE,
        records,
    )


def run(*arguments: str, success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["python3", str(SCRIPT), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if success and result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout)
    if not success and result.returncode == 0:
        raise RuntimeError(f"Command unexpectedly succeeded: {arguments!r}")
    return result


def output(result: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(result.stdout)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="perfect-chaos-root-catalog-") as raw:
        root = Path(raw)
        first_input = root / "first-input.bin"
        first_rejected = root / "first-rejected.bin"
        second_input = root / "second-input.bin"
        second_rejected = root / "second-rejected.bin"
        bad_input = root / "bad-input.bin"

        states = [
            record(3, 2),
            record(2, 3),
            record(1, 2),
            record(0, 1),
            record(0, 2),
        ]
        table(first_input, states[:3])
        table(first_rejected, [states[2]])
        table(second_input, [states[0], states[1], states[3], states[4]])
        table(second_rejected, [states[4]])
        table(bad_input, [states[0], states[2]])

        catalog1 = root / "catalog-1"
        bootstrap = output(run(
            "bootstrap",
            "--input", str(first_input),
            "--rejected", str(first_rejected),
            "--output", str(catalog1),
        ))
        assert bootstrap["counts"]["safeRoots"] == 2
        assert bootstrap["counts"]["rejectedRoots"] == 1
        assert bootstrap["counts"]["classifiedRoots"] == 3

        partition_dir = root / "partition"
        partition = output(run(
            "partition",
            "--frontier", str(second_input),
            "--catalog", str(catalog1),
            "--output", str(partition_dir),
        ))
        assert partition["counts"] == {
            "currentRoots": 4,
            "knownSafeRoots": 2,
            "rejectedHits": 0,
            "unknownRoots": 2,
        }
        assert partition["reusePercent"] == 50.0

        summary = root / "classification.json"
        audit = root / "audit.json"
        classification = {
            "role": "red",
            "fromPieces": BOUNDARY,
            "inputRoots": 2,
            "rejectedRoots": 1,
            "safeInputRoots": 1,
            "classificationComplete": True,
            "policyConflicts": 0,
        }
        summary.write_text(json.dumps(classification))
        audit.write_text(json.dumps(classification))

        catalog2 = root / "catalog-2"
        updated = output(run(
            "update",
            "--catalog", str(catalog1),
            "--unknown", str(partition_dir / "unknown.bin"),
            "--new-rejected", str(second_rejected),
            "--classification-summary", str(summary),
            "--classification-audit", str(audit),
            "--output", str(catalog2),
        ))
        assert updated["counts"]["previousSafeRoots"] == 2
        assert updated["counts"]["newSafeRoots"] == 1
        assert updated["counts"]["safeRoots"] == 3
        assert updated["counts"]["previousRejectedRoots"] == 1
        assert updated["counts"]["newRejectedRoots"] == 1
        assert updated["counts"]["rejectedRoots"] == 2

        verified = output(run("verify", "--directory", str(catalog2)))
        assert verified["status"] == "pass"
        assert verified["safeRoots"] == 3
        assert verified["rejectedRoots"] == 2

        repeated = run(
            "partition",
            "--frontier", str(bad_input),
            "--catalog", str(catalog1),
            "--output", str(root / "bad-partition"),
            success=False,
        )
        assert "previously rejected" in repeated.stderr

        outside = root / "outside-reject.bin"
        table(outside, [states[2]])
        failed_update = run(
            "update",
            "--catalog", str(catalog1),
            "--unknown", str(partition_dir / "unknown.bin"),
            "--new-rejected", str(outside),
            "--output", str(root / "bad-update"),
            success=False,
        )
        assert "outside the unknown input" in failed_update.stderr

        (catalog2 / "unlisted.txt").write_text("tamper\n")
        unlisted = run("verify", "--directory", str(catalog2), success=False)
        assert "unlisted" in unlisted.stderr.lower() or "not exhaustive" in unlisted.stderr.lower()

    print("Perfect Chaos root catalog tests passed.")


if __name__ == "__main__":
    main()
