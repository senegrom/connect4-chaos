#!/usr/bin/env python3
"""Self-contained regression tests for perfect-chaos-audit-round.py."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

FRONTIER_MAGIC = b"C4CFRN1\x00"
POLICY_MAGIC = b"C4CPOL1\x00"
ROLE = 1
FROM = 14
TARGET = 16


def record(size: int, value: int) -> bytes:
    return value.to_bytes(size, "little")


def table(magic: bytes, role: int, boundary: int, record_size: int, records: list[bytes]) -> bytes:
    if any(len(item) != record_size for item in records):
        raise AssertionError("invalid synthetic record size")
    return (
        magic
        + bytes((1, role, boundary, record_size))
        + struct.pack("<I", len(records))
        + b"".join(records)
    )


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def hash_metadata(path: Path, relative: str) -> dict:
    data = path.read_bytes()
    return {
        "path": relative,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def write_sums(root: Path) -> None:
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        relative = path.relative_to(root).as_posix()
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}\n")
    (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def build_fixture(root: Path) -> tuple[Path, Path]:
    prior = root / "prior"
    current = root / "current"
    prior_rejects = [record(19, 1), record(19, 2)]
    new_rejects = [record(19, 3)]
    cumulative = [*prior_rejects, *new_rejects]
    frontier_records = [record(19, 10), record(19, 11)]
    policy_records = [record(20, 20), record(20, 21)]

    prior_table = prior / "red-prepared" / "reject-14.bin"
    prior_table.parent.mkdir(parents=True, exist_ok=True)
    prior_table.write_bytes(table(FRONTIER_MAGIC, ROLE, FROM, 19, prior_rejects))
    write_json(
        prior / "campaign-summary.json",
        {
            "format": "connect4-chaos-frontier-classification-merged-v1",
            "role": "red",
            "fromPieces": FROM,
            "targetPieces": TARGET,
            "classificationComplete": True,
            "policyConflicts": 0,
            "cumulativeRejectedRoots": len(prior_rejects),
        },
    )
    write_sums(prior)

    new_path = current / "new-reject-14.bin"
    cumulative_path = current / "reject-14.bin"
    embedded_path = current / "red-prepared" / "reject-14.bin"
    frontier_path = current / "14-16.frontier.bin"
    policy_path = current / "14-16.policy.bin"
    embedded_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(table(FRONTIER_MAGIC, ROLE, FROM, 19, new_rejects))
    cumulative_bytes = table(FRONTIER_MAGIC, ROLE, FROM, 19, cumulative)
    cumulative_path.write_bytes(cumulative_bytes)
    embedded_path.write_bytes(cumulative_bytes)
    frontier_path.write_bytes(table(FRONTIER_MAGIC, ROLE, TARGET, 19, frontier_records))
    policy_path.write_bytes(table(POLICY_MAGIC, ROLE, TARGET, 20, policy_records))

    classification = {
        "format": "connect4-chaos-frontier-classification-merged-v1",
        "role": "red",
        "fromPieces": FROM,
        "targetPieces": TARGET,
        "shards": 1,
        "inputRoots": 3,
        "rejectedRoots": 1,
        "safeInputRoots": 2,
        "classificationComplete": True,
        "safePolicyEntries": 2,
        "safeFrontierStates": 2,
        "policyConflicts": 0,
        "duplicateRejectedRecords": 0,
        "duplicateFrontierRecords": 0,
        "attempts": 1,
        "splitEvents": 0,
        "maximumSplitDepth": 0,
        "safeLeaves": 1,
        "rejectedLeaves": 1,
        "targetRejectSha256": None,
        "artifacts": {
            "rejected": hash_metadata(new_path, "new-reject-14.bin"),
            "policy": hash_metadata(policy_path, "14-16.policy.bin"),
            "frontier": hash_metadata(frontier_path, "14-16.frontier.bin"),
        },
    }
    write_json(current / "classification.json", classification)
    write_json(
        current / "campaign-summary.json",
        {
            **classification,
            "existingRejectedRoots": 2,
            "newRejectedRoots": 1,
            "cumulativeRejectedRoots": 3,
            "rejectionProgress": 1,
        },
    )
    write_sums(current)
    return prior, current


def invoke(auditor: Path, prior: Path, current: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(auditor),
            "--role",
            "red",
            "--prior-directory",
            str(prior),
            "--current-directory",
            str(current),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def require_pass(result: subprocess.CompletedProcess[str]) -> None:
    if result.returncode != 0:
        raise AssertionError(f"auditor rejected valid fixture:\n{result.stdout}\n{result.stderr}")
    report = json.loads(result.stdout)
    if report.get("status") != "pass" or report.get("cumulativeRejectedRoots") != 3:
        raise AssertionError(f"unexpected audit report: {report}")


def require_fail(result: subprocess.CompletedProcess[str], expected: str) -> None:
    if result.returncode == 0:
        raise AssertionError(f"auditor accepted invalid fixture: {result.stdout}")
    if expected not in result.stderr:
        raise AssertionError(f"missing failure marker {expected!r}: {result.stderr}")


def main() -> int:
    auditor = Path(__file__).with_name("perfect-chaos-audit-round.py")
    if not auditor.is_file():
        raise AssertionError(f"auditor not found: {auditor}")

    with tempfile.TemporaryDirectory(prefix="perfect-chaos-audit-test-") as temporary:
        base = Path(temporary)

        prior, current = build_fixture(base / "valid")
        require_pass(invoke(auditor, prior, current))

        prior, current = build_fixture(base / "rogue")
        (current / "unlisted.bin").write_bytes(b"rogue")
        require_fail(invoke(auditor, prior, current), "SHA256SUMS is not exhaustive")

        prior, current = build_fixture(base / "hash")
        with (current / "14-16.frontier.bin").open("ab") as stream:
            stream.write(b"tamper")
        require_fail(invoke(auditor, prior, current), "SHA-256 mismatch")

        prior, current = build_fixture(base / "overlap")
        overlap = table(FRONTIER_MAGIC, ROLE, FROM, 19, [record(19, 1)])
        (current / "new-reject-14.bin").write_bytes(overlap)
        classification = json.loads((current / "classification.json").read_text())
        classification["artifacts"]["rejected"] = hash_metadata(
            current / "new-reject-14.bin", "new-reject-14.bin"
        )
        write_json(current / "classification.json", classification)
        summary = json.loads((current / "campaign-summary.json").read_text())
        summary["artifacts"] = classification["artifacts"]
        write_json(current / "campaign-summary.json", summary)
        write_sums(current)
        require_fail(invoke(auditor, prior, current), "overlaps the predecessor")

    print("perfect-chaos-audit-round: all regression cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
