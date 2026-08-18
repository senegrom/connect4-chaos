#!/usr/bin/env python3
"""Cache exact safe/losing Perfect Chaos frontier roots between CEGAR rounds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from perfect_chaos_tables import (
    FRONTIER_MAGIC,
    FRONTIER_RECORD_SIZE,
    ROLE_NAMES,
    Table,
    file_summary,
    record_key,
    write_table,
)

FORMAT = "connect4-chaos-root-classification-catalog-v1"
PARTITION_FORMAT = "connect4-chaos-root-classification-partition-v1"
MANIFEST = "catalog-manifest.json"


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is not a regular file: {path}")
    return path.resolve()


def empty_directory(path: Path) -> Path:
    root = path.resolve()
    if root.exists():
        if root.is_symlink() or not root.is_dir() or any(root.iterdir()):
            fail(f"Output directory must be empty and safe: {root}")
    else:
        root.mkdir(parents=True)
    return root


def mirror(mask: int, rows: int, columns: int) -> int:
    stride, column_mask, result = rows + 1, (1 << rows) - 1, 0
    for column in range(columns):
        result |= ((mask >> (column * stride)) & column_mask) << (
            (columns - 1 - column) * stride
        )
    return result


def validate_record(record: bytes, boundary: int, label: str) -> None:
    mover = int.from_bytes(record[:8], "little")
    opponent = int.from_bytes(record[8:16], "little")
    rows, columns, ai_turn = record[16:19]
    if (rows, columns) not in {(6, 7), (7, 6)} or ai_turn not in (0, 1):
        fail(f"{label} contains invalid state metadata.")
    stride, occupied = rows + 1, mover | opponent
    if columns * stride > 63 or occupied >> (columns * stride) or mover & opponent:
        fail(f"{label} contains an invalid packed board.")
    column_mask = (1 << rows) - 1
    for column in range(columns):
        group = occupied >> (column * stride)
        bits = group & column_mask
        if group & (1 << rows) or bits & (bits + 1):
            fail(f"{label} violates sentinel or gravity invariants.")
    reflected = mirror(mover, rows, columns), mirror(opponent, rows, columns)
    if reflected < (mover, opponent) or occupied.bit_count() != boundary:
        fail(f"{label} contains a non-canonical or wrong-boundary root.")


def frontier(path: Path, label: str) -> Table:
    target = regular(path, label)
    payload = target.read_bytes()
    if len(payload) < 16 or payload[:8] != FRONTIER_MAGIC:
        fail(f"{label} has invalid magic.")
    version, role, boundary, record_size = payload[8:12]
    count = int.from_bytes(payload[12:16], "little")
    if version != 1 or role not in ROLE_NAMES or boundary > 42 \
            or record_size != FRONTIER_RECORD_SIZE \
            or len(payload) != 16 + count * FRONTIER_RECORD_SIZE:
        fail(f"{label} has an unsupported or inconsistent header.")
    records, previous = [], None
    for index in range(count):
        offset = 16 + index * FRONTIER_RECORD_SIZE
        record = payload[offset:offset + FRONTIER_RECORD_SIZE]
        validate_record(record, boundary, label)
        key = record_key(record)
        if previous is not None and previous >= key:
            fail(f"{label} records are not strictly sorted.")
        records.append(record)
        previous = key
    return Table(role, boundary, tuple(records))


def same(first: Table, second: Table, label: str) -> None:
    if (first.role, first.boundary) != (second.role, second.boundary):
        fail(f"{label} role or boundary mismatch.")


def keyed(records: Iterable[bytes]) -> dict[tuple[int, int, int, int, int], bytes]:
    source = tuple(records)
    result = {record_key(record): record for record in source}
    if len(result) != len(source):
        fail("A frontier contains duplicate canonical roots.")
    return result


def write_frontier(path: Path, table: Table, records: Iterable[bytes]) -> None:
    write_table(path, FRONTIER_MAGIC, table.role, table.boundary, FRONTIER_RECORD_SIZE, records)


def artifact(path: Path) -> dict[str, int | str]:
    return file_summary(path)


def metadata(path: Path, table: Table) -> dict[str, int | str]:
    return {"records": len(table.records), "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def artifacts(command: str, root: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("perfect-chaos-artifacts.py")), command,
         "--directory", str(root)], check=False, capture_output=True, text=True,
    )
    if result.returncode:
        fail((result.stderr or result.stdout).strip())


def write_catalog(root: Path, template: Table, safe: Iterable[bytes], rejected: Iterable[bytes],
                  counts: dict[str, int], provenance: dict[str, Any], new_safe: Iterable[bytes] | None = None) -> dict[str, Any]:
    safe_path, rejected_path = root / "safe.bin", root / "rejected.bin"
    write_frontier(safe_path, template, safe)
    write_frontier(rejected_path, template, rejected)
    extra = {}
    if new_safe is not None:
        path = root / "new-safe.bin"
        write_frontier(path, template, new_safe)
        extra["newSafe"] = artifact(path)
    manifest = {
        "format": FORMAT, "role": ROLE_NAMES[template.role], "boundary": template.boundary,
        "counts": counts,
        "artifacts": {"safe": artifact(safe_path), "rejected": artifact(rejected_path), **extra},
        "provenance": provenance,
    }
    write_json(root / MANIFEST, manifest)
    artifacts("write", root)
    verify(root)
    return manifest


def load_catalog(root: Path) -> tuple[dict[str, Any], Table, Table]:
    directory = root.resolve()
    if directory.is_symlink() or not directory.is_dir():
        fail(f"Catalog directory is missing or unsafe: {directory}")
    artifacts("verify", directory)
    manifest = json.loads((directory / MANIFEST).read_text())
    if manifest.get("format") != FORMAT:
        fail("Unsupported root catalog format.")
    safe, rejected = frontier(directory / "safe.bin", "Safe catalog"), frontier(directory / "rejected.bin", "Rejected catalog")
    same(safe, rejected, "Catalog")
    safe_keys, rejected_keys = set(keyed(safe.records)), set(keyed(rejected.records))
    if safe_keys & rejected_keys:
        fail("Safe and rejected catalogs overlap.")
    if manifest.get("role") != ROLE_NAMES[safe.role] or manifest.get("boundary") != safe.boundary:
        fail("Catalog metadata does not match its tables.")
    if manifest.get("counts", {}).get("safeRoots") != len(safe.records) \
            or manifest.get("counts", {}).get("rejectedRoots") != len(rejected.records):
        fail("Catalog counts do not match its tables.")
    for name, path in (("safe", directory / "safe.bin"), ("rejected", directory / "rejected.bin")):
        if manifest.get("artifacts", {}).get(name) != artifact(path):
            fail(f"Catalog metadata does not match {path.name}.")
    return manifest, safe, rejected


def bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    root = empty_directory(args.output)
    input_path, rejected_path = regular(args.input, "Input frontier"), regular(args.rejected, "Rejected roots")
    inputs, rejected = frontier(input_path, "Input frontier"), frontier(rejected_path, "Rejected roots")
    same(inputs, rejected, "Bootstrap")
    source, bad = keyed(inputs.records), keyed(rejected.records)
    if not set(bad) <= set(source):
        fail("Rejected roots are outside the classification input.")
    safe = [record for key, record in source.items() if key not in bad]
    return write_catalog(root, inputs, safe, bad.values(), {
        "inputRoots": len(source), "safeRoots": len(safe), "rejectedRoots": len(bad),
        "classifiedRoots": len(source), "newSafeRoots": len(safe), "newRejectedRoots": len(bad),
    }, {"operation": "bootstrap", "input": metadata(input_path, inputs), "rejected": metadata(rejected_path, rejected)})


def partition(args: argparse.Namespace) -> dict[str, Any]:
    root = empty_directory(args.output)
    _catalog, safe, rejected = load_catalog(args.catalog)
    current_path = regular(args.frontier, "Current frontier")
    current = frontier(current_path, "Current frontier")
    same(current, safe, "Partition")
    current_map, safe_keys, rejected_keys = keyed(current.records), set(keyed(safe.records)), set(keyed(rejected.records))
    hits = set(current_map) & rejected_keys
    if hits:
        fail(f"The rebuilt frontier still reaches {len(hits)} previously rejected root(s).")
    known = [record for key, record in current_map.items() if key in safe_keys]
    unknown = [record for key, record in current_map.items() if key not in safe_keys]
    known_path, unknown_path = root / "known-safe.bin", root / "unknown.bin"
    write_frontier(known_path, current, known)
    write_frontier(unknown_path, current, unknown)
    manifest = {
        "format": PARTITION_FORMAT, "role": ROLE_NAMES[current.role], "boundary": current.boundary,
        "counts": {"currentRoots": len(current.records), "knownSafeRoots": len(known), "unknownRoots": len(unknown), "rejectedHits": 0},
        "reusePercent": 100.0 if not current.records else round(100 * len(known) / len(current.records), 6),
        "artifacts": {"knownSafe": artifact(known_path), "unknown": artifact(unknown_path)},
        "provenance": {"frontier": metadata(current_path, current), "catalogManifestSha256": sha256(args.catalog.resolve() / MANIFEST)},
    }
    write_json(root / "partition-manifest.json", manifest)
    artifacts("write", root)
    artifacts("verify", root)
    return manifest


def optional_json(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else json.loads(regular(path, "Classification metadata").read_text())


def partition_scope(unknown_path: Path, unknown: Table) -> dict[str, int] | None:
    directory = unknown_path.parent
    manifest_path = directory / "partition-manifest.json"
    if not manifest_path.exists():
        return None
    artifacts("verify", directory)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != PARTITION_FORMAT \
            or manifest.get("role") != ROLE_NAMES[unknown.role] \
            or manifest.get("boundary") != unknown.boundary:
        fail("Partition metadata does not match the unknown-root table.")
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        fail("Partition counts are missing.")
    required = ("currentRoots", "knownSafeRoots", "unknownRoots", "rejectedHits")
    if any(isinstance(counts.get(field), bool) or not isinstance(counts.get(field), int)
           or counts[field] < 0 for field in required):
        fail("Partition counts are invalid.")
    if counts["unknownRoots"] != len(unknown.records) \
            or counts["rejectedHits"] != 0 \
            or counts["currentRoots"] != counts["knownSafeRoots"] + counts["unknownRoots"]:
        fail("Partition accounting does not match the unknown-root table.")
    known_path = directory / "known-safe.bin"
    known = frontier(known_path, "Known-safe partition")
    same(known, unknown, "Partition")
    if len(known.records) != counts["knownSafeRoots"]:
        fail("Known-safe partition count does not match its manifest.")
    if manifest.get("artifacts", {}).get("knownSafe") != artifact(known_path) \
            or manifest.get("artifacts", {}).get("unknown") != artifact(unknown_path):
        fail("Partition artifact metadata is inconsistent.")
    return {field: counts[field] for field in required}


def classification_scope(document: dict[str, Any], unknown_count: int,
                         partition: dict[str, int] | None) -> tuple[str, int, int]:
    input_roots = document.get("inputRoots")
    if input_roots == unknown_count:
        return "unknown", unknown_count, 0
    if partition is not None and input_roots == partition["currentRoots"]:
        return "full-frontier", partition["currentRoots"], partition["knownSafeRoots"]
    fail("Classification inputRoots matches neither the unknown set nor its full frontier.")


def update(args: argparse.Namespace) -> dict[str, Any]:
    root = empty_directory(args.output)
    _previous, safe, rejected = load_catalog(args.catalog)
    unknown_path, new_bad_path = regular(args.unknown, "Unknown roots"), regular(args.new_rejected, "New rejected roots")
    unknown, new_bad = frontier(unknown_path, "Unknown roots"), frontier(new_bad_path, "New rejected roots")
    same(unknown, safe, "Catalog update"); same(new_bad, safe, "Catalog update")
    safe_map, bad_map, unknown_map, new_bad_map = keyed(safe.records), keyed(rejected.records), keyed(unknown.records), keyed(new_bad.records)
    if set(unknown_map) & (set(safe_map) | set(bad_map)):
        fail("Unknown roots overlap the existing catalog.")
    if not set(new_bad_map) <= set(unknown_map):
        fail("New rejected roots are outside the unknown input.")
    new_safe = {key: record for key, record in unknown_map.items() if key not in new_bad_map}
    partition = partition_scope(unknown_path, unknown)
    scopes = []
    documents = (
        ("summary", optional_json(args.classification_summary),
         {"rejectedRoots": len(new_bad_map), "classificationComplete": True}),
        ("audit", optional_json(args.classification_audit),
         {"newRejectedRoots": len(new_bad_map), "status": "pass"}),
    )
    for label, document, specific in documents:
        if document is None:
            continue
        scope, input_roots, known_safe = classification_scope(document, len(unknown_map), partition)
        scopes.append(scope)
        expected = {
            "role": ROLE_NAMES[unknown.role],
            "fromPieces": unknown.boundary,
            "inputRoots": input_roots,
            "safeInputRoots": known_safe + len(new_safe),
            "policyConflicts": 0,
            **specific,
        }
        for field, value in expected.items():
            if document.get(field) != value:
                fail(f"Classification {label} field {field!r} does not match the catalog update.")
    if len(set(scopes)) > 1:
        fail("Classification summary and audit use different input scopes.")
    classification_scope_name = scopes[0] if scopes else "unverified-metadata"
    safe_map.update(new_safe); bad_map.update(new_bad_map)
    return write_catalog(root, unknown, safe_map.values(), bad_map.values(), {
        "previousSafeRoots": len(safe.records), "previousRejectedRoots": len(rejected.records),
        "unknownRoots": len(unknown_map), "newSafeRoots": len(new_safe), "newRejectedRoots": len(new_bad_map),
        "safeRoots": len(safe_map), "rejectedRoots": len(bad_map), "classifiedRoots": len(safe_map) + len(bad_map),
    }, {"operation": "update", "classificationScope": classification_scope_name,
        "previousManifestSha256": sha256(args.catalog.resolve() / MANIFEST),
        "unknown": metadata(unknown_path, unknown), "newRejected": metadata(new_bad_path, new_bad)}, new_safe.values())


def verify(root: Path) -> dict[str, Any]:
    manifest, safe, rejected = load_catalog(root)
    delta = root.resolve() / "new-safe.bin"
    if delta.exists():
        new_safe = frontier(delta, "New-safe delta"); same(new_safe, safe, "New-safe delta")
        if manifest.get("artifacts", {}).get("newSafe") != artifact(delta) \
                or not set(keyed(new_safe.records)) <= set(keyed(safe.records)):
            fail("New-safe delta is invalid.")
    return {"format": FORMAT, "status": "pass", "role": ROLE_NAMES[safe.role], "boundary": safe.boundary,
            "safeRoots": len(safe.records), "rejectedRoots": len(rejected.records), "manifestSha256": sha256(root.resolve() / MANIFEST)}


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("bootstrap"); first.add_argument("--input", required=True, type=Path); first.add_argument("--rejected", required=True, type=Path); first.add_argument("--output", required=True, type=Path); first.set_defaults(run=bootstrap)
    split = commands.add_parser("partition"); split.add_argument("--frontier", required=True, type=Path); split.add_argument("--catalog", required=True, type=Path); split.add_argument("--output", required=True, type=Path); split.set_defaults(run=partition)
    advance = commands.add_parser("update"); advance.add_argument("--catalog", required=True, type=Path); advance.add_argument("--unknown", required=True, type=Path); advance.add_argument("--new-rejected", required=True, type=Path); advance.add_argument("--classification-summary", type=Path); advance.add_argument("--classification-audit", type=Path); advance.add_argument("--output", required=True, type=Path); advance.set_defaults(run=update)
    check = commands.add_parser("verify"); check.add_argument("--directory", required=True, type=Path); check.set_defaults(run=lambda args: verify(args.directory))
    args = parser.parse_args()
    print(json.dumps(args.run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
