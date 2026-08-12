#!/usr/bin/env python3
"""Write and verify strict, directory-relative SHA-256 artifact manifests."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path, PurePosixPath

DIGEST_LINE = re.compile(r"([0-9a-f]{64})  (.+)")


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
    return root.joinpath(*relative.parts)


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
    temporary.write_text(payload)
    os.replace(temporary, manifest)
    print(f"Wrote {len(records)} checksum record(s) to {manifest}.")


def read_manifest(root: Path, manifest: Path) -> dict[str, str]:
    if manifest.is_symlink() or not manifest.is_file():
        raise RuntimeError(f"Checksum manifest is not a regular file: {manifest}.")
    selected: dict[str, str] = {}
    for line_number, raw in enumerate(manifest.read_text().splitlines(), start=1):
        match = DIGEST_LINE.fullmatch(raw)
        if not match:
            raise RuntimeError(f"Malformed checksum line {line_number} in {manifest}.")
        digest, value = match.groups()
        relative = safe_relative(value, f"manifest entry on line {line_number}")
        canonical = relative.as_posix()
        if canonical in selected:
            raise RuntimeError(f"Duplicate checksum entry: {canonical!r}.")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("write", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--directory", required=True, type=Path)
        subparser.add_argument("--manifest", default="SHA256SUMS")
    args = parser.parse_args()

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
