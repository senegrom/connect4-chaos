#!/usr/bin/env python3
"""Download every deterministic Perfect Chaos shard artifact without pagination gaps.

The official download action currently lists at most 100 artifacts for a pattern. Exact
campaigns use up to 512 shards, so this tool enumerates all pages through the REST API,
binds every artifact to one run and commit, verifies GitHub's archive SHA-256 digest,
and extracts exactly the four expected flat shard files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import stat
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

API_VERSION = "2022-11-28"
EXPECTED_PREFIXES = ("summary", "rejected", "policy", "frontier")


class ShardDownloadError(RuntimeError):
    """Raised when shard artifact identity or archive contents fail closed."""


def expected_files(index: int) -> set[str]:
    return {
        f"summary-{index}.json",
        f"rejected-{index}.bin",
        f"policy-{index}.bin",
        f"frontier-{index}.bin",
    }


def validate_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    run_id: int,
    run_sha: str,
    prefix: str,
    shard_count: int,
) -> list[dict[str, Any]]:
    if shard_count < 1 or shard_count > 512:
        raise ShardDownloadError("shard-count must be between 1 and 512.")
    if not re.fullmatch(r"[0-9a-f]{40}", run_sha):
        raise ShardDownloadError("run-sha must be a lowercase 40-character SHA.")
    pattern = re.compile(re.escape(prefix) + r"(\d+)\Z")
    selected: dict[int, dict[str, Any]] = {}
    for artifact in artifacts:
        name = artifact.get("name")
        match = pattern.fullmatch(name) if isinstance(name, str) else None
        if not match:
            continue
        index = int(match.group(1))
        if index < 0 or index >= shard_count:
            raise ShardDownloadError(f"Shard artifact index is out of range: {name!r}.")
        if index in selected:
            raise ShardDownloadError(f"Duplicate shard artifact index: {index}.")
        if artifact.get("expired") is not False:
            raise ShardDownloadError(f"Shard artifact is expired: {name!r}.")
        workflow = artifact.get("workflow_run")
        if not isinstance(workflow, dict) or workflow.get("id") != run_id \
                or workflow.get("head_sha") != run_sha:
            raise ShardDownloadError(f"Shard artifact has the wrong producer identity: {name!r}.")
        digest = artifact.get("digest")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ShardDownloadError(f"Shard artifact has no valid SHA-256 digest: {name!r}.")
        size = artifact.get("size_in_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ShardDownloadError(f"Shard artifact has an invalid size: {name!r}.")
        artifact_id = artifact.get("id")
        if isinstance(artifact_id, bool) or not isinstance(artifact_id, int) or artifact_id < 1:
            raise ShardDownloadError(f"Shard artifact has an invalid id: {name!r}.")
        selected[index] = {
            "index": index,
            "id": artifact_id,
            "name": name,
            "sha256": digest.removeprefix("sha256:"),
            "sizeInBytes": size,
            "createdAt": artifact.get("created_at"),
            "expiresAt": artifact.get("expires_at"),
        }
    missing = sorted(set(range(shard_count)).difference(selected))
    if missing:
        preview = missing[:16]
        suffix = "..." if len(missing) > len(preview) else ""
        raise ShardDownloadError(f"Missing shard artifact indexes: {preview}{suffix}.")
    return [selected[index] for index in range(shard_count)]


def extract_archive(payload: bytes, *, index: int, output: Path) -> dict[str, Any]:
    expected = expected_files(index)
    try:
        zipped = zipfile.ZipFile(BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ShardDownloadError(f"Shard {index} artifact is not a ZIP archive.") from error
    with zipped:
        infos = zipped.infolist()
        names = [PurePosixPath(info.filename) for info in infos]
        actual = {name.as_posix() for name in names}
        if len(infos) != 4 or actual != expected:
            raise ShardDownloadError(
                f"Shard {index} archive entries differ: {sorted(actual)!r}."
            )
        total_uncompressed = 0
        output.mkdir(parents=True, exist_ok=True)
        for info, relative in zip(infos, names, strict=True):
            mode = (info.external_attr >> 16) & 0o170000
            if info.is_dir() or stat.S_ISLNK(mode) or relative.is_absolute() \
                    or ".." in relative.parts or len(relative.parts) != 1:
                raise ShardDownloadError(f"Shard {index} contains an unsafe ZIP entry.")
            total_uncompressed += info.file_size
            if total_uncompressed > 250_000_000:
                raise ShardDownloadError(f"Shard {index} archive exceeds its size boundary.")
            target = output / relative.name
            if target.exists() or target.is_symlink():
                raise ShardDownloadError(f"Duplicate extracted shard file: {target}.")
            target.write_bytes(zipped.read(info))
    return {"files": sorted(expected), "uncompressedBytes": total_uncompressed}


def request_bytes(url: str, token: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "connect4-chaos-perfect-proof-auditor",
        },
    )
    with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
        return response.read()


def request_json(url: str, token: str) -> dict[str, Any]:
    payload = request_bytes(url, token)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ShardDownloadError(f"GitHub returned invalid JSON for {url}.") from error
    if not isinstance(value, dict):
        raise ShardDownloadError(f"GitHub returned a non-object for {url}.")
    return value


def list_run_artifacts(repository: str, run_id: int, token: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    page = 1
    while True:
        response = request_json(
            f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/artifacts"
            f"?per_page=100&page={page}",
            token,
        )
        batch = response.get("artifacts")
        if not isinstance(batch, list):
            raise ShardDownloadError("GitHub artifact response has no artifacts array.")
        artifacts.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100:
            break
        page += 1
        if page > 100:
            raise ShardDownloadError("Artifact pagination exceeded 100 pages.")
    return artifacts


def download_one(
    row: dict[str, Any],
    *,
    repository: str,
    token: str,
    archive_directory: Path,
    output: Path,
) -> dict[str, Any]:
    index = int(row["index"])
    payload = request_bytes(
        f"https://api.github.com/repos/{repository}/actions/artifacts/{row['id']}/zip",
        token,
    )
    actual = hashlib.sha256(payload).hexdigest()
    if actual != row["sha256"]:
        raise ShardDownloadError(
            f"Shard {index} archive digest mismatch: {actual} != {row['sha256']}."
        )
    archive_directory.mkdir(parents=True, exist_ok=True)
    archive_path = archive_directory / f"{index:03d}.zip"
    archive_path.write_bytes(payload)
    extracted = extract_archive(payload, index=index, output=output)
    return {
        **row,
        "archiveSha256": actual,
        "archiveBytes": len(payload),
        **extracted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--run-sha", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--workers", default=16, type=int)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        raise ShardDownloadError("repository must be an owner/name pair.")
    if args.run_id < 1:
        raise ShardDownloadError("run-id must be positive.")
    if args.workers < 1 or args.workers > 32:
        raise ShardDownloadError("workers must be between 1 and 32.")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ShardDownloadError("GH_TOKEN is required.")
    args.output.mkdir(parents=True, exist_ok=True)
    archive_directory = args.metadata.parent / ".shard-archives"
    artifacts = list_run_artifacts(args.repository, args.run_id, token)
    rows = validate_artifacts(
        artifacts,
        run_id=args.run_id,
        run_sha=args.run_sha,
        prefix=args.artifact_prefix,
        shard_count=args.shard_count,
    )
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                row,
                repository=args.repository,
                token=token,
                archive_directory=archive_directory,
                output=args.output,
            )
            for row in rows
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: item["index"])
    if [item["index"] for item in results] != list(range(args.shard_count)):
        raise ShardDownloadError("Downloaded shard set is incomplete.")
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "format": "connect4-chaos-shard-artifact-download-v1",
        "repository": args.repository,
        "run": args.run_id,
        "runSha": args.run_sha,
        "artifactPrefix": args.artifact_prefix,
        "shardCount": args.shard_count,
        "artifacts": results,
    }
    args.metadata.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "format": summary["format"],
        "run": args.run_id,
        "shards": len(results),
        "files": len(results) * 4,
    }))


if __name__ == "__main__":
    main()
