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
import shutil
import stat
import subprocess
import time
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


def select_artifact_groups(
    artifacts: list[dict[str, Any]],
    *,
    run_id: int,
    run_sha: str,
    prefix: str,
    shard_count: int,
    allow_missing: int = 0,
) -> tuple[list[dict[str, Any]], list[int]]:
    if shard_count < 1 or shard_count > 512:
        raise ShardDownloadError("shard-count must be between 1 and 512.")
    if allow_missing < 0 or allow_missing > 8:
        raise ShardDownloadError("allow-missing must be between 0 and 8.")
    if not re.fullmatch(r"[0-9a-f]{40}", run_sha):
        raise ShardDownloadError("run-sha must be a lowercase 40-character SHA.")
    pattern = re.compile(re.escape(prefix) + r"(\d+)\Z")
    selected: dict[int, list[dict[str, Any]]] = {}
    seen_ids: set[int] = set()
    for artifact in artifacts:
        name = artifact.get("name")
        match = pattern.fullmatch(name) if isinstance(name, str) else None
        if not match:
            continue
        index = int(match.group(1))
        if index < 0 or index >= shard_count:
            raise ShardDownloadError(f"Shard artifact index is out of range: {name!r}.")
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
        if artifact_id in seen_ids:
            raise ShardDownloadError(f"Duplicate shard artifact id: {artifact_id}.")
        seen_ids.add(artifact_id)
        selected.setdefault(index, []).append({
            "index": index,
            "id": artifact_id,
            "name": name,
            "sha256": digest.removeprefix("sha256:"),
            "sizeInBytes": size,
            "createdAt": artifact.get("created_at"),
            "expiresAt": artifact.get("expires_at"),
        })
    missing = sorted(set(range(shard_count)).difference(selected))
    if len(missing) > allow_missing:
        preview = missing[:16]
        suffix = "..." if len(missing) > len(preview) else ""
        raise ShardDownloadError(
            f"Missing {len(missing)} shard artifact indexes: {preview}{suffix}; "
            f"allow-missing is {allow_missing}."
        )
    groups = []
    for index in sorted(selected):
        rows = sorted(selected[index], key=lambda row: row["id"])
        groups.append({"index": index, "artifacts": rows})
    return groups, missing


def select_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    run_id: int,
    run_sha: str,
    prefix: str,
    shard_count: int,
    allow_missing: int = 0,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Compatibility view selecting the newest metadata row for each shard index."""
    groups, missing = select_artifact_groups(
        artifacts,
        run_id=run_id,
        run_sha=run_sha,
        prefix=prefix,
        shard_count=shard_count,
        allow_missing=allow_missing,
    )
    return [group["artifacts"][-1] for group in groups], missing


def validate_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    run_id: int,
    run_sha: str,
    prefix: str,
    shard_count: int,
) -> list[dict[str, Any]]:
    rows, _ = select_artifacts(
        artifacts,
        run_id=run_id,
        run_sha=run_sha,
        prefix=prefix,
        shard_count=shard_count,
        allow_missing=0,
    )
    return rows


def inspect_archive(
    payload: bytes,
    *,
    index: int,
) -> tuple[list[dict[str, Any]], dict[str, bytes], int]:
    expected = expected_files(index)
    try:
        zipped = zipfile.ZipFile(BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ShardDownloadError(f"Shard {index} artifact is not a ZIP archive.") from error
    files: dict[str, bytes] = {}
    manifest: list[dict[str, Any]] = []
    total_uncompressed = 0
    with zipped:
        infos = zipped.infolist()
        names = [PurePosixPath(info.filename) for info in infos]
        actual = {name.as_posix() for name in names}
        if len(infos) != 4 or actual != expected:
            raise ShardDownloadError(
                f"Shard {index} archive entries differ: {sorted(actual)!r}."
            )
        for info, relative in zip(infos, names, strict=True):
            mode = (info.external_attr >> 16) & 0o170000
            if info.is_dir() or stat.S_ISLNK(mode) or relative.is_absolute() \
                    or ".." in relative.parts or len(relative.parts) != 1:
                raise ShardDownloadError(f"Shard {index} contains an unsafe ZIP entry.")
            data = zipped.read(info)
            total_uncompressed += len(data)
            if total_uncompressed > 250_000_000:
                raise ShardDownloadError(f"Shard {index} archive exceeds its size boundary.")
            canonical = relative.as_posix()
            files[canonical] = data
            manifest.append({
                "path": canonical,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    manifest.sort(key=lambda row: row["path"])
    return manifest, files, total_uncompressed


def manifest_digest(manifest: list[dict[str, Any]]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def materialize_files(files: dict[str, bytes], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for relative, data in sorted(files.items()):
        target = output / relative
        if target.exists() or target.is_symlink():
            raise ShardDownloadError(f"Duplicate extracted shard file: {target}.")
        target.write_bytes(data)


def extract_archive(payload: bytes, *, index: int, output: Path) -> dict[str, Any]:
    manifest, files, total_uncompressed = inspect_archive(payload, index=index)
    materialize_files(files, output)
    return {
        "files": sorted(files),
        "uncompressedBytes": total_uncompressed,
        "contentManifestSha256": manifest_digest(manifest),
    }


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


ARTIFACT_DOWNLOAD_ATTEMPTS = 8
ARTIFACT_DOWNLOAD_BASE_DELAY_SECONDS = 2.0
ARTIFACT_DOWNLOAD_MAX_DELAY_SECONDS = 60.0


def artifact_retry_delays(url: str, *, attempts: int = ARTIFACT_DOWNLOAD_ATTEMPTS) -> list[float]:
    """Return a deterministic exponential schedule with small per-URL jitter."""
    if attempts < 1 or attempts > 16:
        raise ShardDownloadError("artifact download attempts must be between 1 and 16.")
    seed = int(hashlib.sha256(url.encode("utf-8")).hexdigest()[:8], 16)
    delays: list[float] = []
    for attempt in range(attempts - 1):
        base = min(
            ARTIFACT_DOWNLOAD_MAX_DELAY_SECONDS,
            ARTIFACT_DOWNLOAD_BASE_DELAY_SECONDS * (2 ** attempt),
        )
        jitter = (seed & 0xF) / 16.0
        delays.append(base + jitter)
    return delays


def request_artifact_bytes(url: str, token: str) -> bytes:
    """Follow GitHub's signed redirect with bounded exponential storage retries."""
    delays = artifact_retry_delays(url)
    errors: list[str] = []
    for attempt in range(len(delays) + 1):
        completed = subprocess.run(
            [
                "curl", "--fail", "--location", "--silent", "--show-error",
                "--retry", "2", "--retry-delay", "1", "--retry-all-errors",
                "--connect-timeout", "30", "--max-time", "240",
                "-H", f"Authorization: Bearer {token}",
                "-H", "Accept: application/vnd.github+json",
                "-H", f"X-GitHub-Api-Version: {API_VERSION}",
                "-H", "User-Agent: connect4-chaos-perfect-proof-auditor",
                url,
            ],
            check=False,
            capture_output=True,
        )
        if completed.returncode == 0:
            return completed.stdout
        details = completed.stderr.decode("utf-8", errors="replace").strip()
        errors.append(f"attempt {attempt + 1}: curl {completed.returncode}: {details}")
        if attempt < len(delays):
            time.sleep(delays[attempt])
    raise ShardDownloadError(
        f"Artifact download failed after {len(errors)} outer attempts; {errors[-1]}"
    )


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


def download_group(
    group: dict[str, Any],
    *,
    repository: str,
    token: str,
    archive_directory: Path,
    verified_directory: Path,
    payloads: dict[int, bytes] | None = None,
) -> dict[str, Any]:
    index = int(group["index"])
    rows = group.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise ShardDownloadError(f"Shard {index} has no selected artifact archives.")
    reference_manifest: list[dict[str, Any]] | None = None
    reference_files: dict[str, bytes] | None = None
    reference_uncompressed = 0
    audited: list[dict[str, Any]] = []
    for row in rows:
        artifact_id = int(row["id"])
        payload = (
            payloads[artifact_id]
            if payloads is not None
            else request_artifact_bytes(
                f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}/zip",
                token,
            )
        )
        actual = hashlib.sha256(payload).hexdigest()
        if actual != row["sha256"]:
            raise ShardDownloadError(
                f"Shard {index} archive digest mismatch: {actual} != {row['sha256']}."
            )
        archive_path = archive_directory / f"{index:03d}-{artifact_id}.zip"
        archive_path.write_bytes(payload)
        manifest, files, total_uncompressed = inspect_archive(payload, index=index)
        if reference_manifest is None:
            reference_manifest = manifest
            reference_files = files
            reference_uncompressed = total_uncompressed
        elif manifest != reference_manifest:
            raise ShardDownloadError(
                f"Same-index shard artifact {artifact_id} is not byte-equivalent "
                f"to the first archive for shard {index}."
            )
        audited.append({
            **row,
            "archiveSha256": actual,
            "archiveBytes": len(payload),
            "uncompressedBytes": total_uncompressed,
            "contentManifestSha256": manifest_digest(manifest),
        })
    assert reference_manifest is not None and reference_files is not None
    materialize_files(reference_files, verified_directory)
    selected = audited[-1]
    return {
        "index": index,
        "matchingArchives": len(audited),
        "selectedArtifactId": selected["id"],
        "contentManifestSha256": manifest_digest(reference_manifest),
        "files": reference_manifest,
        "uncompressedBytes": reference_uncompressed,
        "artifacts": audited,
    }


def validate_output_directory(output: Path) -> None:
    if output.is_symlink():
        raise ShardDownloadError("output must be an absent or empty real directory.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ShardDownloadError("output must be an absent or empty real directory.")


def prepare_private_directory(path: Path, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise ShardDownloadError(f"{label} must not already exist: {path}.")
    path.mkdir(parents=True)


def publish_verified_directory(verified: Path, output: Path) -> None:
    validate_output_directory(output)
    if output.exists():
        output.rmdir()
    verified.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--run-sha", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--shard-count", required=True, type=int)
    parser.add_argument("--allow-missing", default=0, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int)
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
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    archive_directory = args.metadata.parent / ".shard-archives"
    verified_directory = args.output.parent / f".{args.output.name}.verified"
    validate_output_directory(args.output)
    prepare_private_directory(archive_directory, label="archive directory")
    prepare_private_directory(verified_directory, label="verified shard directory")
    try:
        artifacts = list_run_artifacts(args.repository, args.run_id, token)
        groups, missing = select_artifact_groups(
            artifacts,
            run_id=args.run_id,
            run_sha=args.run_sha,
            prefix=args.artifact_prefix,
            shard_count=args.shard_count,
            allow_missing=args.allow_missing,
        )
        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    download_group,
                    group,
                    repository=args.repository,
                    token=token,
                    archive_directory=archive_directory,
                    verified_directory=verified_directory,
                )
                for group in groups
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["index"])
        expected_indexes = [group["index"] for group in groups]
        if [item["index"] for item in results] != expected_indexes:
            raise ShardDownloadError("Downloaded shard set differs from the selected artifacts.")
        publish_verified_directory(verified_directory, args.output)
    except BaseException:
        if verified_directory.exists() and not verified_directory.is_symlink():
            shutil.rmtree(verified_directory)
        raise
    flattened = [artifact for shard in results for artifact in shard["artifacts"]]
    summary = {
        "format": "connect4-chaos-shard-artifact-download-v2",
        "repository": args.repository,
        "run": args.run_id,
        "runSha": args.run_sha,
        "artifactPrefix": args.artifact_prefix,
        "shardCount": args.shard_count,
        "downloadedShards": len(results),
        "matchingArchives": len(flattened),
        "missingShards": missing,
        "shards": results,
        "artifacts": flattened,
    }
    args.metadata.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "format": summary["format"],
        "run": args.run_id,
        "shards": len(results),
        "archives": len(flattened),
        "missing": len(missing),
        "files": len(results) * 4,
    }))


if __name__ == "__main__":
    main()
