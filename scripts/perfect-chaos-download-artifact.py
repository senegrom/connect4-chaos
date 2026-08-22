#!/usr/bin/env python3
"""Download one logical GitHub Actions artifact with rerun-safe equivalence checks.

GitHub retains artifacts from earlier attempts when failed jobs are rerun. That can leave
several unexpired artifacts with the same name. A proof campaign must not select one
silently. This tool binds every matching archive to one run and commit, verifies its
GitHub SHA-256 digest, extracts it safely, and accepts duplicates only when every
extracted file is byte-equivalent. It then publishes one deterministic copy plus an
audit manifest describing every equivalent producer archive.
"""

from __future__ import annotations

import argparse
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
MAX_FILES = 100_000
MAX_UNCOMPRESSED_BYTES = 4_000_000_000


class ArtifactDownloadError(RuntimeError):
    """Raised when named artifact identity, integrity, or equivalence fails closed."""


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
        raise ArtifactDownloadError("artifact download attempts must be between 1 and 16.")
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
    raise ArtifactDownloadError(
        f"Artifact download failed after {len(errors)} outer attempts; {errors[-1]}"
    )


def request_json(url: str, token: str) -> dict[str, Any]:
    payload = request_bytes(url, token)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ArtifactDownloadError(f"GitHub returned invalid JSON for {url}.") from error
    if not isinstance(value, dict):
        raise ArtifactDownloadError(f"GitHub returned a non-object for {url}.")
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
            raise ArtifactDownloadError("GitHub artifact response has no artifacts array.")
        artifacts.extend(item for item in batch if isinstance(item, dict))
        if len(batch) < 100:
            break
        page += 1
        if page > 100:
            raise ArtifactDownloadError("Artifact pagination exceeded 100 pages.")
    return artifacts


def validate_named_artifacts(
    artifacts: list[dict[str, Any]],
    *,
    run_id: int,
    run_sha: str,
    artifact_name: str,
) -> list[dict[str, Any]]:
    if not re.fullmatch(r"[0-9a-f]{40}", run_sha):
        raise ArtifactDownloadError("run-sha must be a lowercase 40-character SHA.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", artifact_name):
        raise ArtifactDownloadError("artifact-name contains unsupported characters.")

    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for artifact in artifacts:
        if artifact.get("name") != artifact_name or artifact.get("expired") is not False:
            continue
        workflow = artifact.get("workflow_run")
        if not isinstance(workflow, dict) or workflow.get("id") != run_id \
                or workflow.get("head_sha") != run_sha:
            raise ArtifactDownloadError(
                f"Artifact {artifact_name!r} has the wrong producer identity."
            )
        digest = artifact.get("digest")
        if not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            raise ArtifactDownloadError(
                f"Artifact {artifact_name!r} has no valid SHA-256 digest."
            )
        size = artifact.get("size_in_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ArtifactDownloadError(f"Artifact {artifact_name!r} has an invalid size.")
        artifact_id = artifact.get("id")
        if isinstance(artifact_id, bool) or not isinstance(artifact_id, int) or artifact_id < 1:
            raise ArtifactDownloadError(f"Artifact {artifact_name!r} has an invalid id.")
        if artifact_id in seen_ids:
            raise ArtifactDownloadError(f"Duplicate artifact id {artifact_id}.")
        seen_ids.add(artifact_id)
        rows.append({
            "id": artifact_id,
            "name": artifact_name,
            "sha256": digest.removeprefix("sha256:"),
            "sizeInBytes": size,
            "createdAt": artifact.get("created_at"),
            "expiresAt": artifact.get("expires_at"),
        })
    if not rows:
        raise ArtifactDownloadError(
            f"No unexpired artifact named {artifact_name!r} exists in run {run_id}."
        )
    rows.sort(key=lambda row: row["id"])
    return rows


def inspect_archive(payload: bytes) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    try:
        zipped = zipfile.ZipFile(BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ArtifactDownloadError("Artifact is not a ZIP archive.") from error

    files: dict[str, bytes] = {}
    manifest: list[dict[str, Any]] = []
    total = 0
    with zipped:
        infos = zipped.infolist()
        if not infos or len(infos) > MAX_FILES:
            raise ArtifactDownloadError("Artifact has an invalid number of ZIP entries.")
        for info in infos:
            name = info.filename
            relative = PurePosixPath(name)
            mode = (info.external_attr >> 16) & 0o170000
            if (
                info.is_dir()
                or stat.S_ISLNK(mode)
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or "\x00" in name
                or not relative.parts
            ):
                raise ArtifactDownloadError(f"Artifact contains an unsafe ZIP entry: {name!r}.")
            canonical = relative.as_posix()
            if canonical in files:
                raise ArtifactDownloadError(f"Artifact contains duplicate entry {canonical!r}.")
            data = zipped.read(info)
            total += len(data)
            if total > MAX_UNCOMPRESSED_BYTES:
                raise ArtifactDownloadError("Artifact exceeds its uncompressed size boundary.")
            files[canonical] = data
            manifest.append({
                "path": canonical,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            })
    manifest.sort(key=lambda row: row["path"])
    return manifest, files


def manifest_digest(manifest: list[dict[str, Any]]) -> str:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def materialize(files: dict[str, bytes], output: Path) -> None:
    if output.exists():
        if output.is_symlink() or not output.is_dir() or any(output.iterdir()):
            raise ArtifactDownloadError("output must be absent or an empty real directory.")
    else:
        output.mkdir(parents=True)
    for relative, data in sorted(files.items()):
        target = output / PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise ArtifactDownloadError(f"Refusing to overwrite extracted file {target}.")
        target.write_bytes(data)


def download_equivalent_artifact(
    *,
    repository: str,
    run_id: int,
    run_sha: str,
    artifact_name: str,
    output: Path,
    metadata: Path,
    token: str,
    artifacts: list[dict[str, Any]] | None = None,
    payloads: dict[int, bytes] | None = None,
) -> dict[str, Any]:
    listed = list_run_artifacts(repository, run_id, token) if artifacts is None else artifacts
    rows = validate_named_artifacts(
        listed,
        run_id=run_id,
        run_sha=run_sha,
        artifact_name=artifact_name,
    )

    archive_directory = metadata.parent / ".named-artifact-archives"
    archive_directory.mkdir(parents=True, exist_ok=True)
    reference_manifest: list[dict[str, Any]] | None = None
    reference_files: dict[str, bytes] | None = None
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
        actual_digest = hashlib.sha256(payload).hexdigest()
        if actual_digest != row["sha256"]:
            raise ArtifactDownloadError(
                f"Artifact {artifact_id} archive digest mismatch: "
                f"{actual_digest} != {row['sha256']}."
            )
        archive_path = archive_directory / f"{artifact_id}.zip"
        archive_path.write_bytes(payload)
        manifest, files = inspect_archive(payload)
        if reference_manifest is None:
            reference_manifest = manifest
            reference_files = files
        elif manifest != reference_manifest:
            raise ArtifactDownloadError(
                f"Same-named artifact {artifact_id} is not byte-equivalent to the first match."
            )
        audited.append({
            **row,
            "archiveSha256": actual_digest,
            "archiveBytes": len(payload),
            "contentManifestSha256": manifest_digest(manifest),
        })

    assert reference_manifest is not None and reference_files is not None
    materialize(reference_files, output)
    selected = audited[-1]
    summary = {
        "format": "connect4-chaos-equivalent-named-artifact-download-v1",
        "repository": repository,
        "run": run_id,
        "runSha": run_sha,
        "artifactName": artifact_name,
        "matchingArchives": len(audited),
        "selectedArtifactId": selected["id"],
        "contentManifestSha256": manifest_digest(reference_manifest),
        "files": reference_manifest,
        "artifacts": audited,
    }
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--run-sha", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        raise ArtifactDownloadError("repository must be an owner/name pair.")
    if args.run_id < 1:
        raise ArtifactDownloadError("run-id must be positive.")
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ArtifactDownloadError("GH_TOKEN is required.")
    summary = download_equivalent_artifact(
        repository=args.repository,
        run_id=args.run_id,
        run_sha=args.run_sha,
        artifact_name=args.artifact_name,
        output=args.output,
        metadata=args.metadata,
        token=token,
    )
    print(json.dumps({
        "format": summary["format"],
        "run": summary["run"],
        "artifact": summary["artifactName"],
        "matchingArchives": summary["matchingArchives"],
        "files": len(summary["files"]),
    }))


if __name__ == "__main__":
    main()
