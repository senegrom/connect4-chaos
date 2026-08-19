#!/usr/bin/env python3
"""Download sharded Classic proof artifacts with exact identity validation.

GitHub's convenience download action is sufficient for ordinary build outputs,
but an exact certificate should bind every fragment to one workflow run and
commit, verify GitHub's archive digest, reject unsafe ZIP entries, and handle
rerun duplicates deterministically.  This tool performs that proof boundary
and writes one extracted directory per artifact name plus a machine-readable
audit manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, NoReturn

FORMAT = "connect4-perfect-classic-fragment-download-v1"
REPOSITORY_RE = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
ARTIFACT_NAME_RE = re.compile(r"[A-Za-z0-9._-]+")
SHA_RE = re.compile(r"[0-9a-f]{40}")
DIGEST_RE = re.compile(r"sha256:([0-9a-f]{64})")


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def require_integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        fail(f"{label} must be an integer in [{minimum}, {maximum}]")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def request_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "connect4-chaos-perfect-classic-fragment-downloader",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            value = json.load(response)
    except (urllib.error.URLError, json.JSONDecodeError) as error:
        fail(f"Could not read GitHub artifact metadata from {url}: {error}")
    if not isinstance(value, dict):
        fail(f"GitHub returned a non-object response for {url}")
    return value


def list_run_artifacts(repository: str, run_id: int, token: str) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    expected_total: int | None = None
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/artifacts"
            f"?per_page=100&page={page}"
        )
        response = request_json(url, token)
        if set(response).difference({"total_count", "artifacts"}):
            # GitHub may add fields in the future; only the two proof-relevant
            # fields are consumed, so unknown response metadata is harmless.
            pass
        total = require_integer(
            response.get("total_count"),
            f"artifact page {page}.total_count",
            0,
            1_000_000,
        )
        if expected_total is None:
            expected_total = total
        elif expected_total != total:
            fail(
                "GitHub artifact total changed during pagination: "
                f"{expected_total} -> {total}"
            )
        values = response.get("artifacts")
        if not isinstance(values, list):
            fail(f"artifact page {page}.artifacts must be an array")
        for artifact in values:
            if not isinstance(artifact, dict):
                fail(f"artifact page {page} contains a non-object record")
            artifacts.append(artifact)
        if len(artifacts) >= total:
            break
        if not values:
            fail(
                f"GitHub reported {total} artifacts but pagination stopped at "
                f"{len(artifacts)}"
            )
        page += 1
        if page > 10_000:
            fail("Artifact pagination exceeded the fail-closed page limit")
    if len(artifacts) != expected_total:
        fail(
            f"Artifact pagination produced {len(artifacts)} records, expected "
            f"{expected_total}"
        )
    return artifacts


def load_offline_index(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        fail(f"Offline artifact index must be a regular file: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"Could not parse offline artifact index {path}: {error}")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        fail("Offline artifact index must contain an array of objects")
    result: list[dict[str, Any]] = []
    for item in value:
        copied = dict(item)
        archive_path = copied.get("archive_path")
        if not isinstance(archive_path, str) or not archive_path:
            fail("Every offline artifact requires archive_path")
        resolved = (path.parent / archive_path).resolve()
        copied["_offline_archive"] = resolved.as_posix()
        result.append(copied)
    return result


def validate_artifact(
    artifact: dict[str, Any],
    *,
    run_id: int,
    run_sha: str,
    prefix: str,
) -> dict[str, Any] | None:
    name = artifact.get("name")
    if not isinstance(name, str) or ARTIFACT_NAME_RE.fullmatch(name) is None:
        fail(f"Artifact has an unsafe name: {name!r}")
    if not name.startswith(prefix):
        return None
    artifact_id = require_integer(artifact.get("id"), f"{name}.id", 1, 10**18)
    if artifact.get("expired") is not False:
        fail(f"Required artifact {name} ({artifact_id}) is expired")
    digest = artifact.get("digest")
    match = DIGEST_RE.fullmatch(digest) if isinstance(digest, str) else None
    if match is None:
        fail(f"Artifact {name} ({artifact_id}) has no valid GitHub SHA-256 digest")
    workflow_run = artifact.get("workflow_run")
    if not isinstance(workflow_run, dict):
        fail(f"Artifact {name} ({artifact_id}) has no workflow_run identity")
    if workflow_run.get("id") != run_id:
        fail(
            f"Artifact {name} ({artifact_id}) belongs to run "
            f"{workflow_run.get('id')}, expected {run_id}"
        )
    if workflow_run.get("head_sha") != run_sha:
        fail(
            f"Artifact {name} ({artifact_id}) belongs to commit "
            f"{workflow_run.get('head_sha')}, expected {run_sha}"
        )
    archive_url = artifact.get("archive_download_url")
    offline_archive = artifact.get("_offline_archive")
    if offline_archive is None and (
        not isinstance(archive_url, str)
        or not archive_url.startswith("https://api.github.com/")
    ):
        fail(f"Artifact {name} ({artifact_id}) has an unsafe archive URL")
    size = artifact.get("size_in_bytes")
    if size is not None:
        require_integer(size, f"{name}.size_in_bytes", 0, 100_000_000_000)
    return {
        "id": artifact_id,
        "name": name,
        "digest": digest,
        "expectedDigest": match.group(1),
        "sizeInBytes": size,
        "archiveUrl": archive_url,
        "offlineArchive": offline_archive,
        "createdAt": artifact.get("created_at"),
        "updatedAt": artifact.get("updated_at"),
    }


def copy_stream(source: BinaryIO, destination: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with destination.open("wb") as output:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        output.flush()
        os.fsync(output.fileno())
    return size, digest.hexdigest()


def obtain_archive(
    artifact: dict[str, Any],
    *,
    token: str | None,
    destination: Path,
) -> tuple[int, str]:
    offline = artifact["offlineArchive"]
    if offline is not None:
        source_path = Path(offline)
        if source_path.is_symlink() or not source_path.is_file():
            fail(f"Offline archive must be a regular file: {source_path}")
        with source_path.open("rb") as source:
            return copy_stream(source, destination)
    if not token:
        fail("GH_TOKEN is required to download GitHub artifacts")
    request = urllib.request.Request(
        artifact["archiveUrl"],
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "connect4-chaos-perfect-classic-fragment-downloader",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as source:
            return copy_stream(source, destination)
    except urllib.error.URLError as error:
        fail(
            f"Could not download artifact {artifact['name']} "
            f"({artifact['id']}): {error}"
        )


def safe_member_path(name: str) -> PurePosixPath:
    if not name or "\x00" in name or "\\" in name:
        fail(f"Unsafe ZIP entry name: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        fail(f"Unsafe ZIP entry path: {name!r}")
    return path


def extract_archive(
    archive: Path,
    destination: Path,
    *,
    maximum_entries: int,
    maximum_uncompressed_bytes: int,
) -> None:
    try:
        handle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as error:
        fail(f"Artifact archive is not a valid ZIP: {archive}: {error}")
    with handle:
        infos = handle.infolist()
        if len(infos) > maximum_entries:
            fail(
                f"Artifact archive contains {len(infos)} entries, exceeding "
                f"the limit {maximum_entries}"
            )
        total = 0
        names: set[str] = set()
        folded: set[str] = set()
        validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
        for info in infos:
            path = safe_member_path(info.filename.rstrip("/") if info.is_dir() else info.filename)
            normalized = path.as_posix()
            if normalized in names:
                fail(f"Artifact archive contains duplicate path {normalized!r}")
            names.add(normalized)
            casefolded = normalized.casefold()
            if casefolded in folded:
                fail(f"Artifact archive contains a case-colliding path {normalized!r}")
            folded.add(casefolded)
            if info.flag_bits & 0x1:
                fail(f"Artifact archive contains encrypted entry {normalized!r}")
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                fail(f"Artifact archive contains symbolic link {normalized!r}")
            if mode and not info.is_dir() and not stat.S_ISREG(mode):
                fail(f"Artifact archive contains unsupported file type {normalized!r}")
            if info.file_size < 0:
                fail(f"Artifact archive contains invalid size for {normalized!r}")
            total += info.file_size
            if total > maximum_uncompressed_bytes:
                fail(
                    "Artifact archive uncompressed size exceeds the limit "
                    f"{maximum_uncompressed_bytes}"
                )
            validated.append((info, path))

        destination.mkdir(parents=True, exist_ok=False)
        root = destination.resolve()
        for info, relative in validated:
            target = destination.joinpath(*relative.parts)
            resolved = target.resolve()
            if root != resolved and root not in resolved.parents:
                fail(f"ZIP entry escaped extraction root: {relative}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info, "r") as source, target.open("xb") as output:
                written = 0
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    written += len(chunk)
                output.flush()
                os.fsync(output.fileno())
            if written != info.file_size:
                fail(
                    f"ZIP entry {relative} extracted {written} bytes, "
                    f"expected {info.file_size}"
                )


def tree_manifest(root: Path, required_basename: str) -> tuple[list[dict[str, Any]], str]:
    files: list[dict[str, Any]] = []
    required: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            fail(f"Extracted artifact contains symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            fail(f"Extracted artifact contains unsupported filesystem entry: {path}")
        relative = path.relative_to(root).as_posix()
        record = {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        files.append(record)
        if path.name == required_basename:
            required.append(relative)
    if len(required) != 1:
        fail(
            f"Artifact {root.name} must contain exactly one {required_basename}; "
            f"found {required}"
        )
    digest = hashlib.sha256(canonical_json(files)).hexdigest()
    return files, digest


def ensure_output(output: Path, replace: bool) -> None:
    if output.is_symlink():
        fail(f"Output path cannot be a symbolic link: {output}")
    if output.exists():
        if not output.is_dir():
            fail(f"Output path must be a directory: {output}")
        if any(output.iterdir()):
            if not replace:
                fail(f"Output directory is not empty: {output}")
            shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)


def download_fragments(
    artifacts: list[dict[str, Any]],
    *,
    repository: str,
    run_id: int,
    run_sha: str,
    prefix: str,
    output: Path,
    token: str | None,
    required_basename: str,
    maximum_entries: int,
    maximum_uncompressed_bytes: int,
    replace: bool,
) -> dict[str, Any]:
    ensure_output(output, replace)
    groups: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[int] = set()
    for raw in artifacts:
        candidate = validate_artifact(raw, run_id=run_id, run_sha=run_sha, prefix=prefix)
        if candidate is None:
            continue
        if candidate["id"] in seen_ids:
            fail(f"Artifact id {candidate['id']} appears more than once")
        seen_ids.add(candidate["id"])
        groups.setdefault(candidate["name"], []).append(candidate)
    if not groups:
        fail(f"No unexpired artifacts beginning with {prefix!r} were found")

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="c4-classic-artifacts-") as directory:
        temporary = Path(directory)
        for name in sorted(groups):
            candidates = sorted(groups[name], key=lambda item: item["id"])
            processed: list[dict[str, Any]] = []
            for candidate in candidates:
                archive = temporary / f"{candidate['id']}.zip"
                size, digest = obtain_archive(candidate, token=token, destination=archive)
                if digest != candidate["expectedDigest"]:
                    fail(
                        f"Artifact {name} ({candidate['id']}) archive digest mismatch: "
                        f"{digest} != {candidate['expectedDigest']}"
                    )
                if candidate["sizeInBytes"] is not None and size != candidate["sizeInBytes"]:
                    fail(
                        f"Artifact {name} ({candidate['id']}) archive size mismatch: "
                        f"{size} != {candidate['sizeInBytes']}"
                    )
                extracted = temporary / f"extract-{candidate['id']}"
                extract_archive(
                    archive,
                    extracted,
                    maximum_entries=maximum_entries,
                    maximum_uncompressed_bytes=maximum_uncompressed_bytes,
                )
                files, content_digest = tree_manifest(extracted, required_basename)
                processed.append(
                    {
                        **candidate,
                        "archiveBytes": size,
                        "archiveSha256": digest,
                        "contentSha256": content_digest,
                        "files": files,
                        "extracted": extracted,
                    }
                )
            content_digests = {item["contentSha256"] for item in processed}
            if len(content_digests) != 1:
                conflict = [
                    {"id": item["id"], "contentSha256": item["contentSha256"]}
                    for item in processed
                ]
                fail(f"Same-name artifact {name!r} has conflicting contents: {conflict}")
            selected = max(processed, key=lambda item: item["id"])
            destination = output / name
            shutil.copytree(selected["extracted"], destination)
            records.append(
                {
                    "name": name,
                    "selectedArtifactId": selected["id"],
                    "contentSha256": selected["contentSha256"],
                    "files": selected["files"],
                    "equivalentArchives": [
                        {
                            "id": item["id"],
                            "digest": item["digest"],
                            "archiveBytes": item["archiveBytes"],
                            "archiveSha256": item["archiveSha256"],
                            "createdAt": item["createdAt"],
                            "updatedAt": item["updatedAt"],
                        }
                        for item in processed
                    ],
                }
            )

    manifest = {
        "format": FORMAT,
        "repository": repository,
        "runId": run_id,
        "runSha": run_sha,
        "artifactPrefix": prefix,
        "requiredBasename": required_basename,
        "artifactNames": len(records),
        "archiveCandidates": sum(len(group) for group in groups.values()),
        "artifacts": records,
    }
    return manifest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-sha", required=True)
    parser.add_argument("--artifact-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--required-basename", default="fragment.json")
    parser.add_argument("--maximum-entries", type=int, default=10_000)
    parser.add_argument("--maximum-uncompressed-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--offline-index", type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if REPOSITORY_RE.fullmatch(arguments.repository) is None:
        fail(f"Unsafe repository identity: {arguments.repository!r}")
    run_id = require_integer(arguments.run_id, "run_id", 1, 10**15)
    if SHA_RE.fullmatch(arguments.run_sha) is None:
        fail("run_sha must be a lowercase 40-character commit SHA")
    if (
        not arguments.artifact_prefix
        or ARTIFACT_NAME_RE.fullmatch(arguments.artifact_prefix) is None
    ):
        fail(f"Unsafe artifact prefix: {arguments.artifact_prefix!r}")
    if (
        not arguments.required_basename
        or "/" in arguments.required_basename
        or "\\" in arguments.required_basename
        or arguments.required_basename in {".", ".."}
    ):
        fail(f"Unsafe required basename: {arguments.required_basename!r}")
    maximum_entries = require_integer(
        arguments.maximum_entries, "maximum_entries", 1, 1_000_000
    )
    maximum_bytes = require_integer(
        arguments.maximum_uncompressed_bytes,
        "maximum_uncompressed_bytes",
        1,
        100_000_000_000,
    )

    token = os.environ.get("GH_TOKEN")
    if arguments.offline_index is not None:
        artifacts = load_offline_index(arguments.offline_index.resolve())
    else:
        if not token:
            fail("GH_TOKEN is required unless --offline-index is supplied")
        artifacts = list_run_artifacts(arguments.repository, run_id, token)

    manifest = download_fragments(
        artifacts,
        repository=arguments.repository,
        run_id=run_id,
        run_sha=arguments.run_sha,
        prefix=arguments.artifact_prefix,
        output=arguments.output.resolve(),
        token=token,
        required_basename=arguments.required_basename,
        maximum_entries=maximum_entries,
        maximum_uncompressed_bytes=maximum_bytes,
        replace=arguments.replace,
    )
    arguments.metadata.parent.mkdir(parents=True, exist_ok=True)
    arguments.metadata.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
