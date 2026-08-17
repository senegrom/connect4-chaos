#!/usr/bin/env python3
"""Use curl for GitHub artifact redirects while retaining Python API pagination."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if old in source:
        if source.count(old) != 1:
            raise RuntimeError(f"Expected one {label} anchor, found {source.count(old)}.")
        return source.replace(old, new, 1)
    if new in source:
        return source
    raise RuntimeError(f"Could not find the {label} anchor.")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch-perfect-chaos-download-shards-curl.py <downloader>")
    path = Path(sys.argv[1])
    source = path.read_text()
    source = replace_once(
        source,
        "import stat\nimport urllib.request\n",
        "import stat\nimport subprocess\nimport urllib.request\n",
        "subprocess import",
    )
    anchor = "\ndef request_json(url: str, token: str) -> dict[str, Any]:\n"
    function = r'''
def request_artifact_bytes(url: str, token: str) -> bytes:
    """Follow GitHub's signed artifact redirect without forwarding auth to storage."""
    completed = subprocess.run(
        [
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "5", "--retry-delay", "1", "--retry-all-errors",
            "-H", f"Authorization: Bearer {token}",
            "-H", "Accept: application/vnd.github+json",
            "-H", f"X-GitHub-Api-Version: {API_VERSION}",
            "-H", "User-Agent: connect4-chaos-perfect-proof-auditor",
            url,
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        details = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ShardDownloadError(
            f"Artifact download failed with curl exit {completed.returncode}: {details}"
        )
    return completed.stdout

'''
    if "def request_artifact_bytes(" not in source:
        if source.count(anchor) != 1:
            raise RuntimeError("Could not find the artifact-request insertion anchor.")
        source = source.replace(anchor, function + anchor, 1)
    source = replace_once(
        source,
        '''    payload = request_bytes(
        f"https://api.github.com/repos/{repository}/actions/artifacts/{row['id']}/zip",
        token,
    )
''',
        '''    payload = request_artifact_bytes(
        f"https://api.github.com/repos/{repository}/actions/artifacts/{row['id']}/zip",
        token,
    )
''',
        "artifact request call",
    )
    path.write_text(source)


if __name__ == "__main__":
    main()
