#!/usr/bin/env python3
"""Build the reviewed production-only PR19 file set in an isolated checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA = "cdd086f4694b544c36074cba1d7614006ffdabb5"
DELETE_PATHS = (
    ".github/workflows/harden-perfect-classic-artifact-downloader.yml",
    ".github/workflows/verify-perfect-classic-artifact-boundary.yml",
)
OUTPUT_FILES = (
    ".github/workflows/compute-perfect-classic-7x7-role1.yml",
    ".github/workflows/compute-perfect-classic-7x7-role2.yml",
    "scripts/perfect-classic-download-fragments.py",
    "tests/perfect-classic-download-fragments.test.js",
)


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        fail(f"Expected one {label}, found {count}.")
    return text.replace(old, new, 1)


def harden_downloader() -> None:
    path = ROOT / "scripts/perfect-classic-download-fragments.py"
    text = path.read_text()
    old = '''    request = urllib.request.Request(
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
'''
    new = '''    request = urllib.request.Request(
        artifact["archiveUrl"],
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "connect4-chaos-perfect-classic-fragment-downloader",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: N802
            return None

    try:
        opener = urllib.request.build_opener(NoRedirect)
        try:
            opener.open(request, timeout=60)
        except urllib.error.HTTPError as redirect:
            if redirect.code not in {301, 302, 303, 307, 308}:
                raise
            signed_url = redirect.headers.get("Location")
        else:
            fail(
                f"Artifact {artifact['name']} ({artifact['id']}) download API "
                "did not return a signed redirect"
            )
        if not isinstance(signed_url, str) or not signed_url.startswith("https://"):
            fail(
                f"Artifact {artifact['name']} ({artifact['id']}) returned an "
                "unsafe signed archive URL"
            )
        signed_request = urllib.request.Request(
            signed_url,
            headers={
                "User-Agent": "connect4-chaos-perfect-classic-fragment-downloader",
            },
        )
        # Deliberately omit Authorization on the cross-origin signed URL.
        with urllib.request.urlopen(signed_request, timeout=300) as source:
            return copy_stream(source, destination)
    except urllib.error.URLError as error:
        fail(
            f"Could not download artifact {artifact['name']} "
            f"({artifact['id']}): {error}"
        )
'''
    text = replace_once(text, old, new, "authenticated archive download block")
    text = replace_once(
        text,
        '    if not name or "\\x00" in name or "\\\\" in name:\n',
        '    if not name or "\\x00" in name or "\\\\" in name or "//" in name:\n',
        "ZIP-entry name guard",
    )
    path.write_text(text, newline="\n")

    test_path = ROOT / "tests/perfect-classic-download-fragments.test.js"
    tests = test_path.read_text()
    marker = "authenticated archive redirects never forward the GitHub token"
    if marker in tests:
        fail("Redirect regression guard already exists unexpectedly.")
    tests += '''

test('authenticated archive redirects never forward the GitHub token', () => {
  const source = readFileSync(SCRIPT, 'utf8');
  assert.match(source, /class NoRedirect/);
  assert.match(source, /Deliberately omit Authorization on the cross-origin signed URL/);
  const signedBlock = source.slice(
    source.indexOf('signed_request = urllib.request.Request('),
    source.indexOf('with urllib.request.urlopen(signed_request'),
  );
  assert.doesNotMatch(signedBlock, /Authorization/);
});
'''
    test_path.write_text(tests, newline="\n")


def install_verified_workflow_downloads() -> dict[str, object]:
    report = ROOT / ".tmp-perfect-classic-fragment-workflow-patch.json"
    try:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/patch-perfect-classic-fragment-downloads.py"),
                "--root",
                str(ROOT),
                "--output",
                str(report),
            ],
            cwd=ROOT,
            check=True,
        )
        value = json.loads(report.read_text())
    finally:
        report.unlink(missing_ok=True)
    if value.get("changed") is not True or len(value.get("workflows", [])) != 2:
        fail(f"Workflow patch did not make the exact two reviewed changes: {value}")
    return value


def delete_one_shot_workflows() -> None:
    for relative in DELETE_PATHS:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            fail(f"Expected one-shot workflow is missing or unsafe: {relative}")
        path.unlink()


def require_exact_scope() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    changed = sorted(line[3:] for line in output if line)
    expected = sorted((*OUTPUT_FILES, *DELETE_PATHS))
    if changed != expected:
        fail(f"Unexpected production scope: {changed}; expected {expected}")
    return changed


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def package(output: Path, workflow_report: dict[str, object]) -> dict[str, object]:
    if output.exists():
        fail(f"Output already exists: {output}")
    files_root = output / "files"
    files = []
    for relative in OUTPUT_FILES:
        source = ROOT / relative
        if source.is_symlink() or not source.is_file():
            fail(f"Final file is missing or unsafe: {relative}")
        destination = files_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        files.append(
            {
                "path": relative,
                "mode": "100644",
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "delete.txt").write_text("\n".join(DELETE_PATHS) + "\n")
    manifest = {
        "format": "connect4-pr19-final-production-files-v1",
        "sourceSha": SOURCE_SHA,
        "files": files,
        "deleted": list(DELETE_PATHS),
        "workflowPatch": workflow_report,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", newline="\n"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != SOURCE_SHA:
        fail(f"Finalizer is bound to {SOURCE_SHA}, not {actual}.")
    harden_downloader()
    workflow_report = install_verified_workflow_downloads()
    delete_one_shot_workflows()
    require_exact_scope()
    manifest = package(arguments.output.resolve(), workflow_report)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
