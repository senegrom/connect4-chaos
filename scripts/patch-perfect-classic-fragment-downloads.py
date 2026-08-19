#!/usr/bin/env python3
"""Install verified fragment downloads in the two 7×7 Classic workflows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, NoReturn

FORMAT = "connect4-perfect-classic-fragment-workflow-patch-v1"
WORKFLOWS = {
    "role1": Path(".github/workflows/compute-perfect-classic-7x7-role1.yml"),
    "role2": Path(".github/workflows/compute-perfect-classic-7x7-role2.yml"),
}
PATTERN_STEP_RE = re.compile(
    r"      - uses: actions/download-artifact@v4\n"
    r"        with: \{ pattern: '([^']+)', path: downloaded \}\n"
)
PERMISSIONS_RE = re.compile(
    r"permissions:\n"
    r"(?:  actions: read\n)?"
    r"  contents: read\n"
)


def fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def load_text(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular file: {path}")
    return path.read_text()


def expected_patterns(role: str) -> list[str]:
    short = "r1" if role == "role1" else "r2"
    return [
        f"c4cert-7x7-{short}-prefix1-*",
        f"c4cert-7x7-{short}-prefix2-*",
        f"c4cert-7x7-{short}-*",
    ]


def verified_step(prefix: str) -> str:
    return f'''      - name: Download and verify exact Classic proof fragments
        shell: bash
        env:
          GH_TOKEN: ${{{{ github.token }}}}
        run: |
          set -euo pipefail
          rm -rf downloaded
          python3 scripts/perfect-classic-download-fragments.py \\
            --repository "$GITHUB_REPOSITORY" \\
            --run-id "$GITHUB_RUN_ID" \\
            --run-sha "$GITHUB_SHA" \\
            --artifact-prefix '{prefix}' \\
            --output downloaded \\
            --metadata downloaded/artifact-download-audit.json
'''


def install_permissions(text: str, path: Path) -> str:
    matches = list(PERMISSIONS_RE.finditer(text))
    if len(matches) != 1:
        fail(f"{path}: expected one read-only permissions block, found {len(matches)}")
    replacement = "permissions:\n  actions: read\n  contents: read\n"
    return text[: matches[0].start()] + replacement + text[matches[0].end() :]


def install_audit_copy(text: str, role: str, path: Path) -> str:
    short = "r1" if role == "role1" else "r2"
    output = f"generated/7x7-{short}-certificate"
    marker = f"cp downloaded/artifact-download-audit.json {output}/fragment-artifact-audit.json"
    if marker in text:
        return text
    # The two reviewed workflows use exactly two equivalent shell layouts:
    # Role 1 redirects on the --output line, while Role 2 continues once
    # before the redirect. Accept those layouts only; any other assembly drift
    # remains a hard failure before the audit record can be omitted.
    pattern = re.compile(
        rf"(          node scripts/perfect-classic-shards\.mjs assemble .*?"
        rf"--output {re.escape(output)} "
        rf"(?:> {re.escape(output)}/assembly\.json"
        rf"|\\\n            > {re.escape(output)}/assembly\.json)\n)",
        re.S,
    )
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        fail(f"{path}: expected one certificate assembly command, found {len(matches)}")
    addition = matches[0].group(1) + f"          {marker}\n"
    text = text[: matches[0].start()] + addition + text[matches[0].end() :]

    verify_anchor = (
        f"          node scripts/perfect-classic-policy.mjs verify-reference \\\n"
        f"            --reference {output}/manifest.json --verify-table-bits 25 \\\n"
    )
    check = f"          test -f {output}/fragment-artifact-audit.json\n"
    if check not in text:
        if text.count(verify_anchor) != 1:
            fail(f"{path}: could not locate the final certificate verification command")
        text = text.replace(verify_anchor, check + verify_anchor, 1)
    return text


def patch_workflow(root: Path, role: str) -> dict[str, Any]:
    relative = WORKFLOWS[role]
    path = root / relative
    original = load_text(path, f"{role} workflow")
    patterns = PATTERN_STEP_RE.findall(original)
    expected = expected_patterns(role)

    if patterns:
        if patterns != expected:
            fail(f"{path}: artifact pattern sequence {patterns} != {expected}")
        index = 0

        def replace(match: re.Match[str]) -> str:
            nonlocal index
            pattern = match.group(1)
            expected_pattern = expected[index]
            index += 1
            if pattern != expected_pattern:
                fail(f"{path}: artifact pattern {pattern!r} != {expected_pattern!r}")
            return verified_step(pattern[:-1])

        updated = PATTERN_STEP_RE.sub(replace, original)
        if index != 3:
            fail(f"{path}: replaced {index} pattern downloads, expected 3")
    else:
        updated = original
        prefixes = [pattern[:-1] for pattern in expected]
        for prefix in prefixes:
            count = updated.count(f"--artifact-prefix '{prefix}'")
            if count != 1:
                fail(
                    f"{path}: patched workflow must contain one downloader for "
                    f"{prefix!r}; found {count}"
                )

    updated = install_permissions(updated, path)
    updated = install_audit_copy(updated, role, path)
    if PATTERN_STEP_RE.search(updated):
        fail(f"{path}: an unverified wildcard artifact download remains")
    if updated.count("scripts/perfect-classic-download-fragments.py") != 3:
        fail(f"{path}: expected three verified fragment download invocations")
    if updated.count("actions/download-artifact@v4") != 1:
        fail(
            f"{path}: expected only the final exact-name certificate download to remain"
        )
    if updated.count("  actions: read\n") != 1:
        fail(f"{path}: actions: read permission is missing or duplicated")

    changed = updated != original
    if changed:
        path.write_text(updated, newline="\n")
    return {
        "role": role,
        "path": relative.as_posix(),
        "changed": changed,
        "verifiedDownloads": 3,
        "remainingExactNameDownloads": 1,
    }


def patch_repository(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        fail(f"Repository root must be a regular directory: {root}")
    records = [patch_workflow(root, role) for role in WORKFLOWS]
    return {
        "format": FORMAT,
        "changed": any(record["changed"] for record in records),
        "workflows": records,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    result = patch_repository(arguments.root.resolve())
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, newline="\n")
    print(encoded, end="")


if __name__ == "__main__":
    main()
