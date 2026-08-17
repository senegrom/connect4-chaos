#!/usr/bin/env python3
"""Align the independent sharded auditor with durable artifact identity."""

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
        raise SystemExit(
            "usage: patch-perfect-chaos-sharded-audit-artifact-identity.py "
            "<sharded-auditor>"
        )
    path = Path(sys.argv[1])
    source = path.read_text()

    source = replace_once(
        source,
        "import json\nimport struct\n",
        "import json\nimport re\nimport struct\n",
        "regular-expression import",
    )
    source = replace_once(
        source,
        'MERGED_FORMAT = "connect4-chaos-frontier-classification-merged-v1"\n',
        'MERGED_FORMAT = "connect4-chaos-frontier-classification-merged-v1"\n'
        'TRANSIENT_DIRECTORY = re.compile(r"\\.incremental-repair-[0-9]+-[0-9]+")\n',
        "transient directory constant",
    )

    helper_anchor = "\ndef verify_sha256sums(root: Path) -> dict[str, str]:\n"
    helper = '''\ndef transient_artifact_path(relative: PurePosixPath) -> bool:\n    """Return whether a path belongs to boundary-labelled exact-repair scratch space."""\n    return any(TRANSIENT_DIRECTORY.fullmatch(part) for part in relative.parts)\n\n\n'''
    if "def transient_artifact_path(" not in source:
        if source.count(helper_anchor) != 1:
            raise RuntimeError("Could not find the sharded checksum helper anchor.")
        source = source.replace(helper_anchor, helper + helper_anchor, 1)

    source = replace_once(
        source,
        "    entries: dict[str, str] = {}\n"
        "    for line_number, raw in enumerate(manifest.read_text(encoding=\"utf-8\").splitlines(), 1):\n",
        "    entries: dict[str, str] = {}\n"
        "    seen: set[str] = set()\n"
        "    for line_number, raw in enumerate(manifest.read_text(encoding=\"utf-8\").splitlines(), 1):\n",
        "duplicate tracking",
    )
    source = replace_once(
        source,
        "        if relative in entries:\n"
        "            fail(f\"Duplicate SHA256SUMS path: {relative}\")\n"
        "        path = root / Path(*pure.parts)\n",
        "        if relative in seen:\n"
        "            fail(f\"Duplicate SHA256SUMS path: {relative}\")\n"
        "        seen.add(relative)\n"
        "        # Legacy staged artifacts may mention hidden exact-repair scratch\n"
        "        # files that GitHub intentionally omitted. Ignore only the explicit\n"
        "        # boundary-labelled scratch namespace; all durable files remain exact.\n"
        "        if transient_artifact_path(pure):\n"
        "            continue\n"
        "        path = root / Path(*pure.parts)\n",
        "legacy transient manifest entry",
    )
    source = replace_once(
        source,
        "    actual = {\n"
        "        path.relative_to(root).as_posix()\n"
        "        for path in files\n"
        "        if path != manifest\n"
        "    }\n",
        "    actual = {\n"
        "        relative\n"
        "        for path in files\n"
        "        if path != manifest\n"
        "        for relative in [path.relative_to(root).as_posix()]\n"
        "        if not transient_artifact_path(PurePosixPath(relative))\n"
        "    }\n",
        "durable actual file set",
    )

    path.write_text(source)


if __name__ == "__main__":
    main()
