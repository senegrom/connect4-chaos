#!/usr/bin/env python3
"""Exclude exact-repair scratch directories from durable artifact identity."""

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
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: patch-perfect-chaos-transient-artifact-identity.py "
            "<artifact-script> <artifact-test>"
        )

    script_path = Path(sys.argv[1])
    test_path = Path(sys.argv[2])
    source = script_path.read_text()

    source = replace_once(
        source,
        'DIGEST_LINE = re.compile(r"([0-9a-f]{64})  (.+)")\n',
        'DIGEST_LINE = re.compile(r"([0-9a-f]{64})  (.+)")\n'
        'TRANSIENT_DIRECTORY = re.compile(r"\\.incremental-repair-[0-9]+-[0-9]+")\n',
        "transient directory constant",
    )

    function_anchor = '''def ensure_no_symlink(root: Path, relative: PurePosixPath) -> Path:\n'''
    function = '''def transient_artifact_path(relative: PurePosixPath) -> bool:\n    """Return whether a path belongs to exact-repair scratch space.\n\n    Incremental repair work directories contain only intermediate partitions and\n    regenerated fragments. Durable policy, frontier, rejection, checkpoint, and\n    replay files live outside them. GitHub excludes hidden directories from\n    artifact uploads by default, so these paths cannot be certificate identity.\n    Only the explicit boundary-labelled scratch convention is excluded.\n    """\n    return any(TRANSIENT_DIRECTORY.fullmatch(part) for part in relative.parts)\n\n\n'''
    if "def transient_artifact_path(" not in source:
        if source.count(function_anchor) != 1:
            raise RuntimeError("Could not find the transient helper insertion anchor.")
        source = source.replace(function_anchor, function + function_anchor, 1)

    source = replace_once(
        source,
        '''        if path == manifest:\n            continue\n        records.append((posix, path))\n''',
        '''        if path == manifest:\n            continue\n        if transient_artifact_path(PurePosixPath(posix)):\n            continue\n        records.append((posix, path))\n''',
        "artifact enumeration",
    )
    source = replace_once(
        source,
        '''    selected: dict[str, str] = {}\n    for line_number, raw in enumerate(manifest.read_text().splitlines(), start=1):\n''',
        '''    selected: dict[str, str] = {}\n    seen: set[str] = set()\n    for line_number, raw in enumerate(manifest.read_text().splitlines(), start=1):\n''',
        "manifest duplicate tracking",
    )
    source = replace_once(
        source,
        '''        canonical = relative.as_posix()\n        if canonical in selected:\n            raise RuntimeError(f"Duplicate checksum entry: {canonical!r}.")\n        target = ensure_no_symlink(root, relative)\n''',
        '''        canonical = relative.as_posix()\n        if canonical in seen:\n            raise RuntimeError(f"Duplicate checksum entry: {canonical!r}.")\n        seen.add(canonical)\n        # Legacy staged artifacts mention hidden exact-repair scratch files that\n        # GitHub correctly omitted. Ignore only this explicit transient namespace.\n        if transient_artifact_path(relative):\n            continue\n        target = ensure_no_symlink(root, relative)\n''',
        "legacy manifest recovery",
    )
    script_path.write_text(source)

    test_source = test_path.read_text()
    test_anchor = "test('artifact verification rejects unlisted files and path traversal', async () => {\n"
    test_block = r'''test('artifact identity excludes only boundary-labelled incremental repair scratch space', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-artifacts-'));
  try {
    const scratch = join(directory, 'yellow', '.incremental-repair-10-12');
    await mkdir(scratch, { recursive: true });
    await writeFile(join(directory, 'proof.bin'), 'proof');
    await writeFile(join(scratch, 'affected-existing-input.bin'), 'scratch');

    await run(['write', '--directory', directory]);
    assert.equal(
      await readFile(join(directory, 'SHA256SUMS'), 'utf8'),
      `${digest('proof')}  proof.bin\n`,
    );

    await rm(join(directory, 'yellow'), { recursive: true, force: true });
    await run(['verify', '--directory', directory]);

    await writeFile(
      join(directory, 'SHA256SUMS'),
      `${digest('scratch')}  yellow/.incremental-repair-10-12/affected-existing-input.bin\n`
        + `${digest('proof')}  proof.bin\n`,
    );
    await run(['verify', '--directory', directory]);

    const nearMatch = join(directory, 'yellow', '.incremental-repair-ten-twelve');
    await mkdir(nearMatch, { recursive: true });
    await writeFile(join(nearMatch, 'unlisted.bin'), 'unlisted');
    await assert.rejects(
      run(['verify', '--directory', directory]),
      /unlisted file/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

'''
    if "artifact identity excludes only boundary-labelled incremental repair scratch space" not in test_source:
        if test_source.count(test_anchor) != 1:
            raise RuntimeError("Could not find the artifact regression insertion anchor.")
        test_source = test_source.replace(test_anchor, test_block + test_anchor, 1)
    test_path.write_text(test_source)


if __name__ == "__main__":
    main()
