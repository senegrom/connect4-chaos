#!/usr/bin/env python3
"""Execute the reviewed journal patch with a current-main signature anchor."""

from __future__ import annotations

import subprocess
from pathlib import Path

SOURCE_BLOB = "20efb7addc9df903617c908ffa091bd59b4fc226"
ROOT = Path(__file__).resolve().parents[1]
source = subprocess.check_output(
    ["git", "-C", str(ROOT), "cat-file", "blob", SOURCE_BLOB],
    text=True,
)
old = '''    text = replace_once(
        text,
        """  shardWorkers = 1,
  allowIncomplete = false,
) {
""",
        """  shardWorkers = 1,
  allowIncomplete = false,
  journal = null,
) {
""",
        label="generate role journal argument",
    )
'''
new = '''    text = replace_once(
        text,
        """async function generateRole(
  binary,
  output,
  roleName,
  boundaries,
  maximumPasses,
  seedDirectory = null,
  shardCount = 1,
  shardFromBoundary = 14,
  minimumStatesPerShard = 2_000_000,
  shardWorkers = 1,
  allowIncomplete = false,
) {
""",
        """async function generateRole(
  binary,
  output,
  roleName,
  boundaries,
  maximumPasses,
  seedDirectory = null,
  shardCount = 1,
  shardFromBoundary = 14,
  minimumStatesPerShard = 2_000_000,
  shardWorkers = 1,
  allowIncomplete = false,
  journal = null,
) {
""",
        label="generate role journal argument",
    )
'''
if source.count(old) != 1:
    raise RuntimeError(f"Expected one reviewed generator edit, got {source.count(old)}.")
source = source.replace(old, new, 1)
namespace = {
    "__name__": "perfect_chaos_journal_patch_source",
    "__file__": str(Path(__file__)),
}
exec(compile(source, str(Path(__file__)), "exec"), namespace)
namespace["main"]()
