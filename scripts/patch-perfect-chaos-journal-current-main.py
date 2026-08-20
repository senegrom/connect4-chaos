#!/usr/bin/env python3
"""Execute the reviewed journal patch with current-main structural anchors."""

from __future__ import annotations

import subprocess
from pathlib import Path

SOURCE_BLOB = "20efb7addc9df903617c908ffa091bd59b4fc226"
ROOT = Path(__file__).resolve().parents[1]
source = subprocess.check_output(
    ["git", "-C", str(ROOT), "cat-file", "blob", SOURCE_BLOB],
    text=True,
)


def replace_generator_block(label: str, replacement: str) -> None:
    global source
    marker = f'        label="{label}",\n    )\n'
    end = source.index(marker) + len(marker)
    start = source.rfind("    text = replace_once(\n", 0, end)
    if start < 0:
        raise RuntimeError(f"Could not find reviewed generator block: {label}.")
    source = source[:start] + replacement + source[end:]


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

replace_generator_block(
    "checkpoint journal output",
    '''    checkpoint_start = text.index("      const checkpoint = await checkpointRole({\\n")
    checkpoint_end = text.index("      return;\\n", checkpoint_start)
    checkpoint_block = text[checkpoint_start:checkpoint_end]
    checkpoint_block = replace_once(
        checkpoint_block,
        """        reuseSeedSegments,
      });
""",
        """        reuseSeedSegments,
        journal,
      });
""",
        label="checkpoint journal argument",
    )
    checkpoint_block = replace_once(
        checkpoint_block,
        "      process.stdout.write(`${JSON.stringify({ compiler, output, checkpoint }, null, 2)}\\\\n`);\\n",
        """      process.stdout.write(`${JSON.stringify({
        compiler,
        output,
        checkpoint,
        ...(journal ? { journal: journal.summary() } : {}),
      }, null, 2)}\\n`);
""",
        label="checkpoint journal output",
    )
    text = text[:checkpoint_start] + checkpoint_block + text[checkpoint_end:]
''',
)

namespace = {
    "__name__": "perfect_chaos_journal_patch_source",
    "__file__": str(Path(__file__)),
}
exec(compile(source, str(Path(__file__)), "exec"), namespace)
namespace["main"]()
