#!/usr/bin/env python3
"""Install a stack-safe Perfect Chaos bridge rejection merge and scale regression."""

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
            "usage: patch-perfect-chaos-bridge-large-rejection-merge.py "
            "<bridge-driver> <bridge-test>"
        )

    bridge_path = Path(sys.argv[1])
    test_path = Path(sys.argv[2])

    bridge = bridge_path.read_text()
    bridge = replace_once(
        bridge,
        "    states.push(...frontier.states);\n",
        "    for (const state of frontier.states) states.push(state);\n",
        "bridge rejection accumulation",
    )
    bridge_path.write_text(bridge)

    test_source = test_path.read_text()
    anchor = "test('frontier decoding fails closed on unsorted states, sentinel bits and wrong counts', () => {\n"
    block = r'''function largeGravityFrontierStates(count) {
  const states = [];
  const heights = Array(7).fill(0);
  const emitHeights = (column, remaining) => {
    if (states.length >= count) return;
    if (column === heights.length) {
      if (remaining !== 0) return;
      const positions = [];
      for (let current = 0; current < heights.length; current += 1) {
        for (let row = 0; row < heights[current]; row += 1) {
          positions.push(current * 7 + row);
        }
      }
      for (let ownership = 0; ownership < 2 ** positions.length; ownership += 1) {
        let mover = 0n;
        let opponent = 0n;
        for (let index = 0; index < positions.length; index += 1) {
          const cell = 1n << BigInt(positions[index]);
          if ((ownership & (1 << index)) !== 0) mover |= cell;
          else opponent |= cell;
        }
        states.push({
          mover,
          opponent,
          rows: 6,
          columns: 7,
          aiTurn: (ownership & 1) !== 0,
        });
        if (states.length >= count) return;
      }
      return;
    }
    const maximum = Math.min(6, remaining);
    for (let height = 0; height <= maximum; height += 1) {
      heights[column] = height;
      emitHeights(column + 1, remaining - height);
      if (states.length >= count) return;
    }
  };
  emitHeights(0, 8);
  if (states.length !== count) {
    throw new Error(`Generated ${states.length} of ${count} large-frontier states.`);
  }
  return states;
}

test('rejection merging remains stack-safe for large deterministic shard outputs', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-large-merge-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const firstPath = join(directory, 'first.bin');
  const secondPath = join(directory, 'second.bin');
  const outputPath = join(directory, 'merged.bin');
  const states = largeGravityFrontierStates(200_000);
  await writeFile(firstPath, encodeChaosFrontier(1, 8, states.slice(0, 180_000)));
  await writeFile(secondPath, encodeChaosFrontier(1, 8, states.slice(180_000)));

  const summary = await mergeChaosRejectionFiles([firstPath, secondPath], outputPath);
  const merged = decodeChaosFrontier(await readFile(outputPath));

  assert.equal(summary.artifact.records, 200_000);
  assert.equal(merged.states.length, 200_000);
});

'''
    if "rejection merging remains stack-safe for large deterministic shard outputs" not in test_source:
        if test_source.count(anchor) != 1:
            raise RuntimeError("Could not find the bridge regression insertion anchor.")
        test_source = test_source.replace(anchor, block + anchor, 1)
    test_path.write_text(test_source)


if __name__ == "__main__":
    main()
