#!/usr/bin/env python3
"""Install a stack-safe Perfect Chaos frontier merge and scale regression."""

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
        raise SystemExit("usage: patch-perfect-chaos-large-frontier-merge.py <prefix-driver>")
    path = Path(sys.argv[1])
    source = path.read_text()

    source = replace_once(
        source,
        "    states.push(...frontier.states);\n",
        "    for (const state of frontier.states) states.push(state);\n",
        "frontier accumulation",
    )

    function_anchor = "\nasync function splitFrontier(path, requestedShards, directory, prefix = '') {\n"
    verification = r'''
async function verifyLargeFrontierMerge(temporary) {
  const directory = join(temporary, 'large-frontier-merge');
  await mkdir(directory, { recursive: true });
  const inputPaths = [
    join(directory, 'first.frontier.bin'),
    join(directory, 'second.frontier.bin'),
  ];
  const boundary = 8;
  const partitionSizes = [180_000, 20_000];
  const totalStates = partitionSizes.reduce((total, count) => total + count, 0);
  const validPositions = [];
  for (let column = 0; column < 7; column += 1) {
    for (let row = 0; row < 6; row += 1) validPositions.push(column * 7 + row);
  }
  const combination = Array.from({ length: boundary }, (_, index) => index);
  const advanceCombination = () => {
    let index = combination.length - 1;
    while (index >= 0
        && combination[index] === validPositions.length - combination.length + index) {
      index -= 1;
    }
    if (index < 0) return false;
    combination[index] += 1;
    for (let next = index + 1; next < combination.length; next += 1) {
      combination[next] = combination[next - 1] + 1;
    }
    return true;
  };

  let produced = 0;
  for (let partition = 0; partition < partitionSizes.length; partition += 1) {
    const states = [];
    for (let index = 0; index < partitionSizes[partition]; index += 1) {
      let mover = 0n;
      for (const position of combination) {
        mover |= 1n << BigInt(validPositions[position]);
      }
      states.push({ mover, opponent: 0n, rows: 6, columns: 7, aiTurn: true });
      produced += 1;
      if (produced < totalStates && !advanceCombination()) {
        throw new Error('Synthetic large-frontier verification exhausted its state space.');
      }
    }
    await writeFile(
      inputPaths[partition],
      encodeFrontier(ROLE_CODES.red, boundary, states),
    );
  }

  const outputPath = join(directory, 'merged.frontier.bin');
  const mergedCount = await mergeFrontiers(outputPath, inputPaths);
  const merged = await readFrontier(outputPath);
  if (mergedCount !== totalStates || merged.count !== totalStates) {
    throw new Error(
      `Large frontier merge retained ${merged.count} of ${totalStates} synthetic states.`,
    );
  }
  return {
    inputFiles: inputPaths.length,
    inputStates: totalStates,
    mergedStates: merged.count,
  };
}
'''
    if "async function verifyLargeFrontierMerge(temporary)" not in source:
        if function_anchor not in source:
            raise RuntimeError("Could not find the splitFrontier insertion anchor.")
        source = source.replace(function_anchor, f"\n{verification}{function_anchor}", 1)

    source = replace_once(
        source,
        "  const sharding = await verifyShardedSmall(binary, temporary);\n",
        "  const largeFrontierMerge = await verifyLargeFrontierMerge(temporary);\n"
        "  const sharding = await verifyShardedSmall(binary, temporary);\n",
        "large-frontier verification call",
    )
    source = replace_once(
        source,
        "    native: native.records,\n    sharding,\n",
        "    native: native.records,\n    largeFrontierMerge,\n    sharding,\n",
        "large-frontier verification result",
    )

    path.write_text(source)


if __name__ == "__main__":
    main()
