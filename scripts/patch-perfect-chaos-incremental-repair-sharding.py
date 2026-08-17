#!/usr/bin/env python3
"""Enable adaptive deterministic sharding inside exact incremental repair."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return source.replace(old, new)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch-perfect-chaos-incremental-repair-sharding.py SOURCE")
    path = Path(sys.argv[1])
    source = path.read_text()
    if "const repairUsesShards = shardCount > 1;" in source:
        print("Incremental repair sharding is already installed.")
        return

    source = replace_once(
        source,
        '''  targetBoundary,
  maximumStateCount,
  outputPolicyPath,''',
        '''  targetBoundary,
  maximumStateCount,
  shardCount = 1,
  minimumStatesPerShard = 2_000_000,
  shardWorkers = 1,
  outputPolicyPath,''',
        "repair parameters",
    )

    source = replace_once(
        source,
        '''    const repaired = await nativeSegment(binary, [
      'extend',
      '--input-frontier', repairInputPath,
      '--frontier-pieces', String(targetBoundary),
      '--maximum-states', String(maximumStateCount),
      '--policy', repairedPolicyPath,
      '--frontier', repairedFrontierPath,
      '--reject-frontier', rejectFrontierPath,
      '--rejected', repairedRejectedPath,
    ]);''',
        '''    const repairUsesShards = shardCount > 1;
    const repaired = repairUsesShards
      ? await shardedNativeExtension({
        binary,
        inputFrontier: repairInputPath,
        targetBoundary,
        maximumStateCount,
        minimumStatesPerShard,
        policyPath: repairedPolicyPath,
        frontierPath: repairedFrontierPath,
        targetReject: rejectFrontierPath,
        rejectedPath: repairedRejectedPath,
        shardCount,
        shardWorkers,
      })
      : await nativeSegment(binary, [
        'extend',
        '--input-frontier', repairInputPath,
        '--frontier-pieces', String(targetBoundary),
        '--maximum-states', String(maximumStateCount),
        '--policy', repairedPolicyPath,
        '--frontier', repairedFrontierPath,
        '--reject-frontier', rejectFrontierPath,
        '--rejected', repairedRejectedPath,
      ]);''',
        "affected repair sharding",
    )

    source = replace_once(
        source,
        '''    const regenerated = await nativeSegment(binary, [
      'extend',
      '--input-frontier', inputFrontierPath,
      '--frontier-pieces', String(targetBoundary),
      '--maximum-states', String(maximumStateCount),
      '--policy', outputPolicyPath,
      '--frontier', outputFrontierPath,
      '--reject-frontier', rejectFrontierPath,
      '--rejected', rejectedPath,
    ]);''',
        '''    const regenerated = shardCount > 1
      ? await shardedNativeExtension({
        binary,
        inputFrontier: inputFrontierPath,
        targetBoundary,
        maximumStateCount,
        minimumStatesPerShard,
        policyPath: outputPolicyPath,
        frontierPath: outputFrontierPath,
        targetReject: rejectFrontierPath,
        rejectedPath,
        shardCount,
        shardWorkers,
      })
      : await nativeSegment(binary, [
        'extend',
        '--input-frontier', inputFrontierPath,
        '--frontier-pieces', String(targetBoundary),
        '--maximum-states', String(maximumStateCount),
        '--policy', outputPolicyPath,
        '--frontier', outputFrontierPath,
        '--reject-frontier', rejectFrontierPath,
        '--rejected', rejectedPath,
      ]);''',
        "fallback sharding",
    )

    source = replace_once(
        source,
        '''      const result = await repairSegment({
        binary,
        workDirectory: join(temporary, 'incremental-segment-repair'),''',
        '''      const shards = integerOption(options.shards, 1, 'shards', 1, 256);
      const minimumStatesPerShard = integerOption(
        options.minimum_states_per_shard,
        2_000_000,
        'minimum-states-per-shard',
        10_000,
        100_000_000,
      );
      const shardWorkers = integerOption(
        options.shard_workers,
        1,
        'shard-workers',
        1,
        32,
      );
      const result = await repairSegment({
        binary,
        workDirectory: join(temporary, 'incremental-segment-repair'),''',
        "repair CLI sharding options",
    )

    source = replace_once(
        source,
        '''        targetBoundary,
        maximumStateCount,
        outputPolicyPath:''',
        '''        targetBoundary,
        maximumStateCount,
        shardCount: shards,
        minimumStatesPerShard,
        shardWorkers,
        outputPolicyPath:''',
        "repair CLI sharding arguments",
    )

    # The integrated verifier already requests two shards. Require the repair
    # itself—not merely the surrounding preparation—to prove that it used the
    # adaptive sharded path.
    source = replace_once(
        source,
        '''  if (repairSummaries.some((summary) => summary.status !== 'safe'
      || summary.fallbackFullRegeneration)) {
    throw new Error('Incremental preparation required an unexpected full fallback.');
  }
  if (!repairSummaries.some((summary) => summary.repairRoots < summary.inputRoots)) {''',
        '''  if (repairSummaries.some((summary) => summary.status !== 'safe'
      || summary.fallbackFullRegeneration)) {
    throw new Error('Incremental preparation required an unexpected full fallback.');
  }
  if (repairSummaries.some((summary) => summary.repairRoots > 0
      && (summary.repair?.format !== 'connect4-chaos-prefix-sharded-certificate-v1'
        || summary.repair?.shardWorkers !== 2))) {
    throw new Error('Incremental preparation did not shard its exact repair roots.');
  }
  if (!repairSummaries.some((summary) => summary.repairRoots < summary.inputRoots)) {''',
        "integrated sharding verification",
    )

    path.write_text(source)


if __name__ == "__main__":
    main()
