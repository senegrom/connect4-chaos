#!/usr/bin/env python3
"""Wire exact incremental segment repair into prepared-prefix synthesis."""

from __future__ import annotations

import sys
from pathlib import Path

REPAIR_SELECTION = r'''      let result = null;
      if (reuseSeedSegments && from > 0 && seedDirectory) {
        const seedRoleDirectory = join(seedDirectory, roleName);
        const boundaryIndex = preparedBoundaries.indexOf(from);
        const seedInputFrom = boundaryIndex <= 0 ? 0 : preparedBoundaries[boundaryIndex - 1];
        const seedInputFrontier = join(
          seedRoleDirectory,
          `${seedInputFrom}-${from}.frontier.bin`,
        );
        const seedPolicy = join(seedRoleDirectory, `${from}-${boundary}.policy.bin`);
        const seedFrontier = join(seedRoleDirectory, `${from}-${boundary}.frontier.bin`);
        const availability = await Promise.all([
          exists(seedInputFrontier),
          exists(seedPolicy),
          exists(seedFrontier),
        ]);
        if (availability.some(Boolean) && !availability.every(Boolean)) {
          throw new Error(`Seed prefix segment ${from}-${boundary} is incomplete for repair.`);
        }
        if (availability.every(Boolean)) {
          const repaired = await repairSegment({
            binary,
            workDirectory: join(roleDirectory, `.incremental-repair-${from}-${boundary}`),
            inputFrontierPath: inputFrontier,
            seedInputFrontierPath: seedInputFrontier,
            seedPolicyPath: seedPolicy,
            seedFrontierPath: seedFrontier,
            rejectFrontierPath: targetReject,
            targetBoundary: boundary,
            maximumStateCount: maximumStates(boundary),
            shardCount,
            minimumStatesPerShard,
            shardWorkers,
            outputPolicyPath: policyPath,
            outputFrontierPath: frontierPath,
            rejectedPath: newRejectPath,
          });
          result = repaired.status === 'safe'
            ? {
              code: 0,
              signal: null,
              stdout: '',
              stderr: '',
              records: [repaired],
            }
            : {
              code: 1,
              signal: null,
              stdout: '',
              stderr: `${repaired.rejectedInputRoots} incrementally repaired root(s) are losing.`,
              records: [repaired],
            };
        }
      }
      const useShards = from > 0 && shardCount > 1 && boundary >= shardFromBoundary;
      if (!result) {
        result = useShards
          ? await shardedNativeExtension({
            binary,
            inputFrontier,
            targetBoundary: boundary,
            maximumStateCount: maximumStates(boundary),
            policyPath,
            frontierPath,
            targetReject,
            rejectedPath: newRejectPath,
            shardCount,
            minimumStatesPerShard,
            shardWorkers,
          })
          : await nativeSegment(binary, args);
      }'''

VERIFY_FUNCTION = r'''async function verifyIncrementalPreparedRepair(binary, temporary) {
  const source = join(temporary, 'sharded-red');
  const seedDirectory = join(temporary, 'incremental-preparation-seed');
  const seedRoleDirectory = join(seedDirectory, 'red');
  await mkdir(seedRoleDirectory, { recursive: true });
  for (const name of [
    '0-4.policy.bin',
    '0-4.frontier.bin',
    '4-6.policy.bin',
    '4-6.frontier.bin',
  ]) {
    await copyFile(join(source, name), join(seedRoleDirectory, name));
  }
  await writeFile(
    join(seedRoleDirectory, 'reject-4.bin'),
    encodeFrontier(ROLE_CODES.red, 4, []),
  );

  const seedInput = join(seedRoleDirectory, '0-4.frontier.bin');
  const seedPolicy = join(seedRoleDirectory, '4-6.policy.bin');
  const seedFrontier = join(seedRoleDirectory, '4-6.frontier.bin');
  const targetStates = (await readFrontier(seedFrontier)).states;
  const candidateReject = join(seedRoleDirectory, 'reject-6.bin');
  const candidateUnaffected = join(temporary, 'candidate-unaffected.bin');
  const candidateAffected = join(temporary, 'candidate-affected.bin');
  let selectedPartition = null;
  for (const state of targetStates) {
    await writeFile(candidateReject, encodeFrontier(ROLE_CODES.red, 6, [state]));
    const partition = await nativeSegment(binary, [
      'partition',
      '--input-frontier', seedInput,
      '--policy', seedPolicy,
      '--reference-frontier', seedFrontier,
      '--reject-frontier', candidateReject,
      '--unaffected', candidateUnaffected,
      '--affected', candidateAffected,
    ]);
    const summary = partition.records.at(-1);
    if (partition.code === 0 && summary?.unaffectedRoots > 0 && summary?.affectedRoots > 0) {
      selectedPartition = summary;
      break;
    }
  }
  if (!selectedPartition) {
    throw new Error('Could not find a partially dependent small reference frontier state.');
  }

  const incrementalOutput = join(temporary, 'incremental-preparation-output');
  const fullOutput = join(temporary, 'full-preparation-output');
  const common = [
    binary,
    null,
    'red',
    [4, 6, 8],
    50,
    seedDirectory,
    2,
    4,
    10_000,
    2,
    false,
  ];
  common[1] = incrementalOutput;
  const incremental = await prepareRole(...common, true);
  common[1] = fullOutput;
  const full = await prepareRole(...common, false);

  const compared = [
    'reject-4.bin',
    'reject-6.bin',
    '0-4.policy.bin',
    '0-4.frontier.bin',
    '4-6.policy.bin',
    '4-6.frontier.bin',
  ];
  for (const name of compared) {
    const incrementalBytes = await readFile(join(incrementalOutput, 'red', name));
    const fullBytes = await readFile(join(fullOutput, 'red', name));
    if (!incrementalBytes.equals(fullBytes)) {
      throw new Error(`Incremental preparation differs from full regeneration at ${name}.`);
    }
  }
  if (JSON.stringify(stable(incremental.replay)) !== JSON.stringify(stable(full.replay))) {
    throw new Error('Incremental preparation replay differs from full regeneration.');
  }
  const repairSummaries = incremental.nativeSummaries.filter(
    (summary) => summary?.format === 'connect4-chaos-incremental-segment-repair-v1',
  );
  if (repairSummaries.length < 1) {
    throw new Error('Incremental preparation never exercised exact segment repair.');
  }
  if (repairSummaries.some((summary) => summary.status !== 'safe'
      || summary.fallbackFullRegeneration)) {
    throw new Error('Incremental preparation required an unexpected full fallback.');
  }
  if (!repairSummaries.some((summary) => summary.repairRoots < summary.inputRoots)) {
    throw new Error('Incremental preparation did not reduce the exact repair root set.');
  }
  return {
    selectedPartition,
    repairSummaries,
    rejectionCounts: incremental.rejected,
    replay: incremental.replay,
  };
}
'''

OLD_SELECTION = r'''      const useShards = from > 0 && shardCount > 1 && boundary >= shardFromBoundary;
      const result = useShards
        ? await shardedNativeExtension({
          binary,
          inputFrontier,
          targetBoundary: boundary,
          maximumStateCount: maximumStates(boundary),
          policyPath,
          frontierPath,
          targetReject,
          rejectedPath: newRejectPath,
          shardCount,
          minimumStatesPerShard,
          shardWorkers,
        })
        : await nativeSegment(binary, args);'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return source.replace(old, new)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch-perfect-chaos-incremental-preparation.py SOURCE")
    path = Path(sys.argv[1])
    source = path.read_text()

    prepare_start = source.index("async function prepareRole(")
    prepare_end = source.index("\nasync function cleanIncompleteRoleDirectory", prepare_start)
    prepare = source[prepare_start:prepare_end]
    if "incrementally repaired root(s) are losing" not in prepare:
        prepare = replace_once(
            prepare,
            OLD_SELECTION,
            REPAIR_SELECTION,
            "prepared segment selection",
        )
        source = source[:prepare_start] + prepare + source[prepare_end:]

    if "async function verifyIncrementalPreparedRepair(" not in source:
        source = replace_once(
            source,
            "async function verifySmall(binary, temporary) {",
            VERIFY_FUNCTION + "\nasync function verifySmall(binary, temporary) {",
            "incremental preparation verifier insertion",
        )
    if "const incrementalPreparation = await verifyIncrementalPreparedRepair" not in source:
        source = replace_once(
            source,
            "  const prefixReuse = await verifyPreparedPrefixReuse(temporary);\n  const policyConflicts = await verifyPolicyConflicts(temporary);",
            "  const prefixReuse = await verifyPreparedPrefixReuse(temporary);\n  const incrementalPreparation = await verifyIncrementalPreparedRepair(binary, temporary);\n  const policyConflicts = await verifyPolicyConflicts(temporary);",
            "incremental preparation verifier call",
        )
        source = replace_once(
            source,
            "    prefixReuse,\n    policyConflicts,",
            "    prefixReuse,\n    incrementalPreparation,\n    policyConflicts,",
            "incremental preparation verifier result",
        )

    path.write_text(source)


if __name__ == "__main__":
    main()
