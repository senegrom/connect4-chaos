#!/usr/bin/env python3
"""Install the incremental exact segment-repair command in the prefix driver."""

from __future__ import annotations

import sys
from pathlib import Path

REPAIR_FUNCTION = r'''async function repairSegment({
  binary,
  workDirectory,
  inputFrontierPath,
  seedInputFrontierPath,
  seedPolicyPath,
  seedFrontierPath,
  rejectFrontierPath,
  targetBoundary,
  maximumStateCount,
  outputPolicyPath,
  outputFrontierPath,
  rejectedPath,
}) {
  await rm(workDirectory, { recursive: true, force: true });
  await mkdir(workDirectory, { recursive: true });
  await Promise.all([
    mkdir(dirname(outputPolicyPath), { recursive: true }),
    mkdir(dirname(outputFrontierPath), { recursive: true }),
    mkdir(dirname(rejectedPath), { recursive: true }),
    rm(outputPolicyPath, { force: true }),
    rm(outputFrontierPath, { force: true }),
    rm(rejectedPath, { force: true }),
  ]);

  const [input, seedInput, seedPolicy, seedFrontier, rejectedBoundary] = await Promise.all([
    readFrontier(inputFrontierPath),
    readFrontier(seedInputFrontierPath),
    readPolicy(seedPolicyPath),
    readFrontier(seedFrontierPath),
    readFrontier(rejectFrontierPath),
  ]);
  if (input.count < 1) throw new Error('Incremental repair requires at least one input root.');
  if (input.role !== seedInput.role || input.role !== seedPolicy.role
      || input.role !== seedFrontier.role || input.role !== rejectedBoundary.role
      || input.boundary !== seedInput.boundary
      || seedPolicy.boundary !== targetBoundary
      || seedFrontier.boundary !== targetBoundary
      || rejectedBoundary.boundary !== targetBoundary
      || input.boundary >= targetBoundary) {
    throw new Error('Incremental repair table metadata does not align.');
  }

  const seedInputKeys = new Set(seedInput.states.map(stateKey));
  const reusableStates = [];
  const freshStates = [];
  for (const state of input.states) {
    (seedInputKeys.has(stateKey(state)) ? reusableStates : freshStates).push(state);
  }

  const reusableInputPath = join(workDirectory, 'reusable-input.bin');
  const freshInputPath = join(workDirectory, 'fresh-input.bin');
  const unaffectedInputPath = join(workDirectory, 'unaffected-input.bin');
  const affectedExistingInputPath = join(workDirectory, 'affected-existing-input.bin');
  await Promise.all([
    writeFile(reusableInputPath, encodeFrontier(input.role, input.boundary, reusableStates)),
    writeFile(freshInputPath, encodeFrontier(input.role, input.boundary, freshStates)),
  ]);

  let partitionSummary = null;
  if (reusableStates.length > 0) {
    const partition = await nativeSegment(binary, [
      'partition',
      '--input-frontier', reusableInputPath,
      '--policy', seedPolicyPath,
      '--reference-frontier', seedFrontierPath,
      '--reject-frontier', rejectFrontierPath,
      '--unaffected', unaffectedInputPath,
      '--affected', affectedExistingInputPath,
    ]);
    if (partition.code !== 0 || partition.records.length !== 1) {
      throw new Error(
        `Exact dependency partition failed.\n${partition.stderr || partition.stdout}`,
      );
    }
    [partitionSummary] = partition.records;
  } else {
    await Promise.all([
      writeFile(unaffectedInputPath, encodeFrontier(input.role, input.boundary, [])),
      writeFile(affectedExistingInputPath, encodeFrontier(input.role, input.boundary, [])),
    ]);
  }

  const unaffectedInput = await readFrontier(unaffectedInputPath);
  const affectedExistingInput = await readFrontier(affectedExistingInputPath);
  if (unaffectedInput.count + affectedExistingInput.count !== reusableStates.length) {
    throw new Error('Dependency partition does not cover every reusable root.');
  }
  const partitionKeys = new Set([
    ...unaffectedInput.states.map(stateKey),
    ...affectedExistingInput.states.map(stateKey),
  ]);
  if (partitionKeys.size !== reusableStates.length
      || reusableStates.some((state) => !partitionKeys.has(stateKey(state)))) {
    throw new Error('Dependency partition is not a disjoint reusable-root cover.');
  }

  const repairInputPath = join(workDirectory, 'repair-input.bin');
  await mergeFrontiers(repairInputPath, [affectedExistingInputPath, freshInputPath]);
  const repairInput = await readFrontier(repairInputPath);

  const unaffectedPolicyPath = join(workDirectory, 'unaffected.policy.bin');
  const unaffectedFrontierPath = join(workDirectory, 'unaffected.frontier.bin');
  let sliceSummary = null;
  if (unaffectedInput.count > 0) {
    const sliced = await nativeSegment(binary, [
      'slice',
      '--input-frontier', unaffectedInputPath,
      '--policy', seedPolicyPath,
      '--reference-frontier', seedFrontierPath,
      '--output-policy', unaffectedPolicyPath,
      '--output-frontier', unaffectedFrontierPath,
    ]);
    if (sliced.code !== 0 || sliced.records.length !== 1) {
      throw new Error(`Safe policy slicing failed.\n${sliced.stderr || sliced.stdout}`);
    }
    [sliceSummary] = sliced.records;
  } else {
    await Promise.all([
      writeFile(unaffectedPolicyPath, encodePolicy(input.role, targetBoundary, [])),
      writeFile(unaffectedFrontierPath, encodeFrontier(input.role, targetBoundary, [])),
    ]);
  }

  const repairedPolicyPath = join(workDirectory, 'repaired.policy.bin');
  const repairedFrontierPath = join(workDirectory, 'repaired.frontier.bin');
  const repairedRejectedPath = join(workDirectory, 'repaired.rejected.bin');
  let repairSummary = null;
  if (repairInput.count > 0) {
    const repaired = await nativeSegment(binary, [
      'extend',
      '--input-frontier', repairInputPath,
      '--frontier-pieces', String(targetBoundary),
      '--maximum-states', String(maximumStateCount),
      '--policy', repairedPolicyPath,
      '--frontier', repairedFrontierPath,
      '--reject-frontier', rejectFrontierPath,
      '--rejected', repairedRejectedPath,
    ]);
    if (repaired.code !== 0) {
      if (!(await exists(repairedRejectedPath))) {
        throw new Error(
          `Affected-root exact repair failed without a rejection certificate.\n`
          + (repaired.stderr || repaired.stdout),
        );
      }
      const rejected = await readFrontier(repairedRejectedPath);
      if (rejected.role !== input.role || rejected.boundary !== input.boundary
          || rejected.count < 1) {
        throw new Error('Affected-root rejection certificate has incompatible metadata.');
      }
      const repairKeys = new Set(repairInput.states.map(stateKey));
      if (rejected.states.some((state) => !repairKeys.has(stateKey(state)))) {
        throw new Error('Affected-root rejection certificate contains an unrelated input root.');
      }
      await copyFile(repairedRejectedPath, rejectedPath);
      return {
        format: 'connect4-chaos-incremental-segment-repair-v1',
        status: 'rejected',
        role: input.role === ROLE_CODES.red ? 'red' : 'yellow',
        fromPieces: input.boundary,
        targetPieces: targetBoundary,
        inputRoots: input.count,
        reusableRoots: reusableStates.length,
        freshRoots: freshStates.length,
        unaffectedRoots: unaffectedInput.count,
        affectedExistingRoots: affectedExistingInput.count,
        repairRoots: repairInput.count,
        rejectedInputRoots: rejected.count,
        partition: partitionSummary,
        slice: sliceSummary,
      };
    }
    repairSummary = repaired.records.at(-1) ?? null;
    if (!repairSummary) throw new Error('Affected-root exact repair returned no summary.');
  } else {
    await Promise.all([
      writeFile(repairedPolicyPath, encodePolicy(input.role, targetBoundary, [])),
      writeFile(repairedFrontierPath, encodeFrontier(input.role, targetBoundary, [])),
    ]);
  }

  let fallbackFullRegeneration = false;
  let fallbackReason = null;
  try {
    await mergePolicies(outputPolicyPath, [unaffectedPolicyPath, repairedPolicyPath]);
    await mergeFrontiers(outputFrontierPath, [unaffectedFrontierPath, repairedFrontierPath]);
  } catch (error) {
    if (!/Conflicting Perfect Chaos policy actions/.test(String(error))) throw error;
    fallbackFullRegeneration = true;
    fallbackReason = String(error);
    await Promise.all([
      rm(outputPolicyPath, { force: true }),
      rm(outputFrontierPath, { force: true }),
      rm(rejectedPath, { force: true }),
    ]);
    const regenerated = await nativeSegment(binary, [
      'extend',
      '--input-frontier', inputFrontierPath,
      '--frontier-pieces', String(targetBoundary),
      '--maximum-states', String(maximumStateCount),
      '--policy', outputPolicyPath,
      '--frontier', outputFrontierPath,
      '--reject-frontier', rejectFrontierPath,
      '--rejected', rejectedPath,
    ]);
    if (regenerated.code !== 0) {
      if (!(await exists(rejectedPath))) {
        throw new Error(
          `Full exact fallback failed without a rejection certificate.\n`
          + (regenerated.stderr || regenerated.stdout),
        );
      }
      const rejected = await readFrontier(rejectedPath);
      return {
        format: 'connect4-chaos-incremental-segment-repair-v1',
        status: 'rejected',
        role: input.role === ROLE_CODES.red ? 'red' : 'yellow',
        fromPieces: input.boundary,
        targetPieces: targetBoundary,
        inputRoots: input.count,
        reusableRoots: reusableStates.length,
        freshRoots: freshStates.length,
        unaffectedRoots: unaffectedInput.count,
        affectedExistingRoots: affectedExistingInput.count,
        repairRoots: input.count,
        rejectedInputRoots: rejected.count,
        fallbackFullRegeneration,
        fallbackReason,
        partition: partitionSummary,
        slice: sliceSummary,
      };
    }
    repairSummary = regenerated.records.at(-1) ?? null;
  }

  const [outputPolicy, outputFrontier] = await Promise.all([
    readPolicy(outputPolicyPath),
    readFrontier(outputFrontierPath),
  ]);
  if (outputPolicy.role !== input.role || outputFrontier.role !== input.role
      || outputPolicy.boundary !== targetBoundary
      || outputFrontier.boundary !== targetBoundary) {
    throw new Error('Incremental repair outputs have incompatible metadata.');
  }
  const rejectedKeys = new Set(rejectedBoundary.states.map(stateKey));
  if (outputFrontier.states.some((state) => rejectedKeys.has(stateKey(state)))) {
    throw new Error('Incremental repair output still reaches a rejected boundary state.');
  }

  const replay = await replaySegment({
    role: input.role,
    inputStates: input.states,
    policyPath: outputPolicyPath,
    frontierPath: outputFrontierPath,
  });
  const verifiedPolicyPath = join(workDirectory, 'verified.policy.bin');
  const verifiedFrontierPath = join(workDirectory, 'verified.frontier.bin');
  const verified = await nativeSegment(binary, [
    'slice',
    '--input-frontier', inputFrontierPath,
    '--policy', outputPolicyPath,
    '--reference-frontier', outputFrontierPath,
    '--output-policy', verifiedPolicyPath,
    '--output-frontier', verifiedFrontierPath,
  ]);
  if (verified.code !== 0 || verified.records.length !== 1) {
    throw new Error(`Merged policy verification failed.\n${verified.stderr || verified.stdout}`);
  }
  const [policyBytes, verifiedPolicyBytes, frontierBytes, verifiedFrontierBytes] = await Promise.all([
    readFile(outputPolicyPath),
    readFile(verifiedPolicyPath),
    readFile(outputFrontierPath),
    readFile(verifiedFrontierPath),
  ]);
  if (!policyBytes.equals(verifiedPolicyBytes) || !frontierBytes.equals(verifiedFrontierBytes)) {
    throw new Error('Merged policy contains unreachable or unreplayed records.');
  }

  return {
    format: 'connect4-chaos-incremental-segment-repair-v1',
    status: 'safe',
    role: input.role === ROLE_CODES.red ? 'red' : 'yellow',
    fromPieces: input.boundary,
    targetPieces: targetBoundary,
    inputRoots: input.count,
    reusableRoots: reusableStates.length,
    freshRoots: freshStates.length,
    unaffectedRoots: unaffectedInput.count,
    affectedExistingRoots: affectedExistingInput.count,
    repairRoots: repairInput.count,
    policyEntries: outputPolicy.count,
    frontierStates: outputFrontier.count,
    fallbackFullRegeneration,
    fallbackReason,
    partition: partitionSummary,
    slice: sliceSummary,
    repair: repairSummary,
    replay,
    nativeVerification: verified.records[0],
  };
}
'''

CLI_BLOCK = r'''    if (options.command === 'repair-segment') {
      const requiredPath = (value, label) => {
        if (typeof value !== 'string' || !value) {
          throw new RangeError(`${label} is required.`);
        }
        return resolve(value);
      };
      const targetBoundary = integerOption(
        options.frontier_pieces,
        undefined,
        'frontier-pieces',
        1,
        42,
      );
      const maximumStateCount = integerOption(
        options.maximum_states,
        10_000_000,
        'maximum-states',
        1,
        100_000_000,
      );
      const result = await repairSegment({
        binary,
        workDirectory: join(temporary, 'incremental-segment-repair'),
        inputFrontierPath: requiredPath(options.input_frontier, 'input-frontier'),
        seedInputFrontierPath: requiredPath(
          options.seed_input_frontier,
          'seed-input-frontier',
        ),
        seedPolicyPath: requiredPath(options.seed_policy, 'seed-policy'),
        seedFrontierPath: requiredPath(options.seed_frontier, 'seed-frontier'),
        rejectFrontierPath: requiredPath(options.reject_frontier, 'reject-frontier'),
        targetBoundary,
        maximumStateCount,
        outputPolicyPath: requiredPath(options.output_policy, 'output-policy'),
        outputFrontierPath: requiredPath(options.output_frontier, 'output-frontier'),
        rejectedPath: requiredPath(options.rejected, 'rejected'),
      });
      process.stdout.write(`${JSON.stringify({ compiler, result }, null, 2)}\n`);
      return;
    }
'''


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one match, found {count}")
    return source.replace(old, new)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch-perfect-chaos-incremental-repair.py SOURCE")
    path = Path(sys.argv[1])
    source = path.read_text()
    if "async function repairSegment({" not in source:
        source = replace_once(
            source,
            "async function cleanIncompleteRoleDirectory(roleDirectory) {",
            REPAIR_FUNCTION + "\nasync function cleanIncompleteRoleDirectory(roleDirectory) {",
            "repair function insertion",
        )
    if "options.command === 'repair-segment'" not in source:
        source = replace_once(
            source,
            "    if (options.command === 'advance-role' || options.command === 'prepare-role') {",
            CLI_BLOCK + "    if (options.command === 'advance-role' || options.command === 'prepare-role') {",
            "repair command insertion",
        )
    path.write_text(source)


if __name__ == "__main__":
    main()
