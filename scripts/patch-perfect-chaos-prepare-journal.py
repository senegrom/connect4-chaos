#!/usr/bin/env python3
"""Extend the content-addressed Perfect Chaos journal into prepare-role."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path("scripts/perfect-chaos-prefix.mjs")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement target, got {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = SCRIPT.read_text()

    text = replace_once(
        text,
        """  allowIncomplete = false,
  reuseSeedSegments = false,
) {
""",
        """  allowIncomplete = false,
  reuseSeedSegments = false,
  journal = null,
) {
""",
        "prepare-role journal argument",
    )

    text = replace_once(
        text,
        """            // segment has been assembled, so replaying this segment here is
            // duplicate work during preparation.
            deferReplay: true,
          });
""",
        """            // segment has been assembled, so replaying this segment here is
            // duplicate work during preparation.
            journal,
            deferReplay: true,
          });
""",
        "prepare repair journal propagation",
    )

    text = replace_once(
        text,
        """      if (!result) {
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
      }
""",
        """      if (!result) {
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
            journal,
          })
          : await journaledSegment(
            journal,
            {
              kind: from === 0 ? 'generate' : 'extend',
              role: roleName,
              fromPieces: from,
              frontierPieces: boundary,
              maximumStates: maximumStates(boundary),
              inputSha256: from === 0 ? null : await sha256OfFile(inputFrontier),
              rejectSha256: await sha256OfFile(targetReject),
            },
            () => nativeSegment(binary, args),
            {
              policyPath,
              frontierPath,
              rejectedPath: from === 0 ? null : newRejectPath,
            },
          );
      }
""",
        "prepare native segment journal",
    )

    text = replace_once(
        text,
        """  outputPolicyPath,
  outputFrontierPath,
  rejectedPath,
  deferReplay = false,
}) {
""",
        """  outputPolicyPath,
  outputFrontierPath,
  rejectedPath,
  journal = null,
  deferReplay = false,
}) {
""",
        "repair journal argument",
    )

    text = replace_once(
        text,
        """    const repaired = repairUsesShards
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
      ]);
""",
        """    const repaired = repairUsesShards
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
        journal,
      })
      : await journaledSegment(
        journal,
        {
          kind: 'incremental-repair-extend',
          role: input.role === ROLE_CODES.red ? 'red' : 'yellow',
          fromPieces: input.boundary,
          frontierPieces: targetBoundary,
          maximumStates: maximumStateCount,
          inputSha256: await sha256OfFile(repairInputPath),
          rejectSha256: await sha256OfFile(rejectFrontierPath),
        },
        () => nativeSegment(binary, [
          'extend',
          '--input-frontier', repairInputPath,
          '--frontier-pieces', String(targetBoundary),
          '--maximum-states', String(maximumStateCount),
          '--policy', repairedPolicyPath,
          '--frontier', repairedFrontierPath,
          '--reject-frontier', rejectFrontierPath,
          '--rejected', repairedRejectedPath,
        ]),
        {
          policyPath: repairedPolicyPath,
          frontierPath: repairedFrontierPath,
          rejectedPath: repairedRejectedPath,
        },
      );
""",
        "affected repair journal",
    )

    text = replace_once(
        text,
        """    const regenerated = shardCount > 1
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
      ]);
""",
        """    const regenerated = shardCount > 1
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
        journal,
      })
      : await journaledSegment(
        journal,
        {
          kind: 'incremental-repair-fallback-extend',
          role: input.role === ROLE_CODES.red ? 'red' : 'yellow',
          fromPieces: input.boundary,
          frontierPieces: targetBoundary,
          maximumStates: maximumStateCount,
          inputSha256: await sha256OfFile(inputFrontierPath),
          rejectSha256: await sha256OfFile(rejectFrontierPath),
        },
        () => nativeSegment(binary, [
          'extend',
          '--input-frontier', inputFrontierPath,
          '--frontier-pieces', String(targetBoundary),
          '--maximum-states', String(maximumStateCount),
          '--policy', outputPolicyPath,
          '--frontier', outputFrontierPath,
          '--reject-frontier', rejectFrontierPath,
          '--rejected', rejectedPath,
        ]),
        {
          policyPath: outputPolicyPath,
          frontierPath: outputFrontierPath,
          rejectedPath,
        },
      );
""",
        "repair fallback journal",
    )

    text = replace_once(
        text,
        """      true,
      reuseSeedSegments,
    )
""",
        """      true,
      reuseSeedSegments,
      journal,
    )
""",
        "checkpoint prepare journal propagation",
    )

    text = replace_once(
        text,
        """      const journal = options.command === 'advance-role'
        ? await createJournal(journalDirectory(options, output), binary)
        : null;
""",
        """      const journal = await createJournal(journalDirectory(options, output), binary);
""",
        "prepare command journal creation",
    )

    journal_test = r'''

  const preparationJournal = await createJournal(
    join(temporary, 'incremental-preparation-journal'),
    binary,
  );
  const journaledFirstOutput = join(temporary, 'journaled-preparation-first');
  const journaledFirst = await prepareRole(
    binary,
    journaledFirstOutput,
    'red',
    [4, 6, 8],
    50,
    seedDirectory,
    2,
    4,
    10_000,
    2,
    false,
    true,
    preparationJournal,
  );
  const freshPreparationJournal = preparationJournal.summary();
  if (freshPreparationJournal.hits !== 0
      || freshPreparationJournal.misses < 1
      || freshPreparationJournal.stores < 1) {
    throw new Error('Fresh incremental preparation did not populate its exact journal.');
  }
  for (const name of compared) {
    const journaledBytes = await readFile(join(journaledFirstOutput, 'red', name));
    const exactBytes = await readFile(join(fullOutput, 'red', name));
    if (!journaledBytes.equals(exactBytes)) {
      throw new Error(`Journaled preparation differs from full regeneration at ${name}.`);
    }
  }

  preparationJournal.resetStatistics();
  const journaledSecondOutput = join(temporary, 'journaled-preparation-second');
  const journaledSecond = await prepareRole(
    binary,
    journaledSecondOutput,
    'red',
    [4, 6, 8],
    50,
    seedDirectory,
    2,
    4,
    10_000,
    2,
    false,
    true,
    preparationJournal,
  );
  const reusedPreparationJournal = preparationJournal.summary();
  if (reusedPreparationJournal.misses !== 0 || reusedPreparationJournal.hits < 1) {
    throw new Error('Incremental preparation did not reuse exact journal entries.');
  }
  if (JSON.stringify(stable(journaledSecond.replay))
      !== JSON.stringify(stable(journaledFirst.replay))) {
    throw new Error('Journal reuse changed the incremental preparation replay.');
  }
  for (const name of compared) {
    const firstBytes = await readFile(join(journaledFirstOutput, 'red', name));
    const secondBytes = await readFile(join(journaledSecondOutput, 'red', name));
    if (!firstBytes.equals(secondBytes)) {
      throw new Error(`Journal reuse changed incremental preparation at ${name}.`);
    }
  }

  const rejectKeyA = journalKey(preparationJournal, {
    kind: 'prepare-rejection-probe',
    rejectSha256: 'a'.repeat(64),
  });
  const rejectKeyB = journalKey(preparationJournal, {
    kind: 'prepare-rejection-probe',
    rejectSha256: 'b'.repeat(64),
  });
  if (rejectKeyA === rejectKeyB) {
    throw new Error('Preparation journal keys are not bound to rejection-table bytes.');
  }

  const corruptedPreparation = await corruptOneJournalOutput(preparationJournal);
  preparationJournal.resetStatistics();
  const recoveredPreparationOutput = join(temporary, 'journaled-preparation-recovered');
  const recoveredPreparation = await prepareRole(
    binary,
    recoveredPreparationOutput,
    'red',
    [4, 6, 8],
    50,
    seedDirectory,
    2,
    4,
    10_000,
    2,
    false,
    true,
    preparationJournal,
  );
  const recoveredPreparationJournal = preparationJournal.summary();
  if (recoveredPreparationJournal.invalidations < 1
      || recoveredPreparationJournal.misses < 1) {
    throw new Error('Corrupt preparation journal output was not invalidated and regenerated.');
  }
  if (JSON.stringify(stable(recoveredPreparation.replay))
      !== JSON.stringify(stable(journaledFirst.replay))) {
    throw new Error('Preparation journal recovery changed the exact replay.');
  }
  for (const name of compared) {
    const freshBytes = await readFile(join(journaledFirstOutput, 'red', name));
    const recoveredBytes = await readFile(join(recoveredPreparationOutput, 'red', name));
    if (!freshBytes.equals(recoveredBytes)) {
      throw new Error(`Preparation journal recovery changed ${name}.`);
    }
  }
'''

    text = replace_once(
        text,
        """  if (!corruptRejected) {
    throw new Error('Deferred seed replay allowed a corrupt policy to reach a checkpoint.');
  }

  return {
    selectedPartition,
    repairSummaries,
    rejectionCounts: incremental.rejected,
    replay: incremental.replay,
  };
""",
        """  if (!corruptRejected) {
    throw new Error('Deferred seed replay allowed a corrupt policy to reach a checkpoint.');
  }
""" + journal_test + """
  return {
    selectedPartition,
    repairSummaries,
    rejectionCounts: incremental.rejected,
    replay: incremental.replay,
    journal: {
      fresh: freshPreparationJournal,
      reused: reusedPreparationJournal,
      corrupted: corruptedPreparation,
      recovered: recoveredPreparationJournal,
    },
  };
""",
        "prepare journal verification",
    )

    SCRIPT.write_text(text, newline="\n")


if __name__ == "__main__":
    main()
