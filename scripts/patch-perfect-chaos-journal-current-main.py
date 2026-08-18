#!/usr/bin/env python3
"""Port a fail-closed, content-addressed prefix journal onto current main."""

from __future__ import annotations

from pathlib import Path

SCRIPT = Path("scripts/perfect-chaos-prefix.mjs")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one replacement target, got {count}.")
    return text.replace(old, new, 1)


def main() -> None:
    text = SCRIPT.read_text()

    text = replace_once(
        text,
        "  readFile,\n  readdir,\n  rm,\n",
        "  readFile,\n  readdir,\n  rename,\n  rm,\n",
        label="rename import",
    )
    text = replace_once(
        text,
        "import { basename, dirname, join, resolve } from 'node:path';\n",
        "import { basename, dirname, join, resolve, sep } from 'node:path';\n",
        label="path import",
    )

    text = replace_once(
        text,
        """  return options;
}

function integerOption(value, fallback, label, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
""",
        """  return options;
}

function journalDirectory(options, output) {
  if (options.journal === 'none' || options.journal === false) return null;
  const outputPath = resolve(output);
  const selected = typeof options.journal === 'string'
    ? resolve(options.journal)
    : `${outputPath}.journal`;
  if (selected === outputPath || selected.startsWith(`${outputPath}${sep}`)) {
    throw new RangeError('The prefix journal must be outside the generated output directory.');
  }
  return selected;
}

function integerOption(value, fallback, label, minimum = 0, maximum = Number.MAX_SAFE_INTEGER) {
""",
        label="journal option",
    )

    text = replace_once(
        text,
        """  minimumStatesPerShard = 2_000_000,
  shardWorkers = 1,
}) {
""",
        """  minimumStatesPerShard = 2_000_000,
  shardWorkers = 1,
  journal = null,
}) {
""",
        label="sharded journal argument",
    )
    text = replace_once(
        text,
        """    const maximumStatesPerShard = Math.max(
      minimumStatesPerShard,
      Math.ceil(maximumStateCount / inputPaths.length),
    );
    const workerCount = Math.max(1, Math.min(shardWorkers, inputPaths.length));
""",
        """    const maximumStatesPerShard = Math.max(
      minimumStatesPerShard,
      Math.ceil(maximumStateCount / inputPaths.length),
    );
    const rejectSha256 = targetReject ? await sha256OfFile(targetReject) : null;
    const workerCount = Math.max(1, Math.min(shardWorkers, inputPaths.length));
""",
        label="sharded reject identity",
    )
    text = replace_once(
        text,
        """      const result = await nativeSegment(binary, [
        'extend',
        '--input-frontier', task.inputPath,
        '--frontier-pieces', String(targetBoundary),
        '--maximum-states', String(maximumStatesPerShard),
        '--policy', shardPolicy,
        '--frontier', shardFrontier,
        ...(targetReject ? ['--reject-frontier', targetReject] : []),
        '--rejected', shardRejected,
      ]);
""",
        """      const result = await journaledSegment(
        journal,
        {
          kind: 'extend-shard',
          frontierPieces: targetBoundary,
          maximumStates: maximumStatesPerShard,
          inputSha256: await sha256OfFile(task.inputPath),
          rejectSha256,
        },
        () => nativeSegment(binary, [
          'extend',
          '--input-frontier', task.inputPath,
          '--frontier-pieces', String(targetBoundary),
          '--maximum-states', String(maximumStatesPerShard),
          '--policy', shardPolicy,
          '--frontier', shardFrontier,
          ...(targetReject ? ['--reject-frontier', targetReject] : []),
          '--rejected', shardRejected,
        ]),
        {
          policyPath: shardPolicy,
          frontierPath: shardFrontier,
          rejectedPath: shardRejected,
        },
      );
""",
        label="journaled shard invocation",
    )

    helpers = r'''
function note(message) {
  process.stderr.write(`[perfect-chaos-prefix ${new Date().toISOString()}] ${message}\n`);
}

async function sha256OfFile(path) {
  return createHash('sha256').update(await readFile(path)).digest('hex');
}

async function createJournal(directory, binary) {
  if (!directory) return null;
  await mkdir(directory, { recursive: true });
  const journal = {
    format: 'connect4-chaos-prefix-journal-v2',
    directory,
    sourceSha256: await sha256OfFile(SOURCE),
    binarySha256: await sha256OfFile(binary),
    hits: 0,
    misses: 0,
    stores: 0,
    invalidations: 0,
    resetStatistics() {
      this.hits = 0;
      this.misses = 0;
      this.stores = 0;
      this.invalidations = 0;
    },
    summary() {
      return {
        format: this.format,
        directory: this.directory,
        sourceSha256: this.sourceSha256,
        binarySha256: this.binarySha256,
        hits: this.hits,
        misses: this.misses,
        stores: this.stores,
        invalidations: this.invalidations,
      };
    },
  };
  return journal;
}

function journalKey(journal, descriptor) {
  const canonical = stable({
    format: journal.format,
    sourceSha256: journal.sourceSha256,
    binarySha256: journal.binarySha256,
    descriptor,
  });
  return createHash('sha256').update(JSON.stringify(canonical)).digest('hex');
}

async function invalidateJournalEntry(journal, entryDirectory) {
  await rm(entryDirectory, { recursive: true, force: true });
  journal.invalidations += 1;
}

async function journalLookup(journal, key, destinations) {
  const entryDirectory = join(journal.directory, key);
  let meta;
  try {
    meta = JSON.parse(await readFile(join(entryDirectory, 'meta.json'), 'utf8'));
  } catch (error) {
    if (error?.code !== 'ENOENT') await invalidateJournalEntry(journal, entryDirectory);
    return null;
  }
  if (meta?.format !== journal.format || meta?.key !== key
      || !Number.isInteger(meta.code) || !Array.isArray(meta.files)
      || !Array.isArray(meta.records)) {
    await invalidateJournalEntry(journal, entryDirectory);
    return null;
  }

  const restored = [];
  try {
    for (const file of meta.files) {
      if (!file || typeof file.name !== 'string'
          || !Number.isInteger(file.bytes) || file.bytes < 0
          || !/^[0-9a-f]{64}$/.test(file.sha256)) {
        throw new Error('Journal file metadata is malformed.');
      }
      const destination = destinations[file.name];
      if (!destination) throw new Error(`Unexpected journal output: ${file.name}.`);
      const bytes = await readFile(join(entryDirectory, file.name));
      const digest = createHash('sha256').update(bytes).digest('hex');
      if (bytes.length !== file.bytes || digest !== file.sha256) {
        throw new Error(`Journal output digest differs: ${file.name}.`);
      }
      restored.push({ destination, bytes });
    }
  } catch {
    await invalidateJournalEntry(journal, entryDirectory);
    return null;
  }

  for (const file of restored) await writeFile(file.destination, file.bytes);
  journal.hits += 1;
  return {
    code: meta.code,
    signal: null,
    stdout: '',
    stderr: typeof meta.stderr === 'string' ? meta.stderr : '',
    records: meta.records,
  };
}

async function journalStore(journal, key, result, storedFiles) {
  const entryDirectory = join(journal.directory, key);
  const temporary = `${entryDirectory}.tmp-${process.pid}-${Math.random().toString(16).slice(2)}`;
  await rm(temporary, { recursive: true, force: true });
  await mkdir(temporary, { recursive: true });
  const files = [];
  try {
    for (const [name, source] of Object.entries(storedFiles)) {
      const bytes = await readFile(source);
      await writeFile(join(temporary, name), bytes);
      files.push({
        name,
        bytes: bytes.length,
        sha256: createHash('sha256').update(bytes).digest('hex'),
      });
    }
    files.sort((first, second) => first.name.localeCompare(second.name));
    await writeFile(join(temporary, 'meta.json'), `${JSON.stringify({
      format: journal.format,
      key,
      code: result.code,
      stderr: result.stderr ?? '',
      records: result.records ?? [],
      files,
    }, null, 2)}\n`);
    try {
      await rename(temporary, entryDirectory);
      journal.stores += 1;
    } catch (error) {
      if (error?.code !== 'EEXIST' && error?.code !== 'ENOTEMPTY') throw error;
      await rm(temporary, { recursive: true, force: true });
    }
  } catch (error) {
    await rm(temporary, { recursive: true, force: true });
    throw error;
  }
}

async function journaledSegment(journal, descriptor, invoke, files) {
  if (!journal) return invoke();
  const destinations = {
    'policy.bin': files.policyPath,
    'frontier.bin': files.frontierPath,
    'rejected.bin': files.rejectedPath,
  };
  await Promise.all(Object.values(destinations)
    .filter((path) => typeof path === 'string')
    .map((path) => rm(path, { force: true })));
  const key = journalKey(journal, descriptor);
  const cached = await journalLookup(journal, key, destinations);
  if (cached) return cached;
  journal.misses += 1;
  const result = await invoke();
  if (result.code === 0) {
    await journalStore(journal, key, result, {
      'policy.bin': files.policyPath,
      'frontier.bin': files.frontierPath,
    });
  } else if (files.rejectedPath && await exists(files.rejectedPath)) {
    await journalStore(journal, key, result, { 'rejected.bin': files.rejectedPath });
  }
  return result;
}

async function corruptOneJournalOutput(journal) {
  const entries = (await readdir(journal.directory, { withFileTypes: true }))
    .filter((entry) => entry.isDirectory() && /^[0-9a-f]{64}$/.test(entry.name))
    .sort((first, second) => first.name.localeCompare(second.name));
  for (const entry of entries) {
    const entryDirectory = join(journal.directory, entry.name);
    const meta = JSON.parse(await readFile(join(entryDirectory, 'meta.json'), 'utf8'));
    const first = meta.files?.[0];
    if (!first?.name) continue;
    await writeFile(join(entryDirectory, first.name), Buffer.from('corrupted journal output'));
    return { entry: entry.name, file: first.name };
  }
  throw new Error('Could not find a journal output to corrupt.');
}

'''
    text = replace_once(
        text,
        """async function initializeRejections(output, roleName, boundaries, seedDirectory) {
""",
        helpers + "async function initializeRejections(output, roleName, boundaries, seedDirectory) {\n",
        label="journal helpers",
    )

    text = replace_once(
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

    old_invocation = """      const useShards = from > 0 && shardCount > 1 && boundary >= shardFromBoundary;
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
        : await nativeSegment(binary, args);
      if (result.code === 0) {
        nativeSummaries.push(result.records.at(-1));
"""
    new_invocation = """      const useShards = from > 0 && shardCount > 1 && boundary >= shardFromBoundary;
      note(`${roleName} pass ${pass}: segment ${from}→${boundary}`
        + `${useShards ? ` (${shardCount} requested shards)` : ''}…`);
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
            rejectSha256: targetReject ? await sha256OfFile(targetReject) : null,
          },
          () => nativeSegment(binary, args),
          {
            policyPath,
            frontierPath,
            rejectedPath: from === 0 ? null : newRejectPath,
          },
        );
      if (result.code === 0) {
        const summary = result.records.at(-1);
        note(`${roleName} pass ${pass}: segment ${from}→${boundary} is safe `
          + `(${summary?.frontierStates ?? summary?.frontier_states ?? '?'} frontier states).`);
        nativeSummaries.push(summary);
"""
    text = replace_once(text, old_invocation, new_invocation, label="generate role invocation")
    text = replace_once(
        text,
        """      if (count <= before) throw new Error(`Prefix refinement at ${from} pieces made no progress.`);
      await rm(previousReject, { force: true });
""",
        """      if (count <= before) throw new Error(`Prefix refinement at ${from} pieces made no progress.`);
      note(`${roleName} pass ${pass}: segment ${from}→${boundary} rejected `
        + `${count - before} new losing roots at ${from} pieces (${count} total).`);
      await rm(previousReject, { force: true });
""",
        label="refinement progress",
    )
    text = replace_once(
        text,
        """    if (restart) continue;
    const replay = await replayRole(output, roleName, boundaries);
""",
        """    if (restart) continue;
    note(`${roleName}: replaying the complete certificate…`);
    const replay = await replayRole(output, roleName, boundaries);
""",
        label="replay progress",
    )

    text = replace_once(
        text,
        """  mode = 'synthesis',
  reuseSeedSegments = false,
}) {
""",
        """  mode = 'synthesis',
  reuseSeedSegments = false,
  journal = null,
}) {
""",
        label="checkpoint journal argument",
    )
    text = replace_once(
        text,
        """      shardWorkers,
      true,
    );
""",
        """      shardWorkers,
      true,
      journal,
    );
""",
        label="checkpoint generate journal",
    )

    text = replace_once(
        text,
        """  minimumStatesPerShard = 2_000_000,
  shardWorkers = 1,
) {
""",
        """  minimumStatesPerShard = 2_000_000,
  shardWorkers = 1,
  journal = null,
) {
""",
        label="reference journal argument",
    )
    text = replace_once(
        text,
        """      minimumStatesPerShard,
      shardWorkers,
    );
""",
        """      minimumStatesPerShard,
      shardWorkers,
      false,
      journal,
    );
""",
        label="reference role journal",
    )

    text = replace_once(
        text,
        """  const generated = join(temporary, 'small-reference');
  const manifest = await generateReference(binary, generated, 8, 20);
  if (manifest.roles.red.replay.segments.at(-1).frontierStates !== 1477
      || manifest.roles.yellow.replay.segments.at(-1).frontierStates !== 4515) {
    throw new Error('Independent replay did not reproduce the eight-piece reference frontiers.');
  }
  return {
""",
        """  const generated = join(temporary, 'small-reference');
  const journal = await createJournal(join(temporary, 'small-journal'), binary);
  const manifest = await generateReference(
    binary, generated, 8, 20, null, 1, 14, 2_000_000, 1, journal,
  );
  if (manifest.roles.red.replay.segments.at(-1).frontierStates !== 1477
      || manifest.roles.yellow.replay.segments.at(-1).frontierStates !== 4515) {
    throw new Error('Independent replay did not reproduce the eight-piece reference frontiers.');
  }
  const freshJournal = journal.summary();
  if (freshJournal.hits !== 0 || freshJournal.misses < 2 || freshJournal.stores < 2) {
    throw new Error('A fresh prefix journal did not record deterministic segment misses.');
  }

  journal.resetStatistics();
  const regenerated = join(temporary, 'small-reference-journaled');
  const rerun = await generateReference(
    binary, regenerated, 8, 20, null, 1, 14, 2_000_000, 1, journal,
  );
  const reusedJournal = journal.summary();
  if (reusedJournal.misses !== 0 || reusedJournal.hits < 2) {
    throw new Error('The prefix journal did not reuse completed deterministic segments.');
  }
  if (JSON.stringify(stable(rerun)) !== JSON.stringify(stable(manifest))) {
    throw new Error('Journal-backed regeneration diverged from the fresh reference.');
  }

  const keyA = journalKey(journal, { kind: 'probe', inputSha256: 'a'.repeat(64) });
  const keyB = journalKey(journal, { kind: 'probe', inputSha256: 'b'.repeat(64) });
  const alteredBinary = { ...journal, binarySha256: '0'.repeat(64) };
  if (keyA === keyB || keyA === journalKey(alteredBinary, {
    kind: 'probe', inputSha256: 'a'.repeat(64),
  })) {
    throw new Error('The prefix journal key is not bound to exact inputs and solver bytes.');
  }

  const corrupted = await corruptOneJournalOutput(journal);
  journal.resetStatistics();
  const recoveredOutput = join(temporary, 'small-reference-recovered');
  const recovered = await generateReference(
    binary, recoveredOutput, 8, 20, null, 1, 14, 2_000_000, 1, journal,
  );
  const recoveredJournal = journal.summary();
  if (recoveredJournal.invalidations < 1 || recoveredJournal.misses < 1
      || recoveredJournal.hits < 1) {
    throw new Error('A corrupted journal entry was not safely invalidated and regenerated.');
  }
  if (JSON.stringify(stable(recovered)) !== JSON.stringify(stable(manifest))) {
    throw new Error('Recovery from a corrupted journal entry changed the certificate.');
  }
  return {
""",
        label="journal verification",
    )
    text = replace_once(
        text,
        """    policyConflicts,
    replay: manifest.roles,
  };
""",
        """    policyConflicts,
    replay: manifest.roles,
    journal: {
      fresh: freshJournal,
      reused: reusedJournal,
      corrupted,
      recovered: recoveredJournal,
    },
  };
""",
        label="journal verification result",
    )

    text = replace_once(
        text,
        """      const checkpoint = await checkpointRole({
        binary,
        output,
""",
        """      const journal = options.command === 'advance-role'
        ? await createJournal(journalDirectory(options, output), binary)
        : null;
      const checkpoint = await checkpointRole({
        binary,
        output,
""",
        label="checkpoint journal creation",
    )
    text = replace_once(
        text,
        """        mode: options.command === 'prepare-role' ? 'preparation' : 'synthesis',
        reuseSeedSegments,
      });
      process.stdout.write(`${JSON.stringify({ compiler, output, checkpoint }, null, 2)}\n`);
""",
        """        mode: options.command === 'prepare-role' ? 'preparation' : 'synthesis',
        reuseSeedSegments,
        journal,
      });
      process.stdout.write(`${JSON.stringify({
        compiler,
        output,
        checkpoint,
        ...(journal ? { journal: journal.summary() } : {}),
      }, null, 2)}\n`);
""",
        label="checkpoint journal output",
    )

    text = replace_once(
        text,
        """      const manifest = await generateReference(
        binary,
        output,
        target,
        passes,
        seedDirectory,
        shards,
        shardFromBoundary,
        2_000_000,
        shardWorkers,
      );
      process.stdout.write(`${JSON.stringify({ compiler, output, manifest }, null, 2)}\n`);
""",
        """      const journal = await createJournal(journalDirectory(options, output), binary);
      const manifest = await generateReference(
        binary,
        output,
        target,
        passes,
        seedDirectory,
        shards,
        shardFromBoundary,
        2_000_000,
        shardWorkers,
        journal,
      );
      process.stdout.write(`${JSON.stringify({
        compiler,
        output,
        manifest,
        ...(journal ? { journal: journal.summary() } : {}),
      }, null, 2)}\n`);
""",
        label="generate journal",
    )

    text = replace_once(
        text,
        """      const passes = integerOption(options.maximum_passes, 500, 'maximum-passes', 1, 10_000);
      const seedDirectory = resolve(options.seed_rejections ?? dirname(referencePath));
      const generated = await generateReference(
        binary,
        output,
        reference.boundaries.at(-1),
        passes,
        seedDirectory,
      );
""",
        """      const passes = integerOption(options.maximum_passes, 500, 'maximum-passes', 1, 10_000);
      const seedDirectory = resolve(options.seed_rejections ?? dirname(referencePath));
      const shards = integerOption(
        options.shards,
        reference.sharding?.count ?? 1,
        'shards',
        1,
        256,
      );
      const shardFromBoundary = integerOption(
        options.shard_from_pieces,
        reference.sharding?.fromBoundary ?? 14,
        'shard-from-pieces',
        2,
        42,
      );
      const shardWorkers = integerOption(
        options.shard_workers,
        reference.sharding?.workers ?? 1,
        'shard-workers',
        1,
        32,
      );
      const journal = await createJournal(journalDirectory(options, output), binary);
      const generated = await generateReference(
        binary,
        output,
        reference.boundaries.at(-1),
        passes,
        seedDirectory,
        shards,
        shardFromBoundary,
        2_000_000,
        shardWorkers,
        journal,
      );
""",
        label="reproduction journal and sharding",
    )
    text = replace_once(
        text,
        """      process.stdout.write(`${JSON.stringify({ compiler, reproduced: referencePath, output }, null, 2)}\n`);
""",
        """      process.stdout.write(`${JSON.stringify({
        compiler,
        reproduced: referencePath,
        output,
        ...(journal ? { journal: journal.summary() } : {}),
      }, null, 2)}\n`);
""",
        label="reproduction journal output",
    )

    if "\t" in text:
        raise RuntimeError("Tabs are forbidden in the patched prefix script.")
    SCRIPT.write_text(text)
    print("Patched content-addressed, corruption-detecting Perfect Chaos prefix journal.")


if __name__ == "__main__":
    main()
