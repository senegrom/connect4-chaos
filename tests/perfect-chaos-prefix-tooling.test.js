import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { buildNative, findCompiler, runProcess } from '../scripts/native-build.mjs';
import {
  createJournal,
  encodeFrontier,
  encodePolicy,
  journalKey,
  journaledSegment,
  replaySegment,
  reproducesReference,
} from '../scripts/perfect-chaos-prefix.mjs';

const SOURCE = fileURLToPath(new URL('../native/perfect-chaos-prefix.cpp', import.meta.url));
const REFERENCE = JSON.parse(readFileSync(
  new URL('../data/perfect-chaos-prefix/manifest.json', import.meta.url), 'utf8'));
const BUILD = Object.freeze({
  sourceSha256: 'a'.repeat(64), compiler: 'c++', compilerVersion: 'test 1', flags: ['-O3'],
});
const EMPTY = { mover: 0n, opponent: 0n, rows: 6, columns: 7, aiTurn: true };

async function temporary(context) {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-prefix-tooling-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  return directory;
}

test('a regeneration reproduces the reference when its certificates and summaries match', () => {
  // What the generator writes: no provenance, and the current source's hash.
  const regenerated = {
    format: REFERENCE.format,
    theorem: REFERENCE.theorem,
    board: REFERENCE.board,
    boundaries: REFERENCE.boundaries,
    sourceSha256: '0'.repeat(64),
    roles: structuredClone(REFERENCE.roles),
    artifacts: structuredClone(REFERENCE.artifacts),
  };
  assert.notDeepEqual(regenerated, REFERENCE, 'the whole manifests can never agree');
  assert.equal(reproducesReference(regenerated, REFERENCE), true);
  regenerated.artifacts.red[0].sha256 = 'f'.repeat(64);
  assert.equal(reproducesReference(regenerated, REFERENCE), false);
  regenerated.artifacts = structuredClone(REFERENCE.artifacts);
  regenerated.roles.yellow.rejected.at14 += 1;
  assert.equal(reproducesReference(regenerated, REFERENCE), false);
});

test('the prefix journal is keyed on the source, compiler and flags, not binary bytes', () => {
  const journal = { format: 'connect4-chaos-prefix-journal-v3', ...BUILD };
  const descriptor = { kind: 'probe', inputSha256: 'b'.repeat(64) };
  const key = journalKey(journal, descriptor);
  assert.equal(journalKey({ ...journal }, descriptor), key, 'rebuilding the same source reuses entries');
  for (const changed of [
    { sourceSha256: 'c'.repeat(64) }, { compiler: 'clang++' },
    { compilerVersion: 'test 2' }, { flags: ['-O2'] },
  ]) {
    assert.notEqual(journalKey({ ...journal, ...changed }, descriptor), key, JSON.stringify(changed));
  }
});

test('the journal records a rejection only for a completed table and a clean exit 1', async (context) => {
  const directory = await temporary(context);
  const journal = await createJournal(join(directory, 'journal'), BUILD);
  const files = {
    policyPath: join(directory, 'segment.policy.bin'),
    frontierPath: join(directory, 'segment.frontier.bin'),
    rejectedPath: join(directory, 'segment.rejected.bin'),
  };
  const descriptor = { kind: 'extend', frontierPieces: 8, inputSha256: 'd'.repeat(64) };
  const table = encodeFrontier(1, 8, [{ ...EMPTY, mover: 1n, opponent: 2n }]);

  // Killed part-way through writing the table: a failure to retry, not a
  // result to replay on every later run.
  let calls = 0;
  const killed = await journaledSegment(journal, descriptor, async () => {
    calls += 1;
    await writeFile(files.rejectedPath, table.subarray(0, table.length - 5));
    return { code: null, signal: 'SIGKILL', stdout: '', stderr: '', records: [] };
  }, files);
  assert.equal(killed.signal, 'SIGKILL');
  assert.equal(journal.stores, 0);

  // Exit 1 with a table that parses is a rejection, and is reused.
  const rejectedRun = async () => {
    calls += 1;
    await writeFile(files.rejectedPath, table);
    return { code: 1, signal: null, stdout: '', stderr: 'losing roots', records: [] };
  };
  assert.equal((await journaledSegment(journal, descriptor, rejectedRun, files)).code, 1);
  assert.equal(journal.stores, 1);
  assert.equal((await journaledSegment(journal, descriptor, rejectedRun, files)).code, 1);
  assert.equal(journal.hits, 1);
  assert.equal(calls, 2, 'the stored rejection must answer without rerunning the segment');

  // A success whose outputs do not parse is refused before it is stored.
  const tornDescriptor = { ...descriptor, inputSha256: 'e'.repeat(64) };
  await assert.rejects(journaledSegment(journal, tornDescriptor, async () => {
    await writeFile(files.policyPath, encodePolicy(1, 8, []).subarray(0, 10));
    await writeFile(files.frontierPath, encodeFrontier(1, 8, []));
    return { code: 0, signal: null, stdout: '', stderr: '', records: [{}] };
  }, files), /invalid magic header|length does not match/);
  assert.equal(journal.stores, 1);
  const entries = await readdir(join(directory, 'journal'));
  assert.equal(entries.length, 1, entries.join(', '));
});

test('a segment replay refuses files whose headers name another boundary', async (context) => {
  const directory = await temporary(context);
  const policyPath = join(directory, '8-10.policy.bin');
  const frontierPath = join(directory, '8-10.frontier.bin');
  await writeFile(policyPath, encodePolicy(1, 12, []));
  await writeFile(frontierPath, encodeFrontier(1, 12, []));
  await assert.rejects(
    replaySegment({ role: 1, inputStates: [EMPTY], policyPath, frontierPath, boundary: 10 }),
    /8-10\.frontier\.bin ends at 12 pieces, but the segment it belongs to ends at 10/,
  );
});

test('the native prefix solver validates its arguments and keeps scratch files in a given directory', async (context) => {
  if (!findCompiler()) {
    context.skip('no C++ compiler available');
    return;
  }
  const { binary } = await buildNative(SOURCE, { name: 'perfect-chaos-prefix' });
  const directory = await temporary(context);

  // The self-test used to write fixed /tmp paths, which do not exist on Windows.
  const verified = await runProcess(binary, ['verify', '--directory', directory]);
  assert.equal(verified.code, 0, verified.stderr);
  assert.equal(verified.stdout.trim().split(/\r?\n/).length, 4);
  assert.deepEqual(await readdir(directory), [], 'the self-test removes its scratch files');
  const undirected = await runProcess(binary, ['verify']);
  assert.equal(undirected.code, 1);
  assert.match(undirected.stderr, /verify requires --directory/);

  // A piece count is stored in a byte: 264 used to become a silent 8.
  for (const pieces of ['264', '0', '43']) {
    const result = await runProcess(binary, ['generate', '--role', 'red', '--frontier-pieces', pieces,
      '--policy', join(directory, 'p.bin'), '--frontier', join(directory, 'f.bin')]);
    assert.equal(result.code, 1, pieces);
    assert.match(result.stderr, /frontier-pieces must be from 1 through 42/, pieces);
  }

  // A full disk surfaces when the last block is flushed on close; the table
  // must not be reported as written. /dev/full exists on Linux only.
  if (existsSync('/dev/full')) {
    const full = await runProcess(binary, ['generate', '--role', 'red', '--frontier-pieces', '2',
      '--policy', '/dev/full', '--frontier', join(directory, 'f.bin')]);
    assert.equal(full.code, 1, full.stdout);
    assert.match(full.stderr, /Policy write failed/);
  }
});
