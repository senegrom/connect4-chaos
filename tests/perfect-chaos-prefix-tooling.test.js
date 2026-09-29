import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { mkdir, mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { buildNative, findCompiler, runProcess } from '../scripts/native-build.mjs';
import {
  createJournal,
  encodeFrontier,
  encodePolicy,
  generateReference,
  initializeRejections,
  journalKey,
  journaledSegment,
  overlapping,
  parseArguments,
  replaySegment,
  reproducesReference,
} from '../scripts/perfect-chaos-prefix.mjs';

const SOURCE = fileURLToPath(new URL('../native/perfect-chaos-prefix.cpp', import.meta.url));
const REFERENCE = JSON.parse(readFileSync(
  new URL('../data/perfect-chaos-prefix/manifest.json', import.meta.url), 'utf8'));
const BUILD = Object.freeze({
  sourceSha256: 'a'.repeat(64), headersSha256: { 'io.hpp': 'd'.repeat(64) }, compiler: 'c++',
  compilerVersion: 'test 1', flags: ['-O3'],
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

test('an option its command does not read is refused, not ignored', () => {
  // A misspelt --reference used to verify the committed catalog and exit 0.
  assert.throws(() => parseArguments(['verify-reference', '--refrence', 'candidate.json']), /has no option --refrence/);
  assert.throws(() => parseArguments(['verify', '--reference', 'candidate.json']), /has no option --reference/);
  assert.throws(() => parseArguments(['generat']), /Unknown command/);
  // The generation command docs/CHAOS_BOUNDED_PROOF.md gives.
  assert.deepEqual(parseArguments(['generate', '--frontier-pieces', '16', '--seed-rejections', 'seeds',
    '--shards', '8', '--shard-from-pieces', '14', '--output', 'out']), {
    command: 'generate', frontier_pieces: '16', seed_rejections: 'seeds', shards: '8',
    shard_from_pieces: '14', output: 'out',
  });
  assert.deepEqual(parseArguments([]), { command: 'verify' });
});

test('generation refuses an output that would delete its seeds or a certificate', async (context) => {
  const directory = await temporary(context);
  const seeds = join(directory, 'seeds');
  const certificate = join(directory, 'certificate');
  const committed = fileURLToPath(new URL('../data/perfect-chaos-prefix', import.meta.url));
  assert.equal(overlapping(seeds, join(directory, 'seedsmore')), false);
  // The seeds, inside them, around them, the committed certificate, and the
  // certificate that reproduce-reference is checking.
  for (const [output, seed, keep] of [[seeds, seeds, []], [join(seeds, 'red'), seeds, []],
    [directory, seeds, []], [committed, null, []], [join(certificate, 'out'), null, [certificate]]]) {
    await assert.rejects(generateReference(null, output, 8, 1, seed, 1, 14, 2_000_000, 1, null, keep), /overlaps/,
      output);
  }
  assert.ok(existsSync(join(committed, 'manifest.json')));
});

test('seed rejections stop at their deepest file; a missing seed or a gap is an error', async (context) => {
  const directory = await temporary(context);
  // The committed certificate seeds reject-8 to reject-14 of a run to 16.
  const committed = fileURLToPath(new URL('../data/perfect-chaos-prefix', import.meta.url));
  const { rejects } = await initializeRejections(join(directory, 'seeded'), 'red', [8, 10, 12, 14, 16], committed);
  assert.deepEqual([...rejects.keys()], [8, 10, 12, 14, 16]);
  assert.deepEqual(readFileSync(rejects.get(14)), readFileSync(join(committed, 'red', 'reject-14.bin')));
  assert.deepEqual(readFileSync(rejects.get(16)), encodeFrontier(1, 16, []));
  // A mistyped --seed-rejections used to run with no seeds at all.
  await assert.rejects(initializeRejections(join(directory, 'typo'), 'red', [8, 10], join(directory, 'no-seeds')),
    /No seed rejections/);
  const gap = join(directory, 'gap');
  await mkdir(join(gap, 'red'), { recursive: true });
  await writeFile(join(gap, 'red', 'reject-8.bin'), encodeFrontier(1, 8, []));
  await writeFile(join(gap, 'red', 'reject-12.bin'), encodeFrontier(1, 12, []));
  await assert.rejects(initializeRejections(join(directory, 'gapped'), 'red', [8, 10, 12], gap),
    /skip boundary 10/);
});

test('the prefix journal is keyed on the source, headers, compiler and flags, not binary bytes', () => {
  const journal = { format: 'connect4-chaos-prefix-journal-v4', ...BUILD };
  const descriptor = { kind: 'probe', inputSha256: 'b'.repeat(64) };
  const key = journalKey(journal, descriptor);
  assert.equal(journalKey({ ...journal }, descriptor), key, 'rebuilding the same source reuses entries');
  for (const changed of [
    { sourceSha256: 'c'.repeat(64) }, { headersSha256: { 'io.hpp': 'e'.repeat(64) } }, { compiler: 'clang++' },
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

test('a segment replay rejects a certificate that differs from the committed one', async (context) => {
  // The committed red 0-8 segment, then one change at a time: every check
  // below used to be exercised only by correct certificates.
  const directory = await temporary(context);
  const committed = new URL('../data/perfect-chaos-prefix/red/', import.meta.url);
  const policy = readFileSync(new URL('0-8.policy.bin', committed));
  const frontier = readFileSync(new URL('0-8.frontier.bin', committed));
  const policyPath = join(directory, '0-8.policy.bin');
  const frontierPath = join(directory, '0-8.frontier.bin');
  const replay = async (policyBytes, frontierBytes = frontier) => {
    await writeFile(policyPath, policyBytes);
    await writeFile(frontierPath, frontierBytes);
    return replaySegment({ role: 1, inputStates: [EMPTY], policyPath, frontierPath, boundary: 8 });
  };
  assert.equal((await replay(policy)).closureStates, 3161);
  // Records follow a 16-byte header whose bytes 12-15 count them; a policy
  // record's action is its byte 18 (0 = drop) and the column its byte 19.
  const moved = Buffer.from(policy);
  if (moved[16 + 18] === 0) moved[16 + 19] = (moved[16 + 19] + 1) % 7;
  else moved.set([0, 3], 16 + 18);
  await assert.rejects(replay(moved), /Replay is missing policy state/);
  const surplus = Buffer.concat([policy, Buffer.alloc(20)]);
  surplus.writeUInt32LE(policy.readUInt32LE(12) + 1, 12);
  // States sort by rows, columns, then mover: a rotated board with every
  // mover bit set sorts last and is never reached.
  surplus.writeBigUInt64LE(0xffffffffffffffffn, policy.length);
  surplus.set([7, 6, 0, 3], policy.length + 16);
  await assert.rejects(replay(surplus), /0-8\.policy\.bin has 1 unreachable records/);
  const short = Buffer.from(frontier.subarray(0, frontier.length - 19));
  short.writeUInt32LE(frontier.readUInt32LE(12) - 1, 12);
  await assert.rejects(replay(policy, short), /Replay frontier mismatch for 0-8\.frontier\.bin/);
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
