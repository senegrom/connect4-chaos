import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { replayPerfectClassicPolicy } from '../scripts/perfect-classic-policy.mjs';
import {
  assemble,
  collectFrontier,
  initialFrontier,
} from '../scripts/perfect-classic-shards.mjs';
import { decodePerfectClassicPolicy } from '../src/perfect-classic-policy.js';

function emptyFragment({ rows, columns, connect, role, handoffRemaining, start, frontier = [] }) {
  const bytes = Buffer.alloc(24);
  bytes.write('C4VPOL1\0', 0, 'binary');
  bytes[8] = 1;
  bytes[9] = rows;
  bytes[10] = columns;
  bytes[11] = connect;
  bytes[12] = role;
  bytes[13] = handoffRemaining;
  bytes[14] = 10;
  bytes.writeInt8(0, 15);
  bytes.writeUInt32LE(0, 16);
  bytes.writeUInt32LE(1, 20);
  return { bytes, start, frontier };
}

async function writeFragment(directory, name, options) {
  const target = join(directory, name);
  await mkdir(target, { recursive: true });
  const fragment = emptyFragment(options);
  const policyPath = join(target, 'fragment.bin');
  await writeFile(policyPath, fragment.bytes);
  const manifest = {
    format: 'connect4-perfect-classic-fragment-manifest-v1',
    rows: options.rows,
    columns: options.columns,
    connect: options.connect,
    role: options.role,
    handoffRemaining: options.handoffRemaining,
    start: fragment.start,
    rootValue: 0,
    entryCount: 0,
    closureStates: 1,
    file: './fragment.bin',
    bytes: fragment.bytes.length,
    sha256: createHash('sha256').update(fragment.bytes).digest('hex'),
    summary: { format: 'connect4-perfect-classic-policy-summary-v1' },
    frontier: fragment.frontier,
  };
  const manifestPath = join(target, 'fragment.json');
  await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
  return manifestPath;
}

test('initial shard frontiers use horizontal canonicalisation', async () => {
  const second = await initialFrontier({ rows: 4, columns: 4, connect: 4, role: 2 });
  assert.deepEqual(second.include.map((entry) => entry.mask), ['1', '32']);

  const first = await initialFrontier({
    rows: 4,
    columns: 4,
    connect: 4,
    role: 1,
    root_column: 1,
  });
  assert.equal(first.include.length, 4);
  assert.ok(first.include.every((entry) => entry.moves === 2));
});

test('frontier collection deduplicates fragment boundaries', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'perfect-classic-shard-frontier-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const common = {
    rows: 4,
    columns: 4,
    connect: 4,
    role: 2,
    handoffRemaining: 16,
    start: { current: '0', mask: '1', moves: 1 },
  };
  const first = await writeFragment(directory, 'first', {
    ...common,
    frontier: [{ current: '32', mask: '35', moves: 3 }],
  });
  const second = await writeFragment(directory, 'second', {
    ...common,
    frontier: [
      { current: '32', mask: '35', moves: 3 },
      { current: '32', mask: '97', moves: 3 },
    ],
  });
  const result = await collectFrontier({ inputs: [first, second] });
  assert.equal(result.fragmentCount, 2);
  assert.deepEqual(result.include.map((entry) => entry.mask), ['35', '97']);
});

test('assembly prunes fragment records and emits a replayable full-root policy', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'perfect-classic-shard-assembly-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const first = await writeFragment(directory, 'first', {
    rows: 4,
    columns: 4,
    connect: 4,
    role: 2,
    handoffRemaining: 16,
    start: { current: '0', mask: '1', moves: 1 },
  });
  const second = await writeFragment(directory, 'second', {
    rows: 4,
    columns: 4,
    connect: 4,
    role: 2,
    handoffRemaining: 16,
    start: { current: '0', mask: '32', moves: 1 },
  });
  const output = join(directory, 'assembled');
  const assembled = await assemble({ inputs: [first, second], root_value: 0, output });
  const entry = assembled.manifest.policies[0];
  assert.equal(entry.entryCount, 0);
  assert.equal(entry.closureStates, 3);

  const bytes = await readFile(join(output, '4x4-c4-role2.bin'));
  const policy = decodePerfectClassicPolicy(bytes, entry);
  const replay = replayPerfectClassicPolicy(policy, { exactTableBits: 14 });
  assert.equal(replay.rootValue, 0);
  assert.equal(replay.closureStates, 3);
});
