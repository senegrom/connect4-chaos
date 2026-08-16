import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';

import { verifyPerfectClassicCatalogParallel } from '../scripts/verify-perfect-classic-parallel.mjs';

function emptyPolicy({ rows, columns, connect, role, handoffRemaining, rootValue, closureStates }) {
  const buffer = Buffer.alloc(24);
  buffer.write('C4VPOL1\0', 0, 'binary');
  buffer[8] = 1;
  buffer[9] = rows;
  buffer[10] = columns;
  buffer[11] = connect;
  buffer[12] = role;
  buffer[13] = handoffRemaining;
  buffer[14] = 10;
  buffer.writeInt8(rootValue, 15);
  buffer.writeUInt32LE(0, 16);
  buffer.writeUInt32LE(closureStates, 20);
  return buffer;
}

function entry(filename, buffer, metadata) {
  return {
    ...metadata,
    entryCount: 0,
    file: `./${filename}`,
    bytes: buffer.length,
    sha256: createHash('sha256').update(buffer).digest('hex'),
  };
}

test('parallel catalog verification independently replays each role', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'perfect-classic-parallel-test-'));
  context.after(() => rm(directory, { recursive: true, force: true }));

  const firstMetadata = {
    rows: 4,
    columns: 4,
    connect: 4,
    role: 1,
    handoffRemaining: 16,
    rootValue: 0,
    closureStates: 1,
  };
  const secondMetadata = {
    ...firstMetadata,
    role: 2,
    closureStates: 3,
  };
  const first = emptyPolicy(firstMetadata);
  const second = emptyPolicy(secondMetadata);
  await writeFile(join(directory, 'first.bin'), first);
  await writeFile(join(directory, 'second.bin'), second);
  const manifestPath = join(directory, 'manifest.json');
  await writeFile(manifestPath, `${JSON.stringify({
    format: 'connect4-perfect-classic-manifest-v1',
    policies: [
      entry('first.bin', first, firstMetadata),
      entry('second.bin', second, secondMetadata),
    ],
  }, null, 2)}\n`);

  const result = await verifyPerfectClassicCatalogParallel({
    reference: manifestPath,
    workers: 2,
    verify_table_bits: 14,
  });
  assert.equal(result.policyCount, 2);
  assert.equal(result.boardCount, 1);
  assert.ok(result.exactNodes > 0);
  assert.deepEqual(result.replay.map((record) => record.role), [1, 2]);
  assert.ok(result.replay.every((record) => record.rootValue === 0));
});

test('parallel catalog verification rejects duplicate policy identities', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'perfect-classic-parallel-duplicate-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const metadata = {
    rows: 4,
    columns: 4,
    connect: 4,
    role: 1,
    handoffRemaining: 16,
    rootValue: 0,
    closureStates: 1,
  };
  const policy = emptyPolicy(metadata);
  await writeFile(join(directory, 'policy.bin'), policy);
  const duplicate = entry('policy.bin', policy, metadata);
  const manifestPath = join(directory, 'manifest.json');
  await writeFile(manifestPath, `${JSON.stringify({
    format: 'connect4-perfect-classic-manifest-v1',
    policies: [duplicate, duplicate],
  })}\n`);

  await assert.rejects(
    verifyPerfectClassicCatalogParallel({ reference: manifestPath, workers: 2 }),
    /Duplicate perfect classic policy/,
  );
});
