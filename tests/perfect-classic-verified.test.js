import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';

import { RED } from '../src/engine.js';
import { loadVerifiedPerfectClassicPolicy } from '../src/perfect-classic-verified.js';

function encodePolicy() {
  const buffer = Buffer.alloc(34);
  Buffer.from('C4VPOL1\0', 'binary').copy(buffer, 0);
  buffer[8] = 1;
  buffer[9] = 4;
  buffer[10] = 4;
  buffer[11] = 4;
  buffer[12] = 1;
  buffer[13] = 0;
  buffer[14] = 10;
  buffer.writeInt8(0, 15);
  buffer.writeUInt32LE(1, 16);
  buffer.writeUInt32LE(1, 20);
  buffer.writeBigUInt64LE(0n, 24);
  buffer[32] = 1 << 1;
  buffer.writeInt8(0, 33);
  return buffer;
}

async function writeFixture(directory, { corruptHash = false, corruptBytes = false } = {}) {
  const policy = encodePolicy();
  if (corruptBytes) policy[33] = 1;
  const policyPath = join(directory, '4x4-c4-role1.bin');
  await writeFile(policyPath, policy);
  const digest = createHash('sha256').update(encodePolicy()).digest('hex');
  const manifest = {
    format: 'connect4-perfect-classic-manifest-v1',
    policies: [{
      rows: 4,
      columns: 4,
      connect: 4,
      role: 1,
      handoffRemaining: 0,
      rootValue: 0,
      entryCount: 1,
      closureStates: 1,
      file: './4x4-c4-role1.bin',
      bytes: policy.length,
      sha256: corruptHash ? '0'.repeat(64) : digest,
    }],
  };
  const manifestPath = join(directory, 'manifest.json');
  await writeFile(manifestPath, `${JSON.stringify(manifest)}\n`);
  return pathToFileURL(manifestPath);
}

test('runtime loader verifies policy bytes and metadata before exposing lookup', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-perfect-classic-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const manifestUrl = await writeFixture(directory);
  const policy = await loadVerifiedPerfectClassicPolicy(4, 4, 4, 1, { manifestUrl });
  assert.equal(policy.entryCount, 1);
  assert.deepEqual(
    policy.lookup(Array.from({ length: 4 }, () => Array(4).fill(0)), RED, RED, RED),
    { action: { type: 'drop', column: 1 }, outcome: 0, mirrored: false },
  );
});

test('runtime loader rejects hash and byte changes fail closed', async (context) => {
  const hashDirectory = await mkdtemp(join(tmpdir(), 'connect4-perfect-classic-hash-'));
  const bytesDirectory = await mkdtemp(join(tmpdir(), 'connect4-perfect-classic-bytes-'));
  context.after(() => Promise.all([
    rm(hashDirectory, { recursive: true, force: true }),
    rm(bytesDirectory, { recursive: true, force: true }),
  ]));

  const badHash = await writeFixture(hashDirectory, { corruptHash: true });
  await assert.rejects(
    loadVerifiedPerfectClassicPolicy(4, 4, 4, 1, { manifestUrl: badHash }),
    /SHA-256/,
  );

  const badBytes = await writeFixture(bytesDirectory, { corruptBytes: true });
  await assert.rejects(
    loadVerifiedPerfectClassicPolicy(4, 4, 4, 1, { manifestUrl: badBytes }),
    /SHA-256/,
  );
});
