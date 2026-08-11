import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

import {
  PERFECT_ROLE_BOTH,
  PERFECT_ROLE_FIRST,
  PERFECT_ROLE_SECOND,
  decodePerfectStrategy,
} from '../src/perfect-strategy.js';
import { decodeStrategy, verifyClosure } from '../scripts/perfect-strategy.mjs';

const strategyUrl = new URL('../assets/perfect-strategy.bin', import.meta.url);
const manifestUrl = new URL('../data/perfect-strategy.manifest.json', import.meta.url);

test('the committed Perfect strategy matches its manifest and closes both starting roles', async () => {
  const [bytes, manifestText] = await Promise.all([
    readFile(strategyUrl),
    readFile(manifestUrl, 'utf8'),
  ]);
  const manifest = JSON.parse(manifestText);
  const sha256 = createHash('sha256').update(bytes).digest('hex');

  const runtime = decodePerfectStrategy(bytes);
  assert.equal(runtime.version, manifest.format);
  assert.equal(runtime.handoffRemaining, manifest.handoffRemaining);
  assert.equal(runtime.roleFlags, PERFECT_ROLE_BOTH);
  assert.equal(runtime.roleFlags, manifest.roleFlags);
  assert.equal(runtime.entryCount, manifest.entryCount);
  assert.equal(runtime.byteLength, manifest.byteLength);
  assert.equal(sha256, manifest.sha256);
  assert.equal(runtime.coversRole(PERFECT_ROLE_FIRST), true);
  assert.equal(runtime.coversRole(PERFECT_ROLE_SECOND), true);

  const root = runtime.lookup(0n);
  assert.ok(root, 'the empty board must be covered when the AI starts');
  assert.equal(root.moveMask, 1 << 3, 'perfect first play must be the centre column');
  assert.equal(root.outcome, 1, 'the first player has a forced win in standard Connect Four');

  const generatorView = decodeStrategy(bytes);
  assert.deepEqual(verifyClosure(generatorView), manifest.closure);
});
