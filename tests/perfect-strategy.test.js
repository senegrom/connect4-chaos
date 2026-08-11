import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildStrategy,
  decodeStrategy,
  verifyClosure,
  STRATEGY_CONSTANTS,
} from '../scripts/perfect-strategy.mjs';

function neutralScores(sequences) {
  return new Map(sequences.map((sequence) => [sequence, Array(7).fill(0)]));
}

test('the deterministic strategy format covers both starting roles to its handoff', async () => {
  const { bytes, manifest } = await buildStrategy({
    handoffRemaining: 40,
    roles: 'both',
    source: 'deterministic test oracle',
    scoreBatch: neutralScores,
  });
  const decoded = decodeStrategy(bytes);

  assert.equal(decoded.version, 1);
  assert.equal(decoded.handoffRemaining, 40);
  assert.equal(decoded.roleFlags, STRATEGY_CONSTANTS.ROLE_FIRST | STRATEGY_CONSTANTS.ROLE_SECOND);
  assert.equal(decoded.entries.length, 5);
  assert.equal(manifest.entryCount, 5);
  assert.deepEqual(verifyClosure(decoded), manifest.closure);
});

test('strategy decoding rejects truncation and multi-move entries', async () => {
  const { bytes } = await buildStrategy({
    handoffRemaining: 41,
    roles: 'first',
    source: 'deterministic test oracle',
    scoreBatch: neutralScores,
  });

  assert.throws(() => decodeStrategy(bytes.subarray(0, bytes.length - 1)), /length/);
  const invalid = bytes.slice();
  invalid[20] = 3;
  assert.throws(() => decodeStrategy(invalid), /exactly one/);
});
