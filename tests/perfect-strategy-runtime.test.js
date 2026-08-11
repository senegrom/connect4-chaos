import test from 'node:test';
import assert from 'node:assert/strict';

import {
  PERFECT_ROLE_BOTH,
  PERFECT_ROLE_FIRST,
  PERFECT_ROLE_SECOND,
  decodePerfectStrategy,
} from '../src/perfect-strategy.js';

function strategyBytes(entries, overrides = {}) {
  const bytes = new Uint8Array(12 + entries.length * 10);
  bytes.set([...Buffer.from('C4PS')], 0);
  const view = new DataView(bytes.buffer);
  view.setUint8(4, overrides.version ?? 1);
  view.setUint8(5, overrides.handoffRemaining ?? 24);
  view.setUint8(6, overrides.entrySize ?? 10);
  view.setUint8(7, overrides.roleFlags ?? PERFECT_ROLE_BOTH);
  view.setUint32(8, entries.length, true);
  entries.forEach((entry, index) => {
    const offset = 12 + index * 10;
    view.setBigUint64(offset, entry.key, true);
    view.setUint8(offset + 8, entry.moveMask);
    view.setInt8(offset + 9, entry.outcome);
  });
  return bytes;
}

test('the runtime strategy decoder validates metadata and performs binary lookup', () => {
  const strategy = decodePerfectStrategy(strategyBytes([
    { key: 3n, moveMask: 1 << 2, outcome: -1 },
    { key: 9n, moveMask: 1 << 4, outcome: 1 },
  ]));

  assert.equal(strategy.handoffRemaining, 24);
  assert.equal(strategy.entryCount, 2);
  assert.equal(strategy.coversRole(PERFECT_ROLE_FIRST), true);
  assert.equal(strategy.coversRole(PERFECT_ROLE_SECOND), true);
  assert.equal(strategy.coversRole(PERFECT_ROLE_BOTH), false);
  assert.equal(strategy.coversRole(0), false);
  assert.deepEqual(strategy.lookup(9n), { key: 9n, moveMask: 1 << 4, outcome: 1 });
  assert.equal(strategy.lookup(8n), null);
});

test('the runtime strategy decoder rejects incomplete or ambiguous policy data', () => {
  assert.throws(
    () => decodePerfectStrategy(strategyBytes([
      { key: 3n, moveMask: (1 << 2) | (1 << 3), outcome: 1 },
    ])),
    /exactly one/,
  );
  assert.throws(
    () => decodePerfectStrategy(strategyBytes([
      { key: 9n, moveMask: 1 << 2, outcome: 1 },
      { key: 3n, moveMask: 1 << 4, outcome: -1 },
    ])),
    /strictly increasing/,
  );
  assert.throws(
    () => decodePerfectStrategy(strategyBytes([], { roleFlags: 0 })),
    /role flags/,
  );
  assert.throws(
    () => decodePerfectStrategy(strategyBytes([
      { key: 1n << 49n, moveMask: 1 << 3, outcome: 1 },
    ])),
    /outside the standard board/,
  );
});
