import assert from 'node:assert/strict';
import test from 'node:test';

import { RED, YELLOW } from '../src/engine.js';
import {
  PERFECT_CLASSIC_ROLE_FIRST,
  decodePerfectClassicPolicy,
  perfectClassicRole,
} from '../src/perfect-classic-policy.js';
import { choosePerfectClassicMove } from '../src/perfect-classic-runtime.js';

function encodePolicy({
  rows = 4,
  columns = 4,
  connect = 4,
  role = PERFECT_CLASSIC_ROLE_FIRST,
  handoffRemaining = 0,
  rootValue = 0,
  closureStates = 1,
  records = [{ key: 0n, moveMask: 1 << 1, outcome: 0 }],
} = {}) {
  const buffer = Buffer.alloc(24 + records.length * 10);
  Buffer.from('C4VPOL1\0', 'binary').copy(buffer, 0);
  buffer[8] = 1;
  buffer[9] = rows;
  buffer[10] = columns;
  buffer[11] = connect;
  buffer[12] = role;
  buffer[13] = handoffRemaining;
  buffer[14] = 10;
  buffer.writeInt8(rootValue, 15);
  buffer.writeUInt32LE(records.length, 16);
  buffer.writeUInt32LE(closureStates, 20);
  records.forEach((record, index) => {
    const offset = 24 + index * 10;
    buffer.writeBigUInt64LE(record.key, offset);
    buffer[offset + 8] = record.moveMask;
    buffer.writeInt8(record.outcome, offset + 9);
  });
  return buffer;
}

function emptyBoard(rows = 4, columns = 4) {
  return Array.from({ length: rows }, () => Array(columns).fill(0));
}

test('variable-board policy decoding validates metadata and returns the canonical action', () => {
  const policy = decodePerfectClassicPolicy(encodePolicy(), {
    rows: 4,
    columns: 4,
    connect: 4,
    role: PERFECT_CLASSIC_ROLE_FIRST,
  });
  assert.equal(policy.entryCount, 1);
  assert.equal(policy.rootValue, 0);
  assert.deepEqual(
    policy.lookup(emptyBoard(), RED, RED, RED),
    {
      action: { type: 'drop', column: 1 },
      outcome: 0,
      mirrored: false,
    },
  );
  assert.equal(policy.lookup(emptyBoard(), RED, RED, YELLOW), null);
  assert.equal(perfectClassicRole(RED, RED), 1);
  assert.equal(perfectClassicRole(RED, YELLOW), 2);
});

test('Perfect classic runtime uses a verified policy without allocating search', () => {
  const policy = decodePerfectClassicPolicy(encodePolicy());
  const updates = [];
  const result = choosePerfectClassicMove({
    board: emptyBoard(),
    currentPlayer: RED,
    startingPlayer: RED,
    connect: 4,
    chaosMode: false,
  }, {
    difficulty: 'perfect',
    aiPlayer: RED,
    perfectClassicPolicy: policy,
    onIteration(update) { updates.push(update); },
  });
  assert.equal(result.solver, 'perfect-classic-policy');
  assert.equal(result.solved, true);
  assert.equal(result.nodes, 0);
  assert.deepEqual(result.action, { type: 'drop', column: 1 });
  assert.deepEqual(updates, [result]);
});

test('Perfect classic runtime hands covered late positions to the exact solver', () => {
  const board = [
    [YELLOW, YELLOW, RED, 0],
    [RED, RED, RED, YELLOW],
    [YELLOW, YELLOW, YELLOW, RED],
    [RED, RED, RED, YELLOW],
  ];
  const policy = decodePerfectClassicPolicy(encodePolicy({
    role: 2,
    handoffRemaining: 1,
    rootValue: 0,
    closureStates: 0,
    records: [],
  }));
  const result = choosePerfectClassicMove({
    board,
    currentPlayer: YELLOW,
    startingPlayer: RED,
    connect: 4,
    chaosMode: false,
  }, {
    difficulty: 'perfect',
    aiPlayer: YELLOW,
    perfectClassicPolicy: policy,
  });
  assert.equal(result.solver, 'classic-exact');
  assert.equal(result.solved, true);
  assert.equal(result.value, 0);
  assert.deepEqual(result.action, { type: 'drop', column: 3 });
});

test('policy decoding rejects ambiguous moves, ordering errors and metadata mismatches', () => {
  assert.throws(
    () => decodePerfectClassicPolicy(encodePolicy({
      records: [{ key: 0n, moveMask: 3, outcome: 0 }],
    })),
    /exactly one legal column bit/,
  );
  assert.throws(
    () => decodePerfectClassicPolicy(encodePolicy({
      records: [
        { key: 2n, moveMask: 2, outcome: 0 },
        { key: 1n, moveMask: 2, outcome: 0 },
      ],
    })),
    /strictly increasing/,
  );
  assert.throws(
    () => decodePerfectClassicPolicy(encodePolicy(), { rows: 5 }),
    /metadata does not match/,
  );
});
