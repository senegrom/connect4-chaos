import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { pathToFileURL } from 'node:url';

import { choosePreparedMove } from '../src/ai-worker.js';
import { RED, YELLOW } from '../src/engine.js';
import {
  PERFECT_CLASSIC_ROLE_FIRST,
  decodePerfectClassicPolicy,
  loadPerfectClassicManifest,
  perfectClassicRole,
} from '../src/perfect-classic-policy.js';
import {
  choosePerfectClassicMove,
  isPerfectClassicVariant,
  usesPerfectClassicPolicy,
} from '../src/perfect-classic-runtime.js';

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

test('variable Perfect play fails closed without a verified policy', () => {
  const position = {
    board: emptyBoard(5, 5),
    currentPlayer: RED,
    startingPlayer: RED,
    connect: 4,
    chaosMode: false,
  };
  assert.equal(isPerfectClassicVariant(position), true);
  assert.throws(
    () => choosePerfectClassicMove(position, {
      difficulty: 'perfect',
      aiPlayer: RED,
    }),
    /policy could not be loaded/,
  );
});

test('a Perfect classic move needs the policy exactly where the catalog hands off', async () => {
  // At or below its handoff a policy is never read, and the worker loads
  // neither it nor the catalog there, so the two must agree on every board.
  const manifest = JSON.parse(await readFile(new URL('../data/perfect-classic/manifest.json', import.meta.url)));
  const withEmptyCells = (rows, columns, empty) => {
    const board = emptyBoard(rows, columns);
    for (let at = 0; at < rows * columns - empty; at += 1) {
      board[rows - 1 - Math.floor(at / columns)][at % columns] = at % 2 ? YELLOW : RED;
    }
    return { board };
  };
  for (const { rows, columns, role, handoffRemaining } of manifest.policies) {
    const label = `${rows}x${columns} role ${role}`;
    assert.equal(usesPerfectClassicPolicy(withEmptyCells(rows, columns, handoffRemaining)), false, label);
    if (handoffRemaining < rows * columns) {
      assert.equal(usesPerfectClassicPolicy(withEmptyCells(rows, columns, handoffRemaining + 1)), true, label);
    }
  }
});

test('Perfect classic play at the handoff needs neither the catalog nor the policy', async () => {
  // Only the exact solver answers there, yet a fresh worker - after Undo,
  // Cancel or two idle minutes - loaded both first, and a failed or stalled
  // load failed the move.
  const options = {
    difficulty: 'perfect',
    aiPlayer: RED,
    manifestUrl: pathToFileURL(join(tmpdir(), 'perfect-classic-unreachable', 'manifest.json')),
  };
  const start = { currentPlayer: RED, startingPlayer: RED, connect: 4, chaosMode: false };
  const result = await choosePreparedMove({ ...start, board: emptyBoard(4, 4) }, options);
  assert.equal(result.solver, 'classic-exact');
  assert.equal(result.solved, true);
  assert.equal(result.value, 0);
  // Above it the policy is still required, and the move fails closed.
  await assert.rejects(choosePreparedMove({ ...start, board: emptyBoard(5, 5) }, options),
    /Could not load the verified Perfect classic policy/);
});

test('standard 6x7 retains the specialised Perfect strategy route', () => {
  const position = {
    board: emptyBoard(6, 7),
    currentPlayer: RED,
    startingPlayer: RED,
    connect: 4,
    chaosMode: false,
  };
  assert.equal(isPerfectClassicVariant(position), false);
  assert.equal(choosePerfectClassicMove(position, { difficulty: 'perfect' }), null);
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

test('policy headers and manifest entries are held to the boards a policy can cover', async (context) => {
  // The checks are the solver's, in a policy's own words. A manifest entry
  // must name its connect length: on 5x5 the solver's default of four would
  // pass for one left out.
  for (const [header, message] of [
    [{ rows: 0 }, /support 1 through 7 rows and columns/],
    [{ rows: 8 }, /support 1 through 7 rows and columns/],
    [{ columns: 8 }, /support 1 through 7 rows and columns/],
    [{ connect: 0 }, /connect length must fit the board/],
    [{ connect: 5 }, /connect length must fit the board/],
  ]) {
    assert.throws(() => decodePerfectClassicPolicy(encodePolicy({ ...header, records: [] })), message,
      JSON.stringify(header));
  }
  const directory = await mkdtemp(join(tmpdir(), 'perfect-classic-manifest-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const entry = { rows: 5, columns: 5, connect: 4, role: 1, file: './5x5.bin', handoffRemaining: 0,
    entryCount: 0, rootValue: 0 };
  const cases = [
    [{ connect: undefined }, /connect length must fit the board/],
    [{ connect: 0 }, /connect length must fit the board/],
    [{ connect: 6 }, /connect length must fit the board/],
    [{ rows: 8, connect: undefined }, /support 1 through 7 rows and columns/],
  ];
  for (const [index, [change, message]] of cases.entries()) {
    const path = join(directory, `manifest-${index}.json`);     // the loader caches by URL
    await writeFile(path, JSON.stringify({
      format: 'connect4-perfect-classic-manifest-v1', policies: [{ ...entry, ...change }],
    }));
    await assert.rejects(loadPerfectClassicManifest(pathToFileURL(path)), message, JSON.stringify(change));
  }
});
