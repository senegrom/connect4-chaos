import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

import {
  PERFECT_CHAOS_CERTIFIED_BOUNDARY,
  PERFECT_CHAOS_ROLE_FIRST,
  PERFECT_CHAOS_ROLE_SECOND,
  decodePerfectChaosPolicy,
  loadPerfectChaosPolicy,
} from '../src/perfect-chaos-prefix.js';
import {
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CCW,
  ACTION_ROTATE_CW,
  RED,
  YELLOW,
  applyAction,
  createBoard,
  legalActions,
  resolveActionOutcome,
} from '../src/engine.js';


const POLICY_ACTION_TYPES = Object.freeze([
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CW,
  ACTION_ROTATE_CCW,
]);

const EXPECTED_IMMEDIATE_WINS = Object.freeze({
  [PERFECT_CHAOS_ROLE_FIRST]: 238_073,
  [PERFECT_CHAOS_ROLE_SECOND]: 692_060,
});

function packedBit(rows, column, rowFromBottom) {
  return 1n << BigInt(column * (rows + 1) + rowFromBottom);
}

function recordBoard(view, offset) {
  const mover = view.getBigUint64(offset, true);
  const opponent = view.getBigUint64(offset + 8, true);
  const rows = view.getUint8(offset + 16);
  const columns = view.getUint8(offset + 17);
  const board = createBoard(rows, columns);
  for (let column = 0; column < columns; column += 1) {
    for (let rowFromBottom = 0; rowFromBottom < rows; rowFromBottom += 1) {
      const mask = packedBit(rows, column, rowFromBottom);
      const row = rows - 1 - rowFromBottom;
      if ((mover & mask) !== 0n) board[row][column] = RED;
      else if ((opponent & mask) !== 0n) board[row][column] = YELLOW;
    }
  }
  return board;
}

function recordAction(view, offset) {
  const type = POLICY_ACTION_TYPES[view.getUint8(offset + 18)];
  return type === ACTION_DROP
    ? { type, column: view.getUint8(offset + 19) }
    : { type };
}

function actionWinsImmediately(board, action) {
  const result = applyAction(board, action, RED);
  if (!result) return false;
  const outcome = resolveActionOutcome(
    result.board,
    4,
    RED,
    action.type,
    action.type === ACTION_DROP ? { row: result.row, column: result.column } : null,
  );
  return outcome.status === 'won' && outcome.winner === RED;
}

const EXPECTED_LAYERS = Object.freeze({
  [PERFECT_CHAOS_ROLE_FIRST]: Object.freeze([
    [0, 8, 1_299],
    [8, 10, 5_058],
    [10, 12, 22_831],
    [12, 14, 92_200],
    [14, 16, 326_031],
  ]),
  [PERFECT_CHAOS_ROLE_SECOND]: Object.freeze([
    [0, 8, 3_863],
    [8, 10, 15_112],
    [10, 12, 67_605],
    [12, 14, 281_707],
    [14, 16, 1_059_068],
  ]),
});

test('the committed Perfect Chaos policy layers decode through sixteen pieces', async () => {
  assert.equal(PERFECT_CHAOS_CERTIFIED_BOUNDARY, 16);
  for (const role of [PERFECT_CHAOS_ROLE_FIRST, PERFECT_CHAOS_ROLE_SECOND]) {
    for (const [fromBoundary, boundary, entryCount] of EXPECTED_LAYERS[role]) {
      const policy = await loadPerfectChaosPolicy(role, fromBoundary);
      assert.equal(policy.role, role);
      assert.equal(policy.fromBoundary, fromBoundary);
      assert.equal(policy.boundary, boundary);
      assert.equal(policy.entryCount, entryCount);
    }
  }
  assert.equal(await loadPerfectChaosPolicy(PERFECT_CHAOS_ROLE_FIRST, 16), null);
});

test('the first policy layer chooses certified centre drops for both starting roles', async () => {
  const first = await loadPerfectChaosPolicy(PERFECT_CHAOS_ROLE_FIRST, 0);
  const second = await loadPerfectChaosPolicy(PERFECT_CHAOS_ROLE_SECOND, 1);

  const empty = createBoard(6, 7);
  assert.deepEqual(first.lookup(empty, RED, RED)?.action, {
    type: ACTION_DROP,
    column: 3,
  });

  const afterHuman = applyAction(empty, { type: ACTION_DROP, column: 2 }, RED).board;
  assert.deepEqual(second.lookup(afterHuman, YELLOW, YELLOW)?.action, {
    type: ACTION_DROP,
    column: 3,
  });
  assert.equal(second.lookup(afterHuman, RED, YELLOW), null);
});

test('the Perfect Chaos policy decoder rejects truncation, wrong roles, and wrong segments', async () => {
  const bytes = await readFile(new URL(
    '../data/perfect-chaos-prefix/red/0-8.policy.bin',
    import.meta.url,
  ));

  assert.throws(
    () => decodePerfectChaosPolicy(bytes.subarray(0, bytes.length - 1)),
    /length mismatch/,
  );
  assert.throws(
    () => decodePerfectChaosPolicy(bytes, PERFECT_CHAOS_ROLE_SECOND),
    /role does not match/,
  );
  assert.throws(
    () => decodePerfectChaosPolicy(bytes, PERFECT_CHAOS_ROLE_FIRST, 10),
    /boundary does not match/,
  );
});


test('every committed policy record takes an immediate win when one exists', async () => {
  for (const role of [PERFECT_CHAOS_ROLE_FIRST, PERFECT_CHAOS_ROLE_SECOND]) {
    const roleDirectory = role === PERFECT_CHAOS_ROLE_FIRST ? 'red' : 'yellow';
    let immediateWins = 0;

    for (const [fromBoundary, boundary, entryCount] of EXPECTED_LAYERS[role]) {
      const bytes = await readFile(new URL(
        `../data/perfect-chaos-prefix/${roleDirectory}/${fromBoundary}-${boundary}.policy.bin`,
        import.meta.url,
      ));
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      assert.equal(view.getUint32(12, true), entryCount);

      for (let index = 0; index < entryCount; index += 1) {
        const offset = 16 + index * 20;
        const board = recordBoard(view, offset);
        const chosen = recordAction(view, offset);
        if (actionWinsImmediately(board, chosen)) {
          immediateWins += 1;
          continue;
        }

        const availableWin = legalActions(board, true).find((action) => (
          actionWinsImmediately(board, action)
        ));
        assert.equal(
          availableWin,
          undefined,
          `${roleDirectory} ${fromBoundary}-${boundary} record ${index} ignores an immediate win`,
        );
      }
    }

    assert.equal(immediateWins, EXPECTED_IMMEDIATE_WINS[role]);
  }
});
