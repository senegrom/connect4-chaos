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

const MANIFEST = JSON.parse(await readFile(new URL(
  '../data/perfect-chaos-prefix/manifest.json',
  import.meta.url,
), 'utf8'));

function roleName(role) {
  if (role === PERFECT_CHAOS_ROLE_FIRST) return 'red';
  if (role === PERFECT_CHAOS_ROLE_SECOND) return 'yellow';
  throw new RangeError(`Unknown Perfect Chaos role ${role}.`);
}

function expectedLayers(role) {
  const name = roleName(role);
  const segments = MANIFEST.roles?.[name]?.replay?.segments;
  assert.ok(Array.isArray(segments) && segments.length > 0, `${name} manifest has no replay segments`);
  return segments.map((segment) => [
    segment.fromPieces,
    segment.frontierPieces,
    segment.policyEntries,
  ]);
}

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

test('the committed Perfect Chaos policy layers match the manifest boundary', async () => {
  const manifestBoundary = MANIFEST.boundaries.at(-1);
  assert.equal(PERFECT_CHAOS_CERTIFIED_BOUNDARY, manifestBoundary);
  for (const role of [PERFECT_CHAOS_ROLE_FIRST, PERFECT_CHAOS_ROLE_SECOND]) {
    for (const [fromBoundary, boundary, entryCount] of expectedLayers(role)) {
      const policy = await loadPerfectChaosPolicy(role, fromBoundary);
      assert.equal(policy.role, role);
      assert.equal(policy.fromBoundary, fromBoundary);
      assert.equal(policy.boundary, boundary);
      assert.equal(policy.entryCount, entryCount);
    }
  }
  assert.equal(await loadPerfectChaosPolicy(PERFECT_CHAOS_ROLE_FIRST, manifestBoundary), null);
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
    const roleDirectory = roleName(role);
    let immediateWins = 0;

    for (const [fromBoundary, boundary, entryCount] of expectedLayers(role)) {
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

    assert.ok(immediateWins > 0, `${roleDirectory} policy never takes an immediate win`);
  }
});
