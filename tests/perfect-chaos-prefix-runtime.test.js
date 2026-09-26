import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  PERFECT_CHAOS_CERTIFIED_BOUNDARY,
  PERFECT_CHAOS_RELEASED_POLICIES,
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
  boardToString,
  createBoard,
  legalActions,
  otherPlayer,
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
      // A released layer is decoded without re-validating each record, as
      // its SHA-256 is pinned; here every one of them must pass validation.
      const bytes = await readFile(new URL(
        `../data/perfect-chaos-prefix/${roleName(role)}/${fromBoundary}-${boundary}.policy.bin`,
        import.meta.url,
      ));
      assert.equal(decodePerfectChaosPolicy(bytes, role, boundary).entryCount, entryCount);
    }
  }
  assert.equal(await loadPerfectChaosPolicy(PERFECT_CHAOS_ROLE_FIRST, manifestBoundary), null);
});

test('the runtime pins exactly the released policy digests of the manifest', () => {
  for (const role of ['red', 'yellow']) {
    const released = Object.fromEntries(MANIFEST.artifacts[role]
      .filter((artifact) => artifact.path.endsWith('.policy.bin'))
      .map((artifact) => [artifact.path, { bytes: artifact.bytes, sha256: artifact.sha256 }]));
    assert.deepEqual(PERFECT_CHAOS_RELEASED_POLICIES[role], released, role);
  }
});

test('the runtime refuses policy bytes that differ from the release', async (context) => {
  const directory = await mkdtemp(join(tmpdir(), 'connect4-chaos-prefix-runtime-'));
  context.after(() => rm(directory, { recursive: true, force: true }));
  const bytes = await readFile(new URL('../data/perfect-chaos-prefix/red/8-10.policy.bin', import.meta.url));

  // Same length, one record changed: only the digest can tell.
  const tampered = Buffer.from(bytes);
  tampered[tampered.length - 1] ^= 1;
  const tamperedPath = join(directory, 'tampered.policy.bin');
  await writeFile(tamperedPath, tampered);
  await assert.rejects(
    loadPerfectChaosPolicy(PERFECT_CHAOS_ROLE_FIRST, 8, pathToFileURL(tamperedPath)),
    /red\/8-10\.policy\.bin does not match the released SHA-256/,
  );

  // A short file is refused by its size, before any hashing or decoding.
  const truncatedPath = join(directory, 'truncated.policy.bin');
  await writeFile(truncatedPath, bytes.subarray(0, bytes.length - 20));
  await assert.rejects(
    loadPerfectChaosPolicy(PERFECT_CHAOS_ROLE_FIRST, 9, pathToFileURL(truncatedPath)),
    new RegExp(`red/8-10\\.policy\\.bin is ${bytes.length - 20} bytes; the release has ${bytes.length}`),
  );
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

test('a mirrored position is played as the mirror image of its certified move', async () => {
  // The records hold only the horizontally canonical side of a position, and
  // a lookup mirrors the answer back for the other side. Every other lookup
  // a test makes is an unmirrored centre drop, so a mirror that stopped
  // swapping the rotations or reflecting a column passed them all, and live
  // Brutal would leave the certified closure. This replays layers 0-10 for
  // both roles against every reply, looking each position up both ways.
  const mirror = (board) => board.map((row) => [...row].reverse());
  const pieces = (board) => board.reduce((sum, row) => sum + row.filter(Boolean).length, 0);
  const exercised = { drop: 0, flip: 0, rotation: 0 };
  for (const [role, ai] of [[PERFECT_CHAOS_ROLE_FIRST, RED], [PERFECT_CHAOS_ROLE_SECOND, YELLOW]]) {
    const layers = [await loadPerfectChaosPolicy(role, 0), await loadPerfectChaosPolicy(role, 8)];
    const lookup = (board) => layers[pieces(board) < 8 ? 0 : 1].lookup(board, ai, ai);
    const seen = new Set();
    const queue = [{ board: createBoard(6, 7), player: RED }];
    while (queue.length > 0) {
      const { board, player } = queue.pop();
      const key = `${player}:${board.length}x${board[0].length}:${boardToString(board)}`;
      if (seen.has(key) || pieces(board) >= 10) continue;
      seen.add(key);
      let actions = legalActions(board, true);
      if (player === ai) {
        const found = lookup(board);
        assert.ok(found, key);
        const reflected = mirror(board);
        if (boardToString(reflected) !== boardToString(board)) {
          const other = lookup(reflected);
          const replayed = other && applyAction(reflected, other.action, ai);
          assert.equal(replayed && boardToString(replayed.board),
            boardToString(mirror(applyAction(board, found.action, ai).board)), key);
          const { action } = found.mirrored ? found : other;
          if (action.type === ACTION_DROP) exercised.drop += action.column === 3 ? 0 : 1;
          else if (action.type === ACTION_FLIP) exercised.flip += 1;
          else exercised.rotation += 1;
        }
        actions = [found.action];
      }
      for (const action of actions) {
        const moved = applyAction(board, action, player);
        const outcome = resolveActionOutcome(moved.board, 4, player, action.type,
          action.type === ACTION_DROP ? { row: moved.row, column: moved.column } : null);
        if (outcome.status === 'playing') queue.push({ board: moved.board, player: otherPlayer(player) });
      }
    }
  }
  // The mirrored side covered every kind of action the records hold.
  assert.ok(exercised.drop > 1000 && exercised.flip > 100 && exercised.rotation > 100, JSON.stringify(exercised));
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
