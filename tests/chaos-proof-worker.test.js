import assert from 'node:assert/strict';
import test from 'node:test';

import { chooseMoveWithChaosProof, choosePreparedMove } from '../src/ai-worker.js';
import { CHAOS_LOSS, CHAOS_WIN } from '../src/chaos-solver.js';
import {
  ACTION_DROP,
  ACTION_FLIP,
  ACTION_ROTATE_CCW,
  ACTION_ROTATE_CW,
  RED,
  YELLOW,
  applyAction,
  createBoard,
  otherPlayer,
  positionKey,
  sameAction,
} from '../src/engine.js';

function position(board, currentPlayer, connect) {
  return {
    board,
    currentPlayer,
    connect,
    chaosMode: true,
    repetitionCounts: [[positionKey(board, currentPlayer, connect, true), 1]],
  };
}

test('the worker entry point returns a converged bounded proof without heuristic search', () => {
  const board = createBoard(2, 2);
  const progress = [];
  const result = chooseMoveWithChaosProof(position(board, RED, 2), {
    difficulty: 'hard',
    aiPlayer: RED,
    chaosExactEmptyThreshold: 0,
    chaosProofDropDepth: 4,
    chaosProofMaximumStates: 1_000,
    useChaosProof: true,
    onIteration(update) {
      progress.push(update);
    },
  });

  assert.equal(result.solver, 'chaos-bounded-proof');
  assert.equal(result.solved, true);
  assert.equal(result.lowerValue, CHAOS_WIN);
  assert.equal(result.upperValue, CHAOS_WIN);
  assert.equal(result.score, CHAOS_WIN);
  assert.ok(result.action);
  assert.equal(progress.length, 1);
  assert.deepEqual(progress[0].action, result.action);
});

test('the worker refuses a proof limit out of range, and a drop depth of 0 switches the proof off', () => {
  // A 2x2 board: the drop depth runs from 0 through its four cells. The
  // search depth is for the runs that end in the ordinary search.
  const choose = (limits) => chooseMoveWithChaosProof(position(createBoard(2, 2), RED, 2), {
    difficulty: 'hard',
    aiPlayer: RED,
    chaosExactEmptyThreshold: 0,
    maximumDepth: 2,
    useChaosProof: true,
    ...limits,
  });
  for (const chaosProofDropDepth of [-1, 5, 1.5, '2']) {
    assert.throws(
      () => choose({ chaosProofDropDepth }),
      { name: 'RangeError', message: 'Chaos proof drop depth must be an integer from 0 through 4.' },
      String(chaosProofDropDepth),
    );
  }
  for (const chaosProofMaximumStates of [0, 2_000_001, 1.5, '1000']) {
    assert.throws(
      () => choose({ chaosProofDropDepth: 4, chaosProofMaximumStates }),
      { name: 'RangeError', message: 'Chaos proof state limit must be an integer from 1 through 2000000.' },
      String(chaosProofMaximumStates),
    );
  }
  // The ends of both ranges are taken, and a limit left out is the
  // difficulty's own.
  assert.equal(choose({ chaosProofDropDepth: 4, chaosProofMaximumStates: 2_000_000 }).solver, 'chaos-bounded-proof');
  assert.ok(choose({}).action);
  // Over a limit of one state the proof fails open to the search.
  assert.notEqual(choose({ chaosProofDropDepth: 4, chaosProofMaximumStates: 1 }).solver, 'chaos-bounded-proof');
  const off = choose({ chaosProofDropDepth: 0 });
  assert.notEqual(off.solver, 'chaos-bounded-proof');
  assert.equal(off.chaosProof, undefined);
  assert.ok(off.action);
});

test('the worker never keeps an action that the optimistic proof still loses', () => {
  const board = createBoard(6, 7);
  board[5] = [YELLOW, RED, 0, RED, RED, 0, YELLOW];
  const before = board.map((row) => [...row]);
  const result = chooseMoveWithChaosProof(position(board, YELLOW, 4), {
    difficulty: 'brutal',
    aiPlayer: YELLOW,
    maximumDepth: 1,
    quiescenceDepth: 0,
    chaosTransformBudget: 0,
    chaosExactEmptyThreshold: 0,
    chaosProofDropDepth: 2,
    chaosProofMaximumStates: 20_000,
    useChaosProof: true,
  });

  assert.equal(result.chaosProof.provenLosingActions.length, 7);
  const selected = result.chaosProof.actionBounds.find((entry) => (
    sameAction(entry.action, result.action)
  ));
  assert.ok(selected);
  assert.notEqual(selected.upper, CHAOS_LOSS);
  assert.ok(
    sameAction(result.action, { type: ACTION_DROP, column: 2 })
      || result.action.type === 'rotateCW'
      || result.action.type === 'rotateCCW',
  );
  assert.deepEqual(board, before);
});

test('explicit fixed-depth searches keep legacy behaviour unless proof is enabled', () => {
  const board = createBoard(6, 7);
  board[5] = [YELLOW, RED, 0, RED, RED, 0, YELLOW];
  const result = chooseMoveWithChaosProof(position(board, YELLOW, 4), {
    difficulty: 'brutal',
    aiPlayer: YELLOW,
    maximumDepth: 1,
    chaosExactEmptyThreshold: 0,
  });
  assert.equal(result.chaosProof, undefined);
});

test('a repeat in an earlier piece layer leaves exact Chaos play on', async () => {
  // 4x4 Connect 4 Chaos, Yellow to move. Of its six actions only a drop in
  // column 2 loses; every other one draws. Pieces are never removed, so the
  // empty board seen twice can never recur. It used to switch off the exact
  // solver and the proof for the rest of the round, and the bounded search
  // then played the losing drop.
  const board = [[0, 1, 0, 0], [0, 1, 0, 0], [2, 1, 1, 2], [2, 2, 1, 2]];
  const own = positionKey(board, YELLOW, 4, true);
  const empty = positionKey(createBoard(4, 4), RED, 4, true);
  for (const repetitionCounts of [new Map([[own, 1]]), new Map([[empty, 2], [own, 1]])]) {
    const result = await choosePreparedMove(
      { ...position(board, YELLOW, 4), startingPlayer: RED, repetitionCounts },
      { difficulty: 'brutal', aiPlayer: YELLOW },
    );
    assert.equal(result.solver, 'chaos-exact-graph');
    assert.ok(!sameAction(result.action, { type: ACTION_DROP, column: 2 }), JSON.stringify(result.action));
  }
});

test('a lost Chaos position takes the draw a third occurrence gives', async () => {
  // 4x4 Connect 3 Chaos, Red started. After these moves Yellow is to move:
  // every drop lets Red win at once, rotating anticlockwise completes a Red
  // line and flipping loses within three plies. Rotating clockwise makes a
  // position the round has seen twice, and its third occurrence draws.
  // Red would win on that board if play went on, so only the repetition
  // clause keeps the tactical safety check from swapping the draw for the
  // flip. The history-free bounded proof calls the rotation lost, which is
  // why the worker skips it once the board's own piece layer has repeated.
  const drop = (column) => ({ type: ACTION_DROP, column });
  let board = createBoard(4, 4);
  let currentPlayer = RED;
  const repetitionCounts = new Map([[positionKey(board, currentPlayer, 3, true), 1]]);
  for (const action of [drop(2), { type: ACTION_FLIP }, drop(2), drop(2), { type: ACTION_FLIP }, drop(3),
    drop(1), { type: ACTION_ROTATE_CW }, { type: ACTION_ROTATE_CCW }, { type: ACTION_ROTATE_CW },
    { type: ACTION_ROTATE_CCW }]) {
    board = applyAction(board, action, currentPlayer).board;
    currentPlayer = otherPlayer(currentPlayer);
    const key = positionKey(board, currentPlayer, 3, true);
    repetitionCounts.set(key, (repetitionCounts.get(key) ?? 0) + 1);
  }
  assert.deepEqual(board, [[0, 0, 0, 0], [0, 0, 0, RED], [0, 0, 0, RED], [0, RED, YELLOW, YELLOW]]);
  assert.equal(currentPlayer, YELLOW);
  const rotated = applyAction(board, { type: ACTION_ROTATE_CW }, YELLOW).board;
  assert.equal(repetitionCounts.get(positionKey(rotated, RED, 3, true)), 2);

  for (const difficulty of ['medium', 'hard', 'brutal']) {
    const result = await choosePreparedMove(
      { board, currentPlayer, connect: 3, chaosMode: true, startingPlayer: RED, repetitionCounts },
      { difficulty, aiPlayer: YELLOW },
    );
    assert.deepEqual(result.action, { type: ACTION_ROTATE_CW }, difficulty);
    assert.equal(result.score, 0, difficulty);
  }
});
