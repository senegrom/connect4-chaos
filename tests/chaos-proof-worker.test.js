import assert from 'node:assert/strict';
import test from 'node:test';

import { chooseMoveWithChaosProof, choosePreparedMove } from '../src/ai-worker.js';
import { CHAOS_LOSS, CHAOS_WIN } from '../src/chaos-solver.js';
import {
  ACTION_DROP,
  RED,
  YELLOW,
  createBoard,
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
      { difficulty: 'brutal', aiPlayer: YELLOW, timeBudgetMs: 50 },
    );
    assert.equal(result.solver, 'chaos-exact-graph');
    assert.ok(!sameAction(result.action, { type: ACTION_DROP, column: 2 }), JSON.stringify(result.action));
  }
});
