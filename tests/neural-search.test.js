import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ACTION_DROP, EMPTY, RED, YELLOW, applyAction, createBoard, otherPlayer, resolveActionOutcome,
} from '../src/engine.js';
import { actionIndex, bestAction, searchPosition } from '../src/neural-search.js';
import { simulationsFor } from '../src/neural-runtime.js';

// A network that knows nothing: uniform priors, every position a draw. The
// search must still find forced tactics, because terminal positions are
// resolved by the rules rather than by the network.
function blindNetwork() {
  return async () => ({
    policy: new Float32Array(13),
    value: new Float32Array([0, 1, 0]),        // certain draw
    q: new Float32Array(39),
  });
}

function boardWith(rows, cols, placements) {
  const board = createBoard(rows, cols);
  for (const [row, column, player] of placements) board[row][column] = player;
  return board;
}

test('the search takes an immediate win', async () => {
  // Bottom row (index rows-1) holds three of the mover's pieces.
  const rows = 6;
  const board = boardWith(rows, 7, [
    [rows - 1, 0, RED], [rows - 1, 1, RED], [rows - 1, 2, RED],
    [rows - 1, 4, YELLOW], [rows - 1, 5, YELLOW], [rows - 1, 6, YELLOW],
  ]);
  const result = await searchPosition(
    { board, currentPlayer: RED, connect: 4, chaosMode: false },
    blindNetwork(), { simulations: 64 },
  );
  const move = bestAction(result);
  assert.equal(move.type, ACTION_DROP);
  assert.equal(move.column, 3, 'column 3 completes the line');
});

function yellowWinningPosition() {
  let board = createBoard(6, 7);
  let currentPlayer = RED;
  for (const column of [0, 1, 0, 1, 2, 1, 2]) {
    const applied = applyAction(board, { type: ACTION_DROP, column }, currentPlayer);
    assert.equal(resolveActionOutcome(applied.board, 4, currentPlayer, ACTION_DROP, applied).status, 'playing');
    board = applied.board;
    currentPlayer = otherPlayer(currentPlayer);
  }
  return { board, currentPlayer, connect: 4, chaosMode: false };
}

function assertWinningMove(position, result) {
  const action = bestAction(result);
  const applied = applyAction(position.board, action, position.currentPlayer);
  const outcome = resolveActionOutcome(applied.board, position.connect, position.currentPlayer, action.type, applied);
  assert.equal(outcome.status, 'won');
  assert.equal(outcome.winner, position.currentPlayer);
}

test('the minimum device budget takes a discovered win despite tied visits', async () => {
  const position = yellowWinningPosition();
  const simulations = simulationsFor({ backend: 'wasm', perEvaluation: 800 });
  assert.equal(simulations, 2);
  const result = await searchPosition(position, blindNetwork(), { simulations });
  assert.deepEqual(result.visits.slice(0, 2), [1, 1]);
  assertWinningMove(position, result);
});

test('Move now prefers a discovered win over a more-visited estimated win', async () => {
  const position = yellowWinningPosition();
  const optimisticNetwork = async (_board, mover) => {
    const policy = new Float32Array(13).fill(-Infinity);
    policy[0] = Math.log(0.68);
    policy[1] = Math.log(0.32);
    const value = new Float32Array(mover === YELLOW ? [-Infinity, -Infinity, 0] : [0, -Infinity, -Infinity]);
    const q = new Float32Array(39).fill(-Infinity);
    for (let action = 0; action < 13; action += 1) q[action * 3 + 2] = 0;
    return { policy, value, q };
  };
  let stopChecks = 0;
  const result = await searchPosition(position, optimisticNetwork, {
    simulations: 128, shouldStop: () => ++stopChecks >= 3,
  });
  assert.equal(result.completedSimulations, 3);
  assert.deepEqual(result.visits.slice(0, 2), [2, 1]);
  assertWinningMove(position, result);
});

test('tied nonterminal moves use the searched value for the root player', async () => {
  const position = { board: createBoard(6, 7), currentPlayer: RED, connect: 4, chaosMode: false };
  const evaluator = async (board) => ({
    policy: new Float32Array(13),
    value: new Float32Array(board[5][0] === RED ? [-Infinity, -Infinity, 0] : [0, -Infinity, -Infinity]),
    q: new Float32Array(39),
  });
  const result = await searchPosition(position, evaluator, { simulations: 2 });
  assert.deepEqual(result.visits.slice(0, 2), [1, 1]);
  assert.equal(bestAction(result).column, 1, 'prefer the child evaluated as losing for the opponent');
});

test('the search blocks an immediate threat', async () => {
  const rows = 6;
  // The opponent has three in a column; the mover must cap it.
  const board = boardWith(rows, 7, [
    [rows - 1, 6, YELLOW], [rows - 2, 6, YELLOW], [rows - 3, 6, YELLOW],
    [rows - 1, 0, RED], [rows - 1, 1, RED],
  ]);
  const result = await searchPosition(
    { board, currentPlayer: RED, connect: 4, chaosMode: false },
    blindNetwork(), { simulations: 128 },
  );
  const move = bestAction(result);
  assert.equal(move.type, ACTION_DROP);
  assert.equal(move.column, 6, 'column 6 is the only move that survives');
});

test('visits are spread only over legal actions and sum to the budget', async () => {
  const board = createBoard(4, 4);
  const result = await searchPosition(
    { board, currentPlayer: RED, connect: 3, chaosMode: true },
    blindNetwork(), { simulations: 40 },
  );
  assert.equal(result.visits.length, result.actions.length);
  assert.equal(result.visits.reduce((sum, count) => sum + count, 0), 40);
  // Chaos adds the three transforms to the four drops.
  assert.equal(result.actions.length, 7);
  const indices = new Set(result.actions.map(actionIndex));
  assert.ok(indices.has(10) && indices.has(11) && indices.has(12), 'transforms are available');
  assert.ok(Math.abs(result.policy.reduce((sum, share) => sum + share, 0) - 1) < 1e-9);
});

test('a network preference decides between equal moves', async () => {
  const board = createBoard(6, 7);
  const favouring = async () => {
    const policy = new Float32Array(13);
    policy[5] = 8;                              // a strong preference for column 5
    return { policy, value: new Float32Array([0, 1, 0]), q: new Float32Array(39) };
  };
  const result = await searchPosition(
    { board, currentPlayer: RED, connect: 4, chaosMode: false },
    favouring, { simulations: 48 },
  );
  assert.equal(bestAction(result).column, 5);
});
