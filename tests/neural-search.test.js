import assert from 'node:assert/strict';
import test from 'node:test';

import { ACTION_DROP, EMPTY, RED, YELLOW, createBoard } from '../src/engine.js';
import { actionIndex, bestAction, searchPosition } from '../src/neural-search.js';

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
