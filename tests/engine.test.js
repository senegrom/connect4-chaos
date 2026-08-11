import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ACTION_DROP,
  ACTION_FLIP,
  EMPTY,
  RED,
  YELLOW,
  applyAction,
  flipBoard,
  hasWinFrom,
  normalizeConfig,
  positionKey,
  resolveActionOutcome,
  rotateBoard,
  winningCells,
} from '../src/engine.js';

test('drop actions stack from the bottom without mutating the source board', () => {
  const board = Array.from({ length: 4 }, () => Array(4).fill(EMPTY));
  const first = applyAction(board, { type: ACTION_DROP, column: 2 }, RED);
  const second = applyAction(first.board, { type: ACTION_DROP, column: 2 }, YELLOW);

  assert.equal(first.row, 3);
  assert.equal(second.row, 2);
  assert.equal(board[3][2], EMPTY);
  assert.equal(first.board[3][2], RED);
  assert.equal(second.board[2][2], YELLOW);
});

test('win detection handles horizontal, vertical, and diagonal lines', () => {
  const horizontal = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 1],
  ];
  const vertical = [
    [0, 2, 0, 0],
    [0, 2, 0, 0],
    [0, 2, 0, 0],
    [0, 2, 0, 0],
  ];
  const diagonal = [
    [0, 0, 0, 1],
    [0, 0, 1, 2],
    [0, 1, 2, 2],
    [1, 2, 2, 1],
  ];

  assert.equal(hasWinFrom(horizontal, 3, 2, RED, 4), true);
  assert.equal(hasWinFrom(vertical, 2, 1, YELLOW, 4), true);
  assert.equal(hasWinFrom(diagonal, 1, 2, RED, 4), true);
  assert.equal(winningCells(horizontal, RED, 4).length, 4);
});

test('flipping reverses the board and then reapplies gravity', () => {
  const board = [
    [0, 0],
    [0, 0],
    [1, 0],
    [2, 1],
  ];

  assert.deepEqual(flipBoard(board), [
    [0, 0],
    [0, 0],
    [2, 0],
    [1, 1],
  ]);
});

test('rotation swaps dimensions and reapplies gravity', () => {
  const board = [
    [1, 0, 2],
    [0, 1, 0],
  ];

  assert.deepEqual(rotateBoard(board, 1), [
    [0, 0],
    [0, 1],
    [1, 2],
  ]);
});

test('a chaos transform that gives both players a line is lost by the mover', () => {
  const board = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [2, 2, 2, 0],
    [1, 1, 1, 0],
  ];

  const outcome = resolveActionOutcome(board, 3, RED, ACTION_FLIP);
  assert.equal(outcome.status, 'won');
  assert.equal(outcome.winner, YELLOW);
  assert.equal(outcome.simultaneousWin, true);
});

test('position identity includes the player, connect length, and chaos rules', () => {
  const board = [[0, 0], [1, 2]];
  const base = positionKey(board, RED, 4, false);

  assert.notEqual(base, positionKey(board, YELLOW, 4, false));
  assert.notEqual(base, positionKey(board, RED, 3, false));
  assert.notEqual(base, positionKey(board, RED, 4, true));
});

test('configuration is clamped to playable ranges', () => {
  assert.deepEqual(normalizeConfig({
    rows: 3,
    cols: 40,
    connect: 9,
    opponent: 'unknown',
    startingPlayer: 7,
    chaosMode: 1,
  }), {
    rows: 4,
    cols: 10,
    connect: 6,
    opponent: 'medium',
    startingPlayer: RED,
    chaosMode: true,
  });

  assert.equal(normalizeConfig({ rows: 4, cols: 4, connect: 6 }).connect, 4);
});

test('Perfect AI is accepted only for standard classic Connect Four', () => {
  assert.equal(normalizeConfig({
    rows: 6,
    cols: 7,
    connect: 4,
    opponent: 'perfect',
    chaosMode: false,
  }).opponent, 'perfect');

  for (const incompatible of [
    { rows: 5, cols: 7, connect: 4, chaosMode: false },
    { rows: 6, cols: 8, connect: 4, chaosMode: false },
    { rows: 6, cols: 7, connect: 5, chaosMode: false },
    { rows: 6, cols: 7, connect: 4, chaosMode: true },
  ]) {
    assert.equal(normalizeConfig({ ...incompatible, opponent: 'perfect' }).opponent, 'brutal');
  }
});
