import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ACTION_DROP,
  ACTION_FLIP,
  EMPTY,
  RED,
  YELLOW,
  applyAction,
  createBoard,
  flipBoard,
  hasWinFrom,
  legalActions,
  normalizeConfig,
  otherPlayer,
  positionKey,
  resolveActionOutcome,
  rotateBoard,
  supportsPerfectClassicConfig,
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

test('Perfect AI is accepted for the classic boards with a shipped policy, 4x4 through 7x6', () => {
  for (let rows = 4; rows <= 7; rows += 1) {
    for (let cols = 4; cols <= 7; cols += 1) {
      const shipped = !(rows === 7 && cols === 7);
      assert.equal(supportsPerfectClassicConfig(rows, cols, 4, false), shipped);
      assert.equal(normalizeConfig({
        rows,
        cols,
        connect: 4,
        opponent: 'perfect',
        chaosMode: false,
      }).opponent, shipped ? 'perfect' : 'brutal');
    }
  }

  for (const incompatible of [
    { rows: 4, cols: 8, connect: 4, chaosMode: false },
    { rows: 8, cols: 7, connect: 4, chaosMode: false },
    { rows: 6, cols: 7, connect: 5, chaosMode: false },
    { rows: 6, cols: 7, connect: 4, chaosMode: true },
  ]) {
    assert.equal(supportsPerfectClassicConfig(
      incompatible.rows,
      incompatible.cols,
      incompatible.connect,
      incompatible.chaosMode,
    ), false);
    assert.equal(normalizeConfig({ ...incompatible, opponent: 'perfect' }).opponent, 'brutal');
  }
});

test('player helpers reject invalid player identifiers', () => {
  assert.throws(() => otherPlayer(EMPTY), /Red or Yellow/);
  assert.throws(
    () => applyAction(createBoard(4, 4), { type: ACTION_DROP, column: 0 }, 7),
    /Red or Yellow/,
  );
});

test('finished boards expose no legal actions, including Chaos transforms', () => {
  const board = [
    [RED, YELLOW],
    [YELLOW, RED],
  ];
  assert.deepEqual(legalActions(board, false), []);
  assert.deepEqual(legalActions(board, true), []);
});

test('outcome resolution rejects ambiguous or unknown actions', () => {
  const board = createBoard(4, 4);
  assert.throws(
    () => resolveActionOutcome(board, 4, RED, ACTION_DROP),
    /dropped piece location/,
  );
  assert.throws(
    () => resolveActionOutcome(board, 4, RED, ACTION_DROP, { row: 3, column: 0 }),
    /identify the mover/,
  );
  assert.throws(
    () => resolveActionOutcome(board, 5, RED, ACTION_FLIP),
    /fits the board/,
  );
  assert.throws(
    () => resolveActionOutcome(board, 4, RED, 'teleport'),
    /Unknown action type/,
  );

  const dropped = applyAction(board, { type: ACTION_DROP, column: 0 }, RED);
  assert.equal(
    resolveActionOutcome(dropped.board, 4, RED, ACTION_DROP, {
      row: dropped.row,
      column: dropped.column,
    }).status,
    'playing',
  );
});
