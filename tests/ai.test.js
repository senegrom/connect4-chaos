import test from 'node:test';
import assert from 'node:assert/strict';

import { chooseMove, evaluateBoard } from '../src/ai.js';
import { ACTION_DROP, ACTION_FLIP, RED, YELLOW, applyAction, positionKey } from '../src/engine.js';

function emptyBoard(rows = 6, cols = 7) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

function position(board, overrides = {}) {
  const currentPlayer = overrides.currentPlayer ?? YELLOW;
  const connect = overrides.connect ?? 4;
  const chaosMode = overrides.chaosMode ?? false;
  return {
    board,
    currentPlayer,
    connect,
    chaosMode,
    repetitionCounts: [[positionKey(board, currentPlayer, connect, chaosMode), 1]],
  };
}

test('easy AI takes an immediate win', () => {
  const board = emptyBoard();
  board[5] = [YELLOW, YELLOW, YELLOW, 0, RED, 0, 0];

  const result = chooseMove(position(board), { difficulty: 'easy', random: () => 0 });
  assert.deepEqual(result.action, { type: ACTION_DROP, column: 3 });
});

test('easy AI blocks an immediate human win', () => {
  const board = emptyBoard();
  board[5] = [RED, RED, RED, 0, YELLOW, 0, 0];

  const result = chooseMove(position(board), { difficulty: 'easy', random: () => 0 });
  assert.deepEqual(result.action, { type: ACTION_DROP, column: 3 });
});


test('easy AI recognises a winning chaos transform', () => {
  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, RED, 0, 0],
    [0, 0, YELLOW, YELLOW, RED, 0, 0],
    [YELLOW, YELLOW, RED, RED, RED, 0, 0],
  ];

  const result = chooseMove(position(board, { chaosMode: true }), {
    difficulty: 'easy',
    random: () => 0,
  });
  assert.deepEqual(result.action, { type: ACTION_FLIP });
});

test('iterative-deepening AI returns a legal move within a small budget', () => {
  const board = emptyBoard();
  const result = chooseMove(position(board), {
    difficulty: 'hard',
    timeBudgetMs: 80,
    maximumDepth: 5,
  });

  assert.ok(result.action);
  assert.equal(result.action.type, ACTION_DROP);
  assert.ok(result.action.column >= 0 && result.action.column < 7);
  assert.ok(result.depth >= 1);
  assert.ok(result.nodes > 0);
});

test('the evaluator rewards central control for the AI', () => {
  const centreBoard = emptyBoard();
  const edgeBoard = emptyBoard();
  centreBoard[5][3] = YELLOW;
  edgeBoard[5][0] = YELLOW;

  assert.ok(evaluateBoard(centreBoard, 4, YELLOW) > evaluateBoard(edgeBoard, 4, YELLOW));
});

test('the selected move can be applied to the searched position', () => {
  const board = emptyBoard();
  board[5][3] = RED;
  const result = chooseMove(position(board), {
    difficulty: 'medium',
    timeBudgetMs: 60,
    maximumDepth: 4,
  });

  const applied = applyAction(board, result.action, YELLOW);
  assert.ok(applied);
  assert.equal(applied.board.flat().filter((cell) => cell === YELLOW).length, 1);
});
