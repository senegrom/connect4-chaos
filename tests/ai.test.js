import test from 'node:test';
import assert from 'node:assert/strict';

import { chooseMove, evaluateBoard } from '../src/ai.js';
import {
  ACTION_DROP,
  ACTION_FLIP,
  RED,
  YELLOW,
  applyAction,
  getDropRow,
  hasWinFrom,
  positionKey,
} from '../src/engine.js';

function emptyBoard(rows = 6, cols = 7) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

function winningDropColumns(board, player, connect = 4) {
  const columns = [];
  for (let column = 0; column < board[0].length; column += 1) {
    const row = getDropRow(board, column);
    if (row < 0) continue;
    board[row][column] = player;
    const wins = hasWinFrom(board, row, column, player, connect);
    board[row][column] = 0;
    if (wins) columns.push(column);
  }
  return columns;
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

test('depth-limited AI completes the requested search depth', () => {
  const board = emptyBoard();
  const result = chooseMove(position(board), {
    difficulty: 'hard',
    maximumDepth: 5,
  });

  assert.ok(result.action);
  assert.equal(result.action.type, ACTION_DROP);
  assert.ok(result.action.column >= 0 && result.action.column < 7);
  assert.ok(result.depth >= 1);
  assert.ok(result.nodes > 0);
});

test('search depth is not curtailed by a legacy time budget option', () => {
  const board = emptyBoard(4, 4);
  const result = chooseMove(position(board, { connect: 3 }), { difficulty: 'medium', timeBudgetMs: 0, maximumDepth: 4 });
  assert.equal(result.depth, 4);
  assert.ok(result.action);
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
    maximumDepth: 4,
  });

  const applied = applyAction(board, result.action, YELLOW);
  assert.ok(applied);
  assert.equal(applied.board.flat().filter((cell) => cell === YELLOW).length, 1);
});

test('tactical extension rejects a horizon move that concedes an immediate win', () => {
  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, YELLOW, 0, RED, RED, 0, 0],
    [YELLOW, RED, YELLOW, RED, YELLOW, 0, RED],
    [RED, RED, YELLOW, YELLOW, YELLOW, 0, RED],
    [YELLOW, YELLOW, RED, RED, RED, 0, YELLOW],
  ];

  const result = chooseMove(position(board), {
    difficulty: 'medium',
    maximumDepth: 1,
  });
  const next = applyAction(board, result.action, YELLOW);

  assert.deepEqual(result.action, { type: ACTION_DROP, column: 5 });
  assert.deepEqual(winningDropColumns(next.board, RED), []);
});

test('searched AI never concedes an immediate win when a safe move exists', () => {
  const board = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,YELLOW,0,RED,RED,0,0],[YELLOW,RED,YELLOW,RED,YELLOW,0,RED],[RED,RED,YELLOW,YELLOW,YELLOW,0,RED],[YELLOW,YELLOW,RED,RED,RED,0,YELLOW]];
  for (const difficulty of ['medium', 'hard']) {
    const result = chooseMove(position(board), { difficulty, maximumDepth: 2 });
    const next = applyAction(board, result.action, YELLOW);
    assert.deepEqual(winningDropColumns(next.board, RED), [], difficulty);
  }
});

test('the evaluator values a playable threat above the same floating shape', () => {
  const playable = emptyBoard();
  const floating = emptyBoard();
  playable[4] = [YELLOW, YELLOW, YELLOW, 0, 0, 0, 0];
  playable[5] = [RED, YELLOW, RED, RED, 0, 0, 0];
  floating[4] = [YELLOW, YELLOW, YELLOW, 0, 0, 0, 0];
  floating[5] = [RED, YELLOW, RED, 0, 0, 0, RED];

  assert.ok(evaluateBoard(playable, 4, YELLOW) > evaluateBoard(floating, 4, YELLOW));
});

test('iterative deepening reports completed depths and a principal variation', () => {
  const board = emptyBoard();
  board[5][3] = RED;
  const progress = [];
  const result = chooseMove(position(board), {
    difficulty: 'medium',
    maximumDepth: 6,
    onIteration(update) {
      progress.push(update);
    },
  });

  assert.ok(progress.length >= 1);
  assert.deepEqual(progress.map((update) => update.depth),
    [...progress.map((update) => update.depth)].sort((a, b) => a - b));
  assert.equal(progress.at(-1).depth, result.depth);
  assert.deepEqual(progress.at(-1).action, result.action);
  assert.ok(result.principalVariation.length >= 1);
  assert.deepEqual(result.principalVariation[0], result.action);
});

test('classic search leaves the caller board unchanged', () => {
  const board = emptyBoard();
  board[5] = [RED, YELLOW, RED, YELLOW, 0, 0, 0];
  const before = board.map((row) => [...row]);

  const result = chooseMove(position(board), {
    difficulty: 'brutal',
    maximumDepth: 6,
  });

  assert.ok(result.action);
  assert.deepEqual(board, before);
});

test('searched chaos positions still return a legal action without mutation', () => {
  const board = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, RED, 0, 0],
    [YELLOW, RED, YELLOW, 0],
  ];
  const before = board.map((row) => [...row]);

  const result = chooseMove(position(board, { chaosMode: true }), {
    difficulty: 'medium',
    maximumDepth: 2,
  });
  const applied = applyAction(board, result.action, YELLOW);

  assert.ok(applied);
  assert.deepEqual(board, before);
});
