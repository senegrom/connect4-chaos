import test from 'node:test';
import assert from 'node:assert/strict';
import { ACTION_DROP, RED, YELLOW } from '../src/engine.js';
import {
  boardToBitboard,
  chooseBitboardMove,
  isBitboardPosition,
  possibleNonLosingMoves,
} from '../src/bitboard.js';

function emptyBoard() {
  return Array.from({ length: 6 }, () => Array(7).fill(0));
}

function position(board, currentPlayer = YELLOW, overrides = {}) {
  return { board, currentPlayer, connect: 4, chaosMode: false, ...overrides };
}

function drop(board, column, player) {
  for (let row = 5; row >= 0; row -= 1) {
    if (board[row][column] === 0) {
      board[row][column] = player;
      return row;
    }
  }
  return -1;
}

function winningColumns(board, player) {
  const wins = [];
  const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];
  for (let column = 0; column < 7; column += 1) {
    const copy = board.map((row) => [...row]);
    const row = drop(copy, column, player);
    if (row < 0) continue;
    let won = false;
    for (const [dr, dc] of directions) {
      let count = 1;
      for (const sign of [-1, 1]) {
        for (let step = 1; step < 4; step += 1) {
          const nextRow = row + dr * step * sign;
          const nextColumn = column + dc * step * sign;
          if (copy[nextRow]?.[nextColumn] !== player) break;
          count += 1;
        }
      }
      if (count >= 4) won = true;
    }
    if (won) wins.push(column);
  }
  return wins;
}

test('eligibility is limited to gravity-valid classic 7x6 connect-four positions', () => {
  const board = emptyBoard();
  assert.equal(isBitboardPosition(position(board)), true);
  assert.equal(isBitboardPosition(position(board, YELLOW, { chaosMode: true })), false);
  assert.equal(isBitboardPosition(position(board, YELLOW, { connect: 5 })), false);
  assert.equal(isBitboardPosition(position(Array.from({ length: 5 }, () => Array(7).fill(0)))), false);
  board[4][0] = RED;
  assert.equal(boardToBitboard(board, YELLOW), null);
});

test('the bitboard solver takes immediate wins and blocks immediate losses', () => {
  const winning = emptyBoard();
  winning[5] = [YELLOW, YELLOW, YELLOW, 0, RED, 0, 0];
  assert.deepEqual(
    chooseBitboardMove(position(winning), { maximumDepth: 1 }).action,
    { type: ACTION_DROP, column: 3 },
  );

  const blocking = emptyBoard();
  blocking[5] = [RED, RED, RED, 0, YELLOW, 0, 0];
  assert.deepEqual(
    chooseBitboardMove(position(blocking), { maximumDepth: 1 }).action,
    { type: ACTION_DROP, column: 3 },
  );
});

test('the opening move is centre-first and the caller board is never mutated', () => {
  const board = emptyBoard();
  const before = board.map((row) => [...row]);
  const result = chooseBitboardMove(position(board, RED), { maximumDepth: 8 });
  assert.deepEqual(result.action, { type: ACTION_DROP, column: 3 });
  assert.deepEqual(board, before);
  assert.equal(result.solver, 'bitboard');
  assert.equal(result.depth, 8);
});

test('legacy time-budget options cannot curtail a fixed-depth bitboard search', () => {
  const result = chooseBitboardMove(position(emptyBoard(), RED), {
    maximumDepth: 6,
    timeBudgetMs: 0,
  });
  assert.equal(result.depth, 6);
  assert.ok(result.nodes > 0);
});

test('non-losing move generation rejects a move that permits an immediate reply', () => {
  const board = emptyBoard();
  board[5] = [RED, RED, RED, 0, YELLOW, 0, 0];
  const bitboard = boardToBitboard(board, YELLOW);
  const moves = possibleNonLosingMoves(bitboard);
  assert.notEqual(moves, 0n);
  const result = chooseBitboardMove(position(board), { maximumDepth: 1 });
  const next = board.map((row) => [...row]);
  drop(next, result.action.column, YELLOW);
  assert.deepEqual(winningColumns(next, RED), []);
});

test('late positions are searched all the way to a proven terminal result', () => {
  const board = [
    [0, RED, YELLOW, 0, RED, 0, RED],
    [0, RED, RED, 0, YELLOW, YELLOW, YELLOW],
    [0, RED, YELLOW, YELLOW, RED, RED, RED],
    [0, YELLOW, RED, YELLOW, YELLOW, YELLOW, RED],
    [0, RED, YELLOW, YELLOW, YELLOW, RED, YELLOW],
    [YELLOW, RED, RED, RED, YELLOW, RED, YELLOW],
  ];
  const result = chooseBitboardMove(position(board, RED), {
    difficulty: 'brutal',
    exactThreshold: 8,
  });
  assert.equal(result.solved, true);
  assert.equal(result.depth, 8);
  assert.equal(result.score, 0);
  assert.deepEqual(result.action, { type: ACTION_DROP, column: 3 });
});

test('completed-depth progress is monotone and finishes at the returned depth', () => {
  const updates = [];
  const result = chooseBitboardMove(position(emptyBoard(), RED), {
    maximumDepth: 6,
    onIteration(update) { updates.push(update); },
  });
  assert.deepEqual(updates.map((update) => update.depth), [1, 2, 3, 4, 5, 6]);
  assert.equal(updates.at(-1).depth, result.depth);
  assert.deepEqual(updates.at(-1).action, result.action);
});
