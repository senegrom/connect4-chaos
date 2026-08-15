import assert from 'node:assert/strict';
import test from 'node:test';

import {
  CLASSIC_DRAW,
  CLASSIC_WIN,
  boardToClassicBitboard,
  canonicalClassicPosition,
  createClassicGeometry,
  isExactClassicPosition,
  solveClassicPosition,
} from '../src/classic-solver.js';
import { ACTION_DROP, EMPTY, RED, YELLOW } from '../src/engine.js';

function emptyBoard(rows, columns) {
  return Array.from({ length: rows }, () => Array(columns).fill(EMPTY));
}

function position(board, currentPlayer, connect = 4) {
  return { board, currentPlayer, connect, chaosMode: false };
}

function drop(board, column, player) {
  for (let row = board.length - 1; row >= 0; row -= 1) {
    if (board[row][column] === EMPTY) {
      board[row][column] = player;
      return row;
    }
  }
  return -1;
}

function hasLineFrom(board, row, column, player, connect) {
  for (const [deltaRow, deltaColumn] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    let count = 1;
    for (const sign of [-1, 1]) {
      for (let step = 1; step < connect; step += 1) {
        const nextRow = row + sign * step * deltaRow;
        const nextColumn = column + sign * step * deltaColumn;
        if (board[nextRow]?.[nextColumn] !== player) break;
        count += 1;
      }
    }
    if (count >= connect) return true;
  }
  return false;
}

function boardKey(board, player) {
  return `${player}:${board.map((row) => row.join('')).join('/')}`;
}

function independentValue(board, player, connect, memo = new Map()) {
  const key = boardKey(board, player);
  const cached = memo.get(key);
  if (cached !== undefined) return cached;

  let best = -2;
  for (let column = 0; column < board[0].length; column += 1) {
    const row = drop(board, column, player);
    if (row < 0) continue;
    let value;
    if (hasLineFrom(board, row, column, player, connect)) value = CLASSIC_WIN;
    else {
      const child = independentValue(
        board,
        player === RED ? YELLOW : RED,
        connect,
        memo,
      );
      value = child === CLASSIC_DRAW ? CLASSIC_DRAW : -child;
    }
    board[row][column] = EMPTY;
    if (value > best) best = value;
    if (best === CLASSIC_WIN) break;
  }

  if (best === -2) best = CLASSIC_DRAW;
  memo.set(key, best);
  return best;
}

function reachablePositions(rows, columns, connect) {
  const queue = [[emptyBoard(rows, columns), RED]];
  const seen = new Set();
  const positions = [];
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const [board, player] = queue[cursor];
    const key = boardKey(board, player);
    if (seen.has(key)) continue;
    seen.add(key);
    positions.push([board, player]);

    for (let column = 0; column < columns; column += 1) {
      const child = board.map((row) => [...row]);
      const row = drop(child, column, player);
      if (row < 0 || hasLineFrom(child, row, column, player, connect)) continue;
      queue.push([child, player === RED ? YELLOW : RED]);
    }
  }
  return positions;
}

function mirrorBoard(board) {
  return board.map((row) => [...row].reverse());
}

function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function randomLatePosition(rows, columns, connect, remaining, random) {
  for (let attempt = 0; attempt < 10_000; attempt += 1) {
    const board = emptyBoard(rows, columns);
    let player = RED;
    let moves = 0;
    let terminal = false;
    while (rows * columns - moves > remaining) {
      const legal = [];
      for (let column = 0; column < columns; column += 1) {
        if (board[0][column] === EMPTY) legal.push(column);
      }
      const column = legal[Math.floor(random() * legal.length)];
      const row = drop(board, column, player);
      moves += 1;
      if (hasLineFrom(board, row, column, player, connect)) {
        terminal = true;
        break;
      }
      player = player === RED ? YELLOW : RED;
    }
    if (!terminal) return { board, player };
  }
  throw new Error('Could not construct a non-terminal late classic position.');
}

test('exact classic geometry covers every configured board through 7x7', () => {
  for (let rows = 4; rows <= 7; rows += 1) {
    for (let columns = 4; columns <= 7; columns += 1) {
      for (let connect = 3; connect <= Math.min(6, Math.max(rows, columns)); connect += 1) {
        const geometry = createClassicGeometry(rows, columns, connect);
        assert.equal(geometry.rows, rows);
        assert.equal(geometry.columns, columns);
        assert.equal(geometry.connect, connect);
        assert.ok(geometry.columns * geometry.stride <= 56);
        assert.equal(
          isExactClassicPosition(position(emptyBoard(rows, columns), RED, connect)),
          true,
        );
      }
    }
  }
});

test('canonical keys fold horizontal reflection on rectangular boards', () => {
  const board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, RED, YELLOW, RED, 0, 0, 0],
  ];
  const first = boardToClassicBitboard(board, YELLOW, 4);
  const second = boardToClassicBitboard(mirrorBoard(board), YELLOW, 4);
  assert.equal(
    canonicalClassicPosition(first.geometry, first).key,
    canonicalClassicPosition(second.geometry, second).key,
  );
});

test('exact solver agrees with independent minimax on every reachable tiny position', () => {
  for (const [rows, columns, connect] of [[2, 3, 2], [3, 3, 3]]) {
    const memo = new Map();
    const positions = reachablePositions(rows, columns, connect);
    for (const [board, player] of positions) {
      const expected = independentValue(
        board.map((row) => [...row]),
        player,
        connect,
        memo,
      );
      const actual = solveClassicPosition(position(board, player, connect), {
        tableBits: 12,
        maximumNodes: 2_000_000,
      });
      assert.equal(actual.value, expected, boardKey(board, player));
      assert.equal(actual.solved, true);
      assert.ok(actual.action === null || actual.action.type === ACTION_DROP);
    }
  }
});

test('empty small-board outcomes match independently established exact values', () => {
  const cases = [
    [2, 2, 2, CLASSIC_WIN],
    [3, 3, 3, CLASSIC_DRAW],
    [4, 4, 3, CLASSIC_WIN],
    [4, 4, 4, CLASSIC_DRAW],
    [4, 5, 4, CLASSIC_DRAW],
  ];
  for (const [rows, columns, connect, expected] of cases) {
    const result = solveClassicPosition(
      position(emptyBoard(rows, columns), RED, connect),
      { tableBits: 20, maximumNodes: 5_000_000 },
    );
    assert.equal(result.value, expected, `${rows}x${columns} connect ${connect}`);
    assert.equal(result.solved, true);
  }
});

test('late exact results cover every dimension from 4x4 through 7x7', () => {
  const random = seededRandom(0x7c1a551c);
  for (let rows = 4; rows <= 7; rows += 1) {
    for (let columns = 4; columns <= 7; columns += 1) {
      for (let sample = 0; sample < 3; sample += 1) {
        const { board, player } = randomLatePosition(rows, columns, 4, 6, random);
        const expected = independentValue(
          board.map((row) => [...row]),
          player,
          4,
        );
        const actual = solveClassicPosition(position(board, player), {
          tableBits: 14,
          maximumNodes: 2_000_000,
        });
        assert.equal(actual.value, expected, `${rows}x${columns} sample ${sample}`);
        assert.equal(actual.solved, true);
      }
    }
  }
});

test('exact solving is immutable, reports progress, and fails closed at a node boundary', () => {
  const board = emptyBoard(4, 5);
  const before = board.map((row) => [...row]);
  const updates = [];
  const result = solveClassicPosition(position(board, RED), {
    tableBits: 18,
    maximumNodes: 2_000_000,
    onIteration(update) {
      updates.push(update);
    },
  });
  assert.deepEqual(board, before);
  assert.equal(updates[0].solved, false);
  assert.deepEqual(updates.at(-1), result);
  assert.equal(result.solved, true);
  assert.throws(
    () => solveClassicPosition(position(emptyBoard(5, 5), RED), {
      tableBits: 12,
      maximumNodes: 100,
    }),
    (error) => error?.code === 'CLASSIC_EXACT_NODE_LIMIT' && error.nodes === 101,
  );
});

test('malformed, wrong-turn, and bounded-depth requests cannot claim exact play', () => {
  const floating = emptyBoard(4, 4);
  floating[2][0] = RED;
  assert.equal(boardToClassicBitboard(floating, YELLOW, 4), null);

  const wrongTurn = emptyBoard(4, 4);
  wrongTurn[3][0] = RED;
  assert.throws(
    () => solveClassicPosition(position(wrongTurn, RED)),
    /side to move/,
  );
  assert.throws(
    () => solveClassicPosition(position(emptyBoard(4, 4), RED), { maximumDepth: 4 }),
    /does not accept a bounded-depth override/,
  );
  assert.throws(
    () => solveClassicPosition({
      ...position(emptyBoard(4, 4), RED),
      chaosMode: true,
    }),
    /non-Chaos board/,
  );
});
