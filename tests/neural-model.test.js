import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  ACTION_DROP, RED, YELLOW, createBoard,
} from '../src/engine.js';
import { CANVAS, PLANES, planeBuffer, writePlanes } from '../src/neural-planes.js';
import { bestAction, searchPosition } from '../src/neural-search.js';

const here = dirname(fileURLToPath(import.meta.url));
const modelPath = join(here, '..', 'assets', 'neural', 'model.onnx');

// The exported network, run through the same encoder and search the page
// uses. Skipped when the model is not present, so a checkout without the
// large asset still passes.
let ort = null;
let session = null;
try {
  ort = await import('onnxruntime-web');
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(await readFile(modelPath),
    { executionProviders: ['wasm'] });
} catch {
  session = null;
}

const describe = session ? test : test.skip;

function evaluator() {
  const input = planeBuffer(1);
  return async (board, mover, _actions, connect, chaosMode) => {
    const rows = board.length;
    const cols = board[0].length;
    writePlanes(input, 0, rows, cols, connect, chaosMode, (row, column) => {
      const cell = board[rows - 1 - row][column];
      if (cell === 0) return 0;
      return cell === mover ? 1 : 2;
    });
    const tensor = new ort.Tensor('float32', input, [1, PLANES, CANVAS, CANVAS]);
    const outputs = await session.run({ planes: tensor });
    return { policy: outputs.policy.data, value: outputs.value.data, q: outputs.q.data };
  };
}

describe('the exported network takes an immediate win', async () => {
  const rows = 6;
  const board = createBoard(rows, 7);
  board[rows - 1][0] = RED;
  board[rows - 1][1] = RED;
  board[rows - 1][2] = RED;
  board[rows - 1][4] = YELLOW;
  board[rows - 1][5] = YELLOW;
  board[rows - 1][6] = YELLOW;
  const result = await searchPosition(
    { board, currentPlayer: RED, connect: 4, chaosMode: false },
    evaluator(), { simulations: 24 },
  );
  const move = bestAction(result);
  assert.equal(move.type, ACTION_DROP);
  assert.equal(move.column, 3);
});

describe('the exported network blocks an immediate threat', async () => {
  const rows = 6;
  const board = createBoard(rows, 7);
  board[rows - 1][6] = YELLOW;
  board[rows - 2][6] = YELLOW;
  board[rows - 3][6] = YELLOW;
  board[rows - 1][0] = RED;
  board[rows - 1][1] = RED;
  const result = await searchPosition(
    { board, currentPlayer: RED, connect: 4, chaosMode: false },
    evaluator(), { simulations: 32 },
  );
  assert.equal(bestAction(result).column, 6);
});

describe('the network prefers the centre of an empty board', async () => {
  // Every strong Connect Four player opens in the middle; a network that
  // does not has been fed a mis-encoded board.
  const board = createBoard(6, 7);
  const result = await searchPosition(
    { board, currentPlayer: RED, connect: 4, chaosMode: false },
    evaluator(), { simulations: 48 },
  );
  const move = bestAction(result);
  assert.equal(move.type, ACTION_DROP);
  assert.ok(move.column >= 2 && move.column <= 4,
    `opened in column ${move.column}, which is not near the centre`);
});
