import assert from 'node:assert/strict';
import test from 'node:test';

import {
  ACTION_DROP, RED, YELLOW, createBoard,
} from '../src/engine.js';
import { startBackend } from '../src/neural-runtime.js';
import { bestAction, searchPosition } from '../src/neural-search.js';
import { readModelBytes } from '../scripts/model-source.mjs';

// The network is served from R2 rather than committed, so it may or may not
// be on disk here; `readModelBytes` finds it if it is and answers null if it
// is not. Only that answer skips these tests, so a checkout without the large
// asset still passes - but never in CI, nor when a download was asked for,
// where a missing model is a failure. Any other error (no runtime, an
// unreadable NEURAL_MODEL, a corrupt download) fails the file too.
const NO_LOCAL_MODEL = Symbol('no local model');
const download = process.env.NEURAL_MODEL_DOWNLOAD === '1';
const required = download || Boolean(process.env.CI);

async function readModel() {
  const bytes = await readModelBytes({ allowDownload: download });
  if (bytes) return bytes;
  if (required) {
    throw new Error('The exported network is required here (CI or NEURAL_MODEL_DOWNLOAD=1) but could not be found or downloaded.');
  }
  return NO_LOCAL_MODEL;
}

// The exported network, run through the same encoder and search the page uses.
const model = await readModel();
let backend = null;
if (model !== NO_LOCAL_MODEL) {
  const ort = await import('onnxruntime-web');
  ort.env.wasm.numThreads = 1;
  backend = await startBackend(ort, model, 'wasm');
}

const describe = backend ? test : test.skip;

function evaluator() {
  return backend.evaluate;
}

test.after(async () => { await backend?.session.release(); });

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
