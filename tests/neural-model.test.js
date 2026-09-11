import assert from 'node:assert/strict';
import test from 'node:test';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  ACTION_DROP, RED, YELLOW, createBoard,
} from '../src/engine.js';
import { startBackend } from '../src/neural-runtime.js';
import { bestAction, searchPosition } from '../src/neural-search.js';
import { readModelBytes } from '../scripts/model-source.mjs';

// The network is served from R2 rather than committed, so it may or may not
// be on disk here; `readModelBytes` finds it if it is and answers null if it
// is not, which skips these tests rather than failing them.
const readModel = async () => {
  const bytes = await readModelBytes({ allowDownload: process.env.NEURAL_MODEL_DOWNLOAD === '1' });
  if (!bytes) {
    const missing = new Error('no local model');
    missing.code = 'ENOENT';
    throw missing;
  }
  return bytes;
};

// The exported network, run through the same encoder and search the page
// uses. Skipped when the model is not present, so a checkout without the
// large asset still passes.
let ort = null;
let session = null;
let backend = null;
try {
  ort = await import('onnxruntime-web');
  ort.env.wasm.numThreads = 1;
  backend = await startBackend(ort, await readModel(), 'wasm');
  session = backend.session;
} catch (error) {
  if (!['ERR_MODULE_NOT_FOUND', 'ENOENT'].includes(error.code)) throw error;
  session = null;
}

const describe = session ? test : test.skip;

function evaluator() {
  return backend.evaluate;
}

test.after(async () => { await session?.release(); });

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
