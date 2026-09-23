import test from 'node:test';
import assert from 'node:assert/strict';
import { startBackend } from '../src/neural-runtime.js';
import { ACTIONS, CANVAS, PLANES } from '../src/neural-planes.js';
import { RED, YELLOW, createBoard } from '../src/engine.js';

const AREA = PLANES * CANVAS * CANVAS;

function drop(board, cells) {
  for (const [column, player] of cells) {
    for (let row = board.length - 1; row >= 0; row -= 1) {
      if (board[row][column] === 0) { board[row][column] = player; break; }
    }
  }
  return board;
}

// A stub session that records every input and answers each slot with a
// fingerprint of that slot's own planes, so a position packed into the wrong
// place - or an answer read back from the wrong place - both show up.
function recordingRuntime() {
  const inputs = [];
  class Tensor {
    constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
    dispose() {}
  }
  const session = {
    release() {},
    async run({ planes }) {
      inputs.push({ data: Float32Array.from(planes.data), dims: planes.dims });
      const count = planes.dims[0];
      const policy = new Float32Array(count * ACTIONS);
      for (let at = 0; at < count; at += 1) {
        let fingerprint = 0;
        for (let i = 0; i < AREA; i += 1) fingerprint += planes.data[at * AREA + i] * ((i % 97) + 1);
        policy[at * ACTIONS] = fingerprint;
      }
      return {
        policy: new Tensor('float32', policy),
        value: new Tensor('float32', new Float32Array(count * 3)),
        q: new Tensor('float32', new Float32Array(count * ACTIONS * 3)),
      };
    },
  };
  return { inputs, ort: { Tensor, InferenceSession: { create: async () => session } } };
}

// From 2026-09-10 to 2026-09-23 the batch wrote position `at` at element
// `at` instead of block `at`: every slot after the first reached the network
// empty, and the first held the last position shifted by a few cells. Only
// WebGPU batches, and every test used one position at a time, so nothing
// noticed while the search on WebGPU ran on noise.
test('a batch hands every position to the network in its own slot', async () => {
  const { inputs, ort } = recordingRuntime();
  const network = await startBackend(ort, null, 'webgpu');
  assert.equal(network.batchSize, 8);
  const items = [
    { board: drop(createBoard(6, 7), [[3, RED], [3, YELLOW], [2, RED]]), mover: YELLOW, connect: 4, chaosMode: false },
    { board: drop(createBoard(5, 5), [[0, RED], [1, YELLOW]]), mover: RED, connect: 3, chaosMode: true, repeated: 1 },
    { board: drop(createBoard(10, 10), [[9, YELLOW], [9, RED], [0, RED]]), mover: YELLOW, connect: 5, chaosMode: true, repeated: 2 },
  ];

  inputs.length = 0;
  const batched = await network.evaluateMany(items);
  const [batch] = inputs.splice(0);
  assert.deepEqual(batch.dims, [items.length, PLANES, CANVAS, CANVAS]);

  for (const [at, item] of items.entries()) {
    const single = await network.evaluate(item.board, item.mover, [], item.connect, item.chaosMode, item.repeated ?? 0);
    const [alone] = inputs.splice(0);
    const slot = batch.data.subarray(at * AREA, (at + 1) * AREA);
    assert.deepEqual(Array.from(slot), Array.from(alone.data), `slot ${at} must hold exactly its own position`);
    assert.ok(slot.some((value) => value !== 0), `slot ${at} must not be empty`);
    assert.equal(batched[at].policy[0], single.policy[0], `result ${at} must be read back from its own slot`);
  }
});
