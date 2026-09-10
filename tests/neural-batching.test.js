import assert from 'node:assert/strict';
import test from 'node:test';

import { createNeuralClient } from '../src/neural-client.js';
import { createGpuGuard } from '../src/neural-gpu-guard.js';
import { searchPosition } from '../src/neural-search.js';
import { RED, createBoard } from '../src/engine.js';

// Batch evaluation is a capability of the backend, not an assumption about
// it: a worker whose network cannot batch must never be handed a batch, and
// the search must still play without one.

const output = () => ({ policy: new Float32Array(13), value: new Float32Array(3), q: new Float32Array(39) });

class FakeWorker extends EventTarget {
  calls = [];

  constructor(batched) { super(); this.batched = batched; }

  postMessage(message) {
    this.calls.push(message);
    if (message.kind === 'load') {
      this.send({ kind: 'result', id: message.id,
        result: { backend: 'wasm', perEvaluation: 10, batched: this.batched,
          batchSize: this.batched ? 8 : 1 } });
    } else if (message.kind === 'evaluate') {
      this.send({ kind: 'result', id: message.id, result: output() });
    } else if (message.kind === 'evaluateMany') {
      this.send({ kind: 'result', id: message.id, result: message.args[0].map(output) });
    }
  }

  terminate() {}

  send(data) { this.dispatchEvent(new MessageEvent('message', { data })); }
}

function clientFor(batched) {
  const workers = [];
  const client = createNeuralClient({
    createWorker: () => { const worker = new FakeWorker(batched); workers.push(worker); return worker; },
    guard: createGpuGuard({ getStorage: () => undefined }),
    downloadTimeoutMs: 1000, evaluationTimeoutMs: 1000, idleTimeoutMs: 0,
  });
  return { client, workers };
}

test('a worker that can batch offers batching to the page', async () => {
  const { client } = clientFor(true);
  const network = await client.load();
  assert.equal(typeof network.evaluateMany, 'function');
  const outputs = await network.evaluateMany([{}, {}, {}]);
  assert.equal(outputs.length, 3);
  client.invalidate(network);
});

test('a worker that cannot batch offers only single evaluation', async () => {
  const { client } = clientFor(false);
  const network = await client.load();
  assert.equal(network.evaluateMany, undefined);
  assert.equal(network.batchSize, 1);
  client.invalidate(network);
});

// WebAssembly - every iPhone and iPad - evaluates one position at a time.
// A batch there only makes a single call block that much longer, and the
// warm-up that measures it has to fit inside the startup budget.
test('a backend that wants one position at a time is never handed a batch', async () => {
  const { client } = clientFor(false);
  const network = await client.load();
  const sizes = [];
  const position = { board: createBoard(6, 7), currentPlayer: RED, connect: 4, chaosMode: false };
  await searchPosition(position, (...args) => { sizes.push(1); return network.evaluate(...args); },
    { simulations: 16, batchSize: network.batchSize, evaluateMany: network.evaluateMany ?? null });
  assert.ok(sizes.length > 0);
  client.invalidate(network);
});

test('the search plays without a batch evaluator, one leaf per call', async () => {
  let calls = 0;
  const position = { board: createBoard(6, 7), currentPlayer: RED, connect: 4, chaosMode: false };
  const result = await searchPosition(position, () => { calls += 1; return Promise.resolve(output()); },
    { simulations: 24 });
  assert.equal(result.completedSimulations, 24);
  assert.equal(calls, result.evaluations);
});

test('the search asks a batch evaluator for several leaves at once', async () => {
  const sizes = [];
  const position = { board: createBoard(6, 7), currentPlayer: RED, connect: 4, chaosMode: false };
  const result = await searchPosition(position, () => Promise.resolve(output()), {
    simulations: 64,
    evaluateMany: (items) => { sizes.push(items.length); return Promise.resolve(items.map(output)); },
  });
  assert.equal(result.completedSimulations, 64);
  assert.ok(sizes.length > 0, 'the batch evaluator was never used');
  assert.ok(Math.max(...sizes) > 1, `every call carried one leaf: ${sizes.join(',')}`);
  // Far fewer calls than simulations is the whole point of batching.
  assert.ok(sizes.length < 64 / 2, `${sizes.length} calls for 64 simulations`);
});
