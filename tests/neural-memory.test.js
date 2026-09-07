import test from 'node:test';
import assert from 'node:assert/strict';
import { preferNeuralWasm } from '../src/neural-gpu-guard.js';
import { createNeuralClient } from '../src/neural-client.js';
import { startBackend, manageBackend } from '../src/neural-runtime.js';
import { fetchWithProgress } from '../src/download-gate.js';
import { createBoard } from '../src/engine.js';

const tick = () => new Promise((resolve) => setImmediate(resolve));
const deferred = () => {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
};

test('iPhones and iPads use WASM even when they advertise WebGPU', async () => {
  for (const navigator of [
    { userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 26_6 like Mac OS X)', gpu: {} },
    { userAgent: 'Mozilla/5.0 (iPad; CPU OS 26_6 like Mac OS X)', gpu: {} },
    { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15)', platform: 'MacIntel', maxTouchPoints: 5, gpu: {} },
  ]) {
    assert.equal(preferNeuralWasm(navigator), true);
    let request;
    const worker = new EventTarget();
    worker.terminate = () => {};
    worker.postMessage = (data) => { request = data; };
    const client = createNeuralClient({ createWorker: () => worker, allowWebgpu: !preferNeuralWasm(navigator),
      guard: { avoided: () => false } });
    const loading = client.load();
    assert.equal(request.allowWebgpu, false);
    worker.dispatchEvent(new MessageEvent('message', { data: {
      kind: 'result', id: request.id, result: { backend: 'wasm' },
    } }));
    await loading;
    client.invalidate();
  }
  assert.equal(preferNeuralWasm({ platform: 'MacIntel', maxTouchPoints: 0 }), false);
  assert.equal(preferNeuralWasm({ platform: 'Linux', maxTouchPoints: 5 }), false);
});

test('warm-up and repeated inference dispose every tensor and return independent logits', async () => {
  let live = 0;
  let fail = false;
  class Tensor {
    constructor(_type, data) { this.data = data; live++; }
    dispose() { live--; this.data.fill(NaN); }
  }
  const session = { release() {}, async run() {
    if (fail) throw new Error('inference failed');
    return Object.fromEntries([['policy', 13], ['value', 3], ['q', 39]]
      .map(([name, size]) => [name, new Tensor('float32', new Float32Array(size).fill(1))]));
  } };
  const network = await startBackend({ Tensor, InferenceSession: { create: async () => session } }, null, 'wasm');
  assert.equal(live, 0, 'warm-up must release its inputs and outputs');
  for (let i = 0; i < 100; i++) {
    const output = await network.evaluate(createBoard(6, 7), 1, [], 4, false);
    assert.equal(live, 0);
    assert.ok([...output.policy, ...output.value, ...output.q].every((value) => value === 1));
  }
  fail = true;
  await assert.rejects(network.evaluate(createBoard(6, 7), 1, [], 4, false), /inference failed/);
  assert.equal(live, 0, 'a failed run must release its input too');
});

test('GPU loss drains inference and releases the GPU before creating a CPU session', async () => {
  const run = deferred(), release = deferred(), loss = deferred();
  const events = [];
  const gpu = { backend: 'webgpu', perEvaluation: 50,
    evaluate() { events.push('gpu run'); return run.promise; },
    session: { async release() { events.push('gpu release'); await release.promise; events.push('gpu freed'); } } };
  const cpu = { backend: 'wasm', perEvaluation: 20, evaluate: async () => 'cpu result',
    session: { release() { events.push('cpu release'); } } };
  const network = manageBackend(gpu, async () => { events.push('cpu create'); return cpu; }, { device: { lost: loss.promise } });
  const pending = network.evaluate();
  await tick();
  loss.resolve();
  await tick();
  assert.deepEqual(events, ['gpu run'], 'device loss must not start a concurrent backend');
  run.resolve('gpu result');
  assert.equal(await pending, 'gpu result');
  const next = network.evaluate();
  await tick();
  assert.deepEqual(events, ['gpu run', 'gpu release']);
  release.resolve();
  assert.equal(await next, 'cpu result');
  assert.deepEqual(events, ['gpu run', 'gpu release', 'gpu freed', 'cpu create']);
  assert.equal(network.backend, 'wasm');
  network.dispose();
  await tick();
  assert.equal(events.at(-1), 'cpu release');
});

test('disposal waits for native inference and never creates a fallback afterwards', async () => {
  const run = deferred();
  let released = 0;
  const network = manageBackend({ backend: 'wasm', evaluate: () => run.promise,
    session: { release() { released++; } } }, () => assert.fail('unexpected fallback'));
  const pending = network.evaluate();
  await tick();
  network.dispose(); network.dispose();
  assert.equal(released, 0);
  run.resolve('done');
  await pending; await tick();
  assert.equal(released, 1);
  await assert.rejects(network.evaluate(), /disposed/);
});

test('a failed GPU run retries on CPU only after release, and late fallback is disposed', async () => {
  const replacement = deferred();
  let releasedGpu = 0, releasedCpu = 0, startedCpu = 0;
  const network = manageBackend({ backend: 'webgpu',
    evaluate: async () => { throw new Error('GPU failed'); },
    session: { release() { releasedGpu++; } },
  }, () => {
    assert.equal(releasedGpu, 1);
    startedCpu++;
    return replacement.promise;
  });
  const pending = network.evaluate();
  await tick();
  assert.equal(startedCpu, 1);
  network.dispose();
  replacement.resolve({ backend: 'wasm', session: { release() { releasedCpu++; } } });
  await assert.rejects(pending, /disposed/);
  await tick();
  assert.equal(releasedGpu, 1);
  assert.equal(releasedCpu, 1);
});

test('streaming downloads support cache-only reads and inaccurate size hints', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(new ReadableStream({
    start(controller) {
      controller.enqueue(new Uint8Array([1, 2]));
      controller.enqueue(new Uint8Array([3, 4, 5]));
      controller.close();
    },
  })));
  for (const expectedBytes of [0, 1, 3, 5, 8]) {
    for (const retain of [false, true]) {
      const progress = [];
      const result = await fetchWithProgress('https://example.test/model', (loaded) => progress.push(loaded), { expectedBytes, retain });
      if (retain) assert.deepEqual([...new Uint8Array(result)], [1, 2, 3, 4, 5]);
      else assert.equal(result, null);
      assert.equal(progress.at(-1), 5);
    }
  }
});
