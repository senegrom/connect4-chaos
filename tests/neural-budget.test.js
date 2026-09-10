import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createBoard, immediateWinningActions, legalActions } from '../src/engine.js';
import { createNeuralClient } from '../src/neural-client.js';
import { manageBackend, simulationsFor, recordSearch } from '../src/neural-runtime.js';
import { searchPosition, bestAction } from '../src/neural-search.js';
import { waitFor } from '../src/async-control.js';

const workerSource = readFileSync(new URL('../src/neural-worker.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../src/neural-app.js', import.meta.url), 'utf8');
const output = () => ({ policy: new Float32Array(13), value: new Float32Array(3), q: new Float32Array(39) });

// Run the real page client, worker dispatcher, backend manager and request
// controller. Only native inference and worker transport are replaced. A
// simulated clock makes slow-backend cases deterministic without real delays.
async function harness(t, { failAt = 1, cpuMs = 50, gpuBatchSize = 1 } = {}) {
  let elapsed = 0, gpuCalls = 0, cpuCalls = 0, terminated = false, handler;
  const gpuBatchSizes = [], cpuBatchSizes = [];
  const worker = new EventTarget();
  worker.terminate = () => { terminated = true; };
  worker.postMessage = (data) => queueMicrotask(() => { if (!terminated) void handler({ data }); });
  const workerContext = {
    self: { addEventListener(_kind, callback) { handler = callback; }, postMessage(data) {
      queueMicrotask(() => { if (!terminated) worker.dispatchEvent(new MessageEvent('message', { data })); });
    } },
    loadNeuralNetwork: async (options) => manageBackend({ backend: 'webgpu', perEvaluation: 1, batchSize: gpuBatchSize,
      session: { release() {} }, async evaluate() {
        if (++gpuCalls >= failAt) throw new Error('Injected GPU loss');
        elapsed += 1;
        return output();
      }, async evaluateMany(items) {
        gpuBatchSizes.push(items.length);
        if (++gpuCalls >= failAt) throw new Error('Injected GPU loss');
        elapsed += 1;
        return items.map(output);
      } }, async () => ({ backend: 'wasm', perEvaluation: cpuMs, batchSize: 1,
      session: { release() {} }, async evaluate() {
        cpuCalls++; cpuBatchSizes.push(1); elapsed += cpuMs;
        return output();
      }, async evaluateMany() {
        assert.fail('The CPU backend must receive one position at a time');
      } }), options),
  };
  vm.runInNewContext(workerSource.slice(workerSource.indexOf('let network =')), workerContext);
  const client = createNeuralClient({ createWorker: () => worker,
    guard: { avoided: () => false, failed() {} }, idleTimeoutMs: 0 });
  t.after(() => client.invalidate());
  const network = await client.load();
  const appContext = { neuralLoadState: () => client.state(), loadNeuralNetwork: (options) => client.load(options),
    invalidateNeuralNetwork: (value) => client.invalidate(value), waitFor, searchPosition, bestAction,
    simulationsFor, recordSearch, immediateWinningActions, performance: { now: () => elapsed } };
  vm.runInNewContext(appSource.slice(appSource.indexOf('export async function')).replace('export ', ''), appContext);
  const position = { board: createBoard(10, 10), currentPlayer: 2, connect: 6, chaosMode: false };
  return { network, position, calls: () => ({ gpu: gpuCalls, cpu: cpuCalls, gpuBatchSizes, cpuBatchSizes }),
    async run({ shouldStop = () => false } = {}) {
      let result;
      const fractions = [], searches = [];
      await appContext.runNeuralRequest({ controller: new AbortController(), position }, {
        isCurrent: () => true, shouldStop,
        onSearch: (search) => searches.push(search), onFraction: (fraction) => fractions.push(fraction),
        finish: (value) => { result = value; }, fail: (message) => assert.fail(message),
      });
      assert.ok(legalActions(position.board, false).some((action) => action.column === result.action.column));
      assert.ok(fractions.every((value) => value >= 0 && value <= 1));
      return { result, searches };
    },
  };
}

test('worker fallback transfers CPU timing without overwriting later page calibration', async (t) => {
  const h = await harness(t);
  const p = h.position;
  await h.network.evaluate(p.board, p.currentPlayer, legalActions(p.board, false), p.connect, false);
  assert.equal(h.network.backend, 'wasm');
  assert.equal(h.network.perEvaluation, 50);
  recordSearch(h.network, 1600, 20);
  const calibrated = h.network.perEvaluation;
  await h.network.evaluate(p.board, p.currentPlayer, legalActions(p.board, false), p.connect, false);
  assert.equal(h.network.perEvaluation, calibrated);
});

test('fallback during the root evaluation shrinks the active search to the CPU budget', async (t) => {
  const h = await harness(t);
  const { result, searches } = await h.run();
  assert.equal(result.nodes, 30);
  assert.equal(result.evaluations, 31);
  assert.equal(h.calls().cpu, 31);
  assert.equal(result.elapsedMs, 1550);
  assert.match(searches.at(-1).note, /30 simulations on wasm/);
});

test('late fallback stops after the in-flight simulation if the CPU budget is already spent', async (t) => {
  const h = await harness(t, { failAt: 50 });
  const { result } = await h.run();
  assert.equal(result.nodes, 49);
  assert.equal(result.evaluations, 50);
  assert.equal(h.calls().cpu, 1);
  assert.equal(h.network.perEvaluation, 50, 'fast GPU work must not dilute the CPU measurement');
  assert.equal((await h.run()).result.nodes, 30, 'the next turn also uses the CPU budget');
});

test('very slow fallback keeps minimum lookahead and Move now can still stop earlier', async (t) => {
  const h = await harness(t, { cpuMs: 800 });
  assert.equal((await h.run()).result.nodes, 2);
  assert.equal((await h.run({ shouldStop: () => true })).result.nodes, 1);
});

test('a healthy GPU retains its full search budget', async (t) => {
  const h = await harness(t, { failAt: Infinity });
  const { result } = await h.run();
  assert.equal(result.nodes, 512);
  assert.equal(result.backend, 'webgpu');
  assert.equal(h.calls().cpu, 0);
});

test('fallback at the root changes batch size before collecting any leaves', async (t) => {
  const h = await harness(t, { gpuBatchSize: 8 });
  const { result } = await h.run();
  assert.equal(result.nodes, 30);
  assert.equal(result.evaluations, 31);
  assert.equal(h.network.batchSize, 1);
  assert.equal(h.calls().cpu, 31);
  assert.deepEqual(h.calls().gpuBatchSizes, []);
});

test('an in-flight GPU batch retries as single CPU evaluations and shrinks the active budget', async (t) => {
  const h = await harness(t, { gpuBatchSize: 8, failAt: 2 });
  const { result, searches } = await h.run();
  assert.equal(result.nodes, 30);
  assert.equal(result.evaluations, 31);
  assert.equal(h.network.batchSize, 1);
  assert.equal(h.network.perEvaluation, 50);
  assert.deepEqual(h.calls().gpuBatchSizes, [8]);
  assert.equal(h.calls().cpu, 30);
  assert.ok(h.calls().cpuBatchSizes.every((size) => size === 1));
  assert.match(searches.at(-1).note, /30 simulations on wasm/);
  assert.equal((await h.run()).result.nodes, 30);
});

test('late batch fallback finishes only the in-flight batch after the CPU budget is spent', async (t) => {
  const h = await harness(t, { gpuBatchSize: 8, failAt: 8 });
  const { result } = await h.run();
  assert.equal(result.nodes, 56);
  assert.equal(h.calls().cpu, 8);
  assert.equal(h.network.batchSize, 1);
  assert.equal((await h.run()).result.nodes, 30);
});
