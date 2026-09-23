import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createBoard, immediateWinningActions, legalActions } from '../src/engine.js';
import { createNeuralClient } from '../src/neural-client.js';
import { manageBackend, simulationsFor, recordSearch, searchOverran } from '../src/neural-runtime.js';
import { searchPosition, bestAction } from '../src/neural-search.js';
import { waitFor } from '../src/async-control.js';

const workerSource = readFileSync(new URL('../src/neural-worker.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../src/neural-app.js', import.meta.url), 'utf8');
const output = () => ({ policy: new Float32Array(13), value: new Float32Array(3), q: new Float32Array(39) });

// Run the real page client, worker dispatcher, backend manager and request
// controller. Only native inference and worker transport are replaced. A
// simulated clock makes slow-backend cases deterministic without real delays.
async function harness(t, { failAt = 1, cpuMs = 50, gpuBatchSize = 1, gpuMs = 1, hold = () => null } = {}) {
  let elapsed = 0, gpuCalls = 0, cpuCalls = 0, terminated = false, handler;
  const gpuBatchSizes = [], cpuBatchSizes = [];
  const worker = new EventTarget();
  worker.terminate = () => { terminated = true; };
  worker.postMessage = (data) => queueMicrotask(() => { if (!terminated) void handler({ data }); });
  const workerContext = {
    self: { addEventListener(_kind, callback) { handler = callback; }, postMessage(data) {
      queueMicrotask(() => { if (!terminated) worker.dispatchEvent(new MessageEvent('message', { data })); });
    } },
    // `gpuMs` is what one GPU call costs on the simulated clock; `hold(call)`
    // can keep a call running in the worker until the test lets it finish.
    loadNeuralNetwork: async (options) => manageBackend({ backend: 'webgpu', perEvaluation: 1, batchSize: gpuBatchSize,
      session: { release() {} }, async evaluate() {
        if (++gpuCalls >= failAt) throw new Error('Injected GPU loss');
        await hold(gpuCalls);
        elapsed += gpuMs;
        return output();
      }, async evaluateMany(items) {
        gpuBatchSizes.push(items.length);
        if (++gpuCalls >= failAt) throw new Error('Injected GPU loss');
        await hold(gpuCalls);
        elapsed += gpuMs;
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
    simulationsFor, recordSearch, searchOverran, immediateWinningActions, performance: { now: () => elapsed } };
  vm.runInNewContext(appSource.slice(appSource.indexOf('export async function')).replace('export ', ''), appContext);
  const position = { board: createBoard(10, 10), currentPlayer: 2, connect: 6, chaosMode: false };
  return { network, position, client, terminated: () => terminated, elapsed: () => elapsed,
    calls: () => ({ gpu: gpuCalls, cpu: cpuCalls, gpuBatchSizes, cpuBatchSizes }),
    // One request as the page makes it, reporting whatever it ends with.
    async request({ controller = new AbortController() } = {}) {
      const outcome = {};
      await appContext.runNeuralRequest({ controller, position }, {
        isCurrent: () => !controller.signal.aborted, shouldStop: () => false,
        onSearch() {}, onFraction() {},
        finish: (value) => { outcome.result = value; }, fail: (message) => { outcome.failure = message; },
      });
      return outcome;
    },
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

// Undo or a new round cancels the request while the worker is still running
// one of its network calls. The network is kept for the next move, and that
// move's first call waits for the abandoned one rather than colliding with it.
test('an abandoned search keeps the network, and the next request waits for its evaluation', async (t) => {
  let running, release;
  const reached = new Promise((resolve) => { running = resolve; });
  const held = new Promise((resolve) => { release = resolve; });
  const h = await harness(t, { failAt: Infinity, gpuBatchSize: 8,
    hold: (call) => (call === 3 ? (running(), held) : null) });
  const undo = new AbortController();
  const abandoned = h.request({ controller: undo });
  await reached;
  undo.abort();
  assert.deepEqual(await abandoned, {}, 'a cancelled request neither plays nor fails');
  assert.equal(h.client.state(), 'ready');
  const next = h.request();
  release();
  const { result, failure } = await next;
  assert.equal(failure, undefined);
  assert.equal(result.backend, 'webgpu');
  assert.equal(result.nodes, 512);
  assert.equal(h.terminated(), false, 'one worker served both requests');
});

// The count is sized when the move starts; a GPU that slows down after that
// (saturated by other work, it has spent 30-130 s on one move) must not hold
// the game. Every GPU batch here costs 500 ms on the simulated clock while the
// warm-up promised 1 ms a position, so 512 simulations would take 32.5 s.
test('a search that overruns three budgets stops, and the next move is sized to fit', async (t) => {
  const h = await harness(t, { failAt: Infinity, gpuBatchSize: 8, gpuMs: 500 });
  const { result } = await h.run();
  assert.ok(result.elapsedMs >= 4_500 && result.elapsedMs < 4_500 + 500,
    `stopped after ${result.elapsedMs} ms, within one batch of the cap`);
  assert.ok(result.nodes >= 32 && result.nodes < 512, `${result.nodes} simulations`);
  // Calibrated on what the stopped search really spent, the next move plans
  // fewer simulations and finishes them without reaching the cap.
  const planned = simulationsFor(h.network);
  assert.ok(planned < result.nodes, `${planned} planned after ${result.nodes}`);
  const next = (await h.run()).result;
  assert.equal(next.nodes, planned);
  assert.ok(next.elapsedMs < 4_500, `${next.elapsedMs} ms`);
});

test('the cap never cuts a search below its minimum lookahead', async (t) => {
  const h = await harness(t, { failAt: Infinity, gpuBatchSize: 8, gpuMs: 5_000 });
  const { result } = await h.run();
  assert.equal(result.nodes, 32, 'four batches, although the first already passed the cap');
  assert.equal(result.elapsedMs, 5 * 5_000);
});
