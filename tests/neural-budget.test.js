import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { createBoard, immediateWinningActions, legalActions } from '../src/engine.js';
import { createNeuralClient } from '../src/neural-client.js';
import {
  DOWNLOAD_BYTES, MIN_SIMULATIONS, manageBackend, minimumSimulations, simulationsFor, recordSearch, searchOverran,
  startNetwork, wasmRestart,
} from '../src/neural-runtime.js';
import { searchPosition, bestAction } from '../src/neural-search.js';
import { waitFor } from '../src/async-control.js';

const workerSource = readFileSync(new URL('../src/neural-worker.js', import.meta.url), 'utf8');
const appSource = readFileSync(new URL('../src/neural-app.js', import.meta.url), 'utf8');
const output = () => ({ policy: new Float32Array(13), value: new Float32Array(3), q: new Float32Array(39) });

// The real page client and worker dispatcher around `loadNeuralNetwork`, which
// stands in for the runtime's; only the worker transport is replaced, and a
// worker the client discards is replaced by a fresh one, as on the page.
// `clientOptions` go to the page client. The tab's GPU guard counts the
// failures it is told of, and avoids the GPU once it has heard of one.
function connect(t, loadNeuralNetwork, clientOptions = {}) {
  let gpuFailures = 0, terminated = false;
  const createWorker = () => {
    let ended = false, handler;
    const worker = new EventTarget();
    worker.terminate = () => { ended = true; terminated = true; };
    worker.postMessage = (data) => queueMicrotask(() => { if (!ended) void handler({ data }); });
    vm.runInNewContext(workerSource.slice(workerSource.indexOf('let network =')), {
      self: { addEventListener(_kind, callback) { handler = callback; }, postMessage(data) {
        queueMicrotask(() => { if (!ended) worker.dispatchEvent(new MessageEvent('message', { data })); });
      } },
      loadNeuralNetwork,
    });
    return worker;
  };
  const client = createNeuralClient({ createWorker,
    guard: { avoided: () => gpuFailures > 0, failed() { gpuFailures += 1; } }, idleTimeoutMs: 0, ...clientOptions });
  t.after(() => client.invalidate());
  // `terminated` says whether the client has ended any worker.
  return { client, terminated: () => terminated, gpuFailures: () => gpuFailures };
}

// Run the real page client, worker dispatcher, backend manager and request
// controller. Only native inference and worker transport are replaced. A
// simulated clock makes slow-backend cases deterministic without real delays.
// The network starts on a GPU, or on WebAssembly with `startOn: 'wasm'`.
// `readModel`, when given, makes the fallback the runtime's own wasmRestart,
// reading the model through it, and `rules` replace parts of the position
// every request searches.
async function harness(t, { failAt = 1, cpuMs = 50, gpuBatchSize = 1, gpuMs = 1, gpuWarmUpMs = 1,
  hold = () => null, readModel = null, clientOptions = {}, rules = {}, startOn = 'webgpu' } = {}) {
  let elapsed = 0, gpuCalls = 0, cpuCalls = 0;
  const gpuBatchSizes = [], gpuBatches = [], cpuBoards = [];
  // `gpuWarmUpMs` is what the GPU's warm-up measured for one position, and
  // `gpuMs` what one GPU call costs on the simulated clock; `hold(call)` can
  // keep a call running in the worker until the test lets it finish.
  const gpuBackend = () => ({ backend: 'webgpu', perEvaluation: gpuWarmUpMs, batchSize: gpuBatchSize,
    session: { release() {} }, async evaluate() {
      if (++gpuCalls >= failAt) throw new Error('Injected GPU loss');
      await hold(gpuCalls);
      elapsed += gpuMs;
      return output();
    }, async evaluateMany(items) {
      gpuBatchSizes.push(items.length);
      gpuBatches.push(items.map((item) => item.board));
      if (++gpuCalls >= failAt) throw new Error('Injected GPU loss');
      await hold(gpuCalls);
      elapsed += gpuMs;
      return items.map(output);
    } });
  const cpuBackend = () => ({ backend: 'wasm', perEvaluation: cpuMs, batchSize: 1,
    session: { release() {} }, async evaluate(board) {
      cpuCalls++; cpuBoards.push(board); elapsed += cpuMs;
      return output();
    }, async evaluateMany() {
      assert.fail('The CPU backend must receive one position at a time');
    } });
  const { client, terminated, gpuFailures } = connect(t, async (options) => (startOn === 'wasm'
    ? manageBackend(cpuBackend(), async () => assert.fail('WebAssembly has nothing to fall back to'), options)
    : manageBackend(gpuBackend(), readModel
      ? wasmRestart(null, undefined, options, { readModel, start: async (_ort, _bytes, provider, { onStage }) => {
        assert.equal(provider, 'wasm');
        onStage('create');
        onStage('warmup');
        return cpuBackend();
      } })
      : async () => cpuBackend(), options)), clientOptions);
  const network = await client.load();
  const appContext = { neuralLoadState: () => client.state(), loadNeuralNetwork: (options) => client.load(options),
    invalidateNeuralNetwork: (value) => client.invalidate(value), waitFor, searchPosition, bestAction,
    simulationsFor, recordSearch, searchOverran, minimumSimulations, immediateWinningActions,
    performance: { now: () => elapsed } };
  vm.runInNewContext(appSource.slice(appSource.indexOf('export async function')).replace('export ', ''), appContext);
  const position = { board: createBoard(10, 10), currentPlayer: 2, connect: 5, chaosMode: false, ...rules };
  return { network, position, client, terminated, gpuFailures, elapsed: () => elapsed,
    calls: () => ({ gpu: gpuCalls, cpu: cpuCalls, gpuBatchSizes, gpuBatches, cpuBoards }),
    // One request as the page makes it, reporting whatever it ends with.
    async request({ controller = new AbortController(), isCurrent = () => !controller.signal.aborted } = {}) {
      const outcome = {};
      await appContext.runNeuralRequest({ controller, position }, {
        isCurrent, shouldStop: () => false,
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
      return { result, searches, fractions };
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

test('a GPU fallback reports before it reads the model, then every read and each session stage', async () => {
  const reports = [];
  const restart = wasmRestart({ tag: 'ort' }, undefined, { onFallbackProgress: (report) => reports.push(report) }, {
    async readModel(_signal, onProgress) {
      assert.equal(reports.length, 1, 'the first report goes out before the model is read');
      onProgress(10, 20);
      onProgress(20, 20);
      return 'bytes';
    },
    async start(ort, bytes, provider, { onStage }) {
      assert.deepEqual([ort.tag, bytes, provider], ['ort', 'bytes', 'wasm']);
      onStage('create');
      onStage('warmup');
      return { backend: 'wasm' };
    },
  });
  assert.deepEqual(await restart(), { backend: 'wasm' });
  assert.deepEqual(reports, [
    { stage: 'model', loaded: 0, total: DOWNLOAD_BYTES.model },
    { stage: 'model', loaded: 10, total: 20 },
    { stage: 'model', loaded: 20, total: 20 },
    { stage: 'session', backend: 'wasm', phase: 'create' },
    { stage: 'session', backend: 'wasm', phase: 'warmup' },
  ]);
});

test('a fallback longer than one evaluation deadline survives, and the search says what it is doing', async (t) => {
  // The real worker routes the runtime's fallback reports to the request it
  // is serving, and the page client re-arms that request on each one. Six
  // reads 25 ms apart outlast the 60 ms evaluation deadline twice over.
  const readModel = async (_signal, onProgress) => {
    for (let read = 1; read <= 6; read += 1) {
      await new Promise((resolve) => { setTimeout(resolve, 25); });
      onProgress(read * 17_000_000, 102_000_000);
    }
    return new Uint8Array(0);
  };
  const h = await harness(t, { readModel, clientOptions: { evaluationTimeoutMs: 60, downloadStallMs: 1_000 } });
  const { result, searches } = await h.run();
  assert.equal(result.backend, 'wasm');
  assert.equal(h.terminated(), false, 'the worker was never timed out');
  const notes = searches.map((search) => search.note);
  assert.ok(notes.includes('Neural search · the GPU failed; reading the network for the CPU (51.0 of 102.0 MB)'), notes.join('\n'));
  assert.ok(notes.includes('Neural search · the GPU failed; starting the network on wasm'), notes.join('\n'));
  assert.match(notes.at(-1), /30 simulations on wasm/);
  assert.equal(h.network.onFallbackProgress, null, 'the finished request leaves no listener behind');
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

// One WebAssembly thread here took about 700 ms an evaluation, which the
// budget alone turned into two simulations a move.
test('every device plans the minimum at least, and a faster one keeps its larger budget', () => {
  // Sixteen is the floor asked for; the rest of this file goes by its name.
  assert.equal(simulationsFor({ backend: 'wasm', perEvaluation: 700 }), 16);
  for (const perEvaluation of [100, 300, 700, 5_000, 60_000]) {
    assert.equal(simulationsFor({ backend: 'wasm', perEvaluation }), MIN_SIMULATIONS, `${perEvaluation} ms`);
  }
  // Unmeasured, the CPU gets the minimum, where it had eight; WebGPU keeps 128.
  assert.equal(simulationsFor({ backend: 'wasm' }), MIN_SIMULATIONS);
  assert.equal(simulationsFor(null), MIN_SIMULATIONS);
  assert.equal(simulationsFor({ backend: 'webgpu' }), 128);
  // Above the minimum the budget decides, as before: about 1.5 s, up to 512.
  assert.equal(simulationsFor({ backend: 'wasm', perEvaluation: 1_500 / 17 }), 17);
  assert.equal(simulationsFor({ backend: 'wasm', perEvaluation: 30 }), 50);
  assert.equal(simulationsFor({ backend: 'webgpu', perEvaluation: 1 }), 512);
});

// A backend that evaluates its leaves in batches has its floor in batches,
// eight of them. Sixteen simulations in two batches of eight played fewer
// winning moves than the network's first choice alone, and 64 in eight
// batches as many as any small search did (tests/strength/neural-strength.mjs).
test('a network\'s minimum is sixteen simulations, or eight full batches where that is more', () => {
  const cpu = { backend: 'wasm', batchSize: 1 };
  const gpu = { backend: 'webgpu', batchSize: 8 };
  assert.equal(minimumSimulations(cpu), MIN_SIMULATIONS);
  assert.equal(minimumSimulations(gpu), 64);
  // A network that states no batch evaluates one position at a time.
  assert.equal(minimumSimulations({ backend: 'webgpu' }), MIN_SIMULATIONS);
  assert.equal(minimumSimulations(null), MIN_SIMULATIONS);
  // However slow, each plans its own minimum and no fewer.
  for (const perEvaluation of [100, 700, 60_000]) {
    assert.equal(simulationsFor({ ...cpu, perEvaluation }), MIN_SIMULATIONS, `${perEvaluation} ms on WebAssembly`);
    assert.equal(simulationsFor({ ...gpu, perEvaluation }), 64, `${perEvaluation} ms on WebGPU`);
  }
  // At 30 ms a position the budget buys 50 simulations: the CPU's plan, and
  // fewer than the GPU's eight batches.
  assert.equal(simulationsFor({ ...cpu, perEvaluation: 30 }), 50);
  assert.equal(simulationsFor({ ...gpu, perEvaluation: 30 }), 64);
  // Unmeasured, the CPU gets its minimum and WebGPU keeps 128.
  assert.equal(simulationsFor(cpu), MIN_SIMULATIONS);
  assert.equal(simulationsFor(gpu), 128);
  // Above its minimum the budget decides for a GPU too.
  assert.equal(simulationsFor({ ...gpu, perEvaluation: 1_500 / 65 }), 65);
  assert.equal(simulationsFor({ ...gpu, perEvaluation: 1 }), 512);
  // The rule is in batches, whatever their size: a wider batch would lift the
  // minimum past the unmeasured default as well.
  const wide = { backend: 'webgpu', batchSize: 32 };
  assert.equal(minimumSimulations(wide), 256);
  assert.equal(simulationsFor(wide), 256);
  assert.equal(simulationsFor({ ...wide, perEvaluation: 30 }), 256);
});

test('an overrun never stops a search before the minimum, whatever count it was given', () => {
  for (const simulations of [1, 2, 8, 15, 16, 24, 32, 512]) {
    assert.equal(searchOverran(Infinity, MIN_SIMULATIONS - 1, simulations), false, `${simulations} planned`);
  }
  assert.equal(searchOverran(Infinity, MIN_SIMULATIONS, 8), true);
  assert.equal(searchOverran(Infinity, MIN_SIMULATIONS, 24), false, 'a search planned for 24 runs them all');
  assert.equal(searchOverran(Infinity, 24, 24), true);
  assert.equal(searchOverran(Infinity, 31, 512), false);
  assert.equal(searchOverran(Infinity, 32, 512), true);
  assert.equal(searchOverran(4_499, 512, 512), false, 'not before three budgets');
  assert.equal(searchOverran(4_500, 32, 512), true);
  // The minimum is the network's: eight batches on WebGPU, which no overrun
  // cuts into, where 32 simulations used to be enough to stop.
  const gpu = { backend: 'webgpu', batchSize: 8 };
  for (const simulations of [16, 32, 64, 512]) {
    assert.equal(searchOverran(Infinity, 63, simulations, gpu), false, `${simulations} planned on WebGPU`);
    assert.equal(searchOverran(Infinity, 64, simulations, gpu), true, `${simulations} planned on WebGPU`);
  }
  assert.equal(searchOverran(4_499, 512, 512, gpu), false, 'not before three budgets');
  // WebAssembly keeps the bounds it had.
  const cpu = { backend: 'wasm', batchSize: 1 };
  assert.equal(searchOverran(Infinity, MIN_SIMULATIONS - 1, 512, cpu), false);
  assert.equal(searchOverran(Infinity, MIN_SIMULATIONS, MIN_SIMULATIONS, cpu), true);
  assert.equal(searchOverran(Infinity, 31, 512, cpu), false);
  assert.equal(searchOverran(Infinity, 32, 512, cpu), true);
});

// The floor is on the work, not on the time: a device this slow takes longer
// over its move, and Move now cannot make it search less.
test('a very slow fallback still searches the minimum, and Move now cannot cut it shorter', async (t) => {
  const h = await harness(t, { cpuMs: 800 });
  assert.equal((await h.run()).result.nodes, MIN_SIMULATIONS);
  const { result, searches } = await h.run({ shouldStop: () => true });
  assert.equal(result.nodes, MIN_SIMULATIONS);
  // The root and sixteen simulations, 800 ms each on the simulated clock.
  assert.equal(result.elapsedMs, (MIN_SIMULATIONS + 1) * 800);
  assert.equal(searches.at(-1).note, `Moving after ${MIN_SIMULATIONS} simulations, the minimum, on wasm`);
});

// The GPU fails on its first batch after the root. The CPU it moves to
// affords two simulations in the budget, and the batch it retries leaves
// eight done: the lowered count still keeps the minimum.
test('a fallback mid-search to a very slow CPU still searches the minimum', async (t) => {
  const h = await harness(t, { gpuBatchSize: 8, failAt: 2, cpuMs: 800 });
  const { result, searches } = await h.run();
  assert.equal(result.backend, 'wasm');
  assert.equal(result.nodes, MIN_SIMULATIONS);
  assert.deepEqual(h.calls().gpuBatchSizes, [8]);
  assert.equal(h.calls().cpu, MIN_SIMULATIONS, 'the dropped batch of eight, then eight more');
  assert.equal(searches.at(-1).note, `Neural search · up to ${MIN_SIMULATIONS} simulations on wasm`);
});

// A GPU too slow for the budget plans its own minimum, eight batches. When it
// fails on its second, the search goes by the CPU's budget and minimum, not
// by the 64 it began with: the batch the CPU retries makes sixteen.
test('a GPU at its minimum that fails mid-search finishes at the CPU\'s minimum', async (t) => {
  const h = await harness(t, { gpuBatchSize: 8, gpuWarmUpMs: 700, failAt: 3, cpuMs: 800 });
  const { result, searches } = await h.run();
  assert.deepEqual(searches.map(({ note }) => note), [
    'Neural search · up to 64 simulations on webgpu',
    `Neural search · up to ${MIN_SIMULATIONS} simulations on wasm`,
  ]);
  assert.equal(result.backend, 'wasm');
  assert.equal(result.nodes, MIN_SIMULATIONS);
  assert.deepEqual(h.calls().gpuBatchSizes, [8, 8]);
  assert.equal(h.calls().cpu, 8, 'the dropped batch of eight, and nothing more');
});

// Pressed early, Move now lowers the count to the minimum, and the search
// tells the page at its next simulation: the note says so, and the bar
// measures against the new count. On WebAssembly the minimum is sixteen.
test('Move now before the minimum runs on to exactly the minimum, and says so', async (t) => {
  const h = await harness(t, { startOn: 'wasm' });
  // Pressed after the root's evaluation and three simulations.
  const { result, searches, fractions } = await h.run({ shouldStop: () => h.calls().cpu > 3 });
  assert.equal(result.nodes, MIN_SIMULATIONS);
  assert.equal(result.evaluations, MIN_SIMULATIONS + 1);
  assert.deepEqual(searches.map(({ note, fraction }) => [note, fraction]), [
    ['Neural search · up to 30 simulations on wasm', undefined],
    [`Moving after ${MIN_SIMULATIONS} simulations, the minimum, on wasm`, 3 / MIN_SIMULATIONS],
  ]);
  assert.deepEqual(fractions, [1 / 30, 2 / 30, 3 / 30,
    ...Array.from({ length: MIN_SIMULATIONS - 3 }, (_, at) => (at + 4) / MIN_SIMULATIONS)]);
});

// On WebGPU the minimum is eight batches, each of them full, and the note
// names their 64 simulations. Sixteen, the minimum before, ended this search
// after the second batch.
test('Move now before the minimum on WebGPU runs on to eight full batches, and says how many', async (t) => {
  const h = await harness(t, { failAt: Infinity, gpuBatchSize: 8 });
  // Pressed after the root and the first batch of eight.
  const { result, searches, fractions } = await h.run({ shouldStop: () => h.calls().gpu > 1 });
  assert.equal(result.nodes, 64);
  assert.deepEqual(h.calls().gpuBatchSizes, new Array(8).fill(8));
  assert.equal(h.calls().gpu, 9, 'the root and eight batches');
  assert.deepEqual(searches.map(({ note, fraction }) => [note, fraction]), [
    ['Neural search · up to 512 simulations on webgpu', undefined],
    ['Moving after 64 simulations, the minimum, on webgpu', 8 / 64],
  ]);
  assert.deepEqual(fractions, [8 / 512, ...[16, 24, 32, 40, 48, 56, 64].map((done) => done / 64)]);
});

// The GPU fails on the batch after the press, and the search finishes on a
// CPU whose budget is 30 simulations. The move was waiting for the GPU's
// minimum and now waits for the CPU's, sixteen and not its budget, which the
// batch the CPU retried already reaches; the note for the new backend still
// says the move is coming.
test('a fallback after Move now takes the CPU\'s minimum and still says the move is coming', async (t) => {
  const h = await harness(t, { failAt: 3, gpuBatchSize: 8 });
  // Pressed after the root and the first batch of eight.
  const { result, searches } = await h.run({ shouldStop: () => h.calls().gpu > 1 });
  assert.equal(result.backend, 'wasm');
  assert.equal(result.nodes, MIN_SIMULATIONS);
  assert.equal(h.calls().cpu, 8, 'the dropped batch of eight, and nothing more');
  assert.deepEqual(searches.map(({ note }) => note), [
    'Neural search · up to 512 simulations on webgpu',
    'Moving after 64 simulations, the minimum, on webgpu',
    `Moving after ${MIN_SIMULATIONS} simulations, the minimum, on wasm`,
  ]);
});

// WebGPU evaluates eight leaves a call, and a batch runs whole. On a board
// this small the first batch finds only four, so the batch that passes the
// minimum ends past it: never before it, and less than one batch after.
test('Move now before the minimum on WebGPU ends with the batch that reaches it', async (t) => {
  const h = await harness(t, { failAt: Infinity, gpuBatchSize: 8, rules: { board: createBoard(4, 4), connect: 4 } });
  const { result, searches } = await h.run({ shouldStop: () => h.calls().gpu > 1 });
  assert.deepEqual(h.calls().gpuBatchSizes, [4, ...new Array(8).fill(8)]);
  assert.ok(result.nodes >= 64 && result.nodes < 64 + 8, `${result.nodes} simulations`);
  assert.equal(searches.at(-1).note, 'Moving after 64 simulations, the minimum, on webgpu');
});

test('Move now after the minimum stops the search at its next simulation', async (t) => {
  const h = await harness(t, { failAt: Infinity });
  const { result, searches } = await h.run({ shouldStop: () => h.calls().gpu > 20 });
  assert.equal(result.nodes, 20);
  assert.equal(searches.length, 1, 'nothing more to say: the move comes at once');
});

// Undo, a new round or a hidden tab: the request is no longer current, and
// its search stops at once, minimum or not.
test('a request that is no longer current stops before the minimum', async (t) => {
  const h = await harness(t, { failAt: Infinity });
  const outcome = await h.request({ isCurrent: () => h.calls().gpu < 4 });
  assert.deepEqual(outcome, {}, 'it neither plays nor fails');
  assert.equal(h.calls().gpu, 4, 'the root and three simulations, and nothing after');
});

// The worker moves a failing GPU to WebAssembly by itself, so the game plays
// on either way. Only the tab's guard remembers the failure once that worker
// is released - two idle minutes, another opponent, a Retry - and without it
// the next worker starts on the same GPU again.
test('a GPU that fails during a search reaches the tab guard once', async (t) => {
  const h = await harness(t);
  const { result } = await h.run();
  assert.equal(result.backend, 'wasm');
  assert.equal(h.gpuFailures(), 1);
});

test('a healthy GPU retains its full search budget', async (t) => {
  const h = await harness(t, { failAt: Infinity });
  const { result } = await h.run();
  assert.equal(result.nodes, 512);
  assert.equal(result.backend, 'webgpu');
  assert.equal(h.calls().cpu, 0);
  assert.equal(h.gpuFailures(), 0, 'nor is it reported as failed');
});

// The runtime as a start sees it. `create(provider, call)` decides each
// session it is asked for, in order: it can throw, or never settle, as the
// runtime does when its own files fail to arrive; otherwise the session
// answers every position with zeros. `created` lists each one's provider.
function runtime(create = () => {}) {
  class Tensor {
    constructor(_type, data, dims) { this.data = data; this.dims = dims; }
  }
  const session = { release() {}, async run({ planes }) {
    return Object.fromEntries([['policy', 13], ['value', 3], ['q', 39]]
      .map(([name, size]) => [name, new Tensor('float32', new Float32Array(planes.dims[0] * size))]));
  } };
  const created = [];
  const ort = { env: { wasm: {} }, Tensor, InferenceSession: { async create(_model, options) {
    created.push(options.executionProviders.join());
    await create(created.at(-1), created.length);
    return session;
  } } };
  return { ort, created };
}

// A start as the page makes it, on a browser that offers a GPU adapter: the
// real client and worker around the runtime's own startNetwork, which imports
// its runtime through `importRuntime`.
const ownNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
function start(t, importRuntime, clientOptions) {
  t.after(() => {
    if (ownNavigator) Object.defineProperty(globalThis, 'navigator', ownNavigator);
    else delete globalThis.navigator;
  });
  Object.defineProperty(globalThis, 'navigator', {
    value: { gpu: { requestAdapter: async () => ({}) } }, configurable: true,
  });
  return connect(t, (options) => startNetwork(null, undefined, options.onProgress, options, { importRuntime }),
    clientOptions);
}

// The runtime imports its loader and fetches its WebAssembly when its first
// session is created. That used to be the network's own session, so on a GPU
// a file that failed to arrive counted as a failed GPU, and the tab kept off
// WebGPU for a day over a network blip. The runtime now starts first, on no
// backend's account: whichever session is first here fails as that does. Its
// own module, which the load imports, never counted against the GPU.
test('a runtime file that fails to arrive during a WebGPU start fails the load, not the GPU', async (t) => {
  const failure = 'Failed to fetch dynamically imported module';
  for (const [file, imported] of [['the runtime\'s module', true], ['its loader or its WebAssembly', false]]) {
    let attempts = 0;
    const { ort, created } = runtime((provider) => {
      if (!imported && attempts === 1) {
        throw new Error(`no available backend found. ERR: [${provider}] TypeError: ${failure}`);
      }
    });
    const h = start(t, async () => {
      attempts += 1;
      if (imported && attempts === 1) throw new TypeError(failure);
      return ort;
    });
    await assert.rejects(h.client.load(), new RegExp(failure), file);
    assert.deepEqual(created, imported ? [] : ['wasm'], `${file}: the GPU is never tried`);
    assert.equal(h.gpuFailures(), 0, file);
    assert.equal(h.terminated(), true, `${file}: the worker is discarded`);
    assert.equal(h.client.state(), 'idle', file);
    // Retry starts a fresh worker. The file arrives, and the GPU is tried
    // as on a first visit: the tab holds nothing against it.
    const network = await h.client.load();
    assert.equal(network.backend, 'webgpu', `${file}: Retry`);
    assert.deepEqual(created.slice(-2), ['wasm', 'webgpu'], `${file}: Retry`);
  }
});

// A file that stalls leaves the runtime starting for ever. The 45 s it is
// given were the WebGPU session's, and their end a failed GPU.
test('a runtime that never starts during a WebGPU start times out as a failed load', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { ort, created } = runtime((_provider, call) => (call === 1 ? new Promise(() => {}) : null));
  const h = start(t, async () => ort);
  const loading = assert.rejects(h.client.load(), /The neural runtime did not finish within 45s/);
  await new Promise((resolve) => { setImmediate(resolve); });
  t.mock.timers.tick(45_000);
  await loading;
  assert.deepEqual(created, ['wasm']);
  assert.equal(h.gpuFailures(), 0);
  assert.equal(h.terminated(), true);
});

// With the runtime running, what fails next is the backend itself: a GPU
// that cannot start reaches the guard as it always did, and WebAssembly never.
test('a backend that fails to start on a running runtime reaches the tab guard only when it is the GPU', async (t) => {
  for (const [allowWebgpu, provider] of [[true, 'webgpu'], [false, 'wasm']]) {
    const { ort, created } = runtime((_provider, call) => {
      if (call === 2) throw new Error('Session creation failed.');
    });
    const h = start(t, async () => ort, { allowWebgpu });
    await assert.rejects(h.client.load(), /Session creation failed/);
    assert.deepEqual(created, ['wasm', provider]);
    assert.equal(h.gpuFailures() > 0, allowWebgpu, `a ${provider} start`);
  }
});

// The runtime's own session is no stage of the start: a stage names its
// backend to the page, which holds a failure after one against that backend.
test('a WebGPU start reports the GPU\'s stages alone, after the runtime\'s own session', async (t) => {
  const { ort, created } = runtime();
  const h = start(t, async () => ort);
  const stages = [];
  const network = await h.client.load({ onProgress: (progress) => stages.push(progress) });
  assert.deepEqual(created, ['wasm', 'webgpu']);
  assert.deepEqual(stages, [{ stage: 'session', backend: 'webgpu', phase: 'create' },
    { stage: 'session', backend: 'webgpu', phase: 'warmup' }]);
  assert.deepEqual([network.backend, network.batchSize], ['webgpu', 8]);
  assert.equal(h.gpuFailures(), 0);
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
  // The eight positions of the batch the GPU dropped are the first eight the
  // CPU evaluates, one call each and in the same order.
  assert.deepEqual(h.calls().cpuBoards.slice(0, 8), h.calls().gpuBatches[0]);
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
// the game. Every GPU batch here costs 250 ms on the simulated clock while the
// warm-up promised 1 ms a position, so 512 simulations would take 16 s.
test('a search that overruns three budgets stops, and the next move is sized to fit', async (t) => {
  const h = await harness(t, { failAt: Infinity, gpuBatchSize: 8, gpuMs: 250 });
  const { result } = await h.run();
  assert.ok(result.elapsedMs >= 4_500 && result.elapsedMs < 4_500 + 250,
    `stopped after ${result.elapsedMs} ms, within one batch of the cap`);
  assert.ok(result.nodes >= 64 && result.nodes < 512, `${result.nodes} simulations`);
  // Calibrated on what the stopped search really spent, the next move plans
  // fewer simulations and finishes them without reaching the cap.
  const planned = simulationsFor(h.network);
  assert.ok(planned < result.nodes, `${planned} planned after ${result.nodes}`);
  const next = (await h.run()).result;
  assert.equal(next.nodes, planned);
  assert.ok(next.elapsedMs < 4_500, `${next.elapsedMs} ms`);
});

// At 500 ms a batch not even the minimum fits: the root and eight batches
// take the whole cap. The slowed search is stopped there, and every move
// after it plans the minimum and pays the same nine calls, where the sixteen
// simulations that were the minimum before took three.
test('a GPU too slow for its minimum is held to it, at nine calls a move', async (t) => {
  const h = await harness(t, { failAt: Infinity, gpuBatchSize: 8, gpuMs: 500 });
  const { result } = await h.run();
  assert.deepEqual([result.nodes, result.elapsedMs], [64, 4_500], 'stopped at the cap, with its minimum');
  for (let move = 2; move <= 3; move += 1) {
    assert.equal(simulationsFor(h.network), 64, `move ${move} plans the minimum`);
    const calls = h.calls().gpu;
    const next = (await h.run()).result;
    assert.equal(next.nodes, 64);
    assert.equal(h.calls().gpu - calls, 9, 'the root and eight batches');
    assert.equal(next.elapsedMs, 9 * 500);
  }
});

test('the cap never cuts a search below its minimum', async (t) => {
  const h = await harness(t, { failAt: Infinity, gpuBatchSize: 8, gpuMs: 5_000 });
  const { result } = await h.run();
  assert.equal(result.nodes, 64, 'eight batches, although the first already passed the cap');
  assert.equal(result.elapsedMs, 9 * 5_000);
});

// The minimum changes how few simulations a search runs, never how it
// gathers them: at 64 simulations or more the network is asked for the same
// positions, in the same batches, as a search handed the network's own batch.
test('a search of 64 simulations or more asks the network for the batches it always did', async (t) => {
  for (const [gpuWarmUpMs, planned] of [[700, 64], [15, 100], [1, 512]]) {
    const h = await harness(t, { failAt: Infinity, gpuBatchSize: 8, gpuWarmUpMs });
    assert.equal((await h.run()).result.nodes, planned);
    const batches = [];
    await searchPosition(h.position, async () => output(), { simulations: planned, batchSize: 8,
      evaluateMany: async (items) => { batches.push(items.map((item) => item.board)); return items.map(output); } });
    assert.ok(batches.length >= 8, `${planned} simulations take eight batches at least`);
    assert.deepEqual(h.calls().gpuBatches, batches, `${planned} simulations`);
  }
});

// The download has grown since a comment in the request first put a size on
// it. The prompt computes the size it shows (DOWNLOAD_BYTES), and nothing
// else in the request may state one that can fall behind.
test('the request states no download size of its own', () => {
  assert.doesNotMatch(appSource, /\d\s*MB\b/);
});
