// Loads the exported network in the browser and evaluates positions with it.
//
// Everything is served from this origin: the page's content-security policy
// allows no third-party scripts, so the ONNX runtime is vendored alongside
// the model. Both are fetched only when the neural opponent is first asked
// for a move, because together they are a large download.
//
// WebGPU is used when the browser offers it and it is actually fast here. A
// GPU already busy with other work can be slower than WebAssembly, can take
// minutes to build a session, and can lose its device mid-game, so the
// runtime bounds session creation, measures WebAssembly as well when the
// GPU looks slow, moves to WebAssembly when the device is lost, and skips
// the GPU for a day after a page that ended abruptly while using it.

import { CANVAS, PLANES, planeBuffer, writePlanes } from './neural-planes.js';
import { boardDimensions } from './engine.js';
import { fetchWithProgress } from './download-gate.js';

// Resolved against this module, not the page: a relative specifier in a
// dynamic import is module-relative, so './assets/...' would look inside
// src/ and 404.
const ASSETS = new URL('../assets/neural/', import.meta.url);
const RUNTIME_URL = new URL('ort.webgpu.min.mjs', ASSETS).href;
const MODEL_URL = new URL('model.onnx', ASSETS).href;
const METADATA_URL = new URL('model.json', ASSETS).href;
const LOADER_URL = new URL('ort-wasm-simd-threaded.asyncify.mjs', ASSETS).href;
const WASM_URL = new URL('ort-wasm-simd-threaded.asyncify.wasm', ASSETS).href;
// Sizes as shipped, so the prompt can state them before anything is fetched.
export const DOWNLOAD_BYTES = { model: 47_400_000, runtime: 25_750_000 };

const DOWNLOAD_TIMEOUT_MS = 600_000;  // the page shows progress and offers Cancel meanwhile
const SESSION_TIMEOUT_MS = 45_000;    // a GPU busy elsewhere can stall session creation for minutes
const SLOW_GPU_MS = 40;               // above this per evaluation, WebAssembly is measured as well
const WEBGPU_ACTIVE_KEY = 'connect4-chaos.neural.webgpu-active';
const WEBGPU_CRASH_KEY = 'connect4-chaos.neural.webgpu-crash';
const WEBGPU_AVOID_MS = 24 * 60 * 60 * 1000;
const PROBE_BOARD = Array.from({ length: 6 }, () => new Array(7).fill(0));

let loading = null;
let ready = false;
let controller = null;
const listeners = new Set();

/** 'idle' before any request, 'loading' while in flight, 'ready' after. */
export function neuralLoadState() {
  if (ready) return 'ready';
  return loading ? 'loading' : 'idle';
}

/** Aborts a load in flight; the pending loadNeuralNetwork() rejects. */
export function cancelNeuralLoad() {
  controller?.abort();
}

/** Where the runtime, the model and its metadata are fetched from. */
export function assetUrls() {
  return { runtime: RUNTIME_URL, loader: LOADER_URL, wasm: WASM_URL, model: MODEL_URL, metadata: METADATA_URL, base: ASSETS.href };
}

/**
 * Loads the runtime and the model once, and reports which backend won.
 * Every caller's `onProgress` hears about the one load in flight, so a
 * request that joins a download already running still shows its progress.
 */
export function loadNeuralNetwork(options = {}) {
  if (options.onProgress && !ready) listeners.add(options.onProgress);
  if (!loading) {
    controller = new AbortController();
    loading = load(controller.signal)
      .then((network) => { ready = true; return network; })
      // A failed or timed-out load must not leave its downloads streaming.
      .catch((error) => { controller?.abort(); loading = null; throw error; })
      .finally(() => { listeners.clear(); controller = null; });
  }
  return loading;
}

/** Rejects rather than waiting forever, so a stall reads as an error. */
function withDeadline(promise, what, timeout) {
  let timer = null;
  return Promise.race([
    promise.finally(() => clearTimeout(timer)),
    new Promise((_resolve, reject) => {
      timer = setTimeout(() => reject(new Error(`${what} did not load within ${Math.round(timeout / 1000)}s`)),
        timeout);
    }),
  ]);
}

// --- crash guard --------------------------------------------------------------
// A flag is set while a WebGPU session is the active backend and cleared when
// the page unloads normally. Finding it still set on the next visit means the
// page ended without unloading, which is what a GPU-process crash looks like,
// so the GPU is skipped for a while.

function storage(action) {
  try {
    return action(localStorage);
  } catch {
    return null;
  }
}

function noteAbruptEnd() {
  if (storage((store) => store.getItem(WEBGPU_ACTIVE_KEY)) === null) return;
  storage((store) => {
    store.setItem(WEBGPU_CRASH_KEY, String(Date.now()));
    store.removeItem(WEBGPU_ACTIVE_KEY);
  });
}

function markWebgpuActive(active) {
  storage((store) => (active ? store.setItem(WEBGPU_ACTIVE_KEY, '1') : store.removeItem(WEBGPU_ACTIVE_KEY)));
}

/** True while a recent abrupt end suggests WebGPU crashed this browser. */
export function webgpuAvoided() {
  const stamp = Number(storage((store) => store.getItem(WEBGPU_CRASH_KEY)));
  return Number.isFinite(stamp) && stamp > 0 && Date.now() - stamp < WEBGPU_AVOID_MS;
}

if (typeof window !== 'undefined') {
  noteAbruptEnd();
  window.addEventListener('pagehide', () => markWebgpuActive(false));
}

// --- sessions -----------------------------------------------------------------

async function createSession(ort, modelBytes, provider) {
  let abandoned = false;
  const creation = ort.InferenceSession.create(modelBytes, {
    executionProviders: [provider],
    graphOptimizationLevel: 'all',
  });
  // A creation that outlives its deadline is released when it finally lands.
  creation.then((session) => { if (abandoned) session.release?.(); }, () => {});
  try {
    return await withDeadline(creation, `the ${provider} backend`, SESSION_TIMEOUT_MS);
  } catch (error) {
    abandoned = true;
    throw error;
  }
}

function makeEvaluate(ort, session) {
  const input = planeBuffer(1);
  return async (board, mover, _actions, connect, chaosMode, repeated = 0) => {
    const { rows, cols } = boardDimensions(board);
    // The engine counts rows from the top and the network from the bottom.
    writePlanes(input, 0, rows, cols, connect, chaosMode,
      (row, column) => {
        const cell = board[rows - 1 - row][column];
        if (cell === 0) return 0;
        return cell === mover ? 1 : 2;
      }, repeated >= 1, repeated >= 2);
    const tensor = new ort.Tensor('float32', input, [1, PLANES, CANVAS, CANVAS]);
    const outputs = await session.run({ planes: tensor });
    return {
      policy: outputs.policy.data,
      value: outputs.value.data,
      q: outputs.q.data,
    };
  };
}

async function startBackend(ort, modelBytes, provider) {
  const session = await createSession(ort, modelBytes, provider);
  const evaluate = makeEvaluate(ort, session);
  const perEvaluation = await measureEvaluation(() => evaluate(PROBE_BOARD, 1, [], 4, false));
  return { backend: provider, session, evaluate, perEvaluation };
}

function gpuDevice(ort) {
  try {
    return ort.env.webgpu?.device ?? null;
  } catch {
    return null;
  }
}

async function load(signal) {
  const onProgress = (progress) => {
    for (const listener of listeners) listener(progress);
  };
  // The two big files are streamed first so the page can show a real
  // progress bar. The runtime then finds its WebAssembly in the browser
  // cache, so nothing is fetched twice.
  const progress = { model: 0, runtime: 0, total: DOWNLOAD_BYTES.model + DOWNLOAD_BYTES.runtime };
  const sizes = { model: DOWNLOAD_BYTES.model, runtime: DOWNLOAD_BYTES.runtime };
  const report = (stage) => onProgress({
    stage,
    loaded: progress.model + progress.runtime,
    total: sizes.model + sizes.runtime,
  });
  onProgress({ stage: 'runtime', loaded: 0, total: progress.total });
  const [modelBytes, metadata] = await withDeadline(Promise.all([
    fetchWithProgress(MODEL_URL, (loaded, total) => {
      if (total) sizes.model = total;
      progress.model = loaded;
      report('model');
    }, { signal, expectedBytes: DOWNLOAD_BYTES.model }),
    fetch(METADATA_URL, { signal }).then((response) => (response.ok ? response.json() : null)),
    fetchWithProgress(WASM_URL, (loaded, total) => {
      if (total) sizes.runtime = total;
      progress.runtime = loaded;
      report('runtime');
    }, { signal, expectedBytes: DOWNLOAD_BYTES.runtime })
      .catch((error) => {            // the runtime fetches it itself if this fails
        if (error?.name === 'AbortError') throw error;
        return null;
      }),
  ]), 'the network', DOWNLOAD_TIMEOUT_MS);

  const ort = await withDeadline(import(RUNTIME_URL), 'the neural runtime', SESSION_TIMEOUT_MS);
  ort.env.wasm.wasmPaths = ASSETS.href;
  ort.env.wasm.numThreads = 1;               // no cross-origin isolation on Pages

  // WebGPU first when it is offered and not under suspicion; WebAssembly as
  // the fallback, and as a rival when the GPU measures slow.
  const errors = [];
  let gpu = null;
  if (navigator.gpu && !webgpuAvoided()) {
    onProgress({ stage: 'session', backend: 'webgpu' });
    try {
      gpu = await startBackend(ort, modelBytes, 'webgpu');
    } catch (error) {
      errors.push(error);
    }
  }
  let cpu = null;
  if (!gpu || gpu.perEvaluation > SLOW_GPU_MS) {
    onProgress({ stage: 'session', backend: 'wasm' });
    try {
      cpu = await startBackend(ort, modelBytes, 'wasm');
    } catch (error) {
      errors.push(error);
    }
  }
  if (!gpu && !cpu) throw errors[errors.length - 1] ?? new Error('No execution provider could load the model.');
  let active = gpu && (!cpu || gpu.perEvaluation <= cpu.perEvaluation) ? gpu : cpu;
  const loser = active === gpu ? cpu : gpu;
  loser?.session.release?.();

  const network = {
    backend: active.backend,
    metadata,
    ort,
    perEvaluation: active.perEvaluation,
    evaluate: null,
  };

  // A lost or failing GPU device moves the network to WebAssembly once,
  // mid-game, instead of ending the round with an error.
  let fallingBack = null;
  const fallBackToWasm = () => {
    if (!fallingBack) {
      fallingBack = (async () => {
        const replacement = await startBackend(ort, modelBytes, 'wasm');
        try {
          active.session.release?.();
        } catch {
          // The GPU session may already be gone.
        }
        active = replacement;
        network.backend = 'wasm';
        network.perEvaluation = replacement.perEvaluation;
        markWebgpuActive(false);
      })();
    }
    return fallingBack;
  };
  network.evaluate = async (...args) => {
    try {
      return await active.evaluate(...args);
    } catch (error) {
      if (active.backend !== 'webgpu') throw error;
      await fallBackToWasm();
      return active.evaluate(...args);
    }
  };
  if (active.backend === 'webgpu') {
    markWebgpuActive(true);
    gpuDevice(ort)?.lost?.then(() => { fallBackToWasm().catch(() => {}); }, () => {});
  }
  return network;
}

const WARMUP_EVALUATIONS = 6;
const TIMED_EVALUATIONS = 5;

/**
 * Median time of one evaluation after warm-up, in milliseconds. The first
 * evaluations on WebGPU compile shaders and take far longer than the rest.
 */
export async function measureEvaluation(run) {
  for (let warm = 0; warm < WARMUP_EVALUATIONS; warm += 1) {
    // eslint-disable-next-line no-await-in-loop
    await run();
  }
  const times = [];
  for (let sample = 0; sample < TIMED_EVALUATIONS; sample += 1) {
    const started = performance.now();
    // eslint-disable-next-line no-await-in-loop
    await run();
    times.push(performance.now() - started);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

const BUDGET_MS = 1500;
const MIN_SIMULATIONS = 2;
const MAX_SIMULATIONS = 192;

/**
 * How many simulations fit in about `BUDGET_MS`, given how fast this
 * device evaluates. Deeper search plays better - on solved chaos boards
 * the same network misplays 3.5% of positions with none, 0.9% with 32 and
 * 0.5% with 128 - so the aim is as many as the budget allows.
 */
export function simulationsFor(network, requested) {
  if (Number.isInteger(requested) && requested > 0) return requested;
  const perEvaluation = typeof network === 'object' ? network.perEvaluation : null;
  if (!perEvaluation || !Number.isFinite(perEvaluation) || perEvaluation <= 0) {
    return network?.backend === 'webgpu' ? 128 : 8;
  }
  const affordable = Math.round(BUDGET_MS / perEvaluation);
  return Math.max(MIN_SIMULATIONS, Math.min(MAX_SIMULATIONS, affordable));
}

/**
 * Feeds the measured time of a finished search back into the budget, so a
 * GPU that slows down mid-game (other work starting on it) gets fewer
 * simulations next move rather than a move that takes many seconds.
 */
export function recordSearch(network, elapsedMs, simulations) {
  if (!network || typeof network !== 'object') return;
  if (!(simulations > 0) || !(elapsedMs > 0)) return;
  const perSimulation = elapsedMs / simulations;
  network.perEvaluation = network.perEvaluation > 0
    ? 0.5 * network.perEvaluation + 0.5 * perSimulation
    : perSimulation;
}
