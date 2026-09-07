// Loads the exported network in the browser and evaluates positions with it.
//
// Everything is served from this origin: the page's content-security policy
// allows no third-party scripts, so the ONNX runtime is vendored alongside
// the model. Both are fetched only when the neural opponent is first asked
// for a move, because together they are a large download.
//
// Desktop browsers can use WebGPU; iPhones/iPads use WASM. Session creation is
// bounded and inference timing controls the search budget. A failed GPU is
// released before a CPU replacement starts, keeping only one session alive.
// The page's tab-local guard avoids GPU retries after a confirmed failure.

import { CANVAS, PLANES, planeBuffer, writePlanes } from './neural-planes.js';
import { boardDimensions } from './engine.js';
import { fetchWithProgress } from './download-gate.js';
import { preferNeuralWasm } from './neural-gpu-guard.js';
import { createResourceLoader, releaseResource, throwIfAborted, waitFor } from './async-control.js';

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
export const DOWNLOAD_BYTES = { model: 47_375_662, runtime: 25_749_873 };

const DOWNLOAD_TIMEOUT_MS = 600_000;  // the page shows progress and offers Cancel meanwhile
const SESSION_TIMEOUT_MS = 45_000;    // a GPU busy elsewhere can stall session creation for minutes
const PROBE_BOARD = Array.from({ length: 6 }, () => new Array(7).fill(0));

const loader = createResourceLoader(load);
let backendOptions = {};

/** 'idle' before any request, 'loading' while in flight, 'ready' after. */
export function neuralLoadState() {
  return loader.state();
}

/** Aborts a load in flight; the pending loadNeuralNetwork() rejects. */
export function cancelNeuralLoad() {
  loader.cancel();
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
  if (loader.state() === 'idle') backendOptions = options;
  return loader.load(options);
}

// Storage is managed by the page; workers receive an explicit allowWebgpu flag.
// --- sessions -----------------------------------------------------------------

async function createSession(ort, modelBytes, provider, signal, timeoutMs) {
  throwIfAborted(signal);
  return waitFor(ort.InferenceSession.create(modelBytes, {
    executionProviders: [provider],
    graphOptimizationLevel: 'all',
  }), { signal, timeoutMs, label: `The ${provider} backend`, onLate: releaseResource });
}

function makeEvaluate(ort, session) {
  return async (board, mover, _actions, connect, chaosMode, repeated = 0) => {
    const input = planeBuffer(1);
    const { rows, cols } = boardDimensions(board);
    // The engine counts rows from the top and the network from the bottom.
    writePlanes(input, 0, rows, cols, connect, chaosMode,
      (row, column) => {
        const cell = board[rows - 1 - row][column];
        if (cell === 0) return 0;
        return cell === mover ? 1 : 2;
      }, repeated >= 1, repeated >= 2);
    const tensor = new ort.Tensor('float32', input, [1, PLANES, CANVAS, CANVAS]);
    let outputs;
    try {
      outputs = await session.run({ planes: tensor });
      // Only these 55 numbers escape inference. Copy them before disposing
      // native outputs, so no tensor/resource remains owned by the search.
      return {
        policy: outputs.policy.data.slice(),
        value: outputs.value.data.slice(),
        q: outputs.q.data.slice(),
      };
    } finally {
      releaseResource(tensor);
      for (const output of Object.values(outputs ?? {})) releaseResource(output);
    }
  };
}

export async function startBackend(ort, modelBytes, provider, {
  signal: parentSignal, timeoutMs = SESSION_TIMEOUT_MS, onStage = () => {},
} = {}) {
  throwIfAborted(parentSignal);
  const controller = new AbortController();
  const abort = () => controller.abort();
  parentSignal?.addEventListener('abort', abort, { once: true });
  const { signal } = controller;
  let session;
  let measurement;
  try {
    onStage('create');
    session = await createSession(ort, modelBytes, provider, signal, timeoutMs);
    const evaluate = makeEvaluate(ort, session);
    onStage('warmup');
    measurement = measureEvaluation(() => evaluate(PROBE_BOARD, 1, [], 4, false), { signal });
    const perEvaluation = await waitFor(measurement, {
      signal, timeoutMs, label: `The ${provider} warm-up`,
    });
    return { backend: provider, session, evaluate, perEvaluation };
  } catch (error) {
    controller.abort(); // Also stop warm-up after a deadline, not just user cancellation.
    if (session) {
      // Do not release a session underneath a pending native evaluation.
      Promise.resolve(measurement).then(() => releaseResource(session), () => releaseResource(session));
    }
    throw error;
  } finally {
    parentSignal?.removeEventListener('abort', abort);
  }
}

function gpuDevice(ort) {
  try {
    return ort.env.webgpu?.device ?? null;
  } catch {
    return null;
  }
}

async function load(signal, onProgress) {
  const options = backendOptions;
  const backendStage = (backend) => (phase) => onProgress({ stage: 'session', backend, phase });
  throwIfAborted(signal);
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
  let [modelBytes, metadata] = await waitFor(Promise.all([
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
    }, { signal, expectedBytes: DOWNLOAD_BYTES.runtime, retain: false })
      .catch((error) => {            // the runtime fetches it itself if this fails
        if (error?.name === 'AbortError') throw error;
        return null;
      }),
  ]), { signal, timeoutMs: DOWNLOAD_TIMEOUT_MS, label: 'The network' });

  const ort = await waitFor(import(RUNTIME_URL), {
    signal, timeoutMs: SESSION_TIMEOUT_MS, label: 'The neural runtime',
  });
  ort.env.wasm.wasmPaths = ASSETS.href;
  ort.env.wasm.numThreads = 1;               // no cross-origin isolation on Pages

  // Never hold two heavyweight sessions just to compare their speed. A timed
  // out GPU startup may still be running natively; let the page kill that
  // worker before Retry starts WASM in a fresh one.
  const provider = globalThis.navigator?.gpu && !preferNeuralWasm() && options.allowWebgpu !== false ? 'webgpu' : 'wasm';
  let active;
  try {
    active = await startBackend(ort, modelBytes, provider, { signal, onStage: backendStage(provider) });
  } catch (error) {
    if (provider === 'webgpu' && !signal.aborted) options.onBackendFailure?.(error);
    throw error;
  }
  // WASM sessions already own their weights. Only a live GPU needs the model
  // bytes for a future CPU fallback; do not pin another 47 MB for CPU games.
  if (provider === 'wasm') modelBytes = null;
  return manageBackend(active, async () => {
    const replacement = await startBackend(ort, modelBytes, 'wasm', { signal, onStage: backendStage('wasm') });
    modelBytes = null;
    return replacement;
  }, { ...options, metadata, ort, device: provider === 'webgpu' ? gpuDevice(ort) : null });
}

/** Serialize inference, GPU loss and disposal; at most one native session lives. */
export function manageBackend(active, restartOnWasm, options = {}) {
  let disposed = false;
  let deviceLost = false;
  let evaluationQueue = Promise.resolve();
  let session = active.session;
  const releaseSession = async () => {
    const previous = session;
    session = null;
    try { await previous?.release?.(); } catch { /* already lost */ }
  };
  const network = {
    backend: active.backend,
    metadata: options.metadata,
    ort: options.ort,
    perEvaluation: active.perEvaluation,
    evaluate: null,
    dispose() {
      if (disposed) return;
      disposed = true;
      // Native run/release must never overlap, including during fallback.
      void evaluationQueue.then(releaseSession);
    },
  };

  // A lost or failing GPU device moves the network to WebAssembly once,
  // mid-game, instead of ending the round with an error.
  let fallingBack = null;
  const fallBackToWasm = () => {
    if (disposed) return Promise.reject(new Error('Network was disposed'));
    if (!fallingBack) {
      fallingBack = (async () => {
        // A device-lost callback must not race a native inference. This runs
        // only inside evaluationQueue, after any in-flight evaluation drains.
        await releaseSession();
        if (disposed) throw new Error('Network was disposed');
        const replacement = await restartOnWasm();
        if (disposed) { releaseResource(replacement.session); throw new Error('Network was disposed'); }
        active = replacement;
        session = replacement.session;
        network.backend = 'wasm';
        network.perEvaluation = replacement.perEvaluation;
        options.onBackend?.('wasm');
      })();
    }
    return fallingBack;
  };
  const evaluateCurrent = async (...args) => {
    if (disposed) throw new Error('Network was disposed');
    if (deviceLost && active.backend === 'webgpu') await fallBackToWasm();
    try {
      return await active.evaluate(...args);
    } catch (error) {
      if (active.backend !== 'webgpu') throw error;
      options.onBackendFailure?.(error);
      await fallBackToWasm();
      return active.evaluate(...args);
    }
  };
  network.evaluate = (...args) => {
    const result = evaluationQueue.then(() => evaluateCurrent(...args));
    evaluationQueue = result.catch(() => {});
    return result;
  };
  if (active.backend === 'webgpu') {
    options.device?.lost?.then(() => {
      if (disposed) return;
      deviceLost = true;
      options.onBackendFailure?.(new Error('WebGPU device lost'));
    }, () => {});
  }
  options.onBackend?.(network.backend);
  return network;
}

const WARMUP_EVALUATIONS = 6;
const TIMED_EVALUATIONS = 5;

/**
 * Median time of one evaluation after warm-up, in milliseconds. The first
 * evaluations on WebGPU compile shaders and take far longer than the rest.
 */
export async function measureEvaluation(run, { signal } = {}) {
  for (let warm = 0; warm < WARMUP_EVALUATIONS; warm += 1) {
    // eslint-disable-next-line no-await-in-loop
    throwIfAborted(signal);
    await run();
    throwIfAborted(signal);
  }
  const times = [];
  for (let sample = 0; sample < TIMED_EVALUATIONS; sample += 1) {
    const started = performance.now();
    // eslint-disable-next-line no-await-in-loop
    throwIfAborted(signal);
    await run();
    throwIfAborted(signal);
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
export function recordSearch(network, elapsedMs, evaluations) {
  if (!network || typeof network !== 'object') return;
  if (!(evaluations > 0) || !(elapsedMs > 0)) return;
  const perSimulation = elapsedMs / evaluations;
  network.perEvaluation = network.perEvaluation > 0
    ? 0.5 * network.perEvaluation + 0.5 * perSimulation
    : perSimulation;
}
