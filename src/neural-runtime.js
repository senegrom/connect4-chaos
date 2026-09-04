// Loads the exported network in the browser and evaluates positions with it.
//
// Everything is served from this origin: the page's content-security policy
// allows no third-party scripts, so the ONNX runtime is vendored alongside
// the model. Both are fetched only when the neural opponent is first asked
// for a move, because together they are a large download.
//
// WebGPU is used when the browser offers it and falls back to WebAssembly,
// which is perhaps twenty times slower here; the search budget adapts to
// whichever is available so a move still arrives promptly.

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
const WASM_URL = new URL('ort-wasm-simd-threaded.jsep.wasm', ASSETS).href;
const LOAD_TIMEOUT_MS = 180000;
// Sizes as shipped, so the prompt can state them before anything is fetched.
export const DOWNLOAD_BYTES = { model: 47_400_000, runtime: 27_800_000 };

let loading = null;

/** True once the network has been asked for in this page. */
export function isNeuralLoaded() {
  return loading !== null;
}

/** Where the runtime, the model and its metadata are fetched from. */
export function assetUrls() {
  return { runtime: RUNTIME_URL, model: MODEL_URL, metadata: METADATA_URL, base: ASSETS.href };
}

/** Loads the runtime and the model once, and reports which backend won. */
export function loadNeuralNetwork(options = {}) {
  if (!loading) loading = load(options).catch((error) => { loading = null; throw error; });
  return loading;
}

/** Rejects rather than waiting forever, so a stall reads as an error. */
function withDeadline(promise, what, timeout = LOAD_TIMEOUT_MS) {
  return Promise.race([
    promise,
    new Promise((_resolve, reject) => {
      setTimeout(() => reject(new Error(`${what} did not load within ${timeout / 1000}s`)),
        timeout);
    }),
  ]);
}

async function load(options) {
  const onProgress = options.onProgress ?? (() => {});
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
    }),
    fetch(METADATA_URL).then((response) => (response.ok ? response.json() : null)),
    fetchWithProgress(WASM_URL, (loaded, total) => {
      if (total) sizes.runtime = total;
      progress.runtime = loaded;
      report('runtime');
    }).catch(() => null),          // the runtime fetches it itself if this fails
  ]), 'the network');

  const ort = await withDeadline(import(RUNTIME_URL), 'the neural runtime');
  ort.env.wasm.wasmPaths = ASSETS.href;
  ort.env.wasm.numThreads = 1;               // no cross-origin isolation on Pages

  let session = null;
  let backend = 'wasm';
  const wanted = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
  let lastError = null;
  for (const provider of wanted) {
    try {
      onProgress({ stage: 'session', backend: provider });
      session = await withDeadline(ort.InferenceSession.create(modelBytes, {
        executionProviders: [provider],
        graphOptimizationLevel: 'all',
      }), `the ${provider} backend`);
      backend = provider;
      break;
    } catch (error) {
      lastError = error;
    }
  }
  if (!session) throw lastError ?? new Error('No execution provider could load the model.');

  const input = planeBuffer(1);
  const evaluate = async (board, mover, _actions, connect, chaosMode) => {
    const { rows, cols } = boardDimensions(board);
    // The engine counts rows from the top and the network from the bottom.
    writePlanes(input, 0, rows, cols, connect, chaosMode,
      (row, column) => {
        const cell = board[rows - 1 - row][column];
        if (cell === 0) return 0;
        return cell === mover ? 1 : 2;
      });
    const tensor = new ort.Tensor('float32', input, [1, PLANES, CANVAS, CANVAS]);
    const outputs = await session.run({ planes: tensor });
    return {
      policy: outputs.policy.data,
      value: outputs.value.data,
      q: outputs.q.data,
    };
  };

  // How long one evaluation takes decides how many the search can afford:
  // WebGPU runs this network in a few milliseconds, WebAssembly in a couple
  // of hundred, and a fixed budget would make one of them unusable.
  const probeBoard = Array.from({ length: 6 }, () => new Array(7).fill(0));
  const started = performance.now();
  for (let probe = 0; probe < 3; probe += 1) {
    // eslint-disable-next-line no-await-in-loop
    await evaluate(probeBoard, 1, [], 4, false);
  }
  const perEvaluation = (performance.now() - started) / 3;

  return { evaluate, backend, metadata, ort, perEvaluation };
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
