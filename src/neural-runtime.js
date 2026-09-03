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

const RUNTIME_URL = './assets/neural/ort.webgpu.min.mjs';
const MODEL_URL = './assets/neural/model.onnx';
const METADATA_URL = './assets/neural/model.json';

let loading = null;

/** Loads the runtime and the model once, and reports which backend won. */
export function loadNeuralNetwork(options = {}) {
  if (!loading) loading = load(options).catch((error) => { loading = null; throw error; });
  return loading;
}

async function load(options) {
  const onProgress = options.onProgress ?? (() => {});
  onProgress({ stage: 'runtime' });
  const ort = await import(RUNTIME_URL);
  ort.env.wasm.wasmPaths = './assets/neural/';
  ort.env.wasm.numThreads = 1;               // no cross-origin isolation on Pages

  onProgress({ stage: 'model' });
  const [modelBytes, metadata] = await Promise.all([
    fetch(MODEL_URL).then((response) => {
      if (!response.ok) throw new Error(`Could not fetch the model: ${response.status}`);
      return response.arrayBuffer();
    }),
    fetch(METADATA_URL).then((response) => (response.ok ? response.json() : null)),
  ]);

  let session = null;
  let backend = 'wasm';
  const wanted = navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
  let lastError = null;
  for (const provider of wanted) {
    try {
      onProgress({ stage: 'session', backend: provider });
      session = await ort.InferenceSession.create(modelBytes, {
        executionProviders: [provider],
        graphOptimizationLevel: 'all',
      });
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
