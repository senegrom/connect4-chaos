// A dedicated worker, not ORT's blob/proxy worker: compatible with worker-src
// 'self', and it supports WebGPU as well as CPU WASM without blocking input.
import { loadNeuralNetwork } from './neural-runtime.js';

let network = null;
let busy = false;
self.addEventListener('message', async ({ data }) => {
  const { id, kind } = data ?? {};
  if (!Number.isSafeInteger(id)) return;
  if (busy) {
    self.postMessage({ id, kind: 'error', error: 'Overlapping neural worker requests.' });
    return;
  }
  busy = true;
  try {
    let result;
    if (kind === 'load') {
      network = await loadNeuralNetwork({
        allowWebgpu: data.allowWebgpu === true,
        onProgress: (progress) => self.postMessage({ id, kind: 'progress', progress }),
        onBackendFailure: () => self.postMessage({ kind: 'gpu-failure' }),
        onBackend: (backend) => self.postMessage({ kind: 'backend', backend }),
      });
      // The client mirrors what this backend can actually do: a network
      // without batch evaluation must not be offered one.
      result = { backend: network.backend, perEvaluation: network.perEvaluation,
        metadata: network.metadata, batched: typeof network.evaluateMany === 'function' };
    } else if (kind === 'evaluate' && network && Array.isArray(data.args)) {
      result = await network.evaluate(...data.args);
    } else if (kind === 'evaluateMany' && network?.evaluateMany && Array.isArray(data.args?.[0])) {
      // One message per batch of leaves rather than one per position: the
      // round trip costs about as much as the evaluation it carries.
      result = await network.evaluateMany(data.args[0]);
    } else throw new Error('Invalid neural worker request.');
    self.postMessage({ id, kind: 'result', result, backend: network.backend });
  } catch (error) {
    self.postMessage({ id, kind: 'error', error: error?.message || String(error) });
  } finally { busy = false; }
});
