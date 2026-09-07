// Only messages and small tensors cross this boundary. Model loading, warm-up
// and every native CPU/GPU evaluation live in a same-origin module worker.
// The watchdog lives on the PAGE so even a synchronous WASM stall is killable.
import { throwIfAborted, waitFor } from './async-control.js';
import { gpuGuard, preferNeuralWasm } from './neural-gpu-guard.js';
export { DOWNLOAD_BYTES, recordSearch, simulationsFor } from './neural-runtime.js';

export function createNeuralClient({
  createWorker = () => new Worker(new URL('./neural-worker.js', import.meta.url), { type: 'module' }),
  guard = gpuGuard,
  allowWebgpu = !preferNeuralWasm(),
  downloadTimeoutMs = 600_000,
  evaluationTimeoutMs = 45_000,
} = {}) {
  let current = null;
  let nextId = 0;

  function discard(target, error = new DOMException('Cancelled', 'AbortError'), failed = false) {
    if (!target || target.dead) return;
    target.dead = true;
    if (current === target) current = null;
    if (failed && target.backend === 'webgpu') guard.failed();
    target.worker.terminate();
    for (const pending of [...target.pending.values()]) pending.finish(error);
    target.listeners.clear();
  }

  function call(target, kind, payload, timeoutMs) {
    if (target.dead) return Promise.reject(new Error('Neural worker was replaced. Retry the move.'));
    const id = ++nextId;
    return new Promise((resolve, reject) => {
      let timer;
      const pending = {
        kind,
        finish(error, result) {
          clearTimeout(timer);
          target.pending.delete(id);
          if (error) reject(error); else resolve(result);
        },
        arm(milliseconds) {
          clearTimeout(timer);
          timer = setTimeout(() => discard(target,
            new Error(`${kind === 'load' ? 'Neural startup' : 'Network evaluation'} timed out. Retry to start a fresh worker.`), true), milliseconds);
        },
      };
      target.pending.set(id, pending);
      pending.arm(timeoutMs);
      try { target.worker.postMessage({ id, kind, ...payload }); }
      catch (error) { discard(target, error); }
    });
  }

  function start() {
    const target = { worker: createWorker(), pending: new Map(), listeners: new Set(),
      backend: null, network: null, ready: null, dead: false };
    current = target;
    target.worker.addEventListener('message', ({ data }) => {
      if (target.dead || current !== target) return;
      if (data?.kind === 'gpu-failure') { guard.failed(); return; }
      if (data?.kind === 'backend') { target.backend = data.backend; return; }
      const pending = target.pending.get(data?.id);
      if (!pending) return; // An old/unknown response can never satisfy a newer request.
      if (data.kind === 'progress' && pending.kind === 'load') {
        if (data.progress?.stage === 'session') {
          target.backend = data.progress.backend;
          pending.arm(evaluationTimeoutMs); // separate creation and warm-up phases
        }
        for (const listener of target.listeners) {
          try { listener(data.progress); } catch { /* telemetry does not affect inference */ }
        }
      } else if (data.kind === 'result') {
        if (data.backend) target.backend = data.backend;
        pending.finish(null, data.result);
      } else {
        discard(target, new Error(data.error || 'Unreadable neural worker response.'), true);
      }
    });
    target.worker.addEventListener('error', (event) => {
      event.preventDefault?.();
      discard(target, new Error(event.message || 'Neural worker failed. Retry to restart it.'), true);
    });
    target.worker.addEventListener('messageerror', () => discard(target,
      new Error('Unreadable neural worker message. Retry to restart it.'), true));
    target.ready = call(target, 'load', { allowWebgpu: allowWebgpu && !guard.avoided() }, downloadTimeoutMs).then((info) => {
      if (target.dead) throw new DOMException('Cancelled', 'AbortError');
      target.backend = info.backend;
      target.network = {
        ...info,
        async evaluate(...args) {
          const result = await call(target, 'evaluate', { args }, evaluationTimeoutMs);
          target.network.backend = target.backend;
          return result;
        },
        dispose() { discard(target); },
      };
      return target.network;
    });
    return target;
  }

  return {
    state: () => current?.network ? 'ready' : current ? 'loading' : 'idle',
    cancel() { if (current && !current.network) discard(current); },
    invalidate(network) {
      if (!current || (network && current.network !== network)) return;
      discard(current);
    },
    async load({ signal, onProgress } = {}) {
      throwIfAborted(signal);
      const target = current ?? start();
      const abort = () => discard(target);
      signal?.addEventListener('abort', abort, { once: true });
      if (onProgress) target.listeners.add(onProgress);
      try { return await waitFor(target.ready, { signal }); }
      finally {
        signal?.removeEventListener('abort', abort);
        target.listeners.delete(onProgress);
      }
    },
  };
}

const client = createNeuralClient();
export const neuralLoadState = () => client.state();
export const cancelNeuralLoad = () => client.cancel();
export const invalidateNeuralNetwork = (network) => client.invalidate(network);
export const loadNeuralNetwork = (options) => client.load(options);
