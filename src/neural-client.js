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
  // A download is bounded by silence, not by its length: every progress
  // message re-arms this. It outlasts the worker's own 60 s stall limit plus
  // the 45 s it gives the runtime to import, so the worker's clearer error
  // arrives first and this only catches a worker that has stopped answering.
  downloadStallMs = 120_000,
  evaluationTimeoutMs = 45_000,
  idleTimeoutMs = 120_000,
} = {}) {
  let current = null;
  let nextId = 0;

  function retainIdle(target) {
    clearTimeout(target.idleTimer);
    if (target.dead || !target.network || target.pending.size || !(idleTimeoutMs > 0)) return;
    target.idleTimer = setTimeout(() => discard(target), idleTimeoutMs);
    target.idleTimer?.unref?.(); // do not keep headless tools open for an idle worker
  }

  function discard(target, error = new DOMException('Cancelled', 'AbortError'), failed = false) {
    if (!target || target.dead) return;
    target.dead = true;
    clearTimeout(target.idleTimer);
    if (current === target) current = null;
    if (failed && target.backend === 'webgpu') guard.failed();
    target.worker.terminate();
    for (const pending of [...target.pending.values()]) pending.finish(error);
    target.listeners.clear();
  }

  function call(target, kind, payload, timeoutMs) {
    if (target.dead) return Promise.reject(new Error('Neural worker was replaced. Retry the move.'));
    clearTimeout(target.idleTimer);
    const id = ++nextId;
    return new Promise((resolve, reject) => {
      let timer;
      const pending = {
        kind,
        send() {
          if (target.dead) return;
          target.sending = pending;
          pending.arm(timeoutMs);
          try { target.worker.postMessage({ id, kind, ...payload }); }
          catch (error) { discard(target, error); }
        },
        finish(error, result) {
          clearTimeout(timer);
          target.pending.delete(id);
          if (target.sending === pending) {
            target.sending = null;
            target.waiting.shift()?.send();
          }
          retainIdle(target);
          if (error) reject(error); else resolve(result);
        },
        arm(milliseconds) {
          clearTimeout(timer);
          timer = setTimeout(() => discard(target,
            new Error(`${kind === 'load' ? 'Neural startup' : 'Network evaluation'} timed out. Retry to start a fresh worker.`), true), milliseconds);
        },
      };
      target.pending.set(id, pending);
      // The worker runs one request at a time and answers an overlapping one
      // with an error, which would discard it as a failed GPU. A search that
      // Undo or a new round abandoned can still have its last evaluation
      // running there, so the next request waits for it rather than
      // colliding with it. Each deadline starts when its request is sent.
      if (target.sending) target.waiting.push(pending);
      else pending.send();
    });
  }

  function start() {
    const target = { worker: createWorker(), pending: new Map(), listeners: new Set(),
      sending: null, waiting: [], backend: null, network: null, ready: null, dead: false };
    current = target;
    target.worker.addEventListener('message', ({ data }) => {
      if (target.dead || current !== target) return;
      if (data?.kind === 'gpu-failure') { guard.failed(); return; }
      if (data?.kind === 'backend') { target.backend = data.backend; return; }
      const pending = target.pending.get(data?.id);
      if (!pending) return; // An old/unknown response can never satisfy a newer request.
      if (data.kind === 'progress') {
        // A load reports its download and session stages; so does an
        // evaluation whose failing GPU moved the network to WebAssembly,
        // which re-reads the model and builds a session inside that one
        // request. Each stage gets the deadline the load would have: one
        // evaluation deadline for the whole fallback used to expire first.
        if (data.progress?.stage === 'session') {
          target.backend = data.progress.backend;
          pending.arm(evaluationTimeoutMs); // separate creation and warm-up phases
        } else pending.arm(downloadStallMs); // bytes are arriving, however slowly
        if (pending.kind === 'load') {
          for (const listener of target.listeners) {
            try { listener(data.progress); } catch { /* telemetry does not affect inference */ }
          }
        }
      } else if (data.kind === 'result') {
        if (data.backend && target.network && target.network.backend !== data.backend) {
          target.network.backend = data.backend;
          // A replacement backend has its own warm-up measurement. Preserve
          // page-side search calibration on ordinary same-backend replies.
          target.network.perEvaluation = data.perEvaluation;
        }
        if (target.network && Number.isInteger(data.batchSize) && data.batchSize > 0) {
          target.network.batchSize = data.batchSize;
        }
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
    target.ready = call(target, 'load', { allowWebgpu: allowWebgpu && !guard.avoided() }, downloadStallMs).then((info) => {
      if (target.dead) throw new DOMException('Cancelled', 'AbortError');
      target.backend = info.backend;
      target.network = {
        ...info,
        async evaluate(...args) {
          const result = await call(target, 'evaluate', { args }, evaluationTimeoutMs);
          target.network.backend = target.backend;
          return result;
        },
        ...(info.batched && (info.batchSize ?? 1) > 1 ? {
          async evaluateMany(items) {
            const result = await call(target, 'evaluateMany', { args: [items] }, evaluationTimeoutMs);
            target.network.backend = target.backend;
            return result;
          },
        } : {}),
        dispose() { discard(target); },
      };
      retainIdle(target);
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
    // A caller that gives up - a hidden tab, Undo, a new round - only stops
    // waiting; the download goes on. The model is cached only once it has
    // arrived whole, and a gzip body cannot resume, so ending the worker
    // here restarted the download from its first byte on the next move.
    // cancel() and invalidate() still end it, and once the network is ready
    // the idle timer releases it if nobody asks again.
    async load({ signal, onProgress } = {}) {
      throwIfAborted(signal);
      const target = current ?? start();
      retainIdle(target);
      if (onProgress) target.listeners.add(onProgress);
      try { return await waitFor(target.ready, { signal }); }
      finally { target.listeners.delete(onProgress); }
    },
  };
}

const client = createNeuralClient();
export const neuralLoadState = () => client.state();
export const cancelNeuralLoad = () => client.cancel();
export const invalidateNeuralNetwork = (network) => client.invalidate(network);
export const loadNeuralNetwork = (options) => client.load(options);
