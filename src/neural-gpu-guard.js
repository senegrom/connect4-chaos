// Record actual backend failures, not another tab's active session. No shared
// localStorage marker is read or written. Storage denial must not stop play.
const FAILURE_KEY = 'connect4-chaos.neural.gpu-failure.v2';
const AVOID_MS = 24 * 60 * 60 * 1000;

// WebGPU availability is not a stability test. Keep this large model off the
// iOS GPU path, where a process kill bypasses both error handlers and watchdogs.
// iPadOS can identify itself as a Mac when requesting desktop websites.
export function preferNeuralWasm(navigator = globalThis.navigator) {
  return /iPhone|iPad|iPod/.test(navigator?.userAgent ?? '')
    || (/Mac/.test(navigator?.platform ?? '') && navigator?.maxTouchPoints > 1);
}

export function createGpuGuard({ getStorage = () => globalThis.sessionStorage, now = Date.now } = {}) {
  let failedAt = null;
  const read = () => {
    try {
      const value = getStorage()?.getItem(FAILURE_KEY);
      return value == null ? failedAt : Number(value);
    } catch { return failedAt; }
  };
  return {
    avoided() {
      const stamp = read();
      const age = now() - stamp;
      return stamp !== null && Number.isFinite(stamp) && age >= 0 && age < AVOID_MS;
    },
    failed() {
      failedAt = now();
      try { getStorage()?.setItem(FAILURE_KEY, String(failedAt)); } catch { /* session-only fallback */ }
    },
  };
}

export const gpuGuard = createGpuGuard();
