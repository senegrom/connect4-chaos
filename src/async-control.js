/** Abort promptly, but dispose a non-cancellable resource if it arrives late. */
export function throwIfAborted(signal) {
  if (signal?.aborted) throw signal.reason ?? new DOMException('Cancelled', 'AbortError');
}

export function releaseResource(resource) {
  try {
    const release = resource?.release ?? resource?.dispose;
    Promise.resolve(release?.call(resource)).catch(() => {});
  } catch {
    // A lost device may already have released the resource.
  }
}

export function waitFor(promise, { signal, timeoutMs, label = 'Operation', onLate } = {}) {
  return new Promise((resolve, reject) => {
    let settled = false;
    let abandoned = false;
    let timer;
    const finish = (callback, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      callback(value);
    };
    const abort = () => {
      abandoned = true;
      finish(reject, signal.reason ?? new DOMException('Cancelled', 'AbortError'));
    };
    Promise.resolve(promise).then((value) => {
      if (abandoned) {
        try { Promise.resolve(onLate?.(value)).catch(() => {}); } catch { /* best effort */ }
      } else finish(resolve, value);
    }, (error) => finish(reject, error));
    if (signal?.aborted) { abort(); return; }
    signal?.addEventListener('abort', abort, { once: true });
    if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
      timer = setTimeout(() => {
        abandoned = true;
        finish(reject, new Error(`${label} did not finish within ${Math.round(timeoutMs / 1000)}s`));
      }, timeoutMs);
    }
  });
}

/** A cancelled attempt can never publish a resource or clear a newer attempt. */
export function createResourceLoader(start, dispose = releaseResource) {
  let attempt = null;
  let resource = null;
  return {
    state: () => resource ? 'ready' : attempt ? 'loading' : 'idle',
    cancel() {
      const previous = attempt;
      attempt = null;
      previous?.controller.abort();
    },
    load({ onProgress } = {}) {
      if (resource) return Promise.resolve(resource);
      if (!attempt) {
        const current = { controller: new AbortController(), listeners: new Set(), promise: null };
        attempt = current;
        const report = (progress) => {
          if (attempt !== current || current.controller.signal.aborted) return;
          for (const listener of current.listeners) listener(progress);
        };
        current.promise = waitFor(Promise.resolve().then(() => {
          throwIfAborted(current.controller.signal);
          return start(current.controller.signal, report);
        }), { signal: current.controller.signal, onLate: dispose }).then((value) => {
          if (current.controller.signal.aborted) { dispose(value); throwIfAborted(current.controller.signal); }
          resource = value;
          return value;
        }).catch((error) => {
          current.controller.abort();
          throw error;
        }).finally(() => {
          current.listeners.clear();
          if (attempt === current) attempt = null;
        });
      }
      const current = attempt;
      if (onProgress) current.listeners.add(onProgress);
      return current.promise.finally(() => current.listeners.delete(onProgress));
    },
  };
}
