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
        // Named, so a caller can tell work that ran out of time, and may
        // still be running, from work that failed.
        finish(reject, new DOMException(
          `${label} did not finish within ${Math.round(timeoutMs / 1000)}s`, 'TimeoutError'));
      }, timeoutMs);
    }
  });
}
