/** Bounded data transport, shared by every AI catalog and binary loader.
 *
 * A load is bounded by silence, not by its length: every chunk that arrives
 * re-arms the deadline, so a slow connection still finishes and a stalled one
 * still fails. A fixed total deadline set a speed below which a large table
 * could never load - the 21 MB Brutal 6x7 Chaos layer needed 2.8 Mbit/s to
 * beat 60 s - and each Retry started again from the first byte. */
export const DATA_LOAD_TIMEOUT_MS = 60_000;
export const CATALOG_LOAD_TIMEOUT_MS = 10_000;

export function abortError() { return new DOMException('Data loading was cancelled.', 'AbortError'); }

async function streamedBytes(response, arrived, onDataProgress, expectedBytes) {
  const reader = response.body?.getReader?.();
  // Without a stream the deadline armed at the headers bounds the whole body.
  if (!reader) return new Uint8Array(await response.arrayBuffer());
  const encoded = Boolean(response.headers?.get?.('content-encoding'));
  const length = Number(response.headers?.get?.('content-length')) || 0;
  const total = expectedBytes > 0 ? expectedBytes : encoded ? 0 : length;
  const chunks = [];
  let loaded = 0;
  try {
    for (;;) {
      // eslint-disable-next-line no-await-in-loop
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      arrived();
      try { onDataProgress?.(loaded, total); } catch { /* progress cannot fail a load */ }
    }
  } catch (error) {
    void reader.cancel().catch(() => {});
    throw error;
  }
  if (chunks.length === 1) return chunks[0];
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

/** Reads url as bytes, or as JSON. `timeoutMs` is the longest silence
 * allowed; `onDataProgress(loaded, total)` hears every chunk, where total is
 * `expectedBytes` when the caller knows the size and 0 when nobody does. */
export async function readData(url, label, options = {}) {
  const {
    signal, timeoutMs = DATA_LOAD_TIMEOUT_MS, json = false, onDataProgress = null, expectedBytes = 0,
  } = options;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new RangeError('Data timeout must be positive.');
  if (signal?.aborted) throw abortError();
  const target = url instanceof URL ? url : new URL(String(url), import.meta.url);
  const controller = new AbortController();
  let deadline;
  let fail;
  const interrupted = new Promise((_, reject) => { fail = reject; });
  const onAbort = () => { fail(abortError()); controller.abort(); };
  signal?.addEventListener('abort', onAbort, { once: true });
  const arrived = () => {
    clearTimeout(deadline);
    deadline = setTimeout(() => {
      fail(new Error(`${label} did not finish loading: nothing arrived for `
        + `${Math.max(1, Math.round(timeoutMs / 1000))} seconds.`));
      controller.abort();
    }, timeoutMs);
  };
  arrived();
  const work = async () => {
    if (target.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
      const { readFile } = await import('node:fs/promises');
      const bytes = await readFile(target, { signal: controller.signal });
      return json ? JSON.parse(bytes.toString('utf8')) : new Uint8Array(bytes);
    }
    const response = await fetch(target, { signal: controller.signal });
    if (!response.ok) throw new Error(`Could not load ${label.toLowerCase()} (${response.status}).`);
    arrived();
    // Catalogs are small: the deadline armed at the headers covers them.
    // Promise.race also bounds a transport that never reacts to the abort.
    return json ? response.json() : streamedBytes(response, arrived, onDataProgress, expectedBytes);
  };
  try { return await Promise.race([work(), interrupted]); }
  finally {
    clearTimeout(deadline);
    signal?.removeEventListener('abort', onAbort);
  }
}

/** Cache successes across requests; pending loads belong to their abort scope. */
export function cachedDataLoad(cache, key, loader, { signal, force = false } = {}) {
  if (signal?.aborted) return Promise.reject(abortError());
  const previous = force ? null : cache.get(key);
  if (previous?.ready) return Promise.resolve(previous.value);
  if (previous && previous.signal === signal && !previous.signal?.aborted) return previous.promise;
  const entry = { signal, ready: false };
  entry.promise = Promise.resolve().then(loader).then((value) => {
    if (signal?.aborted) throw abortError();
    entry.ready = true;
    entry.value = value;
    return value;
  }).catch((error) => {
    if (cache.get(key) === entry) cache.delete(key);
    throw error;
  });
  cache.set(key, entry);
  return entry.promise;
}

/** Lives on the page, so even a worker stalled during decoding can be replaced.
 * Bounded by silence too: the returned stop() ends it, and stop.touch()
 * re-arms it whenever the worker reports that data is still arriving. A
 * long-running *search* has no deadline: only its loading phase is bounded. */
export function loadingWatchdog(onTimeout, { signal, timeoutMs = 65_000 } = {}) {
  let timer;
  let stopped = Boolean(signal?.aborted);
  const stop = () => {
    stopped = true;
    clearTimeout(timer);
    signal?.removeEventListener('abort', stop);
  };
  const arm = () => {
    clearTimeout(timer);
    timer = setTimeout(() => { stop(); onTimeout(); }, timeoutMs);
  };
  stop.touch = () => { if (!stopped) arm(); };
  if (stopped) return stop;
  arm();
  signal?.addEventListener('abort', stop, { once: true });
  return stop;
}
