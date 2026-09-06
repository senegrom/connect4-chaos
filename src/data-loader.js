/** Bounded data transport, shared by every AI catalog and binary loader. */
export const DATA_LOAD_TIMEOUT_MS = 60_000;
export const CATALOG_LOAD_TIMEOUT_MS = 10_000;

export function abortError() { return new DOMException('Data loading was cancelled.', 'AbortError'); }

export async function readData(url, label, options = {}) {
  const { signal, timeoutMs = DATA_LOAD_TIMEOUT_MS, json = false } = options;
  if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new RangeError('Data timeout must be positive.');
  if (signal?.aborted) throw abortError();
  const target = url instanceof URL ? url : new URL(String(url), import.meta.url);
  const controller = new AbortController();
  let deadline;
  let onAbort;
  const interrupted = new Promise((_, reject) => {
    onAbort = () => { reject(abortError()); controller.abort(); };
    signal?.addEventListener('abort', onAbort, { once: true });
    deadline = setTimeout(() => {
      reject(new Error(`${label} did not finish loading within ${Math.max(1, Math.round(timeoutMs / 1000))} seconds.`));
      controller.abort();
    }, timeoutMs);
  });
  const work = async () => {
    if (target.protocol === 'file:' && typeof process !== 'undefined' && process.versions?.node) {
      const { readFile } = await import('node:fs/promises');
      const bytes = await readFile(target, { signal: controller.signal });
      return json ? JSON.parse(bytes.toString('utf8')) : new Uint8Array(bytes);
    }
    const response = await fetch(target, { signal: controller.signal });
    if (!response.ok) throw new Error(`Could not load ${label.toLowerCase()} (${response.status}).`);
    // The same deadline protects headers AND body consumption. Promise.race
    // also bounds a broken transport that never reacts to AbortController.
    return json ? response.json() : new Uint8Array(await response.arrayBuffer());
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
 * A long-running *search* has no deadline: only its loading phase is bounded. */
export function loadingWatchdog(onTimeout, { signal, timeoutMs = 65_000 } = {}) {
  if (signal?.aborted) return () => {};
  let timer;
  const clear = () => { clearTimeout(timer); signal?.removeEventListener('abort', clear); };
  timer = setTimeout(() => { clear(); onTimeout(); }, timeoutMs);
  signal?.addEventListener('abort', clear, { once: true });
  return clear;
}
